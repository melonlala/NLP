#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from pytorch_transformers import BertTokenizer,BertModel
from model.mycrf import CRF


class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.bert = BertClassifier(num_tags= config.num_tags)
        self.output_layer = TaggingFNNDecoder(768, config.num_tags, config.tag_pad_idx) # hidden -> num_tags


    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        utt = batch.utt

        hiddens = self.bert(utt, tag_ids, tag_mask)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)
        
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = prob[0,i,:].cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.2, num_tags=None):
        super(BertClassifier, self).__init__()
        bert_path = '/home/zgjgroup/psy/nlp/chinese-bert-wwm-ext'
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, special_tokens=True)
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, utt, mask, labels=None):
        input_id = []
        with torch.no_grad():
            for u in utt:
                id = []
                for c in u:
                    id.append(self.tokenizer.convert_tokens_to_ids(c))
                id = id + [0] * (50 - len(id))
                input_id.append(torch.tensor(id))
            input_id = torch.stack(input_id).to('cuda:0')
        outputs = self.bert(input_id, token_type_ids=labels.to(dtype=torch.int32), attention_mask=mask)
        seq_output = outputs[0]
        output = self.dropout(seq_output)
        return output

class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        
    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens) 
        prob = self.crf.decode(logits, mask)
        if labels is not None:
            loss = -self.crf.forward(logits, labels, mask)
            return prob, loss
        return (prob, )
