#coding=utf-8
import argparse
import sys
from typing import NoReturn
class Arguments:
    '''Just for type hints. Don't initialize it.'''

    def __init__(self) -> NoReturn:
        raise NotImplementedError()

    device: str
    lr: float
    max_epoch: int
    batch_size: int
    seed: int
    num_layer: int
    noise: bool


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default='0', help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--noise', type=int, default='1')
    arg_parser.add_argument('--anti_noise', action='store_true')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', default='GRU', choices=['LSTM', 'GRU', 'RNN'], help='root of data')
    arg_parser.add_argument('--rnn', default='GRU', choices=['LSTM', 'GRU', 'RNN'], help='type of rnn')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    arg_parser.add_argument('--num_layer', default=1, type=int, help='number of layer')
    arg_parser.add_argument('--num_head', default=8, type=int, help='number of head')
    arg_parser.add_argument('--use_crf', action='store_true', help='use crf or not')
    arg_parser.add_argument('--model', type=str, default='transformer', choices=['baseline', 'transformer', 'bert', 'bilstm_crf'], help='model type')
    return arg_parser

arguments  = init_args()