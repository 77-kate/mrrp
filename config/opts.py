import argparse
import random
import torch 
import time 
import os 
from utils.utils import setup_seed
from utils.build_criterion import get_criterion_opts
from utils.build_optimizer import get_optimizer_opts
from utils.build_model import get_model_opts
from utils.build_dataloader import get_dataset_opts
from utils.lr_scheduler import get_scheduler_opts

def general_opts(parser):
    group = parser.add_argument_group('General Options')

    group.add_argument('--log-interval', type=int, default=1, help='After how many iterations, we should print logs')
    group.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    group.add_argument('--seed', type=int, default=1, help='dataset split (fold No.)')
    group.add_argument('--config-file', type=str, default=None, help='Config file if exists')
    group.add_argument('--msc-eval', action='store_true', default=False, help='Multi-scale evaluation')
    group.add_argument('--save-dir', type=str, default=None, help="Path to save results")
    group.add_argument('--attnmap-weight-dir', type=str, default="None")
    group.add_argument('--device', type=str, default="cpu")
    group.add_argument('--feat-dir', default=False,action='store_true',help='directory of feature')
    group.add_argument('--finetune', action='store_true', default=False)
    group.add_argument('--external-val', action='store_true', default=False,help='load model from checkpoint and conduct validation experiments using eternal dataset.')

    return parser

def visualization_opts(parser):
    group = parser.add_argument_group('Visualization options')
    group.add_argument('--im-or-file', type=str, required=True, help='Name of the image or list of images in file to be visualized')
    group.add_argument('--is-type-file', action='store_true', default=False, help='Is it a file?')
    group.add_argument('--img-extn-vis', type=str, required=True, help='Image extension without dot (example is png)')
    group.add_argument('--vis-res-dir', type=str, default='results_vis', help='Results after visualization')
    group.add_argument('--no-pt-files', action='store_true', default=False, help='Do not save data using torch.save')
    return parser

def get_opts(parser):
    '''General options'''
    parser = general_opts(parser)
    parser = get_optimizer_opts(parser)
    parser = get_criterion_opts(parser)
    parser = get_model_opts(parser)
    parser = get_dataset_opts(parser)
    parser = get_scheduler_opts(parser)
    return parser

def get_config(is_visualization=False):
    parser = argparse.ArgumentParser(description='M3')
    parser = get_opts(parser)
    if is_visualization:
        parser = visualization_opts(parser)
    args = parser.parse_args()
    setup_seed(42) # args.seed
    #torch.set_num_threads(args.data_workers)
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    
    return args, parser