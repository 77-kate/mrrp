import json
import numpy as np
from utils.print_utils import *
import argparse
import glob 
import torch 
import random
import time 
import logging
import matplotlib.pyplot as plt 
import os 

def setup_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # setting environment variables

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    
    # use deterministic algorithms and disable the cuDNN to ensure the reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False

def build_logging(filename):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('PIL').setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class ColorEncoder(object):
    def __init__(self):
        super().__init__()

    def get_colors(self, dataset_name):
        if dataset_name == 'bingli':
            class_colors = [
                (228/ 255.0, 26/ 255.0, 28/ 255.0),
                (55/ 255.0, 126/ 255.0, 184/ 255.0),
                #(77/ 255.0, 175/ 255.0, 74/ 255.0),
                #(152/ 255.0, 78/ 255.0, 163/ 255.0)
            ]

            class_linestyle = ['solid', 'solid']

            return class_colors, class_linestyle
        else:
            raise NotImplementedError

class DictWriter(object):
    def __init__(self, file_name, format='csv'):
        super().__init__()
        assert format in ['csv', 'json', 'txt']

        self.file_name = '{}.{}'.format(file_name, format)
        self.format = format

    def write(self, data_dict: dict):
        if self.format == 'csv':
            import csv
            with open(self.file_name, 'w', newline="") as csv_file:
                writer = csv.writer(csv_file)
                for key, value in data_dict.items():
                    writer.writerow([key, value])
        elif self.format == 'json':
            import json
            with open(self.file_name, 'w') as fp:
                json.dump(data_dict, fp, indent=4, sort_keys=True)
        else:
            with open(self.file_name, 'w') as txt_file:
                for key, value in data_dict.items():
                    line = '{} : {}\n'.format(key, value)
                    txt_file.write(line)

def save_checkpoint(epoch, model_state, optimizer_state, save_dir, is_best=False, is_last=False, printer=print):
    # save the last epoch checkpoint
    if is_last and is_best:
        model_fname = '{}/model_last_best_epo{:0d}.pth'.format(save_dir,epoch)
    elif is_last:
        model_fname = '{}/model_last_epo{:0d}.pth'.format(save_dir,epoch)
    # save the best checkpoint
    else:
        model_fname = '{}/model_best.pth'.format(save_dir,epoch)

    torch.save(model_state, model_fname)
    print_info_message('Checkpoint saved at: {}'.format(model_fname), printer)
    # print_log_message(f'Saved Model state\n{torch.load(model_fname)}', printer=printer) # checked Right
    return model_fname

def load_checkpoint(ckpt_fname, device='cpu'):
    model_state = torch.load(ckpt_fname, map_location=device)
    return model_state

def save_arguments(args, save_loc, json_file_name='arguments.json', printer=print):
    argparse_dict = vars(args)
    arg_fname = '{}/{}'.format(save_loc, json_file_name)
    writer = DictWriter(file_name=arg_fname, format='json')
    writer.write(argparse_dict)
    print_log_message('Arguments are dumped here: {}'.format(arg_fname), printer)


def load_arguments(parser, dumped_arg_loc, json_file_name='arguments.json'):
    arg_fname = '{}/{}'.format(dumped_arg_loc, json_file_name)
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    with open(arg_fname, 'r') as fp:
        json_dict = json.load(fp)
        parser.set_defaults(**json_dict)

        updated_args = parser.parse_args()

    return updated_args


def load_arguments_file(parser, arg_fname):
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    with open(arg_fname, 'r') as fp:
        json_dict = json.load(fp)
        parser.set_defaults(**json_dict)
        updated_args = parser.parse_args()

    return updated_args

def plot_results(res_dict, plot_file, prekey=0):
    train_res = [key for key in res_dict.keys() if 'Training' in key]
    valid_res = [key for key in res_dict.keys() if 'Test' in key]
    plot_results_helper(res_dict, train_res, plot_file+'_train.jpg', prekey)
    plot_results_helper(res_dict, valid_res, plot_file+'_test.jpg', prekey)
    
def plot_results_helper(res_dict, res_lst, plot_file, prekey=0):
    N = len(res_lst)
    nc = N//2 if N%2==0 else (N//2+1)
    _, axarr = plt.subplots(2, nc, figsize=(5*nc, 5))
    # _, axarr = plt.subplots(3, 1, figsize=(5*nc, 5))

    axarr = axarr.flatten()
    for i, key in enumerate(res_lst):
        axarr[i].plot(np.array(range(len(res_dict[key])))+prekey, res_dict[key], label=key)
        axarr[i].legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()