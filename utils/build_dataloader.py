import torch
import numpy as np
from utils.print_utils import *
from torch.utils.data import  WeightedRandomSampler,DataLoader
from data_loader import *

def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_MIL(batch):
    data = {}
    for key in batch[0].keys():
        if torch.is_tensor(batch[0][key]):
            # try:
            data[key] = torch.stack([item[key] for item in batch])
            # except RuntimeError:
            #     print(key,batch[0]['slide_wsi_id'],batch[31]['slide_wsi_id']) 
                # yx_clinical_feat F1105525-1_M4 Z1016172-10_M4; Z1307965-15_M1 Z1535152-12_M4; Z1701334-14_M4 F1201922-1_M4

        elif isinstance(batch[0][key], float) or isinstance(batch[0][key], int):
            data[key] = torch.tensor([item[key] for item in batch])
        else:
            data[key] = [item[key] for item in batch]
    
    return data

def build_data_loader(opts, printer=print):
    split_file = os.path.join(opts.split_file, f'fold_{opts.seed}.csv')
    
    if opts.modal == "blyx":
        train_ds = BLYXDataset(opts, split="train", split_file=split_file)
        test_ds = BLYXDataset(opts, split="test", split_file=split_file)
        
    elif opts.modal == "bl":
        train_ds = BLDataset(opts, split="train", split_file=split_file)
        test_ds = BLDataset(opts, split="test", split_file=split_file)

    elif opts.modal == "yx":
        train_ds = YXDataset(opts, split="train", split_file=split_file)
        test_ds = YXDataset(opts, split="test", split_file=split_file)
    elif opts.modal == "pi":
        train_ds = PIDataset(opts, split="train", split_file=split_file)
        test_ds = PIDataset(opts, split="test", split_file=split_file)
    else:
        print_error_message('Dataset for {} modal not supported yet'.format(opts.modal), printer)
    
    diag_labels = train_ds.RFSCensor
    train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, drop_last=True,
                        pin_memory=True, num_workers=opts.data_workers,
                        worker_init_fn=worker_init_fn, collate_fn=collate_MIL)
    test_dl = DataLoader(test_ds, batch_size=opts.batch_size, shuffle=False, drop_last=True, 
                        pin_memory=True, num_workers=opts.data_workers,  
                        worker_init_fn=worker_init_fn, collate_fn=collate_MIL)

    # compute class-weights for balancing dataset
    diag_classes = len(np.unique(diag_labels))
    if opts.class_weights:
        class_weights = np.histogram(diag_labels, bins=diag_classes)[0] # the number of samples per class
        class_weights = np.array(class_weights) / sum(class_weights) # frequency of each class
        for i in range(diag_classes):
            class_weights[i] = round(np.log(1.0 / class_weights[i]), 5)
    else:
        class_weights = np.ones(diag_classes, dtype=np.float)
    
    return train_dl, test_dl, diag_classes, class_weights

def get_dataset_opts(parser):
    group = parser.add_argument_group('Dataset general details')
    group.add_argument('--dataset', type=str, default='breast', help='Dataset name')
    group.add_argument('--split-file', type=str, default="")
    group.add_argument('--bootstrap-train',action='store_true',default=False,help="Conduct bootstrap training")

    group.add_argument('--bl-bag-feat-dir',type=str, default="",
                       help='directory of bag features') 
    group.add_argument('--bl-rad-file', type=str, default='', help='radiomics feature file')
    group.add_argument('--bl-clin-file', type=str, default='', help='BL_clinical_report feature file')
    group.add_argument('--bl-num-bags', type=int, default=1, help='Number of bags for running')

    group.add_argument('--yx-img-feat-dir',type=str, default='',help='directory of image features')
    group.add_argument('--yx-rad-file', type=str, default='', help='directory of US radiomics features')
    group.add_argument('--yx-clin-file', type=str, default='', help='YX_clinical_report feature file')
    group.add_argument('--yx-num-lesions', type=int, default=1, help='Number of lesions for running')
    
    group.add_argument('--batch-size', type=int, default=32, help='Batch size')
    group.add_argument('--data-workers', type=int, default=2, help='Number of workers for data loading')
    group.add_argument('--class-weights', default=True, help='Compute normalized to address class-imbalance')
    return parser