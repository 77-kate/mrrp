from torch.utils.data import Dataset
import torch 
import h5py
import numpy as np
from data_loader.utils import *
from utils.print_utils import *
import pandas as pd
from copy import deepcopy
import torch.nn as nn
from datetime import datetime
from numpy.random import RandomState

class BLYXDataset(Dataset):
    def __init__(self, opts, split, printer=print, split_file=None):
        super().__init__()

        self.split = split
        self.opts = opts
        self.printer = printer
        self.bl_clin_data = None
        self.yx_radiomics_data = self.yx_clin_data = None
        pd_data = pd.read_csv(split_file)
        if opts.split_file: # given single split file for trian/validation split by 'split_info'
            pd_data = pd_data[pd_data["split_info"] == split].reset_index(drop=True)
            if opts.bootstrap_train and split=="train":
                n_samples = len(pd_data)
                bootstrap_rng = RandomState()
                pd_data = pd_data.sample(n=n_samples, replace=True, random_state=bootstrap_rng).reset_index(drop=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pd_data.to_csv(os.path.join("/home/lh/Ours/data/breast/bootstrap_train",f"{timestamp}.csv"), index=False)
            
        for column in pd_data.columns:
            setattr(self, column, pd_data[column].tolist())
        self.label_lst = self.RFSCensor
        del pd_data
        
        self.bl_bag_features = torch.load(opts.bl_bag_feat_dir, weights_only=False)
        if opts.bl_rad_file.endswith('.csv'):
            self.bl_radiomics_data = pd.read_csv(opts.bl_rad_file, index_col='slide_wsi_id')
        elif opts.bl_rad_file.endswith('.pt'):
            self.bl_radiomics_data = torch.load(opts.bl_rad_file, weights_only=False)

        if opts.bl_clin_file.endswith('.csv'):
            self.bl_clin_data = pd.read_csv(opts.bl_clin_file, index_col='slide_wsi_id')
        elif opts.bl_clin_file.endswith('.pt'):
            self.bl_clin_data = torch.load(opts.bl_clin_file, weights_only=False)

        if self.opts.yx_img_feat_dir.endswith('.csv'):
            self.yx_img_feat = pd.read_csv(opts.yx_img_feat_dir).set_index('Image')
        elif self.opts.yx_img_feat_dir.endswith('.pt'):
            self.yx_img_feat = torch.load(opts.yx_img_feat_dir, weights_only=False)

        if opts.use_yx_rad: 
            if opts.yx_rad_file.endswith('.csv'):
                self.yx_radiomics_data = pd.read_csv(opts.yx_rad_file).set_index('Image')
        if opts.use_yx_clin: 
            if opts.yx_clin_file.endswith('.pt'):
                self.yx_clin_data = torch.load(opts.yx_clin_file, weights_only=False)

        print_log_message('Samples in {}: {}'.format(split_file, len(self.id)), self.printer)
        print_log_message('-- {} ({:.2f}%) Non-return | {} ({:.2f}%) Return | {} ({:.2f}%) Others'.format(
            sum(np.asarray(self.label_lst)==0), 100.0*sum(np.asarray(self.label_lst)==0)/len(self.id),
            sum(np.asarray(self.label_lst)==1), 100.0*sum(np.asarray(self.label_lst)==1)/len(self.id),
            sum(np.asarray(self.label_lst)==-1), 100.0*sum(np.asarray(self.label_lst)==-1)/len(self.id),
        ), self.printer)

    def __len__(self):
        return len(self.id)

    def _load_bl_data(self, index):
        '''
        read the bag-level feature of WSIs and radiomics features according to the split file.
        '''
        wsi_id = self.slide_wsi_id[index]
        if str(wsi_id) != "nan": 
            bag_feature = self.bl_bag_features[wsi_id]
            # bag_feature = torch.FloatTensor(np.nan_to_num(np.asarray(bag_feature, dtype=np.float32), 0.0)).unsqueeze(dim=0)
            flag = 1

            try:
                if self.opts.bl_rad_file.endswith('.csv'): rad_feat = self.bl_radiomics_data.loc[wsi_id].values
                elif self.opts.bl_rad_file.endswith('.pt'): rad_feat = self.bl_radiomics_data[wsi_id].squeeze()
                bl_rad_flag = 1
            except KeyError:
                rad_feat = torch.zeros(self.opts.bl_omics_features).float()
                bl_rad_flag = 0
            radiomics_feat = torch.FloatTensor(np.nan_to_num(np.asarray(rad_feat, dtype=np.float32), 0.0))

            clinical_feat = torch.zeros(self.opts.bl_clin_features).float()
            if self.bl_clin_data:
                if self.opts.bl_clin_file.endswith('.csv'): 
                    clin_feat = self.bl_clin_data.loc[wsi_id].drop(['silde_id', 'id', 'case_wsi_id', 'slide_wsi_id']).values
                elif self.opts.bl_clin_file.endswith('.pt'): 
                    clin_feat = self.bl_clin_data[wsi_id].squeeze()
                clinical_feat = torch.FloatTensor(np.nan_to_num(np.asarray(clin_feat, dtype=np.float32), 0.0))  
        else:
            flag = 0
            bag_feature = torch.zeros(1, self.opts.bl_bag_features).float()
            radiomics_feat = torch.zeros(self.opts.bl_omics_features).float()
            clinical_feat = torch.zeros(self.opts.bl_clin_features).float()
            bl_rad_flag = 0
            
        return bag_feature, radiomics_feat, clinical_feat, flag, bl_rad_flag

    def _load_yx_data(self, index):
        us_id = self.slide_us_id[index]
        if str(us_id) != str(np.nan):
            flag = 1
            if self.opts.yx_img_feat_dir.endswith('.csv'):
                target_row = self.yx_img_feat.loc[us_id+'.jpg']
                feat_lesions = torch.tensor(np.nan_to_num(np.asarray(target_row.values, dtype=np.float32), 0.0)).float()
            elif self.opts.yx_img_feat_dir.endswith('.pt'):
                try:
                    feat_lesions = self.yx_img_feat[str(self.case_us_id[index])]
                    feat_lesions = torch.tensor(np.nan_to_num(np.asarray(feat_lesions, dtype=np.float32), 0.0)).float()
                except KeyError as e:
                    flag = 0 # US image feature not exist
                    feat_lesions = torch.zeros(self.opts.yx_cnn_features).float()
            else:
                file = os.path.join(self.opts.yx_img_feat_dir, us_id+'.npy')
                feat_lesions = torch.FloatTensor(np.nan_to_num(np.asarray(np.load(file), dtype=np.float32), 0.0)).unsqueeze(dim=0)
            # feat_lesions: torch.Size([1, 64, 16, 16])
            
            radiomics_feat = torch.zeros(self.opts.yx_omics_features).float()
            yx_clinical_feat = torch.zeros(self.opts.yx_clin_features).float()

            if self.opts.use_yx_rad:
                if self.opts.yx_rad_file.endswith('.csv'):
                    target_row = self.yx_radiomics_data.loc[us_id+'.jpg']
                    radiomics_feat = torch.tensor(np.nan_to_num(np.asarray(target_row.values, dtype=np.float32), 0.0)).float()
                else:
                    file = os.path.join(self.opts.yx_rad_file, us_id+'.npy')
                    radiomics_feat = torch.FloatTensor(np.nan_to_num(np.asarray(np.load(file), dtype=np.float32), 0.0)).unsqueeze(dim=0)

            yx_clin_flag = 1
            if self.opts.use_yx_clin:
                if self.opts.yx_clin_file.endswith('.pt'):
                    try:
                        yx_clin_data = self.yx_clin_data[str(self.case_us_id[index])]
                        yx_clinical_feat = torch.tensor(np.nan_to_num(np.asarray(yx_clin_data, dtype=np.float32), 0.0)).float()
                    except KeyError as e:
                        yx_clin_flag = 0 # yx_clinical_feat is synthetic
                else:
                    file = os.path.join(self.opts.yx_clin_file, us_id+'.npy')
                    yx_clinical_feat = torch.FloatTensor(np.nan_to_num(np.asarray(np.load(file), dtype=np.float32), 0.0)).unsqueeze(dim=0)

        else:
            flag = 0 # US image feature not exist
            feat_lesions = torch.zeros(1, self.opts.yx_cnn_features).float()
            radiomics_feat = torch.zeros(self.opts.yx_omics_features).float()
            yx_clinical_feat = torch.zeros(self.opts.yx_clin_features).float()
            yx_clin_flag = 0
        
        return feat_lesions, radiomics_feat, yx_clinical_feat, flag, yx_clin_flag

    def __getitem__(self, index):
        bag_feature, bl_radiomics_feat, bl_clinical_feat, bl_flag, bl_rad_flag = self._load_bl_data(index)
        feat_lesions, yx_radiomics_feat, yx_clinical_feat, yx_flag, yx_clin_flag = self._load_yx_data(index)

        #print(self.yx_pid_lst[index], self.bl_pid_lst[index])
        # assert bl_flag or yx_flag, (self.id_lst[index], self.yx_pid_lst[index], self.bl_pid_lst[index])
        pid = self.id[index]
        return {
            "id": pid,
            'slide_wsi_id':self.slide_wsi_id[index],
            'slide_us_id':self.slide_us_id[index],

            "feat_bag": bag_feature,
            "bl_radiomics_feat": bl_radiomics_feat,
            "bl_rad_flag": bl_rad_flag,
            "bl_clinical_feat": bl_clinical_feat,
            "bl_flag": bl_flag, # if zero, the img feature is missing

            "feat_lesions": feat_lesions,
            "yx_radiomics_feat": yx_radiomics_feat,
            "yx_clinical_feat": yx_clinical_feat,
            "yx_flag": yx_flag, # if zero, the img feature is missing
            "yx_clin_flag": yx_clin_flag,

            "rfs": self.RFS_day[index],
            "label": self.RFSCensor[index],

            # patient information: LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            "LVI": self.LVI[index],
            "HER2": self.HER2[index],
            "Molecularsubtype": self.Molecularsubtype[index],
            "Tumorsize": self.Tumorsize[index],
            "LNstatus": self.LNstatus[index],
            "Clinicalstage": self.Clinicalstage[index],
            "Grade": self.Grade[index],
            "Age": self.Age[index],
            "ER": self.ER[index],
            "PR": self.PR[index],
            "Ki67": self.Ki67[index]
            # "clinical_LVI": self.clinical_embeddings[pid]['LVI'],
            # "clinical_HER2": self.clinical_embeddings[pid]['HER2'],
            # "clinical_Molecularsubtype": self.clinical_embeddings[pid]['Molecularsubtype'],
            # "clinical_Tumorsize": self.clinical_embeddings[pid]['Tumorsize'],
            # "clinical_LNstatus": self.clinical_embeddings[pid]['LNstatus'],
            # "clinical_Clinicalstage": self.clinical_embeddings[pid]['Clinicalstage'],
            # "clinical_Grade": self.clinical_embeddings[pid]['Grade'],
            # "clinical_Age": self.clinical_embeddings[pid]['Age'],
            # "clinical_ER": self.clinical_embeddings[pid]['ER'],
            # "clinical_PR": self.clinical_embeddings[pid]['PR'],
            # "clinical_Ki67": self.clinical_embeddings[pid]['Ki67']
        }
    
class BLDataset(Dataset):
    def __init__(self, opts, split, printer=print, split_file=None):
        super().__init__()

        self.split = split
        self.opts = opts
        self.printer = printer
        self.bl_clin_data = None

        pd_data = pd.read_csv(split_file)
        if opts.split_file: # given single split file for trian/validation split by 'split_info'
            pd_data = pd_data[pd_data["split_info"] == split].reset_index(drop=True)
        for column in pd_data.columns:
            setattr(self, column, pd_data[column].tolist())
        self.label_lst = self.RFSCensor
        del pd_data
        
        self.bl_bag_features = torch.load(self.opts.bl_bag_feat_dir, weights_only=False)

        if opts.bl_rad_file.endswith('.csv'):
            self.bl_radiomics_data = pd.read_csv(opts.bl_rad_file, index_col='slide_wsi_id')
        elif opts.bl_rad_file.endswith('.pt'):
            self.bl_radiomics_data = torch.load(opts.bl_rad_file, weights_only=False)
            
        if opts.bl_clin_file.endswith('.csv'):
            self.bl_clin_data = pd.read_csv(opts.bl_clin_file, index_col='slide_wsi_id')
        elif opts.bl_clin_file.endswith('.pt'):
            self.bl_clin_data = torch.load(opts.bl_clin_file, weights_only=False)
        
        print_log_message('Samples in {}: {}'.format(split_file, len(self.id)), self.printer)
        print_log_message('-- {} ({:.2f}%) Non-return | {} ({:.2f}%) Return | {} ({:.2f}%) Others'.format(
            sum(np.asarray(self.label_lst)==0), 100.0*sum(np.asarray(self.label_lst)==0)/len(self.id),
            sum(np.asarray(self.label_lst)==1), 100.0*sum(np.asarray(self.label_lst)==1)/len(self.id),
            sum(np.asarray(self.label_lst)==-1), 100.0*sum(np.asarray(self.label_lst)==-1)/len(self.id),
        ), self.printer)

    def __len__(self):
        return len(self.id)

    def _load_bl_data(self, index):
        '''
        read the bag-level feature of WSIs and radiomics features according to the split file.
        '''
        wsi_id = self.slide_wsi_id[index]
        if wsi_id != str(np.nan): 
            bag_feature = self.bl_bag_features[wsi_id]
            flag = 1

            try:
                if self.opts.bl_rad_file.endswith('.csv'): rad_feat = self.bl_radiomics_data.loc[wsi_id].values
                elif self.opts.bl_rad_file.endswith('.pt'): rad_feat = self.bl_radiomics_data[wsi_id].squeeze()
                bl_rad_flag = 1
            except KeyError:
                rad_feat = torch.zeros(self.opts.bl_omics_features).float()
                bl_rad_flag = 0
            radiomics_feat = torch.FloatTensor(np.nan_to_num(np.asarray(rad_feat, dtype=np.float32), 0.0)).float()

            clinical_feat = torch.zeros(self.opts.bl_clin_features).float()
            if self.bl_clin_data:
                if self.opts.bl_clin_file.endswith('.csv'): 
                    clin_feat = self.bl_clin_data.loc[wsi_id].drop(['silde_id', 'id', 'case_wsi_id', 'slide_wsi_id']).values
                elif self.opts.bl_clin_file.endswith('.pt'): 
                    clin_feat = self.bl_clin_data[wsi_id].squeeze()
                clinical_feat = torch.FloatTensor(np.nan_to_num(np.asarray(clin_feat, dtype=np.float32), 0.0)).float()
        else:
            flag = 0
            bag_feature = torch.zeros(1, self.opts.bl_bag_features).float()
            radiomics_feat = torch.zeros(self.opts.bl_omics_features).float()
            clinical_feat = torch.zeros(self.opts.bl_clin_features).float()
            bl_rad_flag = 0
            
        return bag_feature, radiomics_feat, clinical_feat, flag, bl_rad_flag

    def __getitem__(self, index):
        bag_feature, bl_radiomics_feat, bl_clinical_feat, bl_flag, bl_rad_flag = self._load_bl_data(index)
        pid = self.id[index]

        return {
            "id": pid,
            'slide_wsi_id':self.slide_wsi_id[index],
            'slide_us_id':self.slide_us_id[index],

            "feat_bag": bag_feature,
            "bl_radiomics_feat": bl_radiomics_feat,
            "bl_rad_flag": bl_rad_flag,
            "bl_clinical_feat": bl_clinical_feat,
            "bl_flag": bl_flag, # if zero, the feature is missing

            "rfs": self.RFS_day[index],
            "label": self.RFSCensor[index],

            # patient information: LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            "LVI": self.LVI[index],
            "HER2": self.HER2[index],
            "Molecularsubtype": self.Molecularsubtype[index],
            "Tumorsize": self.Tumorsize[index],
            "LNstatus": self.LNstatus[index],
            "Clinicalstage": self.Clinicalstage[index],
            "Grade": self.Grade[index],
            "Age": self.Age[index],
            "ER": self.ER[index],
            "PR": self.PR[index],
            "Ki67": self.Ki67[index]
        }


class YXDataset(Dataset):
    def __init__(self, opts, split, printer=print, split_file=None):
        super().__init__()

        self.split = split
        self.opts = opts
        self.printer = printer
        self.yx_radiomics_data = self.yx_clin_data = None

        pd_data = pd.read_csv(split_file)
        if opts.split_file: # given single split file for trian/validation split by 'split_info'
            pd_data = pd_data[pd_data["split_info"] == split].reset_index(drop=True)
        for column in pd_data.columns:
            setattr(self, column, pd_data[column].tolist())
        self.label_lst = self.RFSCensor
        del pd_data

        if self.opts.yx_img_feat_dir.endswith('.csv'):
            self.yx_img_data = pd.read_csv(self.opts.yx_img_feat_dir).set_index('Image')
        elif self.opts.yx_img_feat_dir.endswith('.pt'):
            self.yx_img_data = torch.load(self.opts.yx_img_feat_dir, weights_only=False)

        if self.opts.use_yx_rad: 
            if opts.yx_rad_file.endswith('.csv'):
                self.yx_radiomics_data = pd.read_csv(self.opts.yx_rad_file).set_index('Image')
        if self.opts.use_yx_clin: 
            if opts.yx_clin_file.endswith('.pt'):
                self.yx_clin_data = torch.load(self.opts.yx_clin_file, weights_only=False)

        print_log_message('Samples in {}: {}'.format(split_file, len(self.id)), self.printer)
        print_log_message('-- {} ({:.2f}%) Non-return | {} ({:.2f}%) Return | {} ({:.2f}%) Others'.format(
            sum(np.asarray(self.label_lst)==0), 100.0*sum(np.asarray(self.label_lst)==0)/len(self.id),
            sum(np.asarray(self.label_lst)==1), 100.0*sum(np.asarray(self.label_lst)==1)/len(self.id),
            sum(np.asarray(self.label_lst)==-1), 100.0*sum(np.asarray(self.label_lst)==-1)/len(self.id),
        ), self.printer)

    def __len__(self):
        return len(self.id)

    def _load_yx_data(self, index):
        us_id = self.slide_us_id[index]
        if us_id != str(np.nan):
            flag = 1
            if self.opts.yx_img_feat_dir.endswith('.csv'):
                target_row = self.yx_img_data.loc[us_id+'.jpg']
                feat_lesions = torch.tensor(np.nan_to_num(np.asarray(target_row.values, dtype=np.float32), 0.0)).float()
            elif self.opts.yx_img_feat_dir.endswith('.pt'):
                # yx_img_data = self.yx_img_data[str(self.case_us_id[index])]
                # feat_lesions = torch.tensor(np.nan_to_num(np.asarray(yx_img_data, dtype=np.float32), 0.0)).float()
                try:
                    feat_lesions = self.yx_img_data[str(self.case_us_id[index])]
                    feat_lesions = torch.tensor(np.nan_to_num(np.asarray(feat_lesions, dtype=np.float32), 0.0)).float()
                except KeyError as e:
                    flag = 0 # US image feature not exist
                    feat_lesions = torch.zeros(self.opts.yx_cnn_features).float()
            else:
                file = os.path.join(self.opts.yx_img_feat_dir, us_id+'.npy')
                feat_lesions = torch.FloatTensor(np.nan_to_num(np.asarray(np.load(file), dtype=np.float32), 0.0)).unsqueeze(dim=0)
            
            
            radiomics_feat = torch.zeros(self.opts.yx_omics_features).float()
            yx_clinical_feat = torch.zeros(self.opts.yx_clin_features).float()
            if self.opts.use_yx_rad: 
                if self.opts.yx_rad_file.endswith('.csv'):
                    target_row = self.yx_radiomics_data.loc[us_id+'.jpg']
                    radiomics_feat = torch.tensor(np.nan_to_num(np.asarray(target_row.values, dtype=np.float32), 0.0)).float()
                else:
                    file = os.path.join(self.opts.yx_rad_file, us_id+'.npy')
                    radiomics_feat = torch.FloatTensor(np.nan_to_num(np.asarray(np.load(file), dtype=np.float32), 0.0)).unsqueeze(dim=0)

            yx_clin_flag = 1
            if self.opts.use_yx_clin:
                if self.opts.yx_clin_file.endswith('.pt'):
                    try:
                        yx_clin_data = self.yx_clin_data[str(self.case_us_id[index])]
                        yx_clinical_feat = torch.tensor(np.nan_to_num(np.asarray(yx_clin_data, dtype=np.float32), 0.0)).float()
                    except KeyError as e:
                        yx_clin_flag = 0 # yx_clinical_feat is synthetic
                else:
                    file = os.path.join(self.opts.yx_clin_file, us_id+'.npy')
                    yx_clinical_feat = torch.FloatTensor(np.nan_to_num(np.asarray(np.load(file), dtype=np.float32), 0.0)).unsqueeze(dim=0)

                
        else:
            flag = 0 # US image feature not exist
            feat_lesions = torch.zeros(1, self.opts.yx_cnn_features).float()
            radiomics_feat = torch.zeros(self.opts.yx_omics_features).float()
            yx_clinical_feat = torch.zeros(self.opts.yx_clin_features).float()
            yx_clin_flag = 0
        
        return feat_lesions, radiomics_feat, yx_clinical_feat, flag, yx_clin_flag

    def __getitem__(self, index):
        feat_lesions, yx_radiomics_feat, yx_clinical_feat, yx_flag, yx_clin_flag = self._load_yx_data(index)

        pid = self.id[index]
        return {
            "id": pid,
            'slide_wsi_id':self.slide_wsi_id[index],
            'slide_us_id':self.slide_us_id[index],

            "feat_lesions": feat_lesions,
            "yx_radiomics_feat": yx_radiomics_feat,
            "yx_clinical_feat": yx_clinical_feat,
            "yx_flag": yx_flag,
            "yx_clin_flag": yx_clin_flag,

            "rfs": self.RFS_day[index],
            "label": self.RFSCensor[index],

            # patient information: LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            "LVI": self.LVI[index],
            "HER2": self.HER2[index],
            "Molecularsubtype": self.Molecularsubtype[index],
            "Tumorsize": self.Tumorsize[index],
            "LNstatus": self.LNstatus[index],
            "Clinicalstage": self.Clinicalstage[index],
            "Grade": self.Grade[index],
            "Age": self.Age[index],
            "ER": self.ER[index],
            "PR": self.PR[index],
            "Ki67": self.Ki67[index]
        }

class PIDataset(Dataset):
    def __init__(self, opts, split, printer=print, split_file=None):
        super().__init__()

        self.split = split
        self.opts = opts
        self.printer = printer
        pd_data = pd.read_csv(split_file)
        if opts.split_file: # given single split file for trian/validation split by 'split_info'
            pd_data = pd_data[pd_data["split_info"] == split].reset_index(drop=True)
            
        for column in pd_data.columns:
            setattr(self, column, pd_data[column].tolist())
        self.label_lst = self.RFSCensor
        del pd_data
        
        print_log_message('Samples in {}: {}'.format(split_file, len(self.id)), self.printer)
        print_log_message('-- {} ({:.2f}%) Non-return | {} ({:.2f}%) Return | {} ({:.2f}%) Others'.format(
            sum(np.asarray(self.label_lst)==0), 100.0*sum(np.asarray(self.label_lst)==0)/len(self.id),
            sum(np.asarray(self.label_lst)==1), 100.0*sum(np.asarray(self.label_lst)==1)/len(self.id),
            sum(np.asarray(self.label_lst)==-1), 100.0*sum(np.asarray(self.label_lst)==-1)/len(self.id),
        ), self.printer)

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        
        pid = self.id[index]
        return {
            "id": pid,
            'slide_wsi_id':self.slide_wsi_id[index],
            'slide_us_id':self.slide_us_id[index],

            "rfs": self.RFS_day[index],
            "label": self.RFSCensor[index],

            # patient information: LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            "LVI": self.LVI[index],
            "HER2": self.HER2[index],
            "Molecularsubtype": self.Molecularsubtype[index],
            "Tumorsize": self.Tumorsize[index],
            "LNstatus": self.LNstatus[index],
            "Clinicalstage": self.Clinicalstage[index],
            "Grade": self.Grade[index],
            "Age": self.Age[index],
            "ER": self.ER[index],
            "PR": self.PR[index],
            "Ki67": self.Ki67[index]
        }
 