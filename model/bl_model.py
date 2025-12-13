import torch
import torch.nn as nn
from model.nn_layers.ffn import FFN
from model.nn_layers.attn_layers import *
from typing import Optional
from torch import Tensor
from utils.print_utils import *
import os
import numpy as np
import torch.nn.functional as F
from model.feature_extractors.mnasnet import MNASNet
from model.nn_layers.transformer import *
import pandas as pd

class BLModel(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts
        # self.project_bags = nn.Sequential(nn.Linear(opts.bl_bag_features, opts.bl_out_features), nn.LayerNorm(opts.bl_out_features))
        self.project_bags = nn.Linear(opts.bl_bag_features, opts.bl_out_features)

        # self.attn_over_bags = nn.MultiheadAttention(embed_dim=opts.bl_out_features,
        #                         num_heads=1, dropout=opts.bl_dropout, batch_first=True)
        # self.ffn_b2i = FFN(input_dim=opts.bl_out_features, scale=2, p=opts.bl_dropout)
        # self.attn_fn = nn.Softmax(dim=-1)
        # self.attn_dropout = nn.Dropout(p=opts.bl_attn_dropout)
        # self.bags_weight_fn = nn.Linear(opts.bl_out_features, 1, bias=False)

        ## clinical report feature fusion
        # Molecularsubtype,Clinicalstage,Age
        # if opts.split_file:
        #     split_file = os.path.join(opts.split_file, f'fold_{opts.seed}.csv')
        #     pd_data = pd.read_csv(split_file)
        #     pd_data = pd_data[pd_data["split_info"] == 'train'].reset_index(drop=True)
        # else:
        #     pd_data = pd.read_csv(opts.train_file)
        if self.opts.use_bl_clin:
            self.clinical_fc = nn.Linear(opts.bl_clin_features, opts.bl_out_features)
            self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.bl_out_features, num_heads=1, dropout=opts.bl_dropout, batch_first=True)

        #     clinical_attrs = ['Molecularsubtype', 'Clinicalstage']
        #     self.clinical_attrs_tab = {col: pd_data[col].unique().tolist() for col in clinical_attrs}
        #     self.clinical_embeddings = {col: nn.Embedding(len(pd_data[col].unique()), self.opts.bl_out_features).to(self.opts.device)
        #                                 for col in clinical_attrs}
        #     self.clinical_projectors = {'Age': nn.Linear(1, self.opts.bl_out_features).to(self.opts.device)}
        #     del pd_data

        ## radiomics feature fusion
        if self.opts.use_bl_rad:
            self.radiomics_fc = nn.Linear(opts.bl_omics_features, opts.bl_out_features)
            self.radiomics_image_attn = nn.MultiheadAttention(embed_dim=opts.bl_out_features, num_heads=1, dropout=opts.bl_dropout, batch_first=True)

    def energy_function(self, x, weight_fn, need_attn=False):
        # x: (B, N, C)
        x = weight_fn(x).squeeze(dim=-1) # (B, N)
        energy: Tensor[Optional] = None
        if need_attn:
            energy = x
        x = self.attn_fn(x)
        x = self.attn_dropout(x)
        x = x.unsqueeze(dim=-1) # (B, N, 1)
        return x, energy
   
    def parallel_radiomics_clinical(self, image_from_bags, radiomics_feat, clinical_feat):
        radiomics_image_feat, radiomics_attnmap = self.radiomics_image_attn(key=radiomics_feat,
                        query=image_from_bags.squeeze(dim=1), value=radiomics_feat)
        clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat,
                        query=image_from_bags.squeeze(dim=1), value=clinical_feat)
        image_from_bags = image_from_bags.squeeze(dim=1) \
                                + clinical_image_feat.squeeze(dim=1) \
                                + radiomics_image_feat.squeeze(dim=1)
        self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
        return image_from_bags
    
    def parallel_clinical_img(self, image_from_bags, clinical_feat):
        
        clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat,
                        query=image_from_bags.squeeze(dim=1), value=clinical_feat)
        image_from_bags = image_from_bags.squeeze(dim=1) + clinical_image_feat.squeeze(dim=1)        
        self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
        return image_from_bags
    
    def parallel_radiomics_img(self, image_from_bags, radiomics_feat):
        radiomics_image_feat, radiomics_attnmap = self.radiomics_image_attn(key=radiomics_feat,
                        query=image_from_bags.squeeze(dim=1), value=radiomics_feat)
        image_from_bags = image_from_bags.squeeze(dim=1) + radiomics_image_feat.squeeze(dim=1)

        return image_from_bags

    def series_radiomics_clinical(self, image_from_bags, radiomics_feat, clinical_feat):
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=radiomics_feat)
        image_from_bags = image_from_bags + radiomics_image_feat.squeeze(dim=1) 
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=clinical_feat)
        image_from_bags = image_from_bags + clinical_image_feat.squeeze(dim=1)
        return image_from_bags

    def series_clinical_radiomics(self, image_from_bags, radiomics_feat, clinical_feat):
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=clinical_feat)
        image_from_bags = image_from_bags + clinical_image_feat.squeeze(dim=1)
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=radiomics_feat)
        image_from_bags = image_from_bags + radiomics_image_feat.squeeze(dim=1) 
        return image_from_bags

    def forward(self, batch, *args, **kwargs):
        
        self.info_dict = {
            "id": batch["id"],
        }

        image_from_bags = self.project_bags(batch['feat_bag']) # torch.Size([batch size, 1, 512]) bags_from_attn_words

        # bags_attn, bags_attnmap = self.attn_over_bags(key=bags_from_attn_words, query=bags_from_attn_words, value=bags_from_attn_words)
        # self.info_dict["bags_attnmap"] = bags_attnmap[0].detach().cpu().numpy()
        # bags_energy, bags_energy_unnorm = self.energy_function(bags_attn, self.bags_weight_fn)
        # self.info_dict["bags_weight"] = bags_energy[0, ..., 0].detach().cpu().numpy()
        # image_from_bags = torch.sum(bags_attn * bags_energy, dim=-2)
        # image_from_bags = self.ffn_b2i(image_from_bags)

        if self.opts.use_bl_rad and not self.opts.use_bl_clin:
            rad_flag = batch["bl_rad_flag"].unsqueeze(dim=1).float() # (B, 1)
            radiomics_feat = self.radiomics_fc(batch["bl_radiomics_feat"]) # (B, M, C)
            image_from_bags = self.parallel_radiomics_img(image_from_bags, radiomics_feat)

        elif not self.opts.use_bl_rad and self.opts.use_bl_clin:
            # clinical_feat = torch.stack([
            #     self.clinical_embeddings["Molecularsubtype"](torch.tensor([self.clinical_attrs_tab['Molecularsubtype'].index(d) for d in batch["Molecularsubtype"]]).to(self.opts.device)) ,
            #     self.clinical_embeddings["Clinicalstage"](torch.tensor([self.clinical_attrs_tab['Clinicalstage'].index(d) for d in batch["Clinicalstage"]]).to(self.opts.device)) ,
            #     self.clinical_projectors["Age"](batch["Age"].unsqueeze(dim=1))
            # ], dim=1)
            clinical_feat = self.clinical_fc(batch['bl_clinical_feat'])
            image_from_bags = self.parallel_clinical_img(image_from_bags, clinical_feat)

        elif self.opts.use_bl_rad and self.opts.use_bl_clin:
            rad_flag = batch["bl_rad_flag"].unsqueeze(dim=1).float() # (B, 1)
            radiomics_feat = self.radiomics_fc(batch["bl_radiomics_feat"]) # (B, M, C)
            clinical_feat = self.clinical_fc(batch['bl_clinical_feat'])
            if self.opts.feat_fusion_mode == "parallel": # default
                image_from_bags = self.parallel_radiomics_clinical(image_from_bags, radiomics_feat, clinical_feat)
            elif self.opts.feat_fusion_mode == "series_rc":
                image_from_bags = self.series_radiomics_clinical(image_from_bags, radiomics_feat, clinical_feat)
            elif self.opts.feat_fusion_mode == "series_cr":
                image_from_bags = self.series_clinical_radiomics(image_from_bags, radiomics_feat, clinical_feat)
            else:
                raise NotImplementedError
        
        if self.opts.attnmap_weight_dir not in [None, "None"]:
            npz_dir = os.path.join(self.opts.attnmap_weight_dir, "BL")
            os.makedirs(npz_dir, exist_ok=True)
            if batch["bl_flag"][0].item():
                np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
                        bags_attnmap=self.info_dict["bags_attnmap"],
                        bags_weight=self.info_dict["bags_weight"],
                        clinical_weight=self.info_dict["clinical_weight"],
                        )
        
        return {
            "feat": image_from_bags.squeeze(dim=1)
        }


class BLModel_OnlyHer2(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts
        self.her2_fc = nn.Linear(4, opts.bl_out_features)

    def forward(self, batch, *args, **kwargs):
        return {
            "feat": self.her2_fc(batch["clinical_bl_her2"]),
            "mask_words_pred": None,
            "mask_bags_pred": None,
        }


class BLModelOnly(BLModel):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts, *args, **kwargs)
        self.classifier = nn.Linear(opts.bl_out_features, opts.n_classes)

    def forward(self, batch, *args, **kwargs):
        bl_results = super().forward(batch, *args, **kwargs)
        bl_pred = self.classifier(bl_results["feat"])
        return {
            "pred": bl_pred,
            "feat": bl_results["feat"],
            "mask_words_pred": bl_results["mask_words_pred"],
            "mask_bags_pred": bl_results["mask_bags_pred"],
        }


class BLModelNoneClinicalRadiomics(BLModel):
    def forward(self, batch, *args, **kwargs):
        # if not self.training:
        # return self.incremental_inference(batch)
        words = batch['feat_words']
        bl_flag = batch["clinical_bl_flag"] # (B, 3,)

        # STEP1: Project CNN encoded words
        # (B, N_b, N_w, C, H_w, W_w) --> (B, N_b, N_w, d)
        B, N_b, N_w, C, H, W = words.shape        
        words_cnn = self.cnn(words.view(B*N_b*N_w, C, H, W))
        bl_flag = bl_flag.view(B, 1, 1, 3, 1, 1).repeat(1, N_b, N_w, 1, *words_cnn.shape[-2:])
        bl_flag = bl_flag.view(B*N_b*N_w, 3, *words_cnn.shape[-2:])
        attn_words_cnn = torch.cat([words_cnn, bl_flag], dim=1)
        attn_words_cnn = torch.sigmoid(self.attn_layer(attn_words_cnn))
        words_cnn = words_cnn * attn_words_cnn
        words_cnn = words_cnn.view(B, N_b, N_w, -1)
        words_cnn = self.project_words(words_cnn)

        # STEP2: Words to Bags (Attn words | CNN words)
        words_cnn = words_cnn.view(B*N_b, N_w, -1)

        words_attn, words_attnmap = self.attn_over_words(key=words_cnn, query=words_cnn, value=words_cnn)
        words_attn = words_attn.view(B, N_b, N_w, -1)
        words_attn_energy, words_attn_energy_unnorm = self.energy_function(words_attn, self.words_weight_fn)
        # (B, N_B, N_W, C) * (B, N_B, N_W, 1) --> (B, N_B, C)
        bags_from_attn_words = torch.sum(words_attn * words_attn_energy, dim=-2) 
        bags_from_attn_words = self.ffn_attn_w2b_lst(bags_from_attn_words)

        mask_words_pred = self.words_classifier(words_attn) # (B, N_B, N_W, 6)

        # STEP3: Bags to Image
        bags_attn, bags_attnmap = self.attn_over_bags(key=bags_from_attn_words, query=bags_from_attn_words, value=bags_from_attn_words)
        bags_energy, bags_energy_unnorm = self.energy_function(bags_attn, self.bags_weight_fn)
        image_from_bags = torch.sum(bags_attn * bags_energy, dim=-2)
        image_from_bags = self.ffn_b2i(image_from_bags)

        mask_bags_pred = self.bags_classifier(bags_attn) # (B, N_B, 6)
        
        return {
            "feat": image_from_bags,
            "mask_words_pred": mask_words_pred,
            "mask_bags_pred": mask_bags_pred,
            #"bl_pred": self.bl_classifier(image_from_bags)
        }