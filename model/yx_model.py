import torch
import torch.nn as nn
from torch.nn import init
from model.nn_layers.ffn import FFN
from model.nn_layers.attn_layers import *
from typing import NamedTuple, Optional
from torch import Tensor
from torchvision import models
import torch.nn.functional as F
from copy import deepcopy
from model.feature_extractors.mnasnet import MNASNet
from model.nn_layers.transformer import *

class YXModel(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts

        # if opts.yx_cnn_name == "mnasnet":
        #     s = 1.0
        #     weight = 'checkpoints/mnasnet_s_1.0_imagenet_224x224.pth'
        #     backbone = MNASNet(alpha=s)
        #     pretrained_dict = torch.load(weight, map_location=torch.device('cpu'))
        #     backbone.load_state_dict(pretrained_dict)
        #     del backbone.classifier
        #     self.cnn = backbone
        # elif opts.yx_cnn_name is None:
        #     self.cnn = None
        # else:
        #     backbone = eval(f"models.{opts.yx_cnn_name}")(pretrained=opts.yx_cnn_pretrained)
        #     # if opts.yx_cnn_name == "mnasnet1_0":
        #     if "mnasnet" in opts.yx_cnn_name:
        #         self.cnn = nn.Sequential(*[*list(backbone.children())[:-1], nn.AdaptiveAvgPool2d(1)])
        #     else:
        #         self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        # self.attn_layer = nn.Conv2d(opts.yx_cnn_features, 1, kernel_size=1, padding=0, stride=1)

        self.cnn_project = nn.Linear(opts.yx_cnn_features, opts.yx_out_features)

        # self.attn_over_lesions = nn.MultiheadAttention(embed_dim=opts.yx_out_features,
        #                         num_heads=1, dropout=opts.yx_dropout, batch_first=True)
        # self.ffn_attn_l2p = FFN(input_dim=opts.yx_out_features, scale=2, p=opts.yx_dropout)
        # self.lesions_weight_fn = nn.Linear(opts.yx_out_features, 1, bias=False)
        # self.attn_dropout = nn.Dropout(p=opts.yx_attn_dropout)
        # self.attn_fn = nn.Softmax(dim=-1)

        # self.lesions_classifier = nn.Linear(opts.yx_out_features, 5)

        # clinical feature fusion (US report feature)
        if opts.use_yx_clin:
            self.clinical_param = nn.Parameter(torch.zeros(1, opts.yx_out_features))
            self.clinical_fc = nn.Linear(opts.yx_clin_features, opts.yx_out_features)
            self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.yx_out_features, num_heads=1, dropout=opts.yx_dropout, batch_first=True)

        # readiomics feature fusion
        if opts.use_yx_rad:
            self.radiomics_fc = nn.Linear(opts.yx_omics_features, opts.yx_out_features)
            self.radiomics_image_attn = nn.MultiheadAttention(embed_dim=opts.yx_out_features, num_heads=1, dropout=opts.yx_dropout, batch_first=True)

        #self.yx_classifier = nn.Linear(opts.yx_out_features, opts.n_classes)

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

    def parallel_radiomics_clinical(self, patient_from_lesions, radiomics_feat, clinical_feat):
        radiomics_image_feat, radiomics_attnmap = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions, value=radiomics_feat)
        clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions, value=clinical_feat)
        patient_from_lesions = patient_from_lesions \
                                + clinical_image_feat \
                                + radiomics_image_feat

        self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
        return patient_from_lesions

    def parallel_clinical_img(self, patient_from_lesions, clinical_feat):
        clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions, value=clinical_feat)
        patient_from_lesions = patient_from_lesions + clinical_image_feat
        self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
        return patient_from_lesions
    
    def parallel_radiomics_img(self, patient_from_lesions, radiomics_feat):
        radiomics_image_feat, radiomics_attnmap = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions, value=radiomics_feat)
        patient_from_lesions = patient_from_lesions + radiomics_image_feat

        return patient_from_lesions

    def series_radiomics_clinical(self, patient_from_lesions, radiomics_feat, clinical_feat):
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions, value=radiomics_feat)
        patient_from_lesions = patient_from_lesions + radiomics_image_feat 
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions, value=clinical_feat)
        patient_from_lesions = patient_from_lesions + clinical_image_feat
        return patient_from_lesions

    def series_clinical_radiomics(self, patient_from_lesions, radiomics_feat, clinical_feat):
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions, value=clinical_feat)
        patient_from_lesions = patient_from_lesions + clinical_image_feat
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions, value=radiomics_feat)
        patient_from_lesions = patient_from_lesions + radiomics_image_feat 
        return patient_from_lesions

    def forward(self, batch, *args, **kwargs):
        lesions = batch["feat_lesions"]
        # print(" batch[feat_lesions]: ", lesions.shape) # torch.Size([32, 1, 64, 16, 16])
        if len(lesions.shape) == 5:
            B, N_l, C, H, W = lesions.shape
            patient_from_lesions = self.cnn_project(lesions.view(B, N_l, -1)).squeeze(dim=1)
        else:
            patient_from_lesions = self.cnn_project(lesions).squeeze(dim=1)

        # B, N_l, C, H, W = lesions.shape       
        # lesions_cnn = lesions.view(B*N_l, C, H, W)
        # attn_lesions_cnn = torch.sigmoid(self.attn_layer(lesions_cnn)) # attn_lesions_cnn
        # lesions_cnn = lesions_cnn * attn_lesions_cnn
        # lesions_cnn = lesions_cnn.view(B, N_l, -1)
        # lesions_cnn = self.cnn_project(lesions_cnn)

        self.info_dict = {"id": batch["id"]}

        # lesions_attn, lesions_attnmap = self.attn_over_lesions(key=lesions_cnn, query=lesions_cnn, value=lesions_cnn)
        # self.info_dict["lesions_attnmap"] = lesions_attnmap[0].detach().cpu().numpy()
        # lesions_attn_energy, lesions_attn_energy_unnorm = self.energy_function(lesions_attn, self.lesions_weight_fn)
        # # (B, N_l, C) x (B, N_l, 1) --> (B, C)
        # self.info_dict["lesions_weight"] = lesions_attn_energy[0, ..., 0].detach().cpu().numpy()
        # patient_from_lesions = torch.sum(lesions_attn * lesions_attn_energy, dim=1)
        # patient_from_lesions = self.ffn_attn_l2p(patient_from_lesions)
        # # lesions_pred = self.lesions_classifier(lesions_attn)
        # self.lesions_attn_energy_unnorm = lesions_attn_energy_unnorm

        if self.opts.use_yx_rad and not self.opts.use_yx_clin:

            yx_radiomics_feat = batch["yx_radiomics_feat"]
            if len(yx_radiomics_feat.shape) == 5:
                yx_radiomics_feat = yx_radiomics_feat.view(self.opts.batch_size, -1)
            radiomics_feat = self.radiomics_fc(yx_radiomics_feat) # (B, yx_omics_features) --> (B, yx_out_features)
            patient_from_lesions = self.parallel_radiomics_img(patient_from_lesions, radiomics_feat)

        elif not self.opts.use_yx_rad and self.opts.use_yx_clin:
            yx_clinical_feat = batch["yx_clinical_feat"]
            if len(yx_clinical_feat.shape) == 5:
                yx_clinical_feat = yx_clinical_feat.view(self.opts.batch_size, -1)
            clin_flag = batch["yx_clin_flag"].unsqueeze(dim=1).float() # (B, 1)
            clinical_feat = self.clinical_fc(yx_clinical_feat) # (B, C)
            clin_syn_feat = self.clinical_param.repeat(clinical_feat.shape[0], 1) # (B, C) zero matrix, synthetic yx clinical feature when yx_flag=0
            clinical_feat = clinical_feat * clin_flag + clin_syn_feat * (1.0 - clin_flag) # (B, C)

            patient_from_lesions = self.parallel_clinical_img(patient_from_lesions, clinical_feat)

        elif self.opts.use_yx_rad and self.opts.use_yx_clin:
            yx_radiomics_feat = batch["yx_radiomics_feat"]
            if len(yx_radiomics_feat.shape) == 5:
                yx_radiomics_feat = yx_radiomics_feat.view(self.opts.batch_size, -1)
            radiomics_feat = self.radiomics_fc(yx_radiomics_feat) # (B, yx_omics_features) --> (B, yx_out_features)
            
            yx_clinical_feat = batch["yx_clinical_feat"]
            if len(yx_clinical_feat.shape) == 5:
                yx_clinical_feat = yx_clinical_feat.view(self.opts.batch_size, -1)
            clin_flag = batch["yx_clin_flag"].unsqueeze(dim=1).float() # (B, 1)
            clinical_feat = self.clinical_fc(yx_clinical_feat) # (B, C)
            clin_syn_feat = self.clinical_param.repeat(clinical_feat.shape[0], 1) # (B, C) zero matrix, synthetic yx clinical feature when yx_flag=0
            clinical_feat = clinical_feat * clin_flag + clin_syn_feat * (1.0 - clin_flag) # (B, C)
        
            if self.opts.feat_fusion_mode == "parallel":
                patient_from_lesions = self.parallel_radiomics_clinical(patient_from_lesions, radiomics_feat, clinical_feat) 
            elif self.opts.feat_fusion_mode == "series_rc":
                patient_from_lesions = self.series_radiomics_clinical(patient_from_lesions, radiomics_feat, clinical_feat)
            elif self.opts.feat_fusion_mode == "series_cr":
                patient_from_lesions = self.series_clinical_radiomics(patient_from_lesions, radiomics_feat, clinical_feat)
            else:
                raise NotImplementedError

        if self.opts.attnmap_weight_dir not in [None, "None"]:
            npz_dir = os.path.join(self.opts.attnmap_weight_dir, "YX")
            os.makedirs(npz_dir, exist_ok=True)
            if batch["yx_flag"][0].item():
                np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
                        lesions_label=self.info_dict["lesions_label"],
                        lesions_attnmap=self.info_dict["lesions_attnmap"],
                        lesions_weight=self.info_dict["lesions_weight"],
                        clinical_weight=self.info_dict["clinical_weight"])
        
        return {
            "feat": patient_from_lesions
        }

class YXModelOnly(YXModel):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts, *args, **kwargs)
        self.classifier = nn.Linear(opts.yx_out_features, opts.n_classes)

    def forward(self, batch, *args, **kwargs):
        yx_results = super().forward(batch, *args, **kwargs)
        yx_pred = self.classifier(yx_results["feat"])
        return {
            "pred": yx_pred,
            "feat": yx_results["feat"],
            "lesions_pred": yx_results["lesions_pred"],
        }

class YXModelNoneClinicalRadiomics(YXModel):
    def forward(self, batch, *args, **kwargs):
        lesions = batch["lesions"] # (B, N_l, 3, H, W)
        yx_flag = batch["clinical_yx_flag"] # (B, 3)

        B, N_l, C, H, W = lesions.shape
        # (B, N_l, 3, H, W) --> (B, N_l, C)
        lesions_cnn = self.cnn(lesions.view(B*N_l, C, H, W))
        yx_flag = yx_flag.view(B, 1, 3, 1, 1).repeat(1, N_l, 1, *lesions_cnn.shape[-2:])
        yx_flag = yx_flag.view(B*N_l, 3, *lesions_cnn.shape[-2:])
        attn_lesions_cnn = torch.cat([lesions_cnn, yx_flag], dim=1)
        attn_lesions_cnn = torch.sigmoid(self.attn_layer(attn_lesions_cnn))
        lesions_cnn = lesions_cnn * attn_lesions_cnn
        lesions_cnn = lesions_cnn.view(B, N_l, -1)
        lesions_cnn = self.cnn_project(lesions_cnn)

        lesions_attn, lesions_attnmap = self.attn_over_lesions(key=lesions_cnn, query=lesions_cnn, value=lesions_cnn)
        lesions_attn_energy, lesions_attn_energy_unnorm = self.energy_function(lesions_attn, self.lesions_weight_fn)
        # (B, N_l, C) x (B, N_l, 1) --> (B, C)
        patient_from_lesions = torch.sum(lesions_attn * lesions_attn_energy, dim=1)
        patient_from_lesions = self.ffn_attn_l2p(patient_from_lesions)

        lesions_pred = self.lesions_classifier(lesions_attn)

        self.lesions_attn_energy_unnorm = lesions_attn_energy_unnorm
        
        return {
            "feat": patient_from_lesions,
            "lesions_pred": lesions_pred,
            #"yx_pred": self.yx_classifier(patient_from_lesions)
        }