from model.bl_model import *
from model.yx_model import *
from model.nn_layers.transformer import *
import json
import pandas as pd

class BLYXModel(nn.Module):
    def __init__(self, opts, bl_model, yx_model):
        super().__init__()
        self.opts = opts
        self.bl_model = bl_model 
        self.yx_model = yx_model 
        # self.bl_fc = nn.Linear(opts.bl_out_features, opts.blyx_out_features) # S17Figb Linear layer
        # self.yx_fc = nn.Linear(opts.yx_out_features, opts.blyx_out_features)
        self.bl_fc = nn.Sequential( nn.Linear(opts.bl_out_features, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features))
        self.yx_fc = nn.Sequential( nn.Linear(opts.yx_out_features, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features))
        self.blyx_fc = nn.Sequential(nn.Linear(opts.blyx_out_features*2, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features)) # for "mumo" "sum" intermodal_fusion mode

        if self.opts.intermodal_fusion == "mumo":
            self.bl_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
            self.yx_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
            # self.bl_spe_fc = nn.Sequential(nn.Linear(opts.blyx_out_features, opts.blyx_out_features), nn.BatchNorm1d(opts.blyx_out_features), nn.ReLU())
            # self.yx_spe_fc = nn.Sequential(nn.Linear(opts.blyx_out_features, opts.blyx_out_features), nn.BatchNorm1d(opts.blyx_out_features), nn.ReLU())
            
            self.yx_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))
            self.bl_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

            if self.opts.use_modal_align:
                self.bl_com_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features) # S17Figb Modal-agnostic layer
                self.yx_com_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
                # self.bl_com_fc = nn.Sequential(nn.Linear(opts.blyx_out_features, opts.blyx_out_features), nn.BatchNorm1d(opts.blyx_out_features), nn.ReLU())
                # self.yx_com_fc = nn.Sequential(nn.Linear(opts.blyx_out_features, opts.blyx_out_features), nn.BatchNorm1d(opts.blyx_out_features), nn.ReLU())
                self.blyx_fc = nn.Sequential(nn.Linear(opts.blyx_out_features*3, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features))
        
        elif self.opts.intermodal_fusion in ["sum","mul"]:
            self.blyx_fc = nn.Sequential(nn.Linear(opts.blyx_out_features, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features))

        # patient information
        # HER2,Molecularsubtype,Tumorsize,LNstatus,Grade,Age,ER,PR,Ki67
        if opts.use_pi_info:
            if opts.split_file:
                split_file = os.path.join(opts.split_file, f'fold_{opts.seed}.csv')
                pd_data = pd.read_csv(split_file)
                pd_data = pd_data[pd_data["split_info"] == 'train'].reset_index(drop=True)
            else:
                pd_data = pd.read_csv(self.opts.train_file)
            clinical_attrs = {'HER2','Tumorsize','LNstatus','Grade','Molecularsubtype'} # 'LVI','Clinicalstage'
            self.clinical_attrs_tab = {col: pd_data[col].unique().tolist() for col in clinical_attrs}
            self.clinical_embeddings = {col: nn.Embedding(len(pd_data[col].unique()), self.opts.blyx_out_features).to(self.opts.device)
                                        for col in clinical_attrs}            
            clinical_attrs = pd_data.columns[14:18].tolist()
            self.clinical_projectors = {col: nn.Linear(1, self.opts.blyx_out_features).to(self.opts.device)
                                        for col in clinical_attrs}
            del pd_data
            self.clinical_ln = nn.LayerNorm(opts.blyx_out_features)
            
            self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, num_heads=opts.blyx_attn_heads, dropout=opts.blyx_dropout, batch_first=True)
        
        self.classifier = nn.Linear(opts.blyx_out_features, 1) # predict the risk score

    def forward(self, batch, *args, **kwargs):
        yx_results = self.yx_model(batch, *args, **kwargs)
        yx_feat = self.yx_fc(yx_results["feat"]) # (B, C) C=blyx_out_features
        bl_results = self.bl_model(batch, *args, **kwargs)
        bl_feat = self.bl_fc(bl_results["feat"]) # (B, C) C=blyx_out_features

        yx_flag = batch["yx_flag"].unsqueeze(dim=1).float() # (B, 1)
        bl_flag = batch["bl_flag"].unsqueeze(dim=1).float() # (B, 1)
        blyx_flag = bl_flag * yx_flag # (B, 1)
        yx_com_feat = bl_com_feat = None
 
        if self.opts.intermodal_fusion == "cat":
            blyx_feat = torch.cat([bl_feat, yx_feat], dim=1) 
        elif self.opts.intermodal_fusion == "sum":
            blyx_feat = torch.add(bl_feat, yx_feat)
        elif self.opts.intermodal_fusion == "mul":
            blyx_feat = torch.multiply(bl_feat, yx_feat)
        elif self.opts.intermodal_fusion == "mumo":
            yx_spe_feat = self.yx_spe_fc(yx_feat) # (B, C)
            yx_syn_feat = self.yx_param.repeat(yx_feat.shape[0], 1) # (B, C) zero matrix, synthetic yx feature when yx_flag=0
            yx_spe_feat = yx_spe_feat * yx_flag + yx_syn_feat * (1.0 - yx_flag) # (B, C)
            
            bl_spe_feat = self.bl_spe_fc(bl_feat) # (B, C)
            bl_syn_feat = self.bl_param.repeat(bl_feat.shape[0], 1)
            bl_spe_feat = bl_spe_feat * bl_flag + bl_syn_feat * (1.0 - bl_flag) # (B, C)
            
            blyx_feat = torch.cat([bl_spe_feat, yx_spe_feat], dim=1)
            
            if self.opts.use_modal_align:
                # Modal-agnostic features alignment
                yx_com_feat = self.yx_com_fc(yx_feat) # (B, C)
                bl_com_feat = self.bl_com_fc(bl_feat) # (B, C)
                com_feat = (yx_com_feat + bl_com_feat) / 2.0 * blyx_flag
                com_feat = com_feat * blyx_flag + yx_com_feat * yx_flag * (1.0 - blyx_flag)
                com_feat = com_feat * blyx_flag + bl_com_feat * bl_flag * (1.0 - blyx_flag)

                blyx_feat = torch.cat([bl_spe_feat, com_feat, yx_spe_feat], dim=1) #  (B, 3 * blyx_out_fea)  
            
        blyx_feat = self.blyx_fc(blyx_feat) # torch.Size([B, blyx_out_feature])
        self.info_dict = {
            "id": batch["id"],
        }

        if self.opts.use_pi_info:
        # patient information: LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            clinical_feat = torch.stack([
                ## self.clinical_embeddings["LVI"](torch.tensor([self.clinical_attrs_tab['LVI'].index(d) for d in batch["LVI"]]).to(self.opts.device)) , # torch.Size([batch_size, 1024])
                self.clinical_embeddings["HER2"](torch.tensor([self.clinical_attrs_tab['HER2'].index(d) for d in batch["HER2"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Molecularsubtype"](torch.tensor([self.clinical_attrs_tab['Molecularsubtype'].index(d) for d in batch["Molecularsubtype"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Tumorsize"](torch.tensor([self.clinical_attrs_tab['Tumorsize'].index(d) for d in batch["Tumorsize"]]).to(self.opts.device)) ,
                self.clinical_embeddings["LNstatus"](torch.tensor([self.clinical_attrs_tab['LNstatus'].index(d) for d in batch["LNstatus"]]).to(self.opts.device)) ,
                ## self.clinical_embeddings["Clinicalstage"](torch.tensor([self.clinical_attrs_tab['Clinicalstage'].index(d) for d in batch["Clinicalstage"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Grade"](torch.tensor([self.clinical_attrs_tab['Grade'].index(d) for d in batch["Grade"]]).to(self.opts.device)) ,
                self.clinical_projectors["Age"](batch["Age"].unsqueeze(dim=1)) , #.view(self.opts.batch_size,1) torch.Size([batch_size, 1024])
                self.clinical_projectors["ER"](batch["ER"].unsqueeze(dim=1)) ,
                self.clinical_projectors["PR"](batch["PR"].unsqueeze(dim=1)) ,
                self.clinical_projectors["Ki67"](batch["Ki67"].unsqueeze(dim=1)) 
            ], dim=1) # (B, 9, blyx_out_fea)
            # print(f'modely.py clinical_feat: {clinical_feat.shape}') # torch.Size([32, 9, 1024])
            clinical_feat = self.clinical_ln(clinical_feat)
            clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
                                    query=blyx_feat.unsqueeze(dim=1), value=clinical_feat)
            # print(f'clinical_attnmap.shape: {clinical_attnmap.shape}') # clinical_attnmap.shape: (32,1,9)
            self.info_dict["clinical_weight"] = clinical_attnmap[:, 0,:].detach().cpu().numpy()
            blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)

        blyx_pred = self.classifier(blyx_feat)

        # save features, one file per patient
        # if self.opts.feat_dir:
        #     npz_dir = self.opts.feat_dir
        #     os.makedirs(npz_dir, exist_ok=True)
        #     np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
        #             bl_feat=bl_spe_feat.detach().cpu().numpy(),
        #             bl_syn_feat=bl_syn_feat.detach().cpu().numpy(),
        #             bl_flag=bl_flag.detach().cpu().numpy(),
        #             com_feat=com_feat.detach().cpu().numpy(),
        #             yx_feat=yx_spe_feat.detach().cpu().numpy(),
        #             yx_syn_feat=yx_syn_feat.detach().cpu().numpy(),
        #             yx_flag=yx_flag.detach().cpu().numpy())

        return {
            "blyx_flag":blyx_flag,
            "risk_score": blyx_pred,
            "feat": blyx_feat,
            "bl_com_feat": bl_com_feat,
            "yx_com_feat": yx_com_feat,
            "blyx_flag": blyx_flag,
        }


class BLPIModel(nn.Module):
    def __init__(self, opts, bl_model):
        super().__init__()
        self.opts = opts
        self.bl_model = bl_model 
        self.bl_fc = nn.Sequential( nn.Linear(opts.bl_out_features, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features))
        self.bl_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
        self.blyx_fc = nn.Sequential(nn.Linear(opts.blyx_out_features, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features))

        self.bl_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

        if opts.use_pi_info:
            # patient information
            # LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            if opts.split_file:
                split_file = os.path.join(opts.split_file, f'fold_{opts.seed}.csv')
                pd_data = pd.read_csv(split_file)
                pd_data = pd_data[pd_data["split_info"] == 'train'].reset_index(drop=True)
            else:
                pd_data = pd.read_csv(self.opts.train_file)
            # clinical_attrs = {'LVI','HER2','Tumorsize','LNstatus','Grade'}
            # self.clinical_attrs_tab = {col: pd_data[col].unique().tolist() for col in clinical_attrs}
            # self.clinical_embeddings = {col: nn.Embedding(len(pd_data[col].unique()), self.opts.blyx_out_features).to(self.opts.device)
            #                             for col in clinical_attrs}
            # clinical_attrs = pd_data.columns[15:18].tolist()
            # self.clinical_projectors = {col: nn.Linear(1, self.opts.blyx_out_features).to(self.opts.device)
            #                             for col in clinical_attrs}
            clinical_attrs = {'HER2','Tumorsize','LNstatus','Grade','Molecularsubtype'}
            self.clinical_attrs_tab = {col: pd_data[col].unique().tolist() for col in clinical_attrs}
            self.clinical_embeddings = {col: nn.Embedding(len(pd_data[col].unique()), self.opts.blyx_out_features).to(self.opts.device)
                                        for col in clinical_attrs}            
            clinical_attrs = pd_data.columns[14:18].tolist()
            self.clinical_projectors = {col: nn.Linear(1, self.opts.blyx_out_features).to(self.opts.device)
                                        for col in clinical_attrs}
            del pd_data
            self.clinical_ln = nn.LayerNorm(opts.blyx_out_features)
            self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, num_heads=opts.blyx_attn_heads, dropout=opts.blyx_dropout, batch_first=True)
        
        self.classifier = nn.Linear(opts.blyx_out_features, 1) # predict the risk score

    def forward(self, batch, *args, **kwargs):
        bl_results = self.bl_model(batch, *args, **kwargs)
        bl_feat = self.bl_fc(bl_results["feat"]) # (B, C) C=blyx_out_features
        
        bl_flag = batch["bl_flag"].unsqueeze(dim=1).float() # (B, 1)
        bl_spe_feat = self.bl_spe_fc(bl_feat) # (B, C)
        bl_syn_feat = self.bl_param.repeat(bl_feat.shape[0], 1)
        bl_spe_feat = bl_spe_feat * bl_flag + bl_syn_feat * (1.0 - bl_flag) # (B, C)

        self.info_dict = {
            "id": batch["id"],
        }

        blyx_feat = self.blyx_fc(bl_spe_feat) # torch.Size([B, C=blyx_out_feature])

        if self.opts.use_pi_info:
        # patient information: LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            clinical_feat = torch.stack([
                self.clinical_embeddings["HER2"](torch.tensor([self.clinical_attrs_tab['HER2'].index(d) for d in batch["HER2"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Molecularsubtype"](torch.tensor([self.clinical_attrs_tab['Molecularsubtype'].index(d) for d in batch["Molecularsubtype"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Tumorsize"](torch.tensor([self.clinical_attrs_tab['Tumorsize'].index(d) for d in batch["Tumorsize"]]).to(self.opts.device)) ,
                self.clinical_embeddings["LNstatus"](torch.tensor([self.clinical_attrs_tab['LNstatus'].index(d) for d in batch["LNstatus"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Grade"](torch.tensor([self.clinical_attrs_tab['Grade'].index(d) for d in batch["Grade"]]).to(self.opts.device)) ,
                self.clinical_projectors["Age"](batch["Age"].unsqueeze(dim=1)) , # torch.Size([batch_size, 1024])
                self.clinical_projectors["ER"](batch["ER"].unsqueeze(dim=1)) ,
                self.clinical_projectors["PR"](batch["PR"].unsqueeze(dim=1)) ,
                self.clinical_projectors["Ki67"](batch["Ki67"].unsqueeze(dim=1)) 
            ], dim=1) # (B, 9, blyx_out_fea)
            clinical_feat = self.clinical_ln(clinical_feat)
            
            clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
                                    query=blyx_feat.unsqueeze(dim=1), value=clinical_feat) 
            self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
            blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)
        
        blyx_pred = self.classifier(blyx_feat)

        # save features, one file per patient
        # if self.opts.feat_dir:
        #     npz_dir = self.opts.feat_dir
        #     os.makedirs(npz_dir, exist_ok=True)
        #     np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
        #             bl_feat=bl_spe_feat.detach().cpu().numpy(),
        #             bl_syn_feat=bl_syn_feat.detach().cpu().numpy(),
        #             bl_flag=bl_flag.detach().cpu().numpy(),
        #             com_feat=com_feat.detach().cpu().numpy(),
        #             yx_feat=yx_spe_feat.detach().cpu().numpy(),
        #             yx_syn_feat=yx_syn_feat.detach().cpu().numpy(),
        #             yx_flag=yx_flag.detach().cpu().numpy())

        # if self.opts.attnmap_weight_dir not in [None, "None"]:
        #     npz_dir = os.path.join(self.opts.attnmap_weight_dir, "BLYX")
        #     os.makedirs(npz_dir, exist_ok=True)
        #     np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
        #             clinical_weight=self.info_dict["clinical_weight"])

        return {
            "risk_score": blyx_pred,
            "feat": blyx_feat
        }
    
class YXPIModel(nn.Module):
    def __init__(self, opts, yx_model):
        super().__init__()
        self.opts = opts
        self.yx_model = yx_model 
        self.yx_fc = nn.Sequential( nn.Linear(opts.yx_out_features, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features))
        self.yx_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
        self.blyx_fc = nn.Sequential(nn.Linear(opts.blyx_out_features, opts.blyx_out_features), nn.LayerNorm(opts.blyx_out_features))

        self.yx_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

        if opts.use_pi_info:
            # patient information
            # LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            if opts.split_file:
                split_file = os.path.join(opts.split_file, f'fold_{opts.seed}.csv')
                pd_data = pd.read_csv(split_file)
                pd_data = pd_data[pd_data["split_info"] == 'train'].reset_index(drop=True)
            else:
                pd_data = pd.read_csv(self.opts.train_file)
            # clinical_attrs = {'LVI','HER2','Tumorsize','LNstatus','Grade'}
            # self.clinical_attrs_tab = {col: pd_data[col].unique().tolist() for col in clinical_attrs}
            # self.clinical_embeddings = {col: nn.Embedding(len(pd_data[col].unique()), self.opts.blyx_out_features).to(self.opts.device)
            #                             for col in clinical_attrs}
            # clinical_attrs = pd_data.columns[15:18].tolist()
            # self.clinical_projectors = {col: nn.Linear(1, self.opts.blyx_out_features).to(self.opts.device)
            #                             for col in clinical_attrs}
            clinical_attrs = {'HER2','Tumorsize','LNstatus','Grade','Molecularsubtype'}
            self.clinical_attrs_tab = {col: pd_data[col].unique().tolist() for col in clinical_attrs}
            self.clinical_embeddings = {col: nn.Embedding(len(pd_data[col].unique()), self.opts.blyx_out_features).to(self.opts.device)
                                        for col in clinical_attrs}            
            clinical_attrs = pd_data.columns[14:18].tolist()
            self.clinical_projectors = {col: nn.Linear(1, self.opts.blyx_out_features).to(self.opts.device)
                                        for col in clinical_attrs}
            del pd_data
            self.clinical_ln = nn.LayerNorm(opts.blyx_out_features)
            
            self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, num_heads=opts.blyx_attn_heads, dropout=opts.blyx_dropout, batch_first=True)
        
        self.classifier = nn.Linear(opts.blyx_out_features, 1) # predict the risk score

    def forward(self, batch, *args, **kwargs):
        yx_results = self.yx_model(batch, *args, **kwargs)
        yx_feat = self.yx_fc(yx_results["feat"]) # (B, C) C=blyx_out_features

        yx_flag = batch["yx_flag"].unsqueeze(dim=1).float() # (B, 1)
        yx_spe_feat = self.yx_spe_fc(yx_feat) # (B, C)
        yx_syn_feat = self.yx_param.repeat(yx_feat.shape[0], 1) # (B, C) zero matrix, synthetic yx feature when yx_flag=0
        yx_spe_feat = yx_spe_feat * yx_flag + yx_syn_feat * (1.0 - yx_flag) # (B, C)
        
        self.info_dict = {
            "id": batch["id"],
        }

        blyx_feat = self.blyx_fc(yx_spe_feat) # torch.Size([B, C=blyx_out_feature])

        if self.opts.use_pi_info:
        # patient information: LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
            clinical_feat = torch.stack([
                self.clinical_embeddings["HER2"](torch.tensor([self.clinical_attrs_tab['HER2'].index(d) for d in batch["HER2"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Molecularsubtype"](torch.tensor([self.clinical_attrs_tab['Molecularsubtype'].index(d) for d in batch["Molecularsubtype"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Tumorsize"](torch.tensor([self.clinical_attrs_tab['Tumorsize'].index(d) for d in batch["Tumorsize"]]).to(self.opts.device)) ,
                self.clinical_embeddings["LNstatus"](torch.tensor([self.clinical_attrs_tab['LNstatus'].index(d) for d in batch["LNstatus"]]).to(self.opts.device)) ,
                self.clinical_embeddings["Grade"](torch.tensor([self.clinical_attrs_tab['Grade'].index(d) for d in batch["Grade"]]).to(self.opts.device)) ,
                self.clinical_projectors["Age"](batch["Age"].unsqueeze(dim=1)) , # torch.Size([batch_size, 1024])
                self.clinical_projectors["ER"](batch["ER"].unsqueeze(dim=1)) ,
                self.clinical_projectors["PR"](batch["PR"].unsqueeze(dim=1)) ,
                self.clinical_projectors["Ki67"](batch["Ki67"].unsqueeze(dim=1)) 
            ], dim=1) # (B, 8, blyx_out_fea)
            clinical_feat = self.clinical_ln(clinical_feat)
            
            clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
                                    query=blyx_feat.unsqueeze(dim=1), value=clinical_feat) # 
            self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
            blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)
        blyx_pred = self.classifier(blyx_feat)

        # save features, one file per patient
        # if self.opts.feat_dir:
        #     npz_dir = self.opts.feat_dir
        #     os.makedirs(npz_dir, exist_ok=True)
        #     np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
        #             bl_feat=bl_spe_feat.detach().cpu().numpy(),
        #             bl_syn_feat=bl_syn_feat.detach().cpu().numpy(),
        #             bl_flag=bl_flag.detach().cpu().numpy(),
        #             com_feat=com_feat.detach().cpu().numpy(),
        #             yx_feat=yx_spe_feat.detach().cpu().numpy(),
        #             yx_syn_feat=yx_syn_feat.detach().cpu().numpy(),
        #             yx_flag=yx_flag.detach().cpu().numpy())

        # if self.opts.attnmap_weight_dir not in [None, "None"]:
        #     npz_dir = os.path.join(self.opts.attnmap_weight_dir, "BLYX")
        #     os.makedirs(npz_dir, exist_ok=True)
        #     np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
        #             clinical_weight=self.info_dict["clinical_weight"])

        return {
            "risk_score": blyx_pred,
            "feat": blyx_feat
        }


class PIModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # patient information
        # HER2,Molecularsubtype,Tumorsize,LNstatus,Grade,Age,ER,PR,Ki67
        if opts.split_file:
            split_file = os.path.join(opts.split_file, f'fold_{opts.seed}.csv')
            pd_data = pd.read_csv(split_file)
            pd_data = pd_data[pd_data["split_info"] == 'train'].reset_index(drop=True)
        else:
            pd_data = pd.read_csv(self.opts.train_file)

        clinical_attrs = {'HER2','Tumorsize','LNstatus','Grade','Molecularsubtype'} # 'LVI','Clinicalstage'
        self.clinical_attrs_tab = {col: pd_data[col].unique().tolist() for col in clinical_attrs}
        self.clinical_embeddings = {col: nn.Embedding(len(pd_data[col].unique()), self.opts.blyx_out_features).to(self.opts.device)
                                    for col in clinical_attrs}            
        clinical_attrs = pd_data.columns[14:18].tolist()
        self.clinical_projectors = {col: nn.Linear(1, self.opts.blyx_out_features).to(self.opts.device)
                                    for col in clinical_attrs}
        del pd_data

        self.clinical_ln = nn.LayerNorm(opts.blyx_out_features)
                
        self.classifier = nn.Linear(opts.blyx_out_features, 1) # predict the risk score

    def forward(self, batch, *args, **kwargs):
        # patient information: LVI,HER2,Molecularsubtype,Tumorsize,LNstatus,Clinicalstage,Grade,Age,ER,PR,Ki67
        clinical_feat = torch.sum(torch.stack([
            ## self.clinical_embeddings["LVI"](torch.tensor([self.clinical_attrs_tab['LVI'].index(d) for d in batch["LVI"]]).to(self.opts.device)) , # torch.Size([batch_size, 1024])
            self.clinical_embeddings["HER2"](torch.tensor([self.clinical_attrs_tab['HER2'].index(d) for d in batch["HER2"]]).to(self.opts.device)) ,
            self.clinical_embeddings["Molecularsubtype"](torch.tensor([self.clinical_attrs_tab['Molecularsubtype'].index(d) for d in batch["Molecularsubtype"]]).to(self.opts.device)) ,
            self.clinical_embeddings["Tumorsize"](torch.tensor([self.clinical_attrs_tab['Tumorsize'].index(d) for d in batch["Tumorsize"]]).to(self.opts.device)) ,
            self.clinical_embeddings["LNstatus"](torch.tensor([self.clinical_attrs_tab['LNstatus'].index(d) for d in batch["LNstatus"]]).to(self.opts.device)) ,
            ## self.clinical_embeddings["Clinicalstage"](torch.tensor([self.clinical_attrs_tab['Clinicalstage'].index(d) for d in batch["Clinicalstage"]]).to(self.opts.device)) ,
            self.clinical_embeddings["Grade"](torch.tensor([self.clinical_attrs_tab['Grade'].index(d) for d in batch["Grade"]]).to(self.opts.device)) ,
            self.clinical_projectors["Age"](batch["Age"].unsqueeze(dim=1)) , #.view(self.opts.batch_size,1) torch.Size([batch_size, 1024])
            self.clinical_projectors["ER"](batch["ER"].unsqueeze(dim=1)) ,
            self.clinical_projectors["PR"](batch["PR"].unsqueeze(dim=1)) ,
            self.clinical_projectors["Ki67"](batch["Ki67"].unsqueeze(dim=1)) 
        ], dim=1),dim=1) # (B, blyx_out_fea)
        # print(f'modely.py clinical_feat: {clinical_feat.shape}') # torch.Size([32, 1024])
        clinical_feat = self.clinical_ln(clinical_feat)

        blyx_pred = self.classifier(clinical_feat)

        return {
            "risk_score": blyx_pred
        }

