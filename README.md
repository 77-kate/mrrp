# MRRP
This repository provides source code to reproduce the experimental results in the paper "Multimodal deep learning with routine clinical data for recurrence risk stratification in HR+/HER2- early breast cancer". This project is based on MuMo (https://github.com/czifan/MuMo).
If you use this code, please cite the paper using the bibtex reference below.
"Xiaoyan Wu, Hong Liu, Jingyan Liu, Bingan Mu, Jianfei Li, Siyu Wang, Fengling Li, Xunxi Lu, Jie Chen, Yulan Peng, et al. Multimodal deep learning with routine clinical data for recurrence risk stratification in HR+/HER2- early breast cancer. Research. 0:DOI:10.34133/research.1136"


## Running Environment
A conda environment file is provided in mrrp.yml. It can be installed by using conda as follows

```
conda env create -f mrrp.yml
```
Installing PyTorch: Instructions to install a PyTorch compatible with the CUDA on your GPUs (or without GPUs) can be found here: https://pytorch.org/get-started/locally.

## Get Started
Below is a sample shell script for external validation:

```
#!/usr/bin/env bash
modal='blyx'
epochs=200
intermodal_fusion="mumo"
blyx_checkpoint='./ckp/model_last_epo200.pth' # the model checkpoint file, if you want to train the model from scratch, DO NOT pass this parameter

bl_rad_file='path_to_your_WSI_morphological_feature.pt' 
bl_clin_file='path_to_your_WSI_topological_feature.pt' 
bl_wsi_file='path_to_your_WSI_deep_feature.pt'

yx_clin_features=1024 # dimension of your Ultra-Sound clinical text feature
yx_clin_file='path_to_your_US_clinical_text_feature.pt'
yx_rad_file='path_to_your_US_omics_feature.csv' # We use Pyradiomics(https://pyradiomics.readthedocs.io/en/latest/#) to extract omics feature for US image
yx_img_feat_dir='path_to_your_US_deep_feature_folder'

split_file="path_to_your_data_splits"

save_dir='path_to_your_result_folder'
if [ ! -d "$save_dir" ];then
  mkdir $save_dir
fi

CUDA_VISIBLE_DEVICES=0 python main.py \
--modal $modal \
--external-val \ # if you want to train the model from scratch, DO NOT pass this parameter
--blyx-checkpoint $blyx_checkpoint \
--use-pi-info \
--intermodal-fusion $intermodal_fusion \
--epochs $epochs \
--save-dir $save_dir \
--use-bl-clin \
--use-bl-rad \
--bl-clin-file $bl_clin_file \
--bl-bag-feat-dir $bl_wsi_file \
--bl-rad-file $bl_rad_file \
--use-yx-rad \
--use-yx-clin \
--yx-clin-features $yx_clin_features \
--yx-clin-file $yx_clin_file \
--split-file $split_file \

```

