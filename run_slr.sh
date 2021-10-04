# #################################
# # MLP-Mixer
# #################################

# ### Step 1: run SLR train
# ## change @config-file based on needed
# python3 main_slr_mlp.py /data/imagenet --model mixer_b16_224 --pretrained --optimization savlr --admm-train --config-file config_mlp_0.9 --batch-size 128

# ### Step 2: run maked retrain
# ## change @config-file based on needed
# ## change @admmtrain-acc based on model from admm train
# python3 main_slr_mlp.py /data/imagenet --model mixer_b16_224 --pretrained --optimization savlr --masked-retrain --admmtrain-acc 76.08 --config-file config_mlp_0.9 --batch-size 128 --ext _tmp3



# ### Step 1: run ADMM train
# python3 main_slr_mlp.py /data/imagenet --model mixer_b16_224 --pretrained --optimization admm --admm-train --config-file config_mlp_0.9 --batch-size 128

# ### Step 2: run maked retrain
# python3 main_slr_mlp.py /data/imagenet --model mixer_b16_224 --pretrained --optimization admm --masked-retrain --admmtrain-acc 76.08 --config-file config_mlp_0.9 --batch-size 128 --ext _tmp3



# #################################
# # Swin Transformer
# #################################

# ### Step 1: run SLR train
# ## change @config-file based on needed
# python3 main_slr_swin.py /data/imagenet --arch swin --model swin_tiny_patch4_window7_224 --pretrained --optimization savlr --admm-train --config-file config_swin_0.7 --batch-size 128

# ### Step 2: run maked retrain
# ## change @config-file based on needed
# ## change @admmtrain-acc based on model from admm train
# python3 main_slr_swin.py /data/imagenet --arch swin --model swin_tiny_patch4_window7_224 --pretrained --optimization savlr --masked-retrain --admmtrain-acc 76.08 --config-file config_swin_0.9 --batch-size 128 --ext _tmp3



# ### Step 1: run ADMM train
# python3 main_slr_swin.py /data/imagenet --arch swin --model swin_tiny_patch4_window7_224 --pretrained --optimization admm --admm-train --config-file config_swin_0.7 --batch-size 128

# ### Step 2: run maked retrain
# python3 main_slr_swin.py /data/imagenet --arch swin --model swin_tiny_patch4_window7_224 --pretrained --optimization admm --masked-retrain --admmtrain-acc 76.08 --config-file config_swin_0.9 --batch-size 128 --ext _tmp3



#################################
# Overall
#################################

### if you have wandb account change main_slr.py line 54 to your own account and repo; if not, doesn't matter

## change the path
save_middle_model_path="./"

### Step 1: run SLR train
## change @config-file based on needed
python3 main_slr.py /data/imagenet --output $save_middle_model_path --arch swin --model swin_tiny_patch4_window7_224 --pretrained --optimization savlr --admm-train --config-file config_swin_0.7 --batch-size 128

### Step 2: run maked retrain
## change @config-file based on needed
## change @admmtrain-acc based on model from admm train
python3 main_slr.py /data/imagenet --output $save_middle_model_path --arch swin --model swin_tiny_patch4_window7_224 --pretrained --optimization savlr --masked-retrain --admmtrain-acc 76.08 --config-file config_swin_0.7 --batch-size 128 --ext _tmp3



### Step 1: run ADMM train
python3 main_slr.py /data/imagenet --output $save_middle_model_path --arch swin --model swin_tiny_patch4_window7_224 --pretrained --optimization admm --admm-train --config-file config_swin_0.7 --batch-size 128

### Step 2: run maked retrain
python3 main_slr.py /data/imagenet --output $save_middle_model_path --arch swin --model swin_tiny_patch4_window7_224 --pretrained --optimization admm --masked-retrain --admmtrain-acc 76.08 --config-file config_swin_0.7 --batch-size 128 --ext _tmp3


