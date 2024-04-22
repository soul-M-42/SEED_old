import os

# Contrastive learning
# os.system('python main_cnn_seed_sepVideo.py --gpu-index 6')
# Extract DE features from the trained base encoder
os.system('python extract_pretrainFeat_seed_normTrain_sepVideo.py --gpu-index 6')
# Normalize the DE features adaptively (online)
os.system('python running_norm_fea2_sepVideo.py')


