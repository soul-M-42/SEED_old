import os

# Contrastive learning
os.system('python main_cnn_seed.py --gpu-index 7')
# Extract DE features from the trained base encoder
os.system('python extract_pretrainFeat_seed_normTrain.py --gpu-index 7')
# Normalize the DE features adaptively (online)
os.system('python running_norm_fea2.py')


