import os
import scipy.io as sio
import numpy as np

save_dir = './runs_seed/raw_batch15_timeLen30_tf16_sf16_tfLen48_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel'
bn_val = 1

# n_total = 3394
n_total = 3407
n_counters = int(np.ceil(n_total / bn_val))

use_features = 'features1_de_1s_all_normTrain'
print(use_features)

n_subs = 45
n_folds = 5

n_per = int(n_subs / n_folds)

decay_rate = 0.990
print(decay_rate)
for fold in range(n_folds):
    print(fold)
    
    data = sio.loadmat(os.path.join(save_dir, str(fold), use_features+'.mat'))['de']
    print(data.shape)

    data[np.isnan(data)] = -30
    val_sub = np.arange(n_per*fold, n_per*(fold+1))
    train_sub = list(set(np.arange(n_subs)) - set(val_sub))
    print(train_sub)
    print(val_sub)
    data_mean = np.mean(np.mean(data[train_sub, :, :], axis=1), axis=0)
    data_var = np.mean(np.var(data[train_sub, :, :], axis=1), axis=0)
    
    data_norm = np.zeros_like(data)
    for sub in range(data.shape[0]):
        running_sum = np.zeros(data.shape[-1])
        running_square = np.zeros(data.shape[-1])
        decay_factor = 1
        for counter in range(n_counters):
            data_one = data[sub, counter*bn_val: (counter+1)*bn_val, :]
            running_sum = running_sum + data_one
            running_mean = running_sum / (counter+1)
            running_square = running_square + data_one**2
            running_var = (running_square - 2 * running_mean * running_sum) / (counter+1) + running_mean**2

            curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
            curr_var = decay_factor*data_var + (1-decay_factor)*running_var
            decay_factor = decay_factor*decay_rate

            data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-5)
            data_norm[sub, counter*bn_val: (counter+1)*bn_val, :] = data_one

    de = {'de': data_norm}
    sio.savemat(os.path.join(save_dir, str(fold), use_features+'_rnPreWeighted%.3f.mat' % decay_rate), de)
            

