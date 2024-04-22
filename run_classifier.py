import os
import pickle
import numpy as np

for weight_decay in [0.005, 0.011, 0.025, 0.056, 0.125]:
    os.system('python main_de_seed.py --gpu-index 5 --weight-decay %f' % weight_decay)

for wd in [0.005, 0.011, 0.025, 0.056, 0.125]:
    result_dir = './runs_seed/raw_batch15_timeLen30_tf16_sf16_tfLen48_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel/wd%.3f' % wd
    with open(os.path.join(result_dir, 'results_finetune.pkl'), 'rb') as f:
        results_finetune = pickle.load(f)
    print('wd%.3f, val loss mean: %.3f, std: %.3f; val acc mean: %.3f, std: %.3f' % (wd,
        np.mean(results_finetune['best_val_loss']), np.std(results_finetune['best_val_loss']),
        np.mean(results_finetune['best_val_acc']), np.std(results_finetune['best_val_acc'])))


