
use_features = 'pretrained';
datadir = './runs_seed/sepVideo_raw_batch9_timeLen30_tf16_sf16_tfLen48_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel'
disp(datadir)
load('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples')
n_samples_train = n_samples(1:9);
n_samples_val = n_samples(10:end);
n_total_train = sum(n_samples_train);
n_total_val = sum(n_samples_val);
augTime = 1;
n_subs = 45;

n_channs = 16;
n_freqs = 16;

n_folds = 5;

decay_rate = 0.990;
for fold = 0: n_folds-1
    tic
    fprintf('\nfold %d\n', fold)

    load(fullfile(datadir, num2str(fold), sprintf('features1_de_1s_all_normTrain_rnPreWeighted%.3f.mat', decay_rate)))
    disp(fullfile(datadir, num2str(fold), sprintf('features1_de_1s_all_normTrain_rnPreWeighted%.3f.mat', decay_rate)))

    de = de_train;
    disp(size(de))

    n_samples_cum = [0, cumsum(n_samples_train)];

    de_lds = zeros(n_subs, n_channs, n_total_train*augTime, n_freqs);
    for sub = 1: n_subs
        fprintf('sub %d\n', sub)
        for aug = 1: augTime
            de_one = reshape(de(sub, n_total_train*(aug-1)+1: n_total_train*aug, :), n_total_train, n_channs, n_freqs);
            de_one = permute(de_one, [2,1,3]);
            for i = 1: length(n_samples_train)
                de_one_vid = de_one(:, n_samples_cum(i)+1: n_samples_cum(i+1), :);
                de_one_vid_lds = lds(de_one_vid);
                de_lds(sub, :, n_total_train*(aug-1) + n_samples_cum(i)+1: n_total_train*(aug-1) + n_samples_cum(i+1), :) = de_one_vid_lds;
            end
        end
    end

    de_lds = reshape(permute(de_lds, [1,3,2,4]), n_subs, n_total_train*augTime, n_channs*n_freqs);
    save(fullfile(datadir, num2str(fold), sprintf('features1_de_1s_all_normTrain_rnPreWeighted%.3f_train_lds.mat', decay_rate)), 'de_lds')
    toc
end

for fold = 0: n_folds-1
    tic
    fprintf('\nfold %d\n', fold)
    load(fullfile(datadir, num2str(fold), sprintf('features1_de_1s_all_normTrain_rnPreWeighted%.3f.mat', decay_rate)))
    disp(fullfile(datadir, num2str(fold), sprintf('features1_de_1s_all_normTrain_rnPreWeighted%.3f.mat', decay_rate)))

    de = de_val;
    disp(size(de))
    
    load('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples')
    n_samples_cum = [0, cumsum(n_samples_val)];

    de_lds = zeros(n_subs, n_channs, n_total_val*augTime, n_freqs);
    for sub = 1: n_subs
        fprintf('sub %d\n', sub)
        for aug = 1: augTime
            de_one = reshape(de(sub, n_total_val*(aug-1)+1: n_total_val*aug, :), n_total_val, n_channs, n_freqs);
            de_one = permute(de_one, [2,1,3]);
            for i = 1: length(n_samples_val)
                de_one_vid = de_one(:, n_samples_cum(i)+1: n_samples_cum(i+1), :);
                de_one_vid_lds = lds(de_one_vid);
                de_lds(sub, :, n_total_val*(aug-1) + n_samples_cum(i)+1: n_total_val*(aug-1) + n_samples_cum(i+1), :) = de_one_vid_lds;
            end
        end
    end

    de_lds = reshape(permute(de_lds, [1,3,2,4]), n_subs, n_total_val*augTime, n_channs*n_freqs);
    save(fullfile(datadir, num2str(fold), sprintf('features1_de_1s_all_normTrain_rnPreWeighted%.3f_val_lds.mat', decay_rate)), 'de_lds')
    toc
end

    
