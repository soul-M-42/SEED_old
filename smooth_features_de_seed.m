clear;
randSeed = 7;
tf = 16;
sf = 16;

datadir = './runs_seed/raw_batch15_timeLen30_tf16_sf16_tfLen48_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel';
disp(datadir)
n_total = 3394;
augTime = 1;
n_subs = 45;
decay_rate = 0.990;

fprintf('decay rate %f\n', decay_rate)
for fold = 0:4
    tic
    fprintf('\nfold %d\n', fold)
    featuredir = fullfile(datadir, num2str(fold), sprintf('features1_de_1s_all_normTrain_rnPreWeighted%.3f.mat', decay_rate));
    load(featuredir)
    disp(featuredir)
    load('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples')
    n_samples_cum = [0, cumsum(n_samples)];
    
    disp(size(de))
    de_lds = zeros(n_subs, sf, n_total*augTime, tf);
    for sub = 1: n_subs
        fprintf('sub %d\n', sub)
        for aug = 1: augTime
            de_one = reshape(de(sub, n_total*(aug-1)+1: n_total*aug, :), n_total, sf, tf);
            de_one = permute(de_one, [2,1,3]);
            for i = 1: length(n_samples)
                de_one_vid = de_one(:, n_samples_cum(i)+1: n_samples_cum(i+1), :);
                de_one_vid_lds = lds(de_one_vid);
                de_lds(sub, :, n_total*(aug-1) + n_samples_cum(i)+1: n_total*(aug-1) + n_samples_cum(i+1), :) = de_one_vid_lds;
            end
        end
    end

    de_lds = reshape(permute(de_lds, [1,3,2,4]), n_subs, n_total*augTime, sf*tf);
    save(fullfile(datadir, num2str(fold), sprintf('features1_de_1s_all_normTrain_rnPreWeighted%.3f_lds.mat', decay_rate)), 'de_lds')
    toc
end
    
