clear; close all
load runs_seed\raw_batch15_timeLen30_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel_all\features1_de_1s_lds_fold1_pre3_lr0.000500_hidden30_chnNorm_accSel\attrs.mat
attrs = cell(1,3);
attrs{1,1} = attrs_pos;
attrs{1,2} = attrs_neut;
attrs{1,3} = attrs_neg;
attrs_inds = zeros(256,3);
attrs_sorted = zeros(256,3);
for i = 1: 3
    attrs_now = attrs{1,i};
    n_samples_all = size(attrs_now, 1) / 45;
    attrs_mean = zeros(45, 256);
    for sub = 1: 45
        attrs_mean(sub, :) = mean(attrs_now(n_samples_all * (sub - 1) + 1: n_samples_all * sub, :));
    end
    
%     corr_mat = corr(attrs_mean');
%     figure; imagesc(corr_mat); colorbar;
    
    
    attrs_grandMean = mean(attrs_mean);
    [attrs_sorted(:, i), attrs_inds(:, i)] = sort(attrs_grandMean);
    attrs_sorted(:, i) = attrs_sorted(end:-1:1, i);
    attrs_inds(:, i) = attrs_inds(end:-1:1, i);
    h = figure('Renderer', 'painters', 'Position', [10 10 900 600]); 
    bar(attrs_sorted(1:256, i));
    
    ylabel('Feature importance', 'FontSize', 20)
    ax=gca;
    ax.XAxis.FontSize = 18;
    ax.YAxis.FontSize = 18;
end
xlabel('Feature index (sorted)', 'FontSize', 20)

time_ind = zeros(256, 3);
spatial_ind = zeros(256, 3);
for i = 1: 3
    for j = 1: 256
        ind = attrs_inds(j, i);
        time_ind(j, i) = ceil(ind / 16);
        spatial_ind(j, i) = ind - floor(ind / 16) * 16;
        if spatial_ind(j, i) == 0
            spatial_ind(j, i) = 16;
        end
    end
end


load runs_seed\raw_batch15_timeLen30_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel_all\pretrain_weights
f = 200*(0:(48/2))/48;
sp = abs(fft(time_weights(11,end:-1:1)));
h = figure('Renderer', 'painters', 'Position', [60 60 1200 300]); 
subplot(1,2,1); plot(1:48, time_weights(11,:), 'linewidth', 3, 'color', 'k'); xlabel('ms', 'FontSize', 26)
ax=gca; ax.XAxis.FontSize = 24; ax.YAxis.FontSize = 24;
subplot(1,2,2); plot(f(1:12), sp(1:12), 'linewidth', 3, 'color', 'k'); xlabel('Hz', 'FontSize', 26)
ax=gca; ax.XAxis.FontSize = 24; ax.YAxis.FontSize = 24;

sp = abs(fft(time_weights(10,end:-1:1)));
h = figure('Renderer', 'painters', 'Position', [60 60 1200 300]); 
subplot(1,2,1); plot(1:48, time_weights(10,:), 'linewidth', 3, 'color', 'k'); xlabel('ms', 'FontSize', 26)
ax=gca; ax.XAxis.FontSize = 24; ax.YAxis.FontSize = 24;
subplot(1,2,2); plot(f(1:12), sp(1:12), 'linewidth', 3, 'color', 'k'); xlabel('Hz', 'FontSize', 26)
ax=gca; ax.XAxis.FontSize = 24; ax.YAxis.FontSize = 24;

load E:\Data\Emotion\SEED\chn_names
layout_file = 'quickcap64.mat';
h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
U_topoplot_new(spatial_weights(7, :)', layout_file, chn_names)
c = colorbar(); c.FontSize=16;
h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
U_topoplot_new(spatial_weights(3, :)', layout_file, chn_names)
c = colorbar(); c.FontSize=16;

h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
U_topoplot_new(spatial_weights(4, :)', layout_file, chn_names)
c = colorbar(); c.FontSize=16;
h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
U_topoplot_new(spatial_weights(16, :)', layout_file, chn_names)
c = colorbar(); c.FontSize=16;

% figure; plot(time_weights(11,:))
% figure; plot(time_weights(10,:))
% figure; plot(time_weights(16,:))
% figure; plot(time_weights(12,:))
% h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
% U_topoplot_new(spatial_weights(4, :)', layout_file, chn_names)
% h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
% U_topoplot_new(spatial_weights(16, :)', layout_file, chn_names)
% h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
% U_topoplot_new(spatial_weights(6, :)', layout_file, chn_names)
% h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
% U_topoplot_new(spatial_weights(3, :)', layout_file, chn_names)
% h = figure('Renderer', 'painters', 'Position', [10 10 400 300]);
% U_topoplot_new(spatial_weights(2, :)', layout_file, chn_names)
