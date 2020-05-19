%%
% Compute features for a set of video files from datasets
% 
close all; 
clear;

% add path
addpath(genpath('include'));

%%
% parameters
algo_name = 'VIDEVAL'; % algorithm name, eg, 'V-BLIINDS'
data_name = 'TEST_VIDEOS';  % dataset name, eg, 'KONVID_1K'
data_path = 'videos'; % dataset video path, eg, 'KONVID_1K/KoNViD_1k_videos'

%%
% create temp dir to store decoded videos
video_tmp = 'tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = 'features';
filelist_csv = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist, 1);
out_feat_name = fullfile(feat_path, [data_name,'_',algo_name,'_feats.mat']);
feats_mat = zeros(num_videos, 60);
%===================================================

%% extract features
% parfor i = 1:num_videos % for parallel speedup
for i = 1:num_videos
    try
        % get video full path
        video_name = fullfile(data_path,  filelist.video_name{i});
        fprintf('\n---\nComputing features for %d-th sequence: %s\n', i, video_name);

        % decode video and store in temp dir
        yuv_name = fullfile(video_tmp, [filelist.video_name{i}, '.yuv']);
        cmd = ['ffmpeg -loglevel error -y -i ', video_name, ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
        system(cmd);  

        % get video meta data
        width = filelist.width(i);
        height = filelist.height(i);
        framerate = round(filelist.framerate(i));

        % calculate video features
        tic
        feats_mat(i,:) = calc_VIDEVAL_feats(yuv_name, width, height, framerate);
        toc
        % clear cache
        delete(yuv_name)
    catch
        feats_mat(i,:) = NaN;
    end
end
% save feature matrix
save(out_feat_name, 'feats_mat');


