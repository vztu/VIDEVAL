%%
% Compute features for a set of video files from datasets
% 
close all;
clear;
warning('off','all');
% add path
addpath(genpath('include'));

%%
% parameters
algo_name = 'VIDEVAL_light720_6fps'; % algorithm name, eg, 'V-BLIINDS'
data_name = 'TEST_VIDEOS';  % dataset name, eg, 'KONVID_1K'

%% parameters for VIDEVAL-light
% max_reso: downscale the frames to a fixed resolution. Used for optimizing
%           speed on high-res videos.
%           min max_reso should be 240.
% frs_per_blk: number of frames sampled per second. Used for optimizing 
%              speed on high-fps videos.
%           min frs_per_blk should be 2.
% Note this may affect the performance.
max_reso = 720;
frs_per_blk = 6;

%% *You need to customize here*
if strcmp(data_name, 'TEST_VIDEOS')
    data_path = 'videos'; % dataset video path
elseif strcmp(data_name, 'KONVID_1K')
    data_path = '/media/ztu/Seagate-ztu-ugc/KONVID_1K/KoNViD_1k_videos';
elseif strcmp(data_name, 'LIVE_VQC')
    data_path = '/media/ztu/Seagate-ztu-ugc/LIVE_VQC/VideoDatabase';
elseif strcmp(data_name, 'YOUTUBE_UGC')
    data_path = '/media/ztu/Seagate-ztu-ugc/YT_UGC/original_videos';
end

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
    % try
        % get video full path and decoded video name
        if strcmp(data_name, 'TEST_VIDEOS')
            video_name = fullfile(data_path,  filelist.video_name{i});
            yuv_name = fullfile(video_tmp, [filelist.video_name{i}, '.yuv']);
        elseif strcmp(data_name, 'KONVID_1K')
            video_name = fullfile(data_path,  [num2str(filelist.flickr_id(i)),'.mp4']);
            yuv_name = fullfile(video_tmp, [num2str(filelist.flickr_id(i)), '.yuv']);
        elseif strcmp(data_name, 'LIVE_VQC')
            video_name = fullfile(data_path, filelist.File{i});
            yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
        elseif strcmp(data_name, 'YOUTUBE_UGC')
            video_name = fullfile(data_path, filelist.category{i},...
                [num2str(filelist.resolution(i)),'P'],[filelist.vid{i},'.mkv']);
            yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
        end
        fprintf('\n---\nComputing features for %d-th sequence: %s\n', i, video_name);

        % decode video and store in temp dir
        cmd = ['ffmpeg -loglevel error -y -i ', video_name, ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
        system(cmd);  

        % get video meta data
        width = filelist.width(i);
        height = filelist.height(i);
        framerate = round(filelist.framerate(i));

        % calculate video features
        tic
        feats_mat(i,:) = calc_VIDEVAL_feats_light(yuv_name, ...
            width, height, framerate, max_reso, frs_per_blk);
        toc
        % clear cache
        delete(yuv_name)
    % catch
        % feats_mat(i,:) = NaN;
    % end
    save(out_feat_name, 'feats_mat');
end
% save feature matrix
save(out_feat_name, 'feats_mat');


