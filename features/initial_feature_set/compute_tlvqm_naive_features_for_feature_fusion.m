%%%%%%%%%%%%%%%%%%%%%%%
% Compute features for a set of video files from datasets
% 
close all; 
clear;
warning('off','all');
addpath(genpath('./lib/vqm/TLVQM_naive_release'));

%% 
% parameters
algo_name = 'TLVQM_naive'; 
data_name = 'KONVID_1K'; 

%% *You need to customize here*
if strcmp(data_name, 'TEST_VIDEOS')
    data_path = 'videos'; % dataset video path
elseif strcmp(data_name, 'KONVID_1K')
    data_path = '/media/ztu/Seagate-ztu-ugc/KONVID_1K/KoNViD_1k_videos';
elseif strcmp(data_name, 'LIVE_VQC')
    data_path = '/media/ztu/Seagate-ztu-ugc/LIVE_VQC/VideoDatabase';
elseif strcmp(data_name, 'YOUTUBE_UGC')
    data_path = '/media/ztu/Seagate-ztu-ugc/YT_UGC/original_videos';
elseif strcmp(data_name, 'LIVE_VQA')
    data_path = '/media/ztu/Seagate-ztu/LIVE_VQA/videos';
end

video_tmp = './video_tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = './mos_feat_files';
filelist_csv = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist,1);
if ~exist(feat_path, 'dir'), mkdir(feat_path); end
out_feat_name = fullfile(feat_path, [data_name,'_',algo_name,'_feats.mat']);
feats_mat = zeros( num_videos, 75 );
%===================================================

tic
for i = 1:num_videos
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
	elseif strcmp(data_name, 'LIVE_VQA')
	    strs = strsplit(filelist.filename{i}, '_');
	    video_name = fullfile(data_path, [strs{1}(1:2), '_Folder'], filelist.filename{i});
	    yuv_name = video_name;
	end
	fprintf('\n---\nComputing features for %d-th sequence: %s\n', i, video_name);
    % decode video and store in video_tmp dir
    if ~strcmp(video_name, yuv_name) 
    cmd = ['ffmpeg -loglevel error -y -i ', video_name, ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
    system(cmd);   
    end
    
    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));
    nb_frames = filelist.nb_frames(i);
    
    % calculate video features
    tic
    feats_mat(i,:) = compute_nrvqa_features(yuv_name, [width, height], framerate);
    toc
    delete(yuv_name)
end
toc
save(out_feat_name, 'feats_mat');
