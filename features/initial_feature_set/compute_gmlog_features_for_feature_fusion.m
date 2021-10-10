%%%%%%%%%%%%%%%%%%%%%%%
% Compute features for a set of video files from datasets
% 
close all; 
clear;
warning('off','all');
addpath(genpath('./lib/iqm/GM-LOG_release'));

%%
% parameters
algo_name = 'GMLOG_feat_sel'; 
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

video_tmp = 'video_tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
filelist_csv = fullfile(feat_path,[data_name,'_metadata.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist,1);
out_path = './feat_sel_mats';
if ~exist(out_path, 'dir'), mkdir(out_path); end
out_feat_name = fullfile(out_path, [data_name,'_',algo_name,'_feats.mat']);
out_feat_frames_name = fullfile(out_path, [data_name,'_',algo_name,'_frames_feats.mat']);
feats_mat = [];
feats_mat_frames = cell( num_videos, 1 );
%===================================================

tic
% parallel
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
    
    % read YUV frame (credit: Dae Yeol Lee)  
    fp_input = fopen(yuv_name, 'r');
    uv_width = width/2; 
    uv_height = height/2;
    feats_frames = [];

    tic
    % calculate every two frames 
    for fr = 1:2:nb_frames-5
%         fr % frame No. printer

        try
            %% Start a file pointer
            fseek(fp_input,(fr-1)*1.5*width*height, 'bof'); % Frame read for 8 bit
            %% Y component 
            %1) read y stream
            y_stream = fread(fp_input, width * height, 'uchar'); % for 8 bit
            % 2) reshape into a plane
            y_plane= reshape(y_stream, width, height).';
            feats_frames(end+1,:) = Grad_LOG_CP_TIP(y_plane);
        catch
            continue
        end
    end
    fclose(fp_input);
    delete(yuv_name)
    feats_mat_frames{i} = feats_frames; 
    
    % compute brisque mean and consistency within each 1-sec chunk!!
    cons_feats = [];
    mean_feats = [];
    n_temp_vecs = length(feats_frames(:,1));
    half_blk_len = floor(framerate/2);
    
    fprintf('Pooling mean and consistency features\n');
    
    for j = 1:half_blk_len:n_temp_vecs - half_blk_len
        
        j_start = j; 
        j_end = j + half_blk_len;
        
        % compute consistency features
        cons_feats = [cons_feats; std(feats_frames(j_start:j_end, :))];
        mean_feats = [mean_feats; mean(feats_frames(j_start:j_end, :))];
    end
    
    feats = [mean(mean_feats) ...
             mean(cons_feats) ];
         
    feats_mat(i,:) = feats;
    toc
    
    
end
toc
save(out_feat_name, 'feats_mat');
save(out_feat_frames_name, 'feats_mat_frames');

