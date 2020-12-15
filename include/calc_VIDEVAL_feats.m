%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  calc_VIDEVAL_feat.m
%  
%  This function extract the VIDEVAL feature vector (60-dim)
%
%
%  Input: 
%           test_video:             Path to the test video file (YUV420 format)
%           width, height:          Resolution of the YUV video 
%           framerate:              number of frames per second
%                       
%  Output:
%           VIDEVAL_all_features:   Resulting VIDEVAL feature vector (60-dim)
%
%

%%
function [VIDEVAL_all_features] = calc_VIDEVAL_feats(test_video, width, height, framerate)
    % Try to open test_video; if cannot, return
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        VIDEVAL_all_features = [];
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    fprintf('Video file size: %d bytes (%d frames)\n',file_length, ...
            floor(file_length/width/height/1.5));
    % get frame number
    frame_start = 1; 
    frame_end = (floor(file_length/width/height/1.5)-5);
    first_frame_loaded = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop through all the frames in the frame_range to compute frame features 
    %
    fprintf('Computing frame features every second frame on frames %d..%d\n', ...
            frame_start, frame_end);
     
    frame_features_all = [];
    for i = frame_start:2:frame_end
        % Read frames i-i, i and i+1 (note that frame_start must be > 0)
        if first_frame_loaded
            prev_YUV_frame = next_YUV_frame;
            this_YUV_frame = YUVread(test_file,[width height],i);
            next_YUV_frame = YUVread(test_file,[width height],i+1);
        else
            prev_YUV_frame = YUVread(test_file,[width height],i-1);
            this_YUV_frame = YUVread(test_file,[width height],i);
            next_YUV_frame = YUVread(test_file,[width height],i+1);
            first_frame_loaded = 1;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute BRISQUE features # 5(std), 9, 23, 29
        % features_all(1:4) #4
        brisque_feats = compute_brisque_features(this_YUV_frame(:,:,1));

        % compute GM-LOG features # 2(std), 4(std), 8, 18(std), 20, 23, 34(std), 40(both) 
        % features_all(5:12) #8
        gmlog_feats = compute_gmlog_features(this_YUV_frame(:,:,1));

        % compute HIGRADE features # 3, 13, 14, 16, 21(both), 28, 30, 33
        % features_all(13:20) #8
        higrade_feats = compute_higrade_features(this_YUV_frame(:,:,1));

        % compute FRIQUEE_lUMA features # 5(std), 8(std), 27(std), 32(std), 58, 62(std), 66, 68(std), 70(both), 71, 72(std) 
        % features_all(21:31) #11
        friquee_luma_feats = compute_friquee_luma_features(this_YUV_frame(:,:,1));

        % compute FRIQUEE_CHRO features # 4  7(std) 15   24   26   47   60   62   72   73   79
        % features_all(32:42) #11
        friquee_chroma_feats = compute_friquee_chroma_features(this_YUV_frame(:,:,:));

        % compute FRIQUEE LMS features # 52 
        % features_all(43:43) # 1
        friquee_lms_feats = compute_friquee_lms_features(this_YUV_frame(:,:,:));

        % compute TLVQM LCF features # 1 5(both) 12 16 18 19(std) 22(std)
        % features_all(44:50) # 7
        tlvqm_lcf_feats = compute_tlvqm_lcf_features(this_YUV_frame./255, ...
                                                     prev_YUV_frame./255, ...
                                                     next_YUV_frame./255); 
        % concat features
        frame_features_all = [frame_features_all; [brisque_feats, gmlog_feats, higrade_feats, ...
            friquee_luma_feats, friquee_chroma_feats, friquee_lms_feats, tlvqm_lcf_feats]];
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute mean and std pooling
    %
    std_feats = [];
    avg_feats = [];
    std_pool_idx = [1, 5, 6, 8, 11, 12, 17, 21, 22, 23, 24, 26, 28, 29, 31, ...
        33, 45, 49, 50];
    avg_pool_idx = [2:4, 7, 9, 10, 12, 13:20, 25, 27, 29, 30, 32, 34:48];
    n_temp_vecs = length(frame_features_all(:,1)); 
    half_framerate = floor(framerate/2);
    
    fprintf('Mean and standard deviation pooling of frame-wise features...\n');
    for j = 1:half_framerate:n_temp_vecs - half_framerate
        j_start = j; 
        j_end = j + half_framerate;
        % mean and std pooling
        std_feats = [std_feats; std(frame_features_all(j_start:j_end, std_pool_idx))];
        avg_feats = [avg_feats; mean(frame_features_all(j_start:j_end, avg_pool_idx))];
    end
    
    % compute TLVQM HCF features # 1 2 17 22 24 30
    tlvqm_hcf_feats = [];
    fprintf('Computing TLVQM-HCF features at 1 fr/sec...\n');
    for fr=floor(framerate/2):framerate:frame_end
        YUV_frame = YUVread(test_file,[width height],fr-1);
        ftrs = compute_tlvqm_hcf_features(YUV_frame./255);
        tlvqm_hcf_feats = [tlvqm_hcf_feats; ftrs];
    end
    
    % Combine feature vectors
    VIDEVAL_all_features = [mean(avg_feats)   ...
                    mean(std_feats) ...
                    mean(tlvqm_hcf_feats)];
    
    fclose(test_file);
    fprintf('Extracting VIDEVAL features for %s finished!\n', test_video);
end

%% compute_brisque_features
function feat = compute_brisque_features(imgray)
% Compute BRISQUE features # 5(std), 9, 23, 29
    window = fspecial('gaussian',7,7/6);
    window = window/sum(sum(window));
    feat = [];
    %% scale1 5, 9
    mu = filter2(window, imgray, 'same');
    mu_sq = mu.*mu;
    sigma = sqrt(abs(filter2(window, imgray.*imgray, 'same') - mu_sq));
    imstruct = (imgray-mu)./(sigma+1);
    shifts = [0 1; 1 0];
    for itr_shift =1:2
        shifted_imstruct = circshift(imstruct,shifts(itr_shift,:));
        pair = imstruct(:).*shifted_imstruct(:);
        [~, leftstd, ~] = estimateaggdparam(pair);
        feat =[feat leftstd^2];
    end
    
    %% scale2 5, 11
    imgray = imresize(imgray,0.5);
    mu = filter2(window, imgray, 'same');
    mu_sq = mu.*mu;
    sigma = sqrt(abs(filter2(window, imgray.*imgray, 'same') - mu_sq));
    imstruct = (imgray-mu)./(sigma+1);
    shifts = [0 1; 1 1];
    for itr_shift =1:2
        shifted_imstruct = circshift(imstruct,shifts(itr_shift,:));
        pair = imstruct(:).*shifted_imstruct(:);
        [alpha, leftstd, ~] = estimateaggdparam(pair);
        if itr_shift == 1
            feat = [feat leftstd^2];
        else 
            feat = [feat alpha];
        end
    end
end

%% compute_gmlog_features
function feat = compute_gmlog_features(imgray)
% compute GM-LOG features # 2(std), 4(std), 8, 18(std), 20, 23, 34(std), 40(both) 
    sigma = 0.5;
    [gx,gy] = gaussian_derivative(imgray,sigma);
    grad_im = sqrt(gx.^2+gy.^2);

    window2 = fspecial('log', 2*ceil(3*sigma)+1, sigma);
    window2 =  window2/sum(abs(window2(:)));
    log_im = abs(filter2(window2, imgray, 'same'));

    ratio = 2.5; % default value 2.5 is the average ratio of GM to LOG on LIVE database
    grad_im = abs(grad_im/ratio);

    %Normalization
    c0 = 4*0.05;
    sigmaN = 2*sigma;
    window1 = fspecial('gaussian',2*ceil(3*sigmaN)+1, sigmaN);
    window1 = window1/sum(window1(:));
    Nmap = sqrt(filter2(window1,mean(cat(3,grad_im,log_im).^2,3),'same'))+c0;
    grad_im = (grad_im)./Nmap;
    log_im = (log_im)./Nmap;
    % remove the borders, which may be the wrong results of a convolution
    % operation
    h = ceil(3*sigmaN);
    grad_im = abs(grad_im(h:end-h+1,h:end-h+1,:));
    log_im = abs(log_im(h:end-h+1,h:end-h+1));

    ctrs{1} = 1:10;ctrs{2} = 1:10;
    % histogram computation
    step1 = 0.20;
    step2 = 0.20;
    grad_qun = ceil(grad_im/step1);
    log_im_qun = ceil(log_im/step2);

    N1 = hist3([grad_qun(:),log_im_qun(:)],ctrs);
    N1 = N1/sum(N1(:));
    NG = sum(N1,2); NL = sum(N1,1);

    alpha1 = 0.0001;
    % condition probability: Grad conditioned on LOG
    cp_GL = N1./(repmat(NL,size(N1,1),1)+alpha1);
    cp_GL_H=  sum(cp_GL,2)';
    cp_GL_H = cp_GL_H/sum(cp_GL_H);
    % condition probability: LOG conditioned on Grad
    cp_LG = N1./(repmat(NG,1,size(N1,2))+alpha1);
    cp_LG_H = sum(cp_LG,1);
    cp_LG_H = cp_LG_H/(sum(cp_LG_H));

    out = [NG', NL, cp_GL_H,cp_LG_H];
    
    feat = out([2, 4, 8, 18, 20, 23, 34, 40]);
end

function [gx,gy] = gaussian_derivative(imd,sigma)
    window1 = fspecial('gaussian',2*ceil(3*sigma)+1+2, sigma);
    winx = window1(2:end-1,2:end-1)-window1(2:end-1,3:end);winx = winx/sum(abs(winx(:)));
    winy = window1(2:end-1,2:end-1)-window1(3:end,2:end-1);winy = winy/sum(abs(winy(:)));
    gx = filter2(winx,imd,'same');
    gy = filter2(winy,imd,'same');
end

%% compute_higrade_features
function feat = compute_higrade_features(imgray)
% compute HIGRADE features # 3, 13, 14, 16, 21(both), 28, 30, 33
[Gmag, ~] = imgradient(imgray);
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));
scale_num = 2;
feat = [];
for scale = 1:scale_num
    if scale == 1
        im = Gmag;
%         ggdpara = [];
        shifts = [0 1;1 0 ; 1 1; -1 1];

        mu = filter2(window, im, 'same');
        mu_sq = mu.*mu;
        sigma = sqrt(abs(filter2(window, im.*im, 'same') - mu_sq));
        structdis = (im-mu)./(sigma+1);
        [sigma_tmp, ~] = gaussian_para_esti(structdis(:));
        feat = [feat sigma_tmp]; %3 
        structdis = log(abs(structdis) + .1);
        for itr_shift = 1:4
            structdis_shift_tmp = circshift(structdis, shifts(itr_shift,:));
            structdis_diff = structdis - structdis_shift_tmp;
            structdis_shift(itr_shift) = {[structdis_shift_tmp]};
        end
        structdis_diff = structdis + structdis_shift{1,3} - structdis_shift{1,1} - structdis_shift{1,2};
        [sigma_tmp_0, alpha_tmp_0] = gaussian_para_esti(structdis_diff(:));
        feat = [feat  sigma_tmp_0 alpha_tmp_0]; % 13 14
        win_tmp_1 = [0 1 0; -1 0 -1; 0 1 0;];
        structdis_diff_1 = filter2(win_tmp_1, structdis, 'same');
        [~, alpha_tmp_1] = gaussian_para_esti(structdis_diff_1(:));%16
        feat = [feat alpha_tmp_1];
    else
        im = Gmag;
        shifts = [0 1;1 0 ; 1 1; -1 1];
        
        mu = filter2(window, im, 'same');
        mu_sq = mu.*mu;
        sigma = sqrt(abs(filter2(window, im.*im, 'same') - mu_sq));
        structdis = (im-mu)./(sigma+1);
        [sigma_tmp, ~] = gaussian_para_esti(structdis(:));
        feat = [feat sigma_tmp]; %3 
        
        for itr_shift = 1:4
            structdis_shift_tmp = circshift(structdis, shifts(itr_shift,:));
            if itr_shift == 3
                structdis_diff = structdis - structdis_shift_tmp;
                [~, alpha_tmp] = gaussian_para_esti(structdis_diff(:));
                feat = [feat alpha_tmp]; % 10
            elseif itr_shift == 4
                structdis_diff = structdis - structdis_shift_tmp;
                [~, alpha_tmp] = gaussian_para_esti(structdis_diff(:));
                feat = [feat alpha_tmp]; % 12
            end
            structdis_shift(itr_shift) = {[structdis_shift_tmp]};
        end
        win_tmp_1 = [0 1 0; -1 0 -1; 0 1 0;];
        structdis_diff_1 = filter2(win_tmp_1, structdis, 'same');
        [sigma_tmp_1, ~] = gaussian_para_esti(structdis_diff_1(:));%15
        feat = [feat sigma_tmp_1];
    end
    Gmag = imresize(Gmag, 0.5);
end
end

%% compute_friquee_luma_features
function feat = compute_friquee_luma_features(imgray)
% # 5(std), 8(std), 27(std), 32(std), 58, 62(std), 66, 68(std), 70(both), 71, 72(std) 
% 1:31 32:62

    scalenum=2;
    imGray1=imgray;
    
    % The array that aggregates all the features from different feature
    % maps.
    feat = [];
    %% 1 scale 31 features x2 = 62
    %% scale1: 5, 8, 27
    [structdis,~] = divisiveNormalization(imgray);
    shifts = [ 0 1;1 0];
    for itr_shift =1:2
        % Construct the product neighborhood map.
        shifted_structdis = circshift(structdis,shifts(itr_shift,:));
        pair = structdis(:).*shifted_structdis(:); % Element wise product.
        if itr_shift == 1
            feat = [feat skewness(pair(:))]; %5
        else
                    % Fit an AGGD and extract its parameters.
            [alpha, leftstd, rightstd] = estimateaggdparam(pair);
            const = (sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
            meanparam = (rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
            feat = [feat meanparam]; %8
        end
    end
    %Extract the sigma field from the image.
    sigmaMap = computeSigmaMap(imgray);
    
    % Compute kurtosis, skewness and average of the sigma field
    feat = [feat mean2(sigmaMap)];%27
    
    %% sclae2: 32,58,62 - 31 = 1, 27,31
    imgray = imresize(imgray,0.5);  
    shifts = [ 0 1];
    for itr_shift =1:1
        % Construct the product neighborhood map.
        shifted_structdis = circshift(structdis,shifts(itr_shift,:));
        pair = structdis(:).*shifted_structdis(:); % Element wise product.
        
        % Fit an AGGD and extract its parameters.
        [alpha, ~, ~] = estimateaggdparam(pair);
%         const = (sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
%         meanparam = (rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
        feat = [feat alpha];%% 1
    end
     %Extract the sigma field from the image.
    sigmaMap = computeSigmaMap(imgray);
    
    % Compute kurtosis, skewness and average of the sigma field
    feat = [feat mean2(sigmaMap)];%27
    
      % Apply DNT
    [imDivNorm,~] = divisiveNormalization(imgray);
    feat = [feat skewness(imDivNorm(:))]; %31

    %% 
    imgray=imGray1;
    %% Original Scales
    
    % 63:64
    % Features from the Yellow color channel map.
%     yFeat=yellowColorChannelMap(rgb);
%     feat = [feat yFeat];
    
    % 65:70 --> 63:68
    % Features from the Difference of Gaussian (DoG) of Sigma Map 
    
    %Get the sigma map of the image.
    sigmaMap = computeSigmaMap(imgray);
    
    % Construct two Gaussian windows where sig2 = 1.5*sig1;
    k=1.5;
    
    window1 = fspecial('gaussian',7,7/6);
    window1 = window1/sum(sum(window1));

    window2 = fspecial('gaussian',7,7*k/6);
    window2 = window2/sum(sum(window2));
    
    %Compute DOG of the sigmaMap
    DoGSigma= filter2((window1-window2), sigmaMap, 'same');
    
    %Compute the sigmaMap of DoGSigma
    DoGSigma1 = computeSigmaMap(DoGSigma);
    
    %DNT of DOG of sigma.
    divNormDoGSigma = divisiveNormalization(DoGSigma);

    %shape, variance, skewness.
    [~,lsigma] = estimateggdparam(divNormDoGSigma(:));
    
    % Apply DNT on DoGSigma1
    divNormDoGSigma1 = divisiveNormalization(DoGSigma1);
    
    feat = [feat lsigma kurtosis(divNormDoGSigma(:)) kurtosis(divNormDoGSigma1(:))];
    
    % 71:74 --> 69:72
    % Laplacian of the Luminance Map
    addpath(genpath(fullfile('include', 'matlabPyrTools')));
    
    % Extract the first laplacian of the image.
    [pyr pind] = buildLpyr(imgray,4);
    res =  pyrBand(pyr, pind, 1);
    
    % Fit a GGD to the laplacian image.
    [alpha, sigma] = estimateggdparam(res(:));
    
    
    feat = [feat alpha sigma];
%     feat = feat([5,8,27,32,58,62,64,66,68,69,70]);
end

%% compute_friquee_chroma_features
function feat = compute_friquee_chroma_features(imyuv)
%% # 4  7(std) 15   24   26   47   60   62   72   73   79
    rgb = ycbcr2rgb(uint8(imyuv));
%     lab=rgb2lab(rgb);
    colorTransform = makecform('srgb2lab');
    lab = applycform(rgb, colorTransform);

    
    % Get the A and B components of the LAB color space.
    A = double(lab(:,:,2));
    B = double(lab(:,:,3));
    
    % Compute the chroma map.
    imChroma = sqrt(A.*A + B.*B);
    
    % Initializations
    scalenum=2;   
    imChroma1=imChroma;
    
    % The array that aggregates all the features from different feature
    % maps.
    feat = [];
    
    %% scale1: 4 7 15 24
    % Applying divisive normalization operation on the given gray scale image.
    [structdis,~] = divisiveNormalization(imChroma);
    
    % The four neighborhood maps that would be constructed [1]
    shifts = [ 0 1;1 0 ; 1 1; -1 1];
    for itr_shift =1:4
        % Construct the product neighborhood map.
        shifted_structdis = circshift(structdis,shifts(itr_shift,:));
        pair = structdis(:).*shifted_structdis(:); % Element wise product.
        
        if itr_shift ==1
        % Fit an AGGD and extract its parameters.
        [~, ~, rightstd] = estimateaggdparam(pair);
        feat = [feat rightstd^2];%4
        elseif itr_shift ==2
        % Fit an AGGD and extract its parameters.
        [alpha, ~, ~] = estimateaggdparam(pair);
        feat = [feat alpha];%7
        elseif itr_shift ==3
        [~, leftstd, ~] = estimateaggdparam(pair);
        feat = [feat  leftstd^2];%15
        elseif itr_shift ==4
          feat = [feat kurtosis(pair(:))];%24
        end
    end
    sigmaMap = computeSigmaMap(imChroma);
    feat = [feat skewness(sigmaMap(:))];%26
    %% scale2:   47   60   62 - 35 = 12    25    27
    imChroma = imresize(imChroma,0.5);
 
    shifts = [1 0];

    % Construct the product neighborhood map.
    shifted_structdis = circshift(structdis,shifts);
    pair = structdis(:).*shifted_structdis(:); % Element wise product.

    feat = [feat kurtosis(pair(:))]; %12

    sigmaMap = computeSigmaMap(imChroma);
    feat = [feat kurtosis(sigmaMap(:)) mean2(sigmaMap)]; %25 27
    % Some features are computedcccccccc at multiple scales %% 1 scale 35 features x2 =
    % 70
    
    imChroma=imChroma1;
    
    %Get the sigma map of the image.
    sigmaMap = computeSigmaMap(imChroma);
        % Construct two Gaussian windows where sig2 = 1.5*sig1;
    k=1.5;
    
    window1 = fspecial('gaussian',7,7/6);
    window1 = window1/sum(sum(window1));

    window2 = fspecial('gaussian',7,7*k/6);
    window2 = window2/sum(sum(window2));
     %Compute DOG of the sigmaMap
    DoGSigma= filter2((window1-window2), sigmaMap, 'same');
    %DNT of DOG of sigma.
    divNormDoGSigma = divisiveNormalization(DoGSigma);
    %shape, variance, skewness.
    [~,lsigma] = estimateggdparam(divNormDoGSigma(:));
    feat = [feat lsigma skewness(divNormDoGSigma(:))];% 72 73
    
    % Laplacian of the Chroma Map
    addpath(genpath(fullfile('include', 'matlabPyrTools')));

    [pyr pind] = buildLpyr(imChroma,4);
    res =  pyrBand(pyr, pind, 1);
    feat = [feat skewness(res(:))];%79
end

%% compute_friquee_lms_features
function feat = compute_friquee_lms_features(imyuv)
    % compute FRIQUEE LMS features # 52 
    rgb = ycbcr2rgb(uint8(imyuv));
    lms = (colorspace('RGB->CAT02 LMS',rgb));
     % Get the M and S components from the LMS color space.
%     LM = double(lms(:,:,2));
    LS = double(lms(:,:,3));
    %FRIQUEE features from the LMS color space.
%     fM = friqueeMS(LM);%1:32
%     fS = friqueeMS(LS);%33:64
    LS = imresize(LS,0.5);
    sigmaMap = computeSigmaMap(LS);
    dnSigmaFeat = debiasedNormalizedFeats(sigmaMap);%8:11 19:20:21
%     featOpp = lmsColorOpponentFeats(lms); % Color opponency features.
    feat = dnSigmaFeat(2);
end

%% compute_tlvqm_lcf_features
function features = compute_tlvqm_lcf_features(this_fr, prev_fr, next_fr)
 % compute TLVQM LCF features # 1 5(both) 12 16 18 19(std) 22(std)
    [height,width,~] = size(this_fr);
    % Initialize parameters 
    bl_size = floor(width/40);  
    src_win = floor(width/40);
    
    % The following computations are done with reduced resolution
    this_fr = imresize(this_fr,0.5); 
    prev_fr = imresize(prev_fr,0.5);  
    next_fr = imresize(next_fr,0.5);     
    [height,width,~] = size(this_fr);   
    this_Y = this_fr(:,:,1);
    prev_Y = prev_fr(:,:,1);
    next_Y = next_fr(:,:,1);
    this_fr = ycbcr2rgb(this_fr);    
    
    % Apply Sobel filter to the frames
    H = [-1 -2 -1; 0 0 0; 1 2 1]./8;
    sob_h_this = imfilter(this_Y,H'); 
    sob_v_this = imfilter(this_Y,H);
    
    % Reset edge pixels in the Sobeled frames
    sob_h_this(1:4,1:width)=0;
    sob_h_this(height-3:height,1:width)=0;
    sob_h_this(1:height,1:4)=0;
    sob_h_this(1:height,width-3:width)=0;
    sob_v_this(1:4,1:width)=0;
    sob_v_this(height-3:height,1:width)=0;
    sob_v_this(1:height,1:4)=0;
    sob_v_this(1:height,width-3:width)=0;
          
    sob_tot = sqrt(sob_v_this.^2+sob_h_this.^2);   
    sob_h_prev = imfilter(prev_Y,H');
    sob_v_prev = imfilter(prev_Y,H); 
    sob_h_next = imfilter(next_Y,H');
    sob_v_next = imfilter(next_Y,H);     
    
    H1 = [1 1 1 1 1;1 1 1 1 1;-2 -2 0 1 1;-2 -2 -2 1 1;-2 -2 -2 1 1]./32;
    H2 = [-2 -2 -2 1 1;-2 -2 -2 1 1;-2 -2 0 1 1;1 1 1 1 1;1 1 1 1 1]./32;
    H3 = [1 1 -2 -2 -2;1 1 -2 -2 -2;1 1 0 -2 -2;1 1 1 1 1;1 1 1 1 1]./32;
    H4 = [1 1 1 1 1;1 1 1 1 1;1 1 0 -2 -2;1 1 -2 -2 -2;1 1 -2 -2 -2]./32;
    
    corner_avg(:,:,1) = abs(imfilter(this_Y, H1));
    corner_avg(:,:,2) = abs(imfilter(this_Y, H2));
    corner_avg(:,:,3) = abs(imfilter(this_Y, H3));
    corner_avg(:,:,4) = abs(imfilter(this_Y, H4));   
    corner_max = max(corner_avg,[],3);
    corner_this = corner_max-min(corner_avg,[],3); 
    
    mot_threshold = 0.01; 
    
%     cor_max = sort(corner_max(:),'ascend');
%     glob_blockiness = 0;
%     if std2(cor_max(1:floor(0.99*end)))>0
%         glob_blockiness = 0.5*((mean(cor_max(1:floor(0.99*end)))/ ...
%                           std2(cor_max(1:floor(0.99*end))))^2);
%     end
       
    % Reset edge pixels in the corner point filtered frame
    corner_this(1:src_win+3,1:width)=0;
    corner_this(height-src_win-2:height,1:width)=0;
    corner_this(1:height,1:src_win+3)=0;
    corner_this(1:height,width-src_win-2:width)=0;
                                              
    corner_this_copy = corner_this(:);   
    key_pix = zeros((height-6)*(width-6),2);
    n_key_pix = 0;
    
    im_y_vec = mod(0:width*height, height)+1;
    im_x_vec = floor((0:width*height-1)/height)+1;
    sob_this_cp = corner_this_copy(corner_this_copy>mot_threshold);
    im_y_vec = im_y_vec(corner_this_copy>mot_threshold);
    im_x_vec = im_x_vec(corner_this_copy>mot_threshold);
    
    % In the following loop, find the key pixels
    [mx,idx] = max(sob_this_cp);
    if ~isempty(idx)
        while mx>mot_threshold
            i = im_y_vec(idx(1));
            j = im_x_vec(idx(1));

            n_key_pix = n_key_pix + 1;
            key_pix(n_key_pix,:) = [i j];

            idx_remove = find(im_y_vec>=i-floor(bl_size) & ...
                              im_y_vec<=i+floor(bl_size) & ...
                              im_x_vec>=j-floor(bl_size) & ...
                              im_x_vec<=j+floor(bl_size));
            sob_this_cp(idx_remove)=[];
            im_y_vec(idx_remove)=[];
            im_x_vec(idx_remove)=[];

            [mx,idx] = max(sob_this_cp);
        end
    end
    key_pix=key_pix(1:n_key_pix,:);
       
    non_mot_area = ones(height, width);
    
    num_mot_points = 0;
    max_mot_points = (height/bl_size)*(width/bl_size);
    
    %tic
    % In the following loop, find the motion vectors for each key pixel
    motion_vec = [];
    
    distance_matrix = ones(2*src_win+1);
    for i=1:2*src_win+1
        for j=1:2*src_win+1
            distance_matrix(i,j) = ...
                sqrt((1+src_win-i).^2+(1+src_win-j).^2)/sqrt(2*src_win^2);
        end
    end
    distances = distance_matrix(:);
    
    uncertain = 0;

    % Loop through the key pixels
    for z = 1:n_key_pix

        tar_y = key_pix(z,1);
        tar_x = key_pix(z,2);
        match_y_bw = tar_y;
        match_x_bw = tar_x;
        match_y_fw = tar_y;
        match_x_fw = tar_x;
        
        surr_win_v_prev = sob_v_prev(tar_y-src_win-2:tar_y+src_win+2, ...
                                     tar_x-src_win-2:tar_x+src_win+2);
        surr_win_h_prev = sob_h_prev(tar_y-src_win-2:tar_y+src_win+2, ...
                                     tar_x-src_win-2:tar_x+src_win+2);
        diff_win_prev = (sob_v_this(tar_y, tar_x)-surr_win_v_prev).^2 + ...
                        (sob_h_this(tar_y, tar_x)-surr_win_h_prev).^2;
                    
        surr_win_v_next = sob_v_next(tar_y-src_win-2:tar_y+src_win+2, ...
                                     tar_x-src_win-2:tar_x+src_win+2);
        surr_win_h_next = sob_h_next(tar_y-src_win-2:tar_y+src_win+2, ...
                                     tar_x-src_win-2:tar_x+src_win+2);
        diff_win_next = (sob_v_this(tar_y, tar_x)-surr_win_v_next).^2 + ...
                        (sob_h_this(tar_y, tar_x)-surr_win_h_next).^2;
                    
        for i=-1:1
            for j=-1:1
                if i~=0 || j~=0
                    diff_win_prev(3:end-2,3:end-2) = ...
                        diff_win_prev(3:end-2,3:end-2) + ...
                        (sob_v_this(tar_y+i, tar_x+j)- ...
                          surr_win_v_prev(3+i:end-2+i,3+j:end-2+j)).^2+ ...
                        (sob_h_this(tar_y+i, tar_x+j)- ...
                          surr_win_h_prev(3+i:end-2+i,3+j:end-2+j)).^2;   
                    diff_win_next(3:end-2,3:end-2) = ...
                        diff_win_next(3:end-2,3:end-2) + ...
                        (sob_v_this(tar_y+i, tar_x+j)- ...
                          surr_win_v_next(3+i:end-2+i,3+j:end-2+j)).^2+...
                        (sob_h_this(tar_y+i, tar_x+j)- ...
                        surr_win_h_next(3+i:end-2+i,3+j:end-2+j)).^2;   
                end
            end
        end
        diff_win_prev = diff_win_prev(3:end-2,3:end-2);
        diff_win_next = diff_win_next(3:end-2,3:end-2);
                    
        orig_diff_bw = diff_win_prev(1+src_win,1+src_win);
        orig_diff_fw = diff_win_next(1+src_win,1+src_win);
 
        diff_bw = diff_win_prev(1+src_win,1+src_win);   
        if orig_diff_bw>0.005
            [sorted,idx] = sort(diff_win_prev(:),'ascend');
            min_diff = orig_diff_bw;
            if length(sorted)>=2
                if sorted(1)<=0.8*sorted(2) || ...
                   distances(idx(1))<distances(idx(2))
                    min_diff = sorted(1);
                else
                    [idx,~] = find(0.8.*diff_win_prev(:)<=sorted(1));
                    [~,idx2] = sort(distances(idx),'ascend');
                    if diff_win_next(idx(idx2(1)))<1.1*sorted(1)
                        min_diff = diff_win_prev(idx(idx2(1)));
                    elseif sorted(1)<diff_bw*0.9
                        min_diff = sorted(1);
                    end
                    uncertain = uncertain + 1;
                end
                if min_diff*1.01<orig_diff_bw
                    [y,x] = find(diff_win_prev==min_diff);
                    match_y_bw = tar_y+y(1)-src_win-1;
                    match_x_bw = tar_x+x(1)-src_win-1;        
                    diff_bw = diff_win_prev(y(1),x(1));
                end
            end
        end
        
        diff_fw = diff_win_next(1+src_win,1+src_win);  
        if orig_diff_fw>0.005
            [sorted,idx] = sort(diff_win_next(:),'ascend');
            min_diff = orig_diff_fw;
            if length(sorted)>=2
                if sorted(1)<0.8*sorted(2) || ...
                   distances(idx(1))<distances(idx(2))
                    min_diff = sorted(1);
                else                
                    [idx,~] = find(0.8.*diff_win_next(:)<=sorted(1));
                    [~,idx2] = sort(distances(idx),'ascend');
                    if diff_win_next(idx(idx2(1)))<1.1*sorted(1)
                        min_diff = diff_win_next(idx(idx2(1)));
                    elseif sorted(1)<diff_fw*0.9
                        min_diff = sorted(1);
                    end
                    uncertain = uncertain + 1;
                end
                if min_diff*1.01<orig_diff_fw
                    [y,x] = find(diff_win_next==min_diff);
                    match_y_fw = tar_y+y(1)-src_win-1;
                    match_x_fw = tar_x+x(1)-src_win-1;        
                    diff_fw = diff_win_next(y(1),x(1));
                end  
            end
        end
             
        % Add motion vector to the list of motion vectors
        if (orig_diff_bw > diff_bw*1.01 && ...
                (tar_y ~= match_y_bw || tar_x ~= match_x_bw)) || ...
           (orig_diff_fw > diff_fw*1.01 && ...
                (tar_y ~= match_y_fw || tar_x ~= match_x_fw))     

            non_mot_area(max(1,tar_y-bl_size):min(height,tar_y+bl_size),...
                max(1,tar_x-bl_size):min(width,tar_x+bl_size))=0;
            non_mot_area(max(1,match_y_bw-bl_size): ...
                min(height,match_y_bw+bl_size),...
                max(1,match_x_bw-bl_size):...
                min(width,match_x_bw+bl_size)) = 0;
            non_mot_area(max(1,match_y_fw-bl_size):...
                min(height,match_y_fw+bl_size),...
                max(1,match_x_fw-bl_size):...
                min(width,match_x_fw+bl_size)) = 0;
        end
        
        num_mot_points = num_mot_points + 1;
        motion_vec = [motion_vec; ...
                      tar_y-match_y_bw tar_x-match_x_bw ...
                      match_y_fw-tar_y match_x_fw-tar_x ...
                      tar_y tar_x ...
                      orig_diff_bw diff_bw ...
                      orig_diff_fw diff_fw];
    end
    %toc 
 % Compute motion point related statistics
%     motion_uncertainty = 0.5*uncertain/max_mot_points;
%     motion_density = 0;
    motion_intensity = 0;
%     std_mot_intensity = 0;
%     avg_mot_pos = 0;
%     avg_mot_sprd = 0;
%     mot_pred_acc = 0;
    mot_y = 0.5;
%     mot_x = 0.5;
    jerkiness = 0;
%     jerk_cons = 0;
    motion_vec_bg = [];
    num_bg_mot_points = 0;
    if num_mot_points>0
%         motion_density = num_mot_points/(width*height/bl_size^2);    
        mot_intensity_vec = sqrt(((motion_vec(:,1)./src_win).^2 + ...
                                  (motion_vec(:,2)./src_win).^2 + ...
                                  (motion_vec(:,3)./src_win).^2 + ...
                                  (motion_vec(:,4)./src_win).^2)./4.0);
        sum_mot_int = sum(mot_intensity_vec);
        motion_intensity = (sum(mot_intensity_vec)/max_mot_points)^0.25;
%         std_mot_intensity = std(mot_intensity_vec);
        
        if sum_mot_int>0
            % Compute motion position in relation with the screen midpoint
%             avg_motp_y = sum(mot_intensity_vec.*motion_vec(:,5))/...
%                            sum_mot_int;
%             std_motp_y = sqrt(sum(mot_intensity_vec.*...
%                            (motion_vec(:,5)-avg_motp_y).^2)/sum_mot_int);
%             avg_mot_pos_y = (avg_motp_y-height/2)/(height/2);
%             sprd_mot_pos_y = std_motp_y/height;  
%             avg_motp_x = sum(mot_intensity_vec.*motion_vec(:,6))/...
%                            sum_mot_int;
%             std_motp_x = sqrt(sum(mot_intensity_vec.*...
%                            (motion_vec(:,6)-avg_motp_x).^2)/sum_mot_int);
%             avg_mot_pos_x = (avg_motp_x-width/2)/(width/2);
%             sprd_mot_pos_x = std_motp_x/width;
% 
%             avg_mot_pos = sqrt(avg_mot_pos_y^2+avg_mot_pos_x^2);  
%             avg_mot_sprd = sqrt(sprd_mot_pos_y^2+sprd_mot_pos_x^2);

            % Mean motion along x and y axis
            mot_y = mean(0.25.*(motion_vec(:,1)+motion_vec(:,3))./ ...
                      src_win+0.5);    
%             mot_x = mean(0.25.*(motion_vec(:,2)+motion_vec(:,4))./ ...
%                       src_win+0.5);

            % Average motion prediction improvement
%             mot_pred_acc_bw = mean(motion_vec(:,7)-motion_vec(:,8));
%             mot_pred_acc_fw = mean(motion_vec(:,9)-motion_vec(:,10));
%             mot_pred_acc = 0.5*(mot_pred_acc_bw+mot_pred_acc_fw).^0.5;

            % Motion jerkiness
            mot_y_diff = 0.5.*(motion_vec(:,1)'-motion_vec(:,3)')./src_win;
            mot_x_diff = 0.5.*(motion_vec(:,2)'-motion_vec(:,4)')./src_win;
            mot_diff = sqrt(mot_y_diff.^2+mot_x_diff.^2);
            jerkiness = mean(mot_diff.^0.5);        
%             jerk_cons = std(mot_diff.^0.5);
        end
        
        avg_mot_x = mean(0.5.*motion_vec(:,2)+0.5.*motion_vec(:,4));
        avg_mot_y = mean(0.5.*motion_vec(:,1)+0.5.*motion_vec(:,3));
        std_mot_x = std(0.5.*motion_vec(:,2)+0.5.*motion_vec(:,4));
        std_mot_y = std(0.5.*motion_vec(:,1)+0.5.*motion_vec(:,3));

        for z=1:num_mot_points
            mot_x_this = 0.5*motion_vec(z,2)+0.5*motion_vec(z,4);
            mot_y_this = 0.5*motion_vec(z,1)+0.5*motion_vec(z,3);
            if mot_x_this > avg_mot_x-std_mot_x && ...
               mot_x_this < avg_mot_x+std_mot_x && ...
               mot_y_this > avg_mot_y-std_mot_y && ...
               mot_y_this < avg_mot_y+std_mot_y

                num_bg_mot_points = num_bg_mot_points + 1;
                motion_vec_bg = [motion_vec_bg; motion_vec(z,:)];
            end
        end
    end
 % Compute motion point related statistics
%     egomotion_density = 0;
%     egomotion_intensity = 0;
    std_egomot_intensity = 0;
%     avg_egomot_pos = 0;
%     avg_egomot_sprd = 0;
%     egomot_pred_acc = 0;
%     mot_y_bg = 0.5;
%     mot_x_bg = 0.5;
    if num_bg_mot_points>0
%         egomotion_density = num_bg_mot_points/(width*height/bl_size^2);    
        bg_mot_intensity_vec = sqrt(((motion_vec_bg(:,1)./src_win).^2 + ...
                                     (motion_vec_bg(:,2)./src_win).^2 + ...
                                     (motion_vec_bg(:,3)./src_win).^2 + ...
                                     (motion_vec_bg(:,4)./src_win).^2)  ...
                                      ./4.0);
%         sum_bg_mot_int = sum(bg_mot_intensity_vec);
%         egomotion_intensity = (sum(bg_mot_intensity_vec)/...
%                                 max_mot_points)^0.25;
        std_egomot_intensity = std(bg_mot_intensity_vec);
        
        % Compute motion position in relation with the screen midpoint
%         if sum_bg_mot_int>0
%             avg_motp_y = sum(bg_mot_intensity_vec.*motion_vec_bg(:,5))/...
%                            sum_bg_mot_int;
%             std_motp_y = sqrt(sum(bg_mot_intensity_vec.*...
%                            (motion_vec_bg(:,5)-avg_motp_y).^2)/...
%                               sum_bg_mot_int);
%             avg_mot_pos_y = (avg_motp_y-height/2)/(height/2);
%             sprd_mot_pos_y = std_motp_y/height;  
%             avg_motp_x = sum(bg_mot_intensity_vec.*motion_vec_bg(:,6))/...
%                            sum_bg_mot_int;
%             std_motp_x = sqrt(sum(bg_mot_intensity_vec.*...
%                            (motion_vec_bg(:,6)-avg_motp_x).^2)/...
%                            sum_bg_mot_int);
%             avg_mot_pos_x = (avg_motp_x-width/2)/(width/2);
%             sprd_mot_pos_x = std_motp_x/width;
% 
%             avg_egomot_pos = sqrt(avg_mot_pos_y^2+avg_mot_pos_x^2);  
%             avg_egomot_sprd = sqrt(sprd_mot_pos_y^2+sprd_mot_pos_x^2);

            % Average egomotion prediction improvement
%             mot_pred_acc_bw = mean(motion_vec_bg(:,7)-motion_vec_bg(:,8));
%             mot_pred_acc_fw = mean(motion_vec_bg(:,9)-motion_vec_bg(:,10));
%             egomot_pred_acc = 0.5*(mot_pred_acc_bw+mot_pred_acc_fw).^0.5;

%             mot_y_bg = mean(0.25.*(motion_vec_bg(:,1)+...
%                                    motion_vec_bg(:,3))./src_win+0.5);    
%             mot_x_bg = mean(0.25.*(motion_vec_bg(:,2)+...
%                                    motion_vec_bg(:,4))./src_win+0.5);        
%         end
    end

%     mot_size = sum(sum(1-non_mot_area));  
    non_mot_size = sum(sum(non_mot_area));  
%  static_area_flicker = 0;
    static_area_flicker_std = 0;
    if non_mot_size>0
        % Sum of the pixel differences in the static area
        static_area_flicker_bw = sum(non_mot_area(:) .* ...
                                 abs(this_Y(:)-prev_Y(:)))/non_mot_size;
        static_area_flicker_fw = sum(non_mot_area(:) .* ...
                                 abs(this_Y(:)-next_Y(:)))/non_mot_size;
        static_area_flicker = 0.5*(static_area_flicker_bw + ...
                                   static_area_flicker_fw);
        % Variance of pixel differences in the static area
        st_diff_bw = abs(this_Y(:)-prev_Y(:));
        st_diff_fw = abs(this_Y(:)-next_Y(:));
        static_area_flicker_std = sum(non_mot_area(:)' .* ...
                                  abs(max([st_diff_bw'; st_diff_fw']) - ...
                                  static_area_flicker))/non_mot_size;
    end
    
    % Spatial activity in the static area
    si = std2(sob_tot).^0.25;
    
    %[blur glob_blockiness si]
    
    % Temporal activity standard deviation in the static area
    ti_prev = mean(abs(this_Y(:)-prev_Y(:)));
    ti_next = mean(abs(this_Y(:)-next_Y(:)));
    ti_mean = mean([ti_prev ti_next]).^0.25;
      
    % Normalize static area size
%     mot_size = mot_size / (width*height);
    features = [motion_intensity,  ...
                std_egomot_intensity, ...
                si, ...
                jerkiness, ...
                ti_mean, ...
                mot_y, ...
                static_area_flicker_std];
end

%% compute_tlvqm_hcf_features
function features = compute_tlvqm_hcf_features(imyuv)
% compute TLVQM HCF features # 1 2    17    22    24    30
% Initializations
    mono_image = imyuv(:,:,1);
    image = ycbcr2rgb(imyuv);
    lab_image = rgb2lab(image);
    [height,width,depth] = size(image);
        
    % Make Sobeled image
    mask = zeros(height,width);
    mask(2:end-1,2:end-1)=1;
    H = [1 2 1; 0 0 0; -1 -2 -1]./8;
        
    % Make Sobeled image in CIELAB color space
    sob_image_lab_x = (imfilter(lab_image(:,:,1)./100.0,H).^2 + ...
                       imfilter(lab_image(:,:,2)./50.0,H).^2 + ...
                       imfilter(lab_image(:,:,3)./50.0,H).^2).*mask;
    sob_image_lab_y = (imfilter(lab_image(:,:,1)./100.0,H').^2 + ...
                       imfilter(lab_image(:,:,2)./50.0,H').^2 + ...
                       imfilter(lab_image(:,:,3)./50.0,H').^2).*mask;
    sob_image = sqrt(sob_image_lab_x+sob_image_lab_y);

    % Compute fetures for different feature groups
%     [a,b,sat_image1] = compute_saturation(mono_image,1);
% %     sat_bright = [a];
%     [a,b,sat_image2] = compute_saturation(mono_image,0);
% %     sat_dark = [a b];
%     sat_image = max(sat_image1, sat_image2);
%     saturation_ftr = [sat_bright sat_dark]; % 5:8
%     saturation_ftr = sat_bright;
    spatial_ftr = spatial_activity_features(sob_image, []);% 1:4
% %     noisiness_ftr = noise_features(mono_image, sat_image, lab_image); % 9:11
%     blockiness_ftr = blockiness_features(sob_image_lab_x.^0.5, ...
%                                          sob_image_lab_y.^0.5); % 12:14
    contrast_color_ftr = contrast_chroma_features(lab_image, []); % 15:18
%     dct_ftr = dct_features(mono_image); % 19:21
    sharpness_ftr = sharpness_features(sob_image); % 22:30

    % Make the HC feature vector
    features = [spatial_ftr   ...
                contrast_color_ftr   ...
                sharpness_ftr];
end

% This function computes the saturation (bright or dark)
function [len,num,segs] = compute_saturation(image, is_bright)

    [height,width] = size(image);

    lens = [];
    num = 0;    
    
    segs = zeros(height,width);
    
    if (is_bright==1 && max(max(image))>0.9) || ...
       (is_bright==0 && min(min(image))<0.1)     
    
        segs = seg_loop(image,segs,3,3,0.05, is_bright);
        for i=1:max(max(segs))
            len = length(find(segs==i));
            if len<50
                segs(find(segs==i))=0;
            else
                lens = [lens len];
                num = num + 1;
            end
        end 
        segs(find(segs>0))=1;
    end
    
    len = sum(lens)/(width*height);
    if num > 0
        num = len / num;
    end

end

% This function is used for segmentation by measure_saturation
function segim = seg_loop(image, segs, wh, ww, interval, is_bright)

    [height,width] = size(image);

    segim = segs;
    
    maxi = max(max(image));
    mini = min(min(image));
    
    for i=1:height-wh+1
        for j=1:width-ww+1
            if (is_bright == 1 && ...
              min(min(image(i:i+wh-1,j:j+ww-1)))>maxi-interval) || ...
              (is_bright == 0 && ...
              max(max(image(i:i+wh-1,j:j+ww-1)))<mini+interval)
            
                maxsg = max(max(segim(i:i+wh-1,j:j+ww-1)));
                if maxsg>0
                    segs_temp = reshape(segim(i:i+wh-1,j:j+ww-1),wh*ww,1);
                    minsg=min(segs_temp(find(segs_temp>0)));
                    segim(i:i+wh-1,j:j+ww-1)=minsg;
                    if minsg<maxsg
                        segim(find(segim==maxsg))=minsg;
                    end
                else
                    segim(i:i+wh-1,j:j+ww-1)=max(max(segim(:,:)))+1;
                end
            end
        end
    end

end

% This function is used to compute noise related features
function out = noise_features(mono_image, sat_im, lab_image)
    
    [height,width] = size(mono_image);

    new_im = zeros(height, width, 3);

    nonsat_pix = 0;
    noise_pix = 0;
    noise_int = [];
    
    % Loop through pixels to find noise pixels
    for i=5:height-4
        for j=5:width-4
            if sat_im(i,j)==0
                surr_pix = mono_image(i-2:i+2,j-2:j+2);
                surr_pix = surr_pix(:);
                surr_pix = [surr_pix(1:12); surr_pix(14:25)];
                if (mono_image(i,j)>max(surr_pix) || ...
                    mono_image(i,j)<min(surr_pix))
                    surr_pix = mono_image(i-4:i+4,j-4:j+4);
                    if std(surr_pix)<0.05
                        new_im(i,j,2) = 1;
                        pix_diff = sqrt( ...
                            (mean(lab_image(i-3:i+3,j-3:j+3,1))-...
                                 lab_image(i,j,1)).^2 + ...
                            (mean(lab_image(i-3:i+3,j-3:j+3,2))-...
                                 lab_image(i,j,2)).^2 + ...
                            (mean(lab_image(i-3:i+3,j-3:j+3,3))-...
                                  lab_image(i,j,3)).^2);
                        noise_int = [noise_int pix_diff/100]; 
                        noise_pix = noise_pix + 1;
                    end
                end
                nonsat_pix = nonsat_pix + 1;
            end
        end
    end

    a = 0;
    b = 0;
    c = 0;
    
    if nonsat_pix > 0 && noise_pix > 0
        % noise density
        a = noise_pix / nonsat_pix;
        b = mean(noise_int);
        c = std(noise_int);
    end
    
    out = [a b c];
       
end

% This function is used to compute spatial activity features
function out = spatial_activity_features(sobel_image, ~)
    
%     [height,width] = size(sobel_image);
%        
%     sob_dists = zeros(1,height*width);
%     sob_dists2 = zeros(height*width,2);
%     sob_str = zeros(1,height*width);
%     sumstr = 0;
%     
%     n = 0;
%     for i=1:height
%         for j=1:width
%             if sat_image(i,j)==0
%                 if sobel_image(i,j)<0.01
%                     sobel_image(i,j)=0;
%                 end
%                 sumstr = sumstr + sobel_image(i,j);
%                 if sobel_image(i,j) > 0
%                     n = n + 1;
%                     sob_str(n) = sobel_image(i,j);
%                     sob_dists(n) = sqrt((i/height-0.5)^2+(j/width-0.5)^2);
%                     sob_dists2(n,1) = i/height-0.5;
%                     sob_dists2(n,2) = j/width-0.5;                   
%                 end
%             end
%         end
%     end  
%     
%     sob_str = sob_str(1:n);
%     sob_dists = sob_dists(1:n);
%     sob_dists2 = sob_dists2(1:n,:);

%     a = 0;
%     b = 0;
%     c = 0;
%     d = 0;
    
%     if ~isempty(sob_str)>0
        a = mean(mean(sobel_image));
        b = std2(sobel_image);
%         d = w_std(sob_dists, sob_str);        
%         mean_y = sum(sob_str'.*sob_dists2(:,1))/sum(sob_str);
%         mean_x = sum(sob_str'.*sob_dists2(:,2))/sum(sob_str);        
%         c = sqrt(mean_y^2+mean_x^2);
%     end
    
%     out = [a b c d];
    out = [ a b];

end

% Function for "weighted standard deviation", used by function
% measure_spatial_activity
function res = w_std(input, weights)

    wg_n = sum(weights);
    wg_input = input.*weights;
    wg_mean = mean(input.*weights);
    
    res = sqrt(sum((wg_input-wg_mean).^2)/wg_n);
end

% This function is used to compute blockiness index
function blockiness = blockiness_features(sob_y, sob_x)
    
    [height,width] = size(sob_y);
       
    hor_tot = zeros(1,height-4);
    ver_tot = zeros(1,width-4);
    
    for i=3:height-2
        hor_tot(i)=mean(sob_y(i,:)-sob_x(i,:));
    end
    for j=3:width-2
        ver_tot(j)=mean(sob_x(:,j)-sob_y(:,j));
    end
    
    % compute autocorrelations
    autocr_hor = zeros(1,23);
    autocr_ver = zeros(1,23);
    for i=0:22
        autocr_hor(i+1) = sum(hor_tot(1:end-i).*hor_tot(1+i:end));
        autocr_ver(i+1) = sum(ver_tot(1:end-i).*ver_tot(1+i:end));
    end
    
    % Find the highest local maximum (other than 0)
    localpeaks = 0;
%     peakdist = 0;
    max_hor = 0;
    max_ver = 0;
    min_hor = autocr_hor(1);
    min_ver = autocr_ver(1);
    max_hor_diff = 0;
    max_ver_diff = 0;
    for i=2:22
        if autocr_hor(i)>max(autocr_hor(i-1),autocr_hor(i+1))
            localpeaks = localpeaks+1/42;
        end
        if autocr_hor(i)<min(autocr_hor(i-1),autocr_hor(i+1)) && ...
                autocr_hor(i)<min_hor
            min_hor = autocr_hor(i);
        elseif autocr_hor(i)>max(autocr_hor(i-1),autocr_hor(i+1)) && ...
                autocr_hor(i)-min_hor>max_hor_diff
            max_hor = autocr_hor(i);
            max_hor_diff = max_hor-min_hor;
%             peakdist = (i-1)/21;
        end
        if autocr_ver(i)>max(autocr_ver(i-1),autocr_ver(i+1))
            localpeaks = localpeaks + 1/42;
        end
        if autocr_ver(i)<min(autocr_ver(i-1),autocr_ver(i+1)) && ...
                autocr_ver(i)<min_ver
            min_ver = autocr_ver(i);
        elseif autocr_ver(i)>max(autocr_ver(i-1),autocr_ver(i+1)) && ...
                autocr_ver(i)-min_ver>max_ver_diff
            max_ver = autocr_ver(i);
            max_ver_diff = max_ver-min_ver;
%             peakdist = (i-1)/21;
        end
    end
    
    a = 0;
    if autocr_hor(1)>0 && autocr_ver(1)>0
        if max_hor>0 && max_ver>0
            a = max((max_hor_diff/autocr_hor(1)), ...
                             (max_ver_diff/autocr_ver(1)))^0.5;
        elseif max_hor>0
            a = (max_hor_diff/autocr_hor(1))^0.5;
        elseif max_ver>0
            a = (max_ver_diff/autocr_ver(1))^0.5;
        end
    end
    
%     b = peakdist;
%     c = localpeaks;
    blockiness = a;
end

% This function is used to compute contrast and chroma related features
function out = contrast_chroma_features(lab_image, ~)

%     a=0;
%     b=0;
    c=0;
%     d=0;
    
%     [height,width,depth] = size(lab_image);
%     yuv_int = floor(lab_image(:,:,1));
    
    %sat_image = sat_image(:);
%     yuv_int2 = yuv_int(sat_image(:)==0);
%     cumu_err = 0;
%     cumu_tar = 0;
%     if ~isempty(yuv_int2)
%         for i=0:100
%             cumu_tar = cumu_tar + 1/100;
%             cumu_err = cumu_err + (sum(yuv_int2<=i)/length(yuv_int2) - ...
%                                    cumu_tar)/100;
%         end
% %         a = (cumu_err+1.0)/2.0;
%         b = 0.5*(1-cumu_err);
%     else
% %         a = 1;
%         b = sum(sum(lab_image(:,:,1)))/50;
%     end
    c = sqrt(mean(mean((lab_image(:,:,2)./50).^2 + ...
         (lab_image(:,:,3)./50).^2)));
%     d = 0;
%     if std2(lab_image(:,:,1))>0
%         d = 0.01*(std2(lab_image(:,:,2))+std2(lab_image(:,:,3)));
%     end
    
    out = [c];
end
    
% This function is used to compute dct derived features
function out = dct_features(im)
    
    % Input is monochrome image
    [height,width] = size(im);
    
    out_im = abs(dct2(im)).^.5;
    
    area1 = imresize(out_im(1:floor(height/2),1:floor(width/2)),0.25);
    area2 = imresize(out_im(1:floor(height/2),...
                            width:-1:width-floor(width/2)+1),0.25);
    area3 = imresize(out_im(height:-1:height-floor(height/2)+1,...
                            1:floor(width/2)),0.25);
    area4 = imresize(out_im(height:-1:height-floor(height/2)+1,...
                            width:-1:width-floor(width/2)+1),0.25);
    a = max(0,max(corr(area1(:),area2(:)),corr(area1(:),area3(:))));
%     b = 0;
%     if mean(area1)>0
%         b = mean(area4)/mean(area1);
%     end
%     c = 0;
%     if max(mean(area2),mean(area3))>0
%         c = min(mean(area2),mean(area3))/max(mean(area2),mean(area3));
%     end
    
    out = [a];
    
end

% This function is used to compute sharpness related features
function out = sharpness_features(im)

    [~, width] = size(im);
    
    % Full HD video could be downsized
    if width>1280
        im = imresize(im,0.5);
    end
    
    H = [-1 -2 -1; 0 0 0; 1 2 1]./8;
    im_s_h = imfilter(im,H');
    im_s_v = imfilter(im,H);
    im_s = sqrt(im_s_h.^2+im_s_v.^2);
    
    [height,width] = size(im_s_h);
    bl_size = 16;
    conv_list = [];
    
    blur_im = zeros(height,width);
    edge_strong = [];
    edge_all = [];
    
    conv_cube = [];
    blurvals = [];
    
    n_blks = 0;
    
    conv_val_tot = zeros(17);
    for y=floor(bl_size/2):bl_size:height-ceil(3*bl_size/2)
        for x=floor(bl_size/2):bl_size:width-ceil(3*bl_size/2)
            
            n_blks = n_blks + 1;
            
            conv_val = zeros(17);
            for i=0:6
                for j=0:6
                    if i==0 || j==0 || i==j
                        weight_h = 1;
                        weight_v = 1;
                        if i~=0 || j~=0
                            weight_h = abs(i)/(abs(i)+abs(j));
                            weight_v = abs(j)/(abs(i)+abs(j));
                        end
                        diff_h = (im_s_h(y+i:y+bl_size+i,   ...
                                         x+j:x+bl_size+j).* ...
                                  im_s_h(y:y+bl_size,       ...
                                         x:x+bl_size));
                        diff_v = (im_s_v(y+i:y+bl_size+i,   ...
                                         x+j:x+bl_size+j).* ...
                                  im_s_v(y:y+bl_size,       ...
                                         x:x+bl_size));
                        conv_val(i+9,j+9) = weight_h*(mean(diff_h(:)))+ ...
                                            weight_v*(mean(diff_v(:)));
                    end
                end
            end
            blur_im(y:y+bl_size-1,x:x+bl_size-1)=0.5;
            edge_all =  [edge_all conv_val(9,9)];
            if conv_val(9,9)>0.0001
                edge_strong =  [edge_strong conv_val(9,9)];
                conv_val=conv_val./conv_val(9,9);
                conv_val_tot = conv_val_tot + conv_val;

                new_conv_v = [];
                for i=1:6
                    new_conv_v = [new_conv_v sum(sum(conv_val(9-i:9+i,...
                                                            9-i:9+i)))- ...
                                             sum(sum(conv_val(10-i:8+i, ...
                                                            10-i:8+i)))];
                end
                if new_conv_v(1)>0
                    new_conv_v=new_conv_v./new_conv_v(1);
                end

                conv_list = [conv_list; new_conv_v];
                conv_cube(:,:,1)=conv_val;
                blurvals = [blurvals std2(im_s(y:y+bl_size, x:x+bl_size))];

                blur_im(y:y+bl_size-1,x:x+bl_size-1) = ...
                                  0.5 + mean(new_conv_v(2:6))/5;
            end
        end
    end

    % Find the sharpest blocks
    blurs_sharp = [];
    blurs_blur = [];
    if ~isempty(blurvals)    
        for i=1:length(blurvals)
            if blurvals(i)>mean(blurvals)
                conv_val_tot = + conv_val_tot + conv_cube(:,:,1);
                blurs_sharp = [blurs_sharp blurvals(i)];
            else
                blurs_blur = [blurs_blur blurvals(i)];
            end
        end
    end
    
    n_sharps = length(blurs_sharp)/n_blks;
    n_blurs = length(blurs_blur)/n_blks;
    mean_sharps = 0;
    mean_blurs = 0;
    if ~isempty(blurs_sharp)
        mean_sharps = mean(blurs_sharp);
    end
    if ~isempty(blurs_blur)
        mean_blurs = mean(blurs_blur);
    end
    
%     if conv_val_tot(9,9)>0
%         conv_val_tot=conv_val_tot./conv_val_tot(9,9);
%     end
    
    new_conv_v = zeros(1,9);
    if ~isempty(edge_strong)>0
        if length(conv_list(:,1))>1
            new_conv_v = mean(conv_list);
        else
            new_conv_v = conv_list;
        end
    end 
       
    % find local min and/or local max
%     localmin=0;
%     localmindist=0;
%     localmax=0;
%     localmaxdist=0;
%     for i=9:14
%         for j=9:14
%             if (i~=9 && j==9) || (j~=9 && i==9) || (i==j && i>9) 
%                 conv_val_comp=conv_val_tot(i-1:i+1,j-1:j+1);
%                 conv_val_comp=conv_val_comp(:);
%                 if i==9
%                     conv_val_comp=conv_val_comp([4 6]);
%                 elseif j==9
%                     conv_val_comp=conv_val_comp([2 8]);
%                 else
%                     conv_val_comp=conv_val_comp([1 9]);
%                 end    
%                 if conv_val_tot(i,j)>max(conv_val_comp) && ...
%                         conv_val_tot(i,j)>localmax
%                     localmax = conv_val_tot(i,j);
% %                     localmaxdist = sqrt((i-9)^2+(j-9)^2);
%                 elseif conv_val_tot(i,j)<min(conv_val_comp) && ...
%                         conv_val_tot(i,j)<localmin
%                     localmin = conv_val_tot(i,j);
% %                     localmindist = sqrt((i-9)^2+(j-9)^2);
%                 end
%             end
%         end
%     end

%     out = [mean(new_conv_v(2:6)) mean(new_conv_v(2:4)) new_conv_v(2) ...           
%            localmaxdist/5 localmindist/5 ...
%            n_sharps n_blurs mean_sharps mean_blurs];
       
    out = [mean(new_conv_v(2:6)) new_conv_v(2) mean_blurs];
end


% Read one frame from YUV file
function YUV = YUVread(f,dim,frnum)

    % This function reads a frame #frnum (0..n-1) from YUV file into an
    % 3D array with Y, U and V components
    
    fseek(f,dim(1)*dim(2)*1.5*frnum,'bof');
    
    % Read Y-component
    Y=fread(f,dim(1)*dim(2),'uchar');
    if length(Y)<dim(1)*dim(2)
        YUV = [];
        return;
    end
    Y=cast(reshape(Y,dim(1),dim(2)),'double');
    
    % Read U-component
    U=fread(f,dim(1)*dim(2)/4,'uchar');
    if length(U)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end
    U=cast(reshape(U,dim(1)/2,dim(2)/2),'double');
    U=imresize(U,2.0);
    
    % Read V-component
    V=fread(f,dim(1)*dim(2)/4,'uchar');
    if length(V)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end    
    V=cast(reshape(V,dim(1)/2,dim(2)/2),'double');
    V=imresize(V,2.0);
    
    % Combine Y, U, and V
    YUV(:,:,1)=Y';
    YUV(:,:,2)=U';
    YUV(:,:,3)=V';
end
    
