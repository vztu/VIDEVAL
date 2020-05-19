function [f, tElapsed] = Complex_DIIVINE_feature(im)
% 82 features are extracted:
% Magnitude 1:16
% Relative Mag: 17:28
%Phase: 29:52
%CW-ssim: 53:82
% Magnitude: 1st scale -> 14 features (MGGD); 2nd scale -> 2 features (MGGD); 
%        finest scale -> 12 features (GGD, relative magnitude, combined orientations)
% Fhase: 24 features (first 2 scales, relative phase, two orientations)
% Scale Correlation: 30 features (1&2, 2&3, 1&3, hpr & 1,2)

num_scale = 3;
num_ori = 6;
if(size(im,3)~=1)
    im = (double(rgb2gray(im)));
else
    im = double(im);
end

% complex steerable pyramid decomposition
[pyr, pind] = buildSCFpyr(im, num_scale, num_ori-1);
% f = [];
tStart_1 = tic;
[band_dnt ind] = complex_divisive_normalized(pyr,pind,num_scale,num_ori,1,1,3,3);

% Magnitude features
shifts = [0 1; 1 0; 1 1; -1 1];
para_diff = [];
para = [];
b_s = [];

for i = 1:num_ori
    b = abs(band_dnt{i});
    b_s = [b_s; b(:)];
    [alpha, beta] = MGGD_ParaEstimate(b(:));
    para = [para; log(alpha) beta];
    
    b_diff = b + circshift(b, shifts(3,:)) - circshift(b, shifts(1,:)) - circshift(b, shifts(2,:));
    [sigma, gam] = gaussian_para_esti(b_diff(:));
    para_diff = [para_diff; sigma gam];
end
[alpha_s, beta_s] = MGGD_ParaEstimate(b_s);
para = [para; log(alpha_s) beta_s];

for s_n = 2:num_scale-1
    b_s = [];
    for ori_n = 1:num_ori
        b_ori = abs(band_dnt{(s_n-1)*num_ori+ori_n});
        b_s = [b_s; b_ori(:)];
    end
    [alpha_s, beta_s] = MGGD_ParaEstimate(b_s);
    para = [para; log(alpha_s) beta_s];
end
tElapsed_1 = toc(tStart_1);

f = [para(:); para_diff(:)]; 

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% horizontal and vertical relative phase
tStart_2 = tic;
ph_rela = RelativePhase_HV(pyr, pind, num_ori);
cell_len = length(ph_rela);
wrap_cauchy = [];
for k = 1:cell_len
    re_phase = ph_rela{k};
    [mu, row, loop_num] = WrapCauchyEstimate(re_phase);
    wrap_cauchy = [wrap_cauchy; row];
end
tElapsed_2 = toc(tStart_2);
f = [f; wrap_cauchy];

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% CW-SSIM
tStart_3 = tic;
K = 0;
cw_ssim_pyr = Scale_CW_SSIM(pyr, pind, num_scale, num_ori, K);
cw_ssim = cw_ssim_pyr';
cw_ssim = cw_ssim(:);
tElapsed_3 = toc(tStart_3);

f = [f; cw_ssim];

tElapsed = [tElapsed_1 tElapsed_2 tElapsed_3];


% sub-function
function phase_relative = RelativePhase_HV(pyr, pind, NumOri)
m = size(pind,1);
shifts = [0 1; 1 0];
for bnum = 2:m-1*NumOri-1
    band = pyrBand(pyr,pind,bnum);
    phase = angle(band);
    for j = 1:size(shifts, 1)
        phase_shift_hori = shift(phase,shifts(j,:));
        phase_rela_hori = phase_shift_hori - phase;
        phase_rela_hori = phase_rela_hori(:,2:end);
        ind_1_hori = phase_rela_hori < -pi;
        ind_2_hori = phase_rela_hori > pi;
        ind_3_hori = (phase_rela_hori >= -pi)&(phase_rela_hori <= pi);
        ph_rela_hori = ind_1_hori .*(phase_rela_hori+2*pi) + ind_2_hori .*(phase_rela_hori-2*pi) + ind_3_hori .*phase_rela_hori;
        phase_relative(bnum - 1, j) = {[ph_rela_hori]};
    end
end
phase_relative = phase_relative(:);

function cw_ssim_pyr = Scale_CW_SSIM(pyr, pind, num_scale, num_ori, K)
m = size(pind,1);
for bnum = 1:m
    band = pyrBand(pyr,pind,bnum);
    suband(bnum) = {[band]};    
end
for scale = 1:num_scale-1
    s1_ind = m - num_ori*scale;
    s2_ind = m - num_ori*(scale+1);    

    for ori = 1:num_ori
        band1 = suband{1,(s1_ind + ori - 1)};
        band2 = suband{1,(s2_ind + ori - 1)};
        cw_ssim_pyr(scale,ori) = cwssim_index_new(imresize(band1,size(band2)), band2, K);
    end
end

scale = 1;
s1_ind = m - num_ori*scale;
s2_ind = m - num_ori*(scale+2);    

for ori = 1:num_ori
    band1 = suband{1,(s1_ind + ori - 1)};
    band2 = suband{1,(s2_ind + ori - 1)};
    cw_ssim_ori(scale, ori) = cwssim_index_new(imresize(band1,size(band2)), band2, K);
end

st = 2; % only calculate HPR band and 1:num_scale+1-st scale subband
hp_band = suband{1,1};
for scale = st:num_scale
    s_ind = m - num_ori*scale;
    for ori = 1:num_ori
        bp_band =  suband{1,(s_ind + ori - 1)};
        cw_ssim_pyr(scale+num_scale-st,ori) = cwssim_index_new(imresize(bp_band,size(hp_band)), hp_band, K);
    end
end
cw_ssim_pyr = [cw_ssim_pyr; cw_ssim_ori];

function [mu, row, k] = WrapCauchyEstimate(AngleMatrix)
u1 = 0.3; u2 = 0.3;
e = .00001;
k = 0;
while (k < 1000)  
    mu1 = u1;
    mu2 = u2;
    w = 1./(1 - mu1*cos(AngleMatrix) - mu2*sin(AngleMatrix));
    num1 = w.*cos(AngleMatrix);
    num2 = w.*sin(AngleMatrix);    
    u1 = sum(num1(:))/sum(w(:));
    u2 = sum(num2(:))/sum(w(:));
    k = k + 1;
    if (abs(u1-mu1)< e)&&(abs(u2-mu2)< e)
        break;
    end
end
if k == 1000
    error('Cauchy data do not converge');
   %mu = 0; row = 0; k =0;
end
mu = atan(u2/u1);
row = (1 - sqrt(1 - u1^2 - u2^2))/sqrt(u1^2 + u2^2);