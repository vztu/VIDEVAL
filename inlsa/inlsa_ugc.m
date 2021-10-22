close all
clear
lib_dir = '../../lib'; addpath(genpath(lib_dir))
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

%Demo showing the use of INLSA.m with made-up data.
randn('state',0);
rand('state',0);

error_ratio=0;  %can be used to allocate arror between MOS and parameters
norm_set=1;     
%experiment 1 is the reference and will not be adjusted
%can change this to 2 or 3 and see what happens
%use the experiment that you trust most as the reference

%Experiment 1  
exp1 = 'YouTube-UGC';
load('YOUTUBE_UGC_NIQE_scores.mat')
MOS_exp1 = MOS';
par_exp1 = y_pred';
nan_indices = isoutlier(par_exp1);
MOS_exp1(nan_indices) = [];
par_exp1(nan_indices) = [];
nclips1=size(MOS_exp1,1);
scale_exp1=[1 5]';
cost_exp1=ones(nclips1,1);
MOS_exp1=min(5,MOS_exp1); %limt MOS values to the proper range
MOS_exp1=max(1,MOS_exp1);

%Experiment 2
exp2 = 'KoNViD-1k';
load('KONVID_1K_NIQE_scores.mat')
% nan_indices = [3,6,10,22,23,46,51,52,68,74,77,99,103,122,136,141,158,173,368,426,467,477,506,563,594,...
%     639,654,657,666,670,671,681,690,697,702,703,710,726,736,764,768,777,786,796,977,990,1012,...
%     1015,1023,1091,1118,1205,1282,1312,1336];
MOS_exp2 = MOS';
par_exp2 = y_pred';
nan_indices = isoutlier(par_exp2);
MOS_exp2(nan_indices) = [];
par_exp2(nan_indices) = [];
nclips2=size(MOS_exp2,1);
scale_exp2=[1 5]';
cost_exp2=ones(nclips2,1);
MOS_exp2=min(5,MOS_exp2);
MOS_exp2=max(1,MOS_exp2);

%Experiment 3
exp3 = 'LIVE-VQC';
load('LIVE_VQC_NIQE_scores.mat')
MOS_exp3 = MOS';
par_exp3 = y_pred';
nan_indices = isoutlier(par_exp3);
MOS_exp3(nan_indices) = [];
par_exp3(nan_indices) = [];
nclips3=size(MOS_exp3,1);
scale_exp3=[0 100]';
cost_exp3=ones(nclips3,1);
MOS_exp3=min(100,MOS_exp3);
MOS_exp3=max(0,MOS_exp3);


% original plot
h1 = figure('NumberTitle', 'off','Position', [0 0 500 400]);

p1 = plot(par_exp1,MOS_exp1, 'o');
set(p1, 'Color', [0 0.4470 0.7410],'LineWidth',2, 'MarkerSize',4.5);
hold on

p2 = plot(par_exp2,MOS_exp2,'+');
set(p2, 'Color', [0.8500 0.3250 0.0980], 'LineWidth',2, 'MarkerSize', 6);
hold on

p3 = plot(par_exp3,MOS_exp3,'x');
set(p3, 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2, 'MarkerSize', 6);
hold off

xlabel('NIQE', 'FontSize', 16)
ylabel('MOS', 'FontSize', 16)
% title('3 Experiments (B,G,R) MOS vs Parameter, Before INLSA')
legend(exp1, exp2, exp3, 'FontSize', 14)
grid minor
set(gca,'FontSize', 14)
set(gca, 'Color', 'none');
set(gcf, 'Color', 'w');
xlim([0,16])
fig_name = fullfile('inlsa_before.pdf');
export_fig(fig_name, '-native', '-painters','-q101');
fig_name = fullfile('inlsa_before.png');
export_fig(fig_name, '-native', '-painters','-q101', '-a4','-m2');


% linear scale
h2 = figure('NumberTitle', 'off','Position', [0 0 500 400]);

p1 = plot(par_exp1,MOS_exp1, 'o');
set(p1, 'Color', [0 0.4470 0.7410],'LineWidth',2, 'MarkerSize',4.5);
hold on

p2 = plot(par_exp2,MOS_exp2,'+');
set(p2, 'Color', [0.8500 0.3250 0.0980], 'LineWidth',2, 'MarkerSize', 6);
hold on

p3 = plot(par_exp3,MOS_exp3./100.*4 + 1,'x');
set(p3, 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2, 'MarkerSize', 6);
hold off

xlabel('NIQE', 'FontSize', 16)
ylabel('MOS', 'FontSize', 16)
% title('3 Experiments (B,G,R) MOS vs Parameter, Before INLSA')
legend(exp1, exp2, exp3, 'FontSize', 14)
grid minor
set(gca,'FontSize', 14)
set(gca, 'Color', 'none');
set(gcf, 'Color', 'w');
xlim([0,16])
fig_name = fullfile('linscale_after.pdf');
export_fig(fig_name, '-native', '-painters','-q101');
fig_name = fullfile('linscale_after.png');
export_fig(fig_name, '-native', '-painters','-q101', '-a4','-m2');


[rhosq,W,BA,MOS_HAT,MOS_TILDA] = inlsa(error_ratio,norm_set, ...
    scale_exp1,cost_exp1,MOS_exp1,par_exp1,...
    scale_exp2,cost_exp2,MOS_exp2,par_exp2,...
    scale_exp3,cost_exp3,MOS_exp3,par_exp3);

W  %paremeter gain and shift
BA %MOS gain and shift for each experiment

% figure(2)
% p1 = plot(par_exp1,MOS_TILDA(1:nclips1),'ob')
% hold on
% plot(par_exp2,MOS_TILDA(1+nclips1:nclips1+nclips2),'og')
% plot(par_exp3,MOS_TILDA(nclips1+nclips2+1:nclips1+nclips2+nclips3),'or')
% grid
% hold off
% xlabel('Parameter Value')
% ylabel('Distortion')
% title('3 Experiments (B,G,R) Distortion vs Parameter, After INLSA')
% legend(exp1, exp2, exp3)
% 
% figure(3)
% plot(par_exp1,5-4*MOS_TILDA(1:nclips1),'ob')
% hold on
% plot(par_exp2,5-4*MOS_TILDA(nclips1+1:nclips1+nclips2),'og')
% plot(par_exp3,5-4*MOS_TILDA(nclips1+nclips2+1:nclips1+nclips2+nclips3),'or')
% grid
% hold off
% xlabel('Parameter Value')
% ylabel('MOS')
% title('3 Experiments (B,G,R) MOS vs Parameter, After INLSA')
% legend(exp1, exp2, exp3)

%Note that the gain and shift in BA must be used in the distortion domain,
%not the MOS domain, this section shows how.  The figure will agree with
%figure 3
h4 = figure('NumberTitle', 'off','Position', [0 0 500 400]);

temp=(max(scale_exp1)-MOS_exp1)/range(scale_exp1); %convert MOS to distortion
temp=BA(1,1)+BA(2,1)*temp; %apply gain and shift produced by INSLA
temp=max(scale_exp1)-range(scale_exp1)*temp; %convert distortion to MOS
p1=plot(par_exp1,temp,'o');
set(p1, 'Color', [0 0.4470 0.7410],'LineWidth',2, 'MarkerSize',4.5);
hold on

temp=(max(scale_exp2)-MOS_exp2)/range(scale_exp2); %convert MOS to distortion
temp=BA(1,2)+BA(2,2)*temp; %apply gain and shift produced by INSLA
temp=max(scale_exp1)-range(scale_exp1)*temp; %convert distortion to MOS
p2=plot(par_exp2,temp,'+');
set(p2, 'MarkerFaceColor', [0.8500 0.3250 0.0980], 'LineWidth',2, 'MarkerSize', 6);
hold on


temp=(max(scale_exp3)-MOS_exp3)/range(scale_exp3); %convert MOS to distortion
temp=BA(1,3)+BA(2,3)*temp; %apply gain and shift produced by INSLA
temp=max(scale_exp1)-range(scale_exp1)*temp; %convert distortion to MOS
p3=plot(par_exp3,temp,'x');
set(p3, 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2, 'MarkerSize', 6);
p3.Color(4) = 0.2;
hold off

xlabel('NIQE', 'FontSize', 16)
ylabel('MOS', 'FontSize', 16)
% title('3 Experiments (B,G,R) MOS vs Parameter, Before INLSA')
legend(exp1, exp2, exp3, 'FontSize', 14)
grid minor
set(gca,'FontSize', 14)
set(gca, 'Color', 'none');
set(gcf, 'Color', 'w');
xlim([0,16])
fig_name = fullfile('inlsa_after.pdf');
export_fig(fig_name, '-native', '-painters','-q101');
fig_name = fullfile('inlsa_after.png');
export_fig(fig_name, '-native', '-painters','-q101', '-a4','-m2');
