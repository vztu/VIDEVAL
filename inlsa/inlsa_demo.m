%Demo showing the use of INLSA.m with made-up data.
randn('state',0);
rand('state',0);

error_ratio=1;  %can be used to allocate arror between MOS and parameters
norm_set=1;     
%experiment 1 is the reference and will not be adjusted
%can change this to 2 or 3 and see what happens
%use the experiment that you trust most as the reference

%Experiment 1
nclips=91;
scale_exp1=[1 5]';
cost_exp1=ones(nclips,1);
par_exp1=.7*randn(nclips,1)+3;  %create random parameter values
MOS_exp1=par_exp1+rand(nclips,1)/2;%create MOS values that are 
%correlated to those random parameter values 
MOS_exp1=min(5,MOS_exp1); %limt MOS values to the proper range
MOS_exp1=max(1,MOS_exp1);

%Experiment 2
nclips=77;
scale_exp2=[1 5]';
cost_exp2=ones(nclips,1);
par_exp2=.8*randn(nclips,1)+3;
MOS_exp2=par_exp2+.5+rand(nclips,1);
MOS_exp2=min(5,MOS_exp2);
MOS_exp2=max(1,MOS_exp2);

%Experiment 3
nclips=140;
scale_exp3=[1 5]';
cost_exp3=ones(nclips,1);
par_exp3=.6*randn(nclips,1)+3;
MOS_exp3=.5*par_exp3+1+rand(nclips,1)/2;
MOS_exp3=min(5,MOS_exp3);
MOS_exp3=max(1,MOS_exp3);

figure(1)
plot(par_exp1,MOS_exp1,'ob')
hold on
plot(par_exp2,MOS_exp2,'og')
plot(par_exp3,MOS_exp3,'or')
grid
hold off
xlabel('Parameter Value')
ylabel('MOS')
title('3 Experiments (B,G,R) MOS vs Parameter, Before INLSA')

[rhosq,W,BA,MOS_HAT,MOS_TILDA] = inlsa(error_ratio,norm_set, ...
    scale_exp1,cost_exp1,MOS_exp1,par_exp1,...
    scale_exp2,cost_exp2,MOS_exp2,par_exp2,...
    scale_exp3,cost_exp3,MOS_exp3,par_exp3);

W  %paremeter gain and shift
BA %MOS gain and shift for each experiment

figure(2)
plot(par_exp1,MOS_TILDA(1:91),'ob')
hold on
plot(par_exp2,MOS_TILDA(91+1:91+77),'og')
plot(par_exp3,MOS_TILDA(91+77+1:91+77+140),'or')
grid
hold off
xlabel('Parameter Value')
ylabel('Distortion')
title('3 Experiments (B,G,R) Distortion vs Parameter, After INLSA')

figure(3)
plot(par_exp1,5-4*MOS_TILDA(1:91),'ob')
hold on
plot(par_exp2,5-4*MOS_TILDA(91+1:91+77),'og')
plot(par_exp3,5-4*MOS_TILDA(91+77+1:91+77+140),'or')
grid
hold off
xlabel('Parameter Value')
ylabel('MOS')
title('3 Experiments (B,G,R) MOS vs Parameter, After INLSA')

%Note that the gain and shift in BA must be used in the distortion domain,
%not the MOS domain, this section shows how.  The figure will agree with
%figure 3
figure(4)

temp=(5-MOS_exp1)/4; %convert MOS to distortion
temp=BA(1,1)+BA(2,1)*temp; %apply gain and shift produced by INSLA
temp=5-4*temp; %convert distortion to MOS
plot(par_exp1,temp,'ob')
hold on

temp=(5-MOS_exp2)/4; %convert MOS to distortion
temp=BA(1,2)+BA(2,2)*temp; %apply gain and shift produced by INSLA
temp=5-4*temp; %convert distortion to MOS
plot(par_exp2,temp,'og')

temp=(5-MOS_exp3)/4; %convert MOS to distortion
temp=BA(1,3)+BA(2,3)*temp; %apply gain and shift produced by INSLA
temp=5-4*temp; %convert distortion to MOS
plot(par_exp3,temp,'or')

grid
hold off
xlabel('Parameter Value')
ylabel('MOS')
title('Same as Fig 4, but calculated the long way')