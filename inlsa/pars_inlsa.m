function [mdl, mdl_wts, scale_factors] = pars_inlsa(par_struct, test_struct, varargin)
% PARS_INLSA
%  Run the Iterative Nested Least Squares Algorithm (INLSA) on a given list
%  of parameters; return the resultant model & inlsa_mos.
% SYNTAX
%  [mdl, mdl_wts, scale_factors] = pars_inlsa(par_struct, test_struct);
%  [] = pars_inlsa(...,'PropertyName',PropertyValue,...);
% SEMANTICS
%  Run INLSA the parameters in par_struct, of the same format as GPars.
%  'test_struct' is the test information, in the same format as GTests.
%  Return the model in 'mdl', in the same format as GPars; the model
%  weights in mdl_wts, with the constant term FIRST, and the offset/gain of
%  each test in 'scale_factors'.
%
%  The following optional parameters are available:
%  'ErrorRatio', #,     The error ratio between subjective & objective
%                                errors.  Defaults to 1.0.
%  'WeighViewers'  Weigh video clips with more viewers more heavily
%  'WeighTests'        Weigh tests equally.  By default, individual clips
%                                weighed equally.
%  'Verbose'              Report on INLSA results.
%  'group',list,            Group the tests named in cell array 'list' into
%                                one test, for purposes of training.  Use
%                                their 'inlsa_mos' subjective data, instead
%                                of usual mos subjective data.  Range of
%                                inlsa_mos assumed to be worst=1, best=0.
%  'NormSet',#,         use the #th test as the norm set, to which all
%                                other data sets are scaled.  Numbering is
%                                of alphabetized list, putting groups first.

verbose = 0;

% Norm_set is the set to which all other data sets are scaled.
norm_set = 1;

%  The error ratio between the subjective and objective errors.  See msdf.m
%  An error ratio of 1.0 means that subjective and objective errors are of
%  equal importance.
error_ratio = 1.0;

% Options to weight clips by number of viewers, & weight tests equally.
option_weight_viewers = 0;
option_weight_tests_equal = 0;

% keep track of requeseted groups.
group = [];
num_groups = 0;

% read in the optional control arguments.
cnt = 1;
while cnt <= nargin-2,
    if strcmp(lower(varargin{cnt}),'errorratio'),
        error_ratio = varargin{cnt+1};
        cnt = cnt + 2;
    elseif strcmp(lower(varargin{cnt}),'weighviewers'),
        option_weight_viewers = 1;
        cnt = cnt + 1;
    elseif strcmp(lower(varargin{cnt}),'weightests'),
        option_weight_tests_equal = 1;
        cnt = cnt + 1;
    elseif strcmp(lower(varargin{cnt}),'verbose'),
        verbose = 1;
        cnt = cnt + 1;
    elseif strcmp(lower(varargin{cnt}),'group'),
        num_groups = num_groups + 1;
        group{num_groups} = varargin{cnt+1};
        cnt = cnt + 2;
    elseif strcmp(lower(varargin{cnt}), 'normset'),
        norm_set = varargin{cnt+1};
        cnt = cnt + 2;
    else
        error('Optional argument not recognized');
    end
end;

% find the number of parameters & clips
[num_pars, num_clips] = size(par_struct.data);

%  Parse the clip names into three cell arrays: test, scene, hrc.
[clip_test, clip_scene, clip_hrc] = pars_parse(par_struct);

% Fill out this with the arguments to msdf, which computes INLSA
eval_string = sprintf('inlsa (error_ratio, norm_set');

% get list of unique tests.  
test = unique(sort(clip_test));

% Remove from this the grouped tests.
if num_groups > 0,
    not_group_test = test;
    test = [];
    for i=1:num_groups,
        list = group{i};
        test{i} = sprintf('group%d', i);
        
        % find the clips associated with this group;
        test_offsets{i} = [];
        for j=1:length(group{i}),
            test_offsets{i} = [test_offsets{i} find(strcmp(list{j},clip_test))];
            not_group_test = not_group_test( find(strcmp(not_group_test,list{j}) == 0) );
        end
        
        %  Setup initial cost functions for fitting so each clip is equally weighted
        num_clips = length(test_offsets{i});
        cost{i} = ones(num_clips,1);
        
        %  Optionally, increase so that the more viewers, the more weight. 
        if option_weight_viewers,
            error('viewer weighting not implemented for groups');
        end
        % Optionally, weight each TEST equally, instead of default weighing
        % each clip equally.
        if option_weight_tests_equal,
            cost{i} = cost{i} ./ num_clips;
        end
        
        % pick off scores
        mos{i} = par_struct.inlsa_mos(test_offsets{i})';
        
        % define worst & best
        range{i} = [ 1; 0 ];
    end
    test((num_groups+1):(num_groups+length(not_group_test))) = not_group_test;
end


% get & organize information on each data set.
for i=1:length(test),
    if i > num_groups,
        % find the offset for this test.
        offset = find_test(test_struct,test(i));
	
        % find the clips associated with each test
        test_offsets{i} = find(strcmp(test{i},clip_test));
	
        % find the number clips for each test
        num_clips = length(test_offsets{i});   
	
        % define worst & best quality values for each subjective data set scale.  
        range{i} = [test_struct(offset).mos_worst ; test_struct(offset).mos_best];
        
        %  Setup initial cost functions for fitting so each clip is equally weighted
        cost{i} = ones(num_clips,1);
        
        %  Optionally, increase so that the more viewers, the more weight. 
        if option_weight_viewers,
            cost{i} = cost{i} * sqrt( test_struct(offset).viewers );
        end
        % Optionally, weight each TEST equally, instead of default weighing
        % each clip equally.
        if option_weight_tests_equal,
            cost{i} = cost{i} ./ num_clips;
        end
    
        % pick off the scores.
        mos{i} = par_struct.mos(test_offsets{i})';
    end
    
    % pick off the parameter data
    par{i} = par_struct.data(:,test_offsets{i})';
    
    % fill out more of the call.
    eval_string = sprintf('%s, range{%d}, cost{%d}, mos{%d}, par{%d}', eval_string, i, i, i, i);
end
eval_string = sprintf('%s )', eval_string);

% Run INLSA .
[sq_corr_coef, w, scale_factors, mos_hat, cost_wt_mos_tilda] = eval( eval_string );

% apply range to scaling factors
for i=1:length(test),
    hold = range{i};
    scale_factors(2,i) = scale_factors(2,i) / (hold(1)-hold(2));
    scale_factors(1,i) = scale_factors(1,i) - scale_factors(2,i) * hold(2);
end

% make inlsa-produced model into a new parameter structure, for return.
hold = sprintf('inlsa_%f', w(1));
for i=2:length(w),
    if w(i) < 0.0
        hold = sprintf('%s%f*[p%d]', hold, w(i), i-1);
    else
        hold = sprintf('%s+%f*[p%d]', hold, w(i), i-1);
    end
end
mdl.par_name{1} = hold;
mdl.clip_name = par_struct.clip_name;
mdl.mos = par_struct.mos;
mdl.data = par_struct.data(1,:) .* 0.0 + w(1);
for cnt=1:(length(w)-1),
    mdl.data = mdl.data + par_struct.data(cnt,:) .* w(cnt+1);
end
mdl.inlsa_mos = par_struct.inlsa_mos * 0;

mdl_wts = w;

% compute INLSA_MOS values
for i=1:length(test),
    % print out correlation for this data set.
    index = pars_find_clip( par_struct, 'test', test{i}, 'index');
    mdl.inlsa_mos(index) = mdl.mos(index) * scale_factors(2,i) + scale_factors(1,i);
end

% report on INLSA results
if verbose,
    tmp= corrcoef(mdl.data, mdl.inlsa_mos);
    fprintf('INLSA model correlation %f\n', tmp(1,2));
    fprintf('Model weights:\n\t%f\n', w(1));
    for i=2:length(w),
        if w(i) < 0.0,
            fprintf('\t%f * %s\n', w(i), par_struct.par_name{i-1});
        else
            fprintf('\t+%f * %s\n', w(i), par_struct.par_name{i-1});
        end
    end
    
    fprintf('\nCorrelation and Subjective scaling factors (A=offset, B=gain) for each data set:\n');
    for i=1:length(test),
        % print out test name, 
        count = fprintf('\t%s', test{i});
        for cnt = count+1:7,
            fprintf(' ');
        end
        
        % print out correlation for this data set.
        temp = pars_find_clip( par_struct, 'test', test{i}, 'index');
        temp = corrcoef(mdl.inlsa_mos(temp), mdl.data(temp));
        fprintf('\tcorr %-5.3g ', temp(1,2));
        
        % print offset and gain for subjective data computed by INLSA
        fprintf('\tA=%8.5f,  B=%8.5f\n', scale_factors(1,i), scale_factors(2,i));
        
        % print out list of tests in this group, if it is a group.
        if i <= num_groups,
            fprintf('\t\t');
            list = group{i};
            for j=1:length(list);
                fprintf('%s ', list{j});
            end
            fprintf('\n');
        end
    end
    
end

