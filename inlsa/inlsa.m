function [varargout] = inlsa(error_ratio,norm_set,varargin);
% INLSA
%  Run the Iterative Nested Least Squares Algorithm (INLSA).  
%  See also 'pars_inlsa'.
%  This function was originally named "mdsf"
% SYNTAX
%  [varargout] = inlsa(error_ratio,norm_set,varargin)
% DEFINITION
%  Fits subjective data to objective parameters.  An iterative approach is used.
%
%  1) Weighted least squares is used to solve for W in the equation MOS_HAT=PAR*W.  This
%  minimizes |COST*(MOS_TILDA - MOS_HAT)|^2, where MOS_TILDA=A*MOS+B
%
%  2) A joint error minimization (JEM) algorithm is used on each data set to
%  solve A*(MOS+MOS_ERROR)+B = (MOS_HAT+MOS_HAT_ERROR).  The JEM algorithm finds
%  A and B such that error_ratio = |COST*MOS_ERROR|/|COST*MOS_HAT_ERROR|, and
%  |COST*MOS_HAT_ERROR| + g*|COST*MOS_ERROR| is minimized for some positive
%  weighting factor g, which is a function of error_ratio.
%
%  If you change the code to call jem2 instead of jem, then step 2 is a bit
%  different.  It looks like this:
%  2) A joint error minimization (JEM) algorithm is used on each data set to
%  solve A*(MOS+MOS_ERROR)+B = (MOS_HAT+MOS_HAT_ERROR).  The JEM algorithm finds
%  A and B such that error_ratio = |COST*MOS_ERROR|/|COST*MOS_HAT_ERROR|, and
%  |COST*MOS_HAT_ERROR| + |COST*MOS_ERROR| is minimized.
%
%  3) If results have changed significantly on this iteration, then go to step 1.
%
%  norm_set - the reference data (all data sets will be scaled to the reference).
%             [0 1] will represent the reference data set scale - see scale terms 
%             below.  Data that falls outside of [0 1] represents quality better
%             quality (< 0) or worse quality (> 1) than the reference.
%
%  error_ratio -	the ratio of objective error to subjective error that will result
%
%  Note that this code contains two normalization options "move&return" and "stayput."
%  This feature was added 12/7/00 by S. Voran.
%
%  Under the "move&return" option, JEM.M or JEM2.M calculates a and b for all data sets,
%  including the reference data set.  Each iteration also includes a normalization step
%  that moves a back to 1 and b back to 0 for the reference data set.  This normalization
%  also moves w and the other a's and b's in a corresponding fashion so that the solution is
%  simply scaled and shifted.
%
%  Under the "stayput" option, JEM.M or JEM2.M calculates a and b for all data sets except
%  the reference data set.  Thus a and b for the reference data set do not change, and no 
%  additional normalization steps are needed.
%
%  While the "stayput" option seems more direct, the "move&return" option seems to converge
%  more quickly, at least in the cases that Voran has tested so far.  It seems that when
%  skipping the JEM step on the reference data set, direct information about the relationship
%  between the reference data set the parameters is ignored.  This information is apparently
%  extracted in a slower and less direct way through the remaining data sets, leading to
%  slower convergence.  Depending on the scores and parameters, the two options may converge
%  to the same solution or to different solutions.  There appears to be no general result
%  on the relative optimality of the two solutions.  If speed is important, one may be
%  forced to use the "move&return" option,  However, if speed is not important, it may 
%  be worth experimenting with both options to see which give the best results for the 
%  current data of interest.
%
%
%  Varargin contains data in the following input format:
%  -----------------------------------------------------
%
%  For data sets i = 1, 2, ..., NUM_SETS, varargin must be organized as follows:
%
%  SCALE(1) - a 2 element row vector [low high]' that gives the values associated 
%             with the low and high quality endpoints of scale 1.  Used to scale 
%             the MOS data for a range of [0 1], where 0 is no impairment and 1 is
%             maximum impairment.
%  COST(1) - a NUM_CLIPS(1) element column vector that gives the cost weighting  
%            for prediction errors of each clip in data set 1.
%  MOS(1) - a NUM_CLIPS(1) element column vector that gives the mean opinion
%           score (MOS) of each clip in data set 1.
%  PAR(1) - a NUM_CLIPS(1) by NUM_PARS matrix that gives the parameter values for
%           each clip.  The NUM_PARS must be the same for all data sets.  Does not
%           include the DC parameter (i.e., ones column vector).
%  .
%  .
%  .
%  SCALE(NUM_SETS) - a 2 element column vector [low high]' that gives the values 
%                    associated with the low and high quality endpoints of the scale
%                    NUM_SETS, where NUM_SETS is the total number of data sets.
%  COST(NUM_SETS) - a NUM_CLIPS(NUM_SETS) element column vector that gives the 
%                   cost weighting for prediction errors of each clip in data set
%                   NUM_SETS.
%  MOS(NUM_SETS) - a NUM_CLIPS(NUM_SETS) element column vector that gives the 
%                  mean opinion score (MOS) of each clip in data set NUM_SETS.
%  PAR(NUM_SETS) - a NUM_CLIPS(NUM_SETS) by NUM_PARS matrix that gives the parameter
%                  values for each clip.
%    
%  If one output argument is requested, the program returns the squared correlation
%  coefficient between the COST weighted MOS_TILDA and the COST weighted predicted 
%  MOS_HAT.  For this calculation, all of the data is treated as one big data set.
%
%  If two output arguments are requested, the program also returns the column 
%  vector W.
%
%  If three output arguments are requested, the program also returns a 2 by NUM_SETS
%  matrix that gives the final [Bi Ai]' scaling factors.
%
%  If four output arguments are requested, the program also returns the COST weighted
%  MOS_HAT array.
%
%  If five output arguments are requested, the program also returns the COST weighted
%  MOS_TILDA array (the mapped MOS values).
%  


%Select a normalization type, could be passed in if desired
%normalization='stayput    '
normalization='move&return';

%  Find the number of data sets that were input
num_sets = length(varargin)/4;

num_clips = zeros(1,num_sets);  %number of clips in each data set
first = zeros(1,num_sets);  %index number of first clip in each data set
last = zeros(1,num_sets);  %index number of last clip in each data set
%  Organize the SCALE, COST, MOS, and PAR arrays by combining the smaller data sets
for i_set = 1:num_sets
   if (i_set == 1)
      scale = varargin{4*i_set-3};
      cost = varargin{4*i_set-2};
      mos = varargin{4*i_set-1};
      par = varargin{4*i_set};
      num_clips(i_set) = size(varargin{4*i_set},1);
      first(i_set) = 1;
      last(i_set) = first(i_set)+num_clips(i_set)-1;
   else
      scale = cat(2,scale,varargin{4*i_set-3});
      cost = cat(1,cost,varargin{4*i_set-2});
      mos = cat(1,mos,varargin{4*i_set-1});
      par = cat(1,par,varargin{4*i_set});
      num_clips(i_set) = size(varargin{4*i_set},1);
      first(i_set) = last(i_set-1)+1;
      last(i_set) = first(i_set)+num_clips(i_set)-1;
   end
end

%  Find the total number of clips, the number of parameters and add 
%  the dc parameter (i.e., ones column vector) to par
[total_clips,num_pars] = size(par);
par = [ones(total_clips,1) par];

%  Scale the MOS data so that it falls within the [0 1] interval
for i_set = 1:num_sets
   this_mos = mos(first(i_set):last(i_set));
   %  Do the gain and DC offset adjustment (zero is good quality, one is bad quality)
   this_mos = (this_mos-scale(2,i_set))/(scale(1,i_set)-scale(2,i_set));
   %  Or do this gain and DC adjustment (zero is bad quality, one is good quality)
   %this_mos = (this_mos-scale(1,i_set))/(scale(2,i_set)-scale(1,i_set));
   mos(first(i_set):last(i_set)) = this_mos;
end

%  Since the MOS data has been scaled between [0 1], the initial a is set equal to 1 
%  and the initial b is set equal to 0.
a_short = ones(1,num_sets);
a_short_new = zeros(1,num_sets);
b_short = zeros(1,num_sets);
b_short_new = zeros(1,num_sets);
for i_set = 1:num_sets
   if (i_set == 1)
      a = ones(num_clips(i_set),1);
      b = zeros(num_clips(i_set),1);
   else
      a = cat(1,a,ones(num_clips(i_set),1));
      b = cat(1,b,zeros(num_clips(i_set),1));
   end
end


%  Normalize the cost vector and compute the cost^2 matrix
cost = cost./norm(cost);
cost2_vec = cost.*cost;
cost2 = sparse(1:total_clips,1:total_clips,cost2_vec,total_clips,total_clips);

%  Interation loop to calculate w, a, and b.  Try a maximum of max_tries to converge
%  to a solution where the change in value of a and b is less than max_change between
%  two iteration loops.
max_tries = 100;
i_try = 1;
max_change = .000001;
this_change = 1.0;

while ((i_try <= max_tries) & (this_change > max_change))
      
   %  Calculate new mos_tilda, w, mos_hat.
   mos_tilda = a.*mos+b;
   
   %***************************************************
   %  Code added by S. Wolf and S. Voran to implement quadratic cost as function of mos_tilda.
   %  Update the cost function, cost_gain is the ratio of cost at 0 and 1 versus cost at 1/2.
   %cost_gain = 3;
   %cost = 4*(cost_gain-1)*(mos_tilda.^2 - mos_tilda + .25)+1;
   %cost = cost./norm(cost);
   %cost2_vec = cost.*cost;
   %cost2 = sparse(diag(cost2_vec));
   %***************************************************
   
   %***************************************************
   %  This code added on 10/4/00 to normalize mos_tilda for unit variance and zero mean
   %mos_tilda = mos_tilda - mean(mos_tilda);
   %mos_tilda = mos_tilda./std(mos_tilda);
   %***************************************************
   
   %***************************************************  
   %  This code added on 10/4/00 to normalize mos_tilda to exactly cover the interval [0,1]
   %mos_tilda = (mos_tilda-min(mos_tilda))/(max(mos_tilda)-min(mos_tilda));
   %***************************************************
   
%    w = inv(par'*cost2*par)*par'*cost2*mos_tilda;
   w = (par'*cost2*par)\(par'*cost2*mos_tilda);
   mos_hat = par*w;
   
   %****************************************************
   %  Code added by S. Wolf and W. Voran to implement quadratic cost as function of mos_hat.
   %  Update the cost function, cost_gain is the ratio of cost at 0 and 1 versus cost at 1/2.
   %cost_gain = 3;
   %cost = 4*(cost_gain-1)*(mos_hat.^2 - mos_hat + .25)+1;
   %cost = cost./norm(cost);
   %cost2_vec = cost.*cost;
   %cost2 = sparse(diag(cost2_vec));
   %****************************************************

   
   %Update the bi and ai, using jem algorithm, given the current mos_hat and mos
   for i_set = 1:num_sets
      if (normalization=='stayput    ')&(i_set==norm_set)
         %Don't update a and b for the reference dataset
         a_short_new(i_set)=a_short(i_set);					
         b_short_new(i_set)=b_short(i_set);
      else
         this_mos = mos(first(i_set):last(i_set));
         this_mos_hat = mos_hat(first(i_set):last(i_set));
         this_cost2 = cost2_vec(first(i_set):last(i_set));	%column vector of costs for current data set
         [tempa,tempb,dud,dud]=jem(this_mos,this_mos_hat,error_ratio,sqrt(this_cost2));
         %[tempa,tempb,dud,dud]=jem2(this_mos,this_mos_hat,error_ratio,sqrt(this_cost2));
         a_short_new(i_set)=tempa;					
         b_short_new(i_set)=tempb;
      end
   end

   
   %*****************************************************
   %This corrected code added on 10/7/00 by S. Voran to keep a=1, b=0 for selected data set
   if normalization=='move&return'
      w(1) = w(1)-b_short_new(norm_set);
      b_short_new = b_short_new-b_short_new(norm_set);
      w = w/a_short_new(norm_set);
      b_short_new = b_short_new/a_short_new(norm_set);
      a_short_new = a_short_new/a_short_new(norm_set);
   end
   %*****************************************************
   
   %  See if a and b have converged to a solution
   this_change = 0;
   for i_set = 1:num_sets
      if (abs(a_short_new(i_set)-a_short(i_set)) > this_change)
         this_change = abs(a_short_new(i_set)-a_short(i_set));
      end
      if (abs(b_short_new(i_set)-b_short(i_set)) > this_change)
         this_change = abs(b_short_new(i_set)-b_short(i_set));
      end
   end

   %  Update a and b
   for i_set = 1:num_sets
         a_short(i_set) = a_short_new(i_set);
         b_short(i_set) = b_short_new(i_set);
      if (i_set == 1)
         a = ones(num_clips(i_set),1)*a_short(i_set);
         b = ones(num_clips(i_set),1)*b_short(i_set);
      else
         a = cat(1,a,ones(num_clips(i_set),1)*a_short(i_set));
         b = cat(1,b,ones(num_clips(i_set),1)*b_short(i_set));
      end
   end
   i_try = i_try +1;
   
end

%  Calculate updated mos_tilda and mos_hat for renormalized a, b, and w
mos_hat=par*w;
mos_tilda=a.*mos+b;

%  Compute the cost-weighted correlation coefficient squared.
temp_mos_hat = mos_hat-mean(mos_hat);
temp_mos_hat = temp_mos_hat/norm(temp_mos_hat);
temp_mos_tilda = mos_tilda-mean(mos_tilda);
temp_mos_tilda = temp_mos_tilda/norm(temp_mos_tilda);

corr_mat = corrcoef(cost.*temp_mos_hat,cost.*temp_mos_tilda);
this_corr = corr_mat(1,2)^2;

%  Assign output arguments
if (nargout == 1)
   varargout{1} = this_corr;
end

if (nargout == 2)
   varargout{1} = this_corr;
   varargout{2} = w;
end

if (nargout == 3)
   varargout{1} = this_corr;
   varargout{2} = w;
   varargout{3} = [b_short;a_short];
end

if (nargout == 4)
   varargout{1} = this_corr;
   varargout{2} = w;
   varargout{3} = [b_short;a_short];
   varargout{4} = mos_hat;
end

if (nargout == 5)
   varargout{1} = this_corr;
   varargout{2} = w;
   varargout{3} = [b_short;a_short];
   varargout{4} = mos_hat;
   varargout{5} = mos_tilda;
end

%--------------------------------------------------------------------------
function [a,b,ex,ey]=jem(x,y,r,c);
%This code solves a*(x+ex)+b=y+ey such that r=|c.*ex|/|c.*ey|,
%and |c.*ex|+g(r)*|c.*ey| is minimized for some positive constant weight g(r).
%x,y, and c may be row or column vectors but must have the same length.
%r must be a scalar, 0<r
%
%Written 11/1/00 by S. Voran
%Documentation expanded on 11/28/00

%Check inputs
[m,n]=size(x);
if min(m,n)>1		%Test to see if x is a vector
   error('x must be a vector')
end
if m==1				%Force x to be a column vector
   x=x';
end

[m,n]=size(y);
if min(m,n)>1		%Test to see if y is a vector
   error('y must be a vector')
end
if m==1				%Force y to be a column vector
   y=y';
end

[m,n]=size(c);
if min(m,n)>1		%Test to see if c is a vector
   error('c must be a vector')
end
if m==1				%Force c to be a column vector
   c=c';
end

if length(x)~=length(y) |length(x)~=length(c)	%Check for consistent lengths of x, y, and c
   error('x, y, and c must have same length')
end

c=c/sqrt(sum(c.^2));		%Normalize cost vector to unit norm

%The Guts
my=sum(c.*c.*y);
mx=sum(c.*c.*x);
tx=c.*(x-mx);
ty=c.*(y-my);

if ty'*ty == 0,
    error('INLSA encountered a divide by zero.  Aborting.');
end

if tx'*ty>=0;
   t=(tx+r*ty);
   t=sqrt((t'*t)/(ty'*ty));
    if (r-t) == 0,
        error('INLSA encountered a divide by zero.  Aborting.');
	end
   a=-1/(r-t);
   b=my-a*mx;
   u=(x+r*(y-b))/(1+a*r);
else
   t=(tx-r*ty);
   t=sqrt((t'*t)/(ty'*ty));
    if (r-t) == 0,
        error('INLSA encountered a divide by zero.  Aborting.');
	end
   a=1/(r-t);
   b=my-a*mx;
   u=(x-r*(y-b))/(1-a*r);
end
ex=u-x;
ey=a*u+b-y;


