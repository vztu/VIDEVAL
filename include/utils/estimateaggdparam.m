function [alpha leftstd rightstd] = estimateaggdparam(vec)


gam   = 0.2:0.001:10;
r_gam = ((gamma(2./gam)).^2)./(gamma(1./gam).*gamma(3./gam));
throwAwayThresh = 0.0;
leftstd            = sqrt(mean((vec(vec<-throwAwayThresh)).^2));
rightstd           = sqrt(mean((vec(vec>throwAwayThresh)).^2));
gammahat           = leftstd/rightstd;

vec1=vec;
rhat               = (mean(abs(vec1)))^2/mean((vec1).^2);
rhatnorm           = (rhat*(gammahat^3 +1)*(gammahat+1))/((gammahat^2 +1)^2);
[min_difference, array_position] = min((r_gam - rhatnorm).^2);
alpha              = gam(array_position);