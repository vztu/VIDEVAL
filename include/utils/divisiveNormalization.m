function [structdis,iMinusMu,mu,sigma]= divisiveNormalization(imdist)
    
    window = fspecial('gaussian',7,7/6);
    window = window/sum(sum(window));
    
    mu = filter2(window, imdist, 'same');
    mu_sq = mu.*mu;
    
    sigma = sqrt(abs(filter2(window, imdist.*imdist, 'same') - mu_sq));
    iMinusMu = (imdist-mu);
    structdis =iMinusMu./(sigma +1);
end