function [subband size_band] = complex_divisive_normalized(pyro,pind,Nsc,Nor,parent,neighbor,blSzX,blSzY)

Nband = size(pind,1)-1;                                    

p = 1;
for scale=1:Nsc-1
    for orien=1:Nor
        nband = (scale-1)*Nor+orien+1; % except the ll
        %      if(prod(band-nband-1) ~=0)
        %            continue;
        %      end
        aux_c = pyrBand(pyro, pind, nband);
        aux = abs(aux_c);
        [Nsy,Nsx] = size(aux);

        prnt = parent & (nband <= Nband-(Nsc-1)*Nor);  % has the subband a parent? 
%       define that only the finest scale has a parent
        BL = zeros(size(aux,1),size(aux,2),1 + prnt);
        BL(:,:,1) = aux;
        if prnt,
            auxp = pyrBand(pyro, pind, nband+Nor);
            %    if nband>Nor+1,     % resample 2x2 the parent if not in the high-pass oriented subbands.
            % 	   auxp = real(imenlarge2(auxp)); % this was uncommented
            auxp = abs(imresize(auxp,2)); %
            %   end
            %  fprintf('parent band and size is %d %d %d \n',nband+Nor,Nsy,Nsx)
            BL(:,:,2) = auxp(1:Nsy,1:Nsx);
        end
        y = BL;
        [nv,nh,nb] = size(y);
        block = [blSzX blSzY];

        nblv = nv-block(1)+1;	% Discard the outer coefficients
        nblh = nh-block(2)+1;   % for the reference (centrral) coefficients (to avoid boundary effects)
        nexp = nblv*nblh;			% number of coefficients considered
        N = prod(block) + prnt; % size of the neighborhood

        Ly = (block(1)-1)/2;		% block(1) and block(2) must be odd!
        Lx = (block(2)-1)/2;
        if (Ly~=floor(Ly))|(Lx~=floor(Lx)),
            error('Spatial dimensions of neighborhood must be odd!');
        end
        Y = zeros(nexp,N);		% It will be the observed signal (rearranged in nexp neighborhoods)
        % Rearrange observed samples in 'nexp' neighborhoods
        n = 0;
        for ny=-Ly:Ly,	% spatial neighbors
            for nx=-Lx:Lx,
                n = n + 1;
                foo = shift(y(:,:,1),[ny nx]);
                foo = foo(Ly+1:Ly+nblv,Lx+1:Lx+nblh);
                Y(:,n) = (foo(:));
            end
        end

        if prnt,	% parent
            n = n + 1;
            foo = y(:,:,2);
            foo = foo(Ly+1:Ly+nblv,Lx+1:Lx+nblh);
            Y(:,n) = (foo(:));
        end

        %      including neighbor
        if neighbor,
            for neib=1:Nor
                if neib == orien
                    continue;
                end
                n=n+1;
                nband1 = (scale-1)*Nor+neib+1; % except the ll
                aux1 = abs(pyrBand(pyro, pind, nband1));
                aux1 = aux1(Ly+1:Ly+nblv,Lx+1:Lx+nblh);
                Y(:,n) = (aux1(:));
            end
        end

        C_x = innerProd(Y)/nexp;
        % C_x is positive definete covariance matrix
        [Q,L] = eig(C_x);
        % correct possible negative eigenvalues, without changing the overall variance
        L = diag(diag(L).*(diag(L)>0))*sum(diag(L))/(sum(diag(L).*(diag(L)>0))+(sum(diag(L).*(diag(L)>0))==0));
        C_x = Q*L*Q';

        o_c = aux_c(Ly+1:Ly+nblv,Lx+1:Lx+nblh);
        o_c = (o_c(:));
%         o_c_r = real(o_c);
%         o_c_i = imag(o_c);
%         o_c_r = o_c_r - mean(o_c_r);
%         o_c_i = o_c_i - mean(o_c_i);
        o_c = o_c - mean(o_c);
        
        tempY = (Y*inv(C_x)).*Y/N;
        z = sqrt(sum(tempY,2));
        ind = find(z~=0);

        g_c = o_c(ind)./z(ind);
        size_band(p,1) = nblv;
        size_band(p,2) = nblh;
        g_c = g_c - mean(g_c);
        g_c_m = reshape(g_c, nblv, nblh);
        subband{p} = g_c_m;
        p = p+1;
        
    end
end
return;