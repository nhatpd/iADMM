function[acc] = Cluster(Z,K,gnd)
    
    %post processing
    [U,S,V] = svd(Z,'econ');
    S = diag(S);
    r = sum(S>1e-4*S(1));
    U = U(:,1:r);S = S(1:r);
    U = U*diag(sqrt(S));
    U = normr(U);
    L = (U*U').^4;

% spectral clustering
%add new 
%     L = (L + L')/2;
    %
    D = diag(1./sqrt(sum(L,2)));
    L = D*L*D;
    [U,S,V] = svd(L);
    V = U(:,1:K);
    V = D*V;

    n = size(V,1);
    M = zeros(K,K,20);
    rand('state',123456789);
    for i=1:size(M,3)
        inds = false(n,1);
        while sum(inds)<K
            j = ceil(rand()*n);
            inds(j) = true;
        end
        M(:,:,i) = V(inds,:);
    end


    %add new
%     idx = kmeans(V,K,'emptyaction','singleton','replicates',20,'display','off');
    %%%
    idx = kmeans(V,K,'emptyaction','singleton','start',M,'display','off');
    acc =  1 - missclassGroups(idx,gnd,K)/length(idx);



    % end
    
%     aveage_acc = acc;
%     disp(['seg acc=' num2str(aveage_acc)]);
end