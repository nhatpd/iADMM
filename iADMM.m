function[output] = iADMM( X, A, B, para )
% iADMM solves the following low-rank representation optimization problem,
% min sum_ig(\sigma_i(Z))+ p(E) + r(Y)
% s.t., X = AZ+ EB + Y 
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
% para.g could be
% 'NN': nuclear norm
% ...
% para.p:
% 'exp21' : sum_i(1-exp(-\theta\|E_i\|_2))
% ...
% para.r:
% 'l2': \|Y\|^2/2
% ...
% Reference: LTK Hien, DN Phan, N Gillis. "A Framework of Inertial Alternating Direction Method of Multipliers for
% Non-Convex Non-Smooth Optimization".
%
% Written by Duy Nhat Phan.
% Latest update February 2021

output.method = 'iADMM';

lambda = para.lambda;
maxIter = para.maxIter;
cf = para.cf;

[d, n] = size(X);
m = size(A,2);
q = size(B,1);
if(isfield(para, 'Z'))
    Z = para.Z;
else
    Z = zeros(m,n);
end
if(isfield(para, 'E'))
    E = para.E;
else
    E = zeros(d,q);
end
if(isfield(para, 'Y'))
    Y = para.Y;
else
    Y = sparse(d,n);
end
if(isfield(para, 'W'))
    W = para.W;
else
    W = Y;
end

AX = A'*X;

obj = zeros(maxIter+1, 1);
Obj = zeros(maxIter+1, 1);
RMSE = zeros(maxIter+1, length(para.nknn));
Time = zeros(maxIter+1, 1);



Ep = E;
AZ = A*Z;
AAZ = A'*AZ;
AAZp = AAZ;
mu = para.mu;
Zp=Z;

EB = E*B; 
EBB = EB*B';
EBBp = EBB;
XB = X*B';

[~, S, ~] = svd(Z, 'econ');
sigma = diag(S);
    
objVal = compute_g(sigma,para) + compute_p(E,para) + compute_r(Y,para);
Obj(1) = objVal;

objVal = objVal + (mu/2)*sum(sum((AZ + EB + cf*Y - X).^2)) + sum(sum((AZ + EB + cf*Y - X).*W));

obj(1) = objVal;

if para.iter_acc == 1
    RMSE(1,:) = compute_acc_test(X,Z,E,para);
end

[~, tmp, ~] = svd(A'*A, 'econ');
L1 = max(diag(tmp));
% L1 = min(1,max(diag(tmp)));

L1 = mu*L1;
if para.g == "exp" || para.g == "log"
    L1 = L1*para.kappa_non;
end

[~, tmp, ~] = svd(B*B', 'econ');
L2 = max(diag(tmp));
% L2 = min(1,max(diag(tmp)));
L2 = mu*L2;

if para.p == "exp" || para.g == "log"
    L2 = L2*para.kappa_non;
end

rho = para.rho;

c = 1;
for i = 1:maxIter
    tt = cputime;
    if para.inertial
        cp = c;
    else
        cp = 1; 
    end
    c = (1 + sqrt(1+4*cp^2))/2;
    bi = (cp - 1)/c;
    if para.g == "exp" || para.g == "log"
        bi = min(bi,(para.kappa_non-1)/(para.kappa_non+1)*0.5*sqrt(para.nuC));
    else
        bi = min(bi,sqrt(para.nuC));
    end
    
    AAZ = A'*AZ;
    grad = mu*((1+bi)*AAZ - bi*AAZp + A'*EB + cf*A'*Y - AX) + A'*W;
    P = (1+bi)*Z - bi*Zp;
    w = compute_w(sigma,para);
    Zp = Z;
    [Z,sigma] = prox_NN(P - grad/L1,lambda*w/L1);
    AZ = A*Z;
    
    AAZp = AAZ;
    
    %update E
    bi = (cp - 1)/c;
    if para.p == "l21" || para.p == "l1"
        bi = min(bi,sqrt(para.nuC));
    else
        bi = min(bi,(para.kappa_non-1)/(para.kappa_non+1)*0.5*sqrt(para.nuC));
    end
    
    EBB = EB*B';
    grad = mu*((1+bi)*EBB - bi*EBBp + AZ*B' + cf*Y*B' - XB) + W*B';
    P = (1+bi)*E - bi*Ep;
    
    Ep = E;
    E = update_E(E,P - grad/L2,para,L2);
    
    EB = E*B;
    EBBp = EBB;
    
    
    %update Y
    if cf~=0
        grad = mu*(AZ + EB +cf*Y- X)+ W;
        Y = update_Y(Y - grad/(mu*cf),para,mu*cf);
    end
    
    W = W + rho*mu*(AZ + EB + cf*Y - X);
    
    objVal = compute_g(sigma,para) + compute_p(E,para) + compute_r(Y,para);
    Obj(i+1) = objVal;

    objVal = objVal + (mu/2)*sum(sum((AZ + EB + cf*Y - X).^2)) + sum(sum((AZ + EB + cf*Y - X).*W));
    
    Time(i+1) = cputime - tt;
    
    obj(i+1) = objVal;
    if para.iter_acc == 1
        fprintf('iter: %d; obj : %0.4d; diff : %0.4d; acc : %0.4d \n',i,objVal, obj(i) - obj(i+1),max(RMSE(i,:),[],2)); 

        RMSE(i+1,:) = compute_acc_test(X,Z,E,para);
    else
        fprintf('iter: %d; obj : %0.4d; diff : %0.4d \n',i,objVal, obj(i) - obj(i+1)); 
    end
    
    if(sum(Time) > para.maxtime)
        break;
    end
end

output.obj = obj(2:(i+1));
output.Obj = Obj(2:(i+1));
output.RMSE = RMSE(2:(i+1),:);
Time = cumsum(Time);
output.Time = Time(2:(i+1));

output.Z = Z;
output.E = E;
output.Rank = nnz(sigma);
output.Y = Y;
output.W = W;

end

function[g] = compute_g(sigma,para)
    if para.g == "exp"
        g = para.lambda*sum(1-exp(-para.theta*sigma));
    end
    
    if para.g =="NN"
        g = para.lambda*sum(sigma);
    end
    
    if para.g == "log"
        g = para.lambda*sum(sigma + para.epsilon);
    end
end

function[p] = compute_p(E, para)

    if para.p == "exp12"
        sqrt_E = sqrt(sum(E.^2,2));
        p = para.gamma*sum(1-exp(-para.theta*sqrt_E));
    end
    
    if para.p == "l21"
        sqrt_E = sqrt(sum(E.^2,1));
        p = para.gamma*sum(sqrt_E);
    end
    
    if para.p == "exp21"
        sqrt_E = sqrt(sum(E.^2,1));
        p = para.gamma*sum(1-exp(-para.theta*sqrt_E));
    end
    
    if para.p == "log21"
        sqrt_E = sqrt(sum(E.^2,1));
        p = para.gamma*sum(sqrt_E + para.epsilon);
    end
    
    if para.p == "log"
        [~, S, ~] = svd(E, 'econ');
        sigma = diag(S);
        p = para.gamma*sum(sigma + para.epsilon);
    end
    
    if para.p == "exp"
        [~, S, ~] = svd(E, 'econ');
        sigma = diag(S);
        p = para.gamma*sum(1-exp(-para.theta*sigma));
    end
    
    if para.p =="NN"
        [~, S, ~] = svd(E, 'econ');
        sigma = diag(S);
        p = para.gamma*sum(sigma);
    end
    
    if para.p == "l1"
        p = para.gamma*sum(sum(abs(E)));
    end
    
end

function[r] = compute_r(Y,para)
    if para.r == "l2"
        r = (para.beta/2)*sum(sum(Y.^2));
    end
    if para.r == "l1"
        r = para.beta*sum(sum(abs(Y)));
    end
end

function[w] = compute_w(sigma,para)
    if para.g == "exp"
        w = para.theta*exp(-para.theta*sigma);
    end
    
    if para.g == "NN"
        w = ones(length(sigma),1);
    end
    
    if para.g == "log"
        w = 1./(sigma + para.epsilon);
    end
end

function[Z,sigma] = prox_NN(grad,w)
    [U, S, V] = svd(grad, 'econ');
    sigma = diag(S);
    sigma = max(sigma-w,0);

    svp = nnz(sigma);
    if svp == 0
        svp = 1;
    end
    Z = U(:,1:svp)*diag(sigma(1:svp))*V(:,1:svp)';
end

function[E] = update_E(E,grad,para,L)
   
    if para.p == "exp12"
        d = size(E,2);
        gammaL = para.gamma/L;
        sqrt_D = sqrt(sum(grad.^2,2));
        sqrt_E = sqrt(sum(E.^2,2));
        w = para.theta*exp(-para.theta*sqrt_E);
%         w = compute_w(sqrt_D,para);
        E(sqrt_D>gammaL*w,:) = grad(sqrt_D>gammaL*w,:).*repmat(1-gammaL*w(sqrt_D>gammaL*w)./sqrt_D(sqrt_D>gammaL*w),1,d); 
        E(sqrt_D<=gammaL*w,:) = 0;
    end

    if para.p == "l21"
        d = size(E,1);
        gammaL = para.gamma/L;
        sqrt_D = sqrt(sum(grad.^2,1));
        E(:,sqrt_D>gammaL) = grad(:,sqrt_D>gammaL).*repmat(1-gammaL./sqrt_D(sqrt_D>gammaL),d,1); 
        E(:,sqrt_D<=gammaL) = 0;
    end
    
    if para.p == "exp21"
        d = size(E,1);
        gammaL = para.gamma/L;
        sqrt_D = sqrt(sum(grad.^2,1));
        sqrt_E = sqrt(sum(grad.^2,1));
        w = para.theta*exp(-para.theta*sqrt_E);
%         w = compute_w(sqrt_D,para);
        E(:,sqrt_D>gammaL*w) = grad(:,sqrt_D>gammaL*w).*repmat(1-gammaL*w(sqrt_D>gammaL*w)./sqrt_D(sqrt_D>gammaL*w),d,1); 
        E(:,sqrt_D<=gammaL*w) = 0;
    end
    
    if para.p == "log21"
        d = size(E,1);
        gammaL = para.gamma/L;
        sqrt_D = sqrt(sum(grad.^2,1));
        sqrt_E = sqrt(sum(E.^2,1));
        w = 1./(sqrt_E + para.epsilon);
        E(:,sqrt_D>gammaL*w) = grad(sqrt_D>gammaL*w,:).*repmat(1-gammaL*w(sqrt_D>gammaL*w)./sqrt_D(sqrt_D>gammaL*w),d,1); 
        E(:,sqrt_D<=gammaL*w) = 0;
    end
    
    if para.p =="exp"
        [~, S, ~] = svd(E, 'econ');
        sigma = diag(S);
        w = para.theta*exp(-para.theta*sigma);
        [E,~] = prox_NN(grad,para.gamma*w/L);
    end
    
    if para.p =="NN"
        [E,~] = prox_NN(grad,para.gamma/L);
    end
    
    if para.p =="log"
        [~, S, ~] = svd(E, 'econ');
        sigma = diag(S);
        w = 1./(sigma + para.epsilon);
        [E,~] = prox_NN(grad,para.gamma*w/L);
    end
    
    if para.p=="l1"
        gammaL = para.gamma/L;
        E = max(abs(grad)-gammaL,0).*sign(grad);
    end
end

function[Y] = update_Y(grad,para,L)

    if para.r == "l2"
        Y = L*grad/(L+para.beta);
    end
    
    if para.r=="l1"
        gammaL = para.beta/L;
        Y = max(abs(grad)-gammaL,0).*sign(grad);
    end
end

