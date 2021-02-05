addpath(genpath('./data'));
dataset = {'Yaleb10'};

%%%%%%%%%%%%set parameters
para.maxtime = 500;
para.g = "NN";
para.p = "exp21";
para.r = "l2";
para.maxIter = 1e8;
para.maxinner = 20;
para.cf = 1;
para.lambda = 1;
para.gamma = 1;
para.beta = 1;
para.theta = 5;
para.epsilon = 0.1;
para.rho = 1;
para.nuC = 1-1e-15;
para.C1 = 0.999999;

tmp = 3*para.rho/(1-abs(1-para.rho))^2;
para.mu = 2*(2+para.C1)*tmp/para.C1;

para.kappa = 1;
para.kappa_non = 1.1;

for i = 1:1
    
data = loadmatfile([dataset{i}, '.mat']);
X = data.X;
gnd = data.cids;
[~, n] = size(X);
gnd = gnd';
K = max(gnd');
Q1 = orth(X');
Q2 = orth(X);
A = X*Q1;
B = Q2'*X;

para.Q1 = Q1;
para.Q2 = Q2;

para.yTrain = gnd;
para.nknn = [1];
para.er = "ncut";
para.iter_acc = 0;
para.K = K;


timeStamp = strcat(datestr(clock,'yyyy-mm-dd_HH-MM-ss'));


para.inertial = 1; %iADMM with extrapolation
out{1} = iADMM(X, A, B, para );
para.inertial = 0;% iADMM without extrapolation
out{2} = iADMM(X, A, B, para );
out{3} = linearizedADMM(X, A, B, para );


close all;
LEGEND = categorical({'iADMM-mm','ADMM-mm','linearizedADMM'});
ERROR = [];
for j = 1:3
    ERROR = [ERROR,1-compute_acc_test(X,out{j}.Z,out{j}.E,para)];
end
figure;
subplot(1, 2, 1);
hold on;
bar(LEGEND,ERROR)
xlabel('Method');
ylabel('Error');
title(dataset{i})

figure;
subplot(1, 2, 1);
hold on;
plot(out{1}.Time, out{1}.obj, 'r', 'LineWidth',3);

hold on;
plot(out{2}.Time, out{2}.obj, 'b', 'LineWidth',3);
hold on;
plot(out{3}.Time, out{3}.obj, 'g', 'LineWidth',3); 
xlabel('Time (s)');
ylabel('Objective value');
legend('iADMM-mm','ADMM-mm','linearizedADMM');
title(dataset{i})


end


% save(['result/', dataset{i}, strcat(timeStamp,'.mat')], 'out'); 



