function[acc] = compute_acc_test(X,Z,E,para)
    if para.er == "knn"
        
        E = E*para.Q2';
        Train = E*X;
        Test = E*para.Test;
        ks = para.nknn;
        acc = zeros(1,length(ks));
        for j = 1:length(ks)
            k = ks(j);
            Mdl = fitcknn(Train',para.yTrain,'NumNeighbors',k,'Standardize',1);
            ypred = predict(Mdl,Test');
            acc(j) = sum(abs(ypred-para.yTest)<0.1)/length(ypred);
        end
    end
    
    if para.er =="ncut"
        Z = para.Q1*Z;
        if sum(sum(Z))==0
            acc = 0;
        else
            acc = Cluster(Z,para.K,para.yTrain);
        end
    end
end