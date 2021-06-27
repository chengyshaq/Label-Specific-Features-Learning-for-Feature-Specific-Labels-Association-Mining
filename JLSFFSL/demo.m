%% Make experiments repeatedly
% rng(1);
clear all;clc
%% Add necessary pathes
addpath('data','eval');

%% Choose a dataset
dataset  =   'Image';
load([dataset,'.mat']);

%% set para
para.alpha = 2.^-10;
para.beta = 2.^-2;
para.gamma = 2.^2;
para.C = 2.^2;
para.maxIter = 100;
para.miniLossMargin = 0.0001;
L2Norm = 1;
%% Set data
if exist('train_data','var')==1
    data    = [train_data;test_data];
    target  = [train_target,test_target];
    target(target==0) = -1;
    clear train_data test_data train_target test_target
else
    target(target==0) = -1;
end

if L2Norm == 1
    temp_data = data;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
    if sum(sum(isnan(temp_data)))>0
        temp_data = data+eps;
        temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
    end
else
    temp_data = data;
end
clear data

%% Perform n-fold cross validation
num_fold = 10; Results = zeros(6,num_fold);
indices = crossvalind('Kfold',size(temp_data,1),num_fold);
for i = 1:num_fold
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test;
    train_data = temp_data(train,:);
    train_target = target(:,train);
    test_data = temp_data(test,:);
    test_target = target(:,test);
    tic;
    % IG
    fl_ent = IG_significance(train_data,train_target')';
    [HammingLoss,OneError,Coverage,RankingLoss,Average_Precision,Outputs,Pre_Labels] = JLSFFSL(train_data,train_target,test_data,test_target,fl_ent,para);
    Results(1,i) = toc;
    Results(2:end,i) = [HammingLoss,OneError,Coverage,RankingLoss,Average_Precision];
end

%% Show the experimental results
meanResults = squeeze(mean(Results,2));
stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));
% printmat([meanResults,stdResults],dataset,'Time HammingLoss Accuracy ExactMatch MacroF1 MicroF1','Mean Std.');
printmat([meanResults,stdResults],dataset,'Time HammingLoss OneError Coverage RankingLoss Average_Precision','Mean Std.');
