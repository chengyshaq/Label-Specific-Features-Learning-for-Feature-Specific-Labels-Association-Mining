function [HammingLoss,OneError,Coverage,RankingLoss,Average_Precision,Outputs,Pre_Labels] = JLSFFSL(train_data,train_target,test_data,test_target,fl_ent,para)

alpha   = para.alpha;
beta    = para.beta;
gamma   = para.gamma;
C       = para.C;
maxIter = para.maxIter;
miniLossMargin = para.miniLossMargin;

%% training
X = train_data;
Y = train_target';
n = size(Y,1);

Omega_train = kernel_matrix(X, gamma);
W_s=((Omega_train+speye(n)/C)\Y);

W_s_1 = W_s;

% Label Density Correlation

R     = pdist2( fl_ent+eps,fl_ent+eps, 'cosine' );
% R     = fl_ent*fl_ent';

iter    = 1;
oldloss = 0;

Lip = sqrt(2*(norm(Omega_train)^2 + 2*norm(alpha*R)^2));

bk = 1;
bk_1 = 1;

%% proximal gradient
while iter <= maxIter
    
    W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
    %Gw_s_k = W_s_k - 1/Lip * ((XTX*W_s_k - XTY) + alpha * W_s_k*R);
    Gw_s_k = W_s_k - 1/Lip * ((Omega_train*W_s_k - Y) + alpha * W_s_k*R);
    bk_1   = bk;
    bk     = (1 + sqrt(4*bk^2 + 1))/2;
    W_s_1  = W_s;
    W_s    = softthres(Gw_s_k,beta/Lip);
    
    %predictionLoss = trace((X*W_s - Y)'*(X*W_s - Y));
    predictionLoss = trace((Omega_train*W_s - Y)'*(Omega_train*W_s - Y));
    correlation     = trace(R*W_s'*W_s);
    sparsity    = sum(sum(W_s~=0));
    totalloss = predictionLoss + alpha*correlation + beta*sparsity;
    
    if abs(oldloss - totalloss) <= miniLossMargin
        disp(iter);
        break;
    elseif totalloss <=0
        break;
    else
        oldloss = totalloss;
    end
    
    iter=iter+1;
end
W = W_s;
toc
%% testing
Xt = test_data;
Omega_test = kernel_matrix(X, gamma, Xt);
Outputs = Omega_test'*W;
Pre_Labels = sign(Outputs');

HammingLoss=Hamming_loss(Pre_Labels,test_target);

Outputs = Outputs';
OneError=One_error(Outputs,test_target);
Coverage = coverage(Outputs,test_target);
RankingLoss=Ranking_loss(Outputs,test_target);
Average_Precision=Average_precision(Outputs,test_target);
% Accuracy=MultiLabelAccuracyEvaluation(Pre_Labels,test_target);
% ExactMatch = Exact_match(Pre_Labels,test_target);
% MacroF1 = Macro_F1(test_target,Pre_Labels);
% MicroF1 = Micro_F1(test_target,Pre_Labels);
