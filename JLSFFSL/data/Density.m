
if exist('train_data','var')==1
    data    = [train_data;test_data];
    target  = [train_target,test_target];
    target(target==0) = -1;
    clear train_data test_data train_target test_target
else
    target(target==0) = -1;
end
LD = sum(target==1)/sum(target~=0);