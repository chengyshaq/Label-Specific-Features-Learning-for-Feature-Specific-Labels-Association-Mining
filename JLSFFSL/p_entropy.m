function [res] = p_entropy( vector )

[uidx,v,single] = unique( vector, 'rows' );%vector=uidx(single);uidx排序后的特征值（去除相同特征），single特征值的位置
count = zeros(size(uidx,1),1);
for k=1:size(vector,1)%特征总得数量
    count( single(k), 1 ) = count( single(k), 1 ) + 1;
end
res = -( (count/size(vector,1))'*log2( (count/size(vector,1)) ) );
