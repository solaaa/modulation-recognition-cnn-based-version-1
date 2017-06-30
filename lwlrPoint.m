% 对单个点计算估计值
function yHatPoint = lwlrPoint(point, xMat, yMat, k)
     [row , ~] = size(xMat);
     weights = zeros(row, row);
     for i = 1:1:row
         diffMat = point - xMat(i, :);
         weights(i,i) = exp(diffMat * diffMat.' / (-2.0 * (k ^ 2)));    %计算权重对角矩阵
     end
     xTx = xMat.' * (weights * xMat);   % 奇异矩阵不能计算
     if det(xTx) == 0
         disp('This matrix is singular, cannot do inverse');
     end
     theta = xTx^-1 * (xMat.' * (weights * yMat));          %计算回归系数
     yHatPoint = point * theta;
end
