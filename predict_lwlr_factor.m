% according to semultaneous gradient to predict the LWLR factor
function [ factor ] = predict_lwlr_factor(x, y)
sum = 0;
for i = 2:128
    sum = sum + abs(y(i) - y(i-1)) + abs(x(i) - x(i-1));
end

 p = [-0.234, 1.4549, -3.4053, 3.6493, -1.7388, 0.4138, 0.0727];
 
     factor = 0;
     for j = 1:7
         factor = factor + p(j)*(sum^(7-j));
     end

end
