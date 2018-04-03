function z = encoder(x_in, W,b, x_max, x_min)

T = normz(x_in, x_max, x_min);
for i = 1:numel(W)/2
    T = T*W{i} + b{i};
%     T = tanh(T);
    T = sigmoid(T);
end

z = T;
end

function x = normz(x, x_max, x_min)

x = (x-x_min)/(x_max-x_min);

end

