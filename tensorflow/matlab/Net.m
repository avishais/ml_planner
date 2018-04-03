function x_out = Net(x_in, W,b, x_max, x_min)

T = x_in;%normz(x_in, x_max, x_min);
% disp(T)
for i = 1:numel(W)
    T = T*W{i} + b{i};
%     T = tanh(T);
    T = sigmoid(T);
    if i==4
        disp(T);
    end
%     disp(T);
    T;
end

disp(T)
% disp(denormz(T, x_max, x_min));

end

function x = normz(x, x_max, x_min)

x = (x-x_min)/(x_max-x_min);

end

function x = denormz(x, x_max, x_min)

x = x*(x_max-x_min) + x_min;

end