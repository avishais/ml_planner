function x_out = decoder(z_in, W,b, x_max, x_min)

T = z_in;
for i = numel(W)/2:numel(W)
    T = T*W{i} + b{i};
    %     T = tanh(T);
    T = sigmoid(T);
end

x_out = denormz(T, x_max, x_min);
end

function x = denormz(x, x_max, x_min)

x = x*(x_max-x_min) + x_min;

end