clear all

R = load('../net_abb3.netxt');
n = R(1);
R(1) = [];

%%
% Weights
for i = 1:n
    r = R(1);
    w = R(2);
    R(1:2)=[];
    
    Wt = R(1:r*w);
    R(1:r*w) = [];
    
    W{i} = reshape(Wt, [w, r])';
end

% Biases
for i = 1:n
    w = R(1);
    R(1)=[];
    
    bt = R(1:w);
    R(1:w) = [];
    
    b{i} = bt;
end

x_max = R(1); x_min = R(2);

%% Test net

% X = load('inout_5bars.txt');
% 
% Xi = X(1:end,1:5);
% Z = X(:,6:7);
% Xo = X(:,8:12);
% 
% 
% %%
% clc
% x_in = Xi(1,:);
% Net(x_in, W,b, x_max, x_min);
% disp(Z(1,:))
% disp(Xo(1,:));
