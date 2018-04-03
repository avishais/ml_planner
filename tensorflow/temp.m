X1 = load('hparam_abb.txt'); 
X2 = load('hparam_abb_skopt.txt');
X = [X1(:,2:end); X2];
X = [(1:size(X,1))' X];

X(isnan(X(:,6)),:)=[];

lr = X(:,2);
lm = max(lr);

r = linspace(0,lm,20);
b = zeros(length(r),1);
for i = 2:length(r)
    M = X(X(:,2)>r(i-1) & X(:,2)<r(i), :);
    b(i-1) = mean(M(:,6));
end

plot(r(2:end),b(2:end))

%%
for i = 1:3
    c(i) = mean(X(X(:,5)==i,6));
end

plot(1:3, c, 'ko');

%%
for i = 1:16
    d(i) = mean(X(X(:,4)==i,6));
end

% plot(1:16, d, 'ko');
plot3(X(:,3),X(:,4),X(:,6),'o');
grid on
zlim([0 0.05]);