X = load('inout_5bars.txt');

Xi = X(:,1:5);
Z = X(:,6:7);
Xo = X(:,8:12);

%%

figure(1)
for j = 1:5
    subplot(2,3,j)
    hist([Xi(:,j) Xo(:,j)]);
    title(num2str(j));
end

figure(2)
plot(Z(:,1),Z(:,2),'k.')