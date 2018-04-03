X = load('abb_samples_noJL (copy).txt');
X = X(1:1e5,:);

%%
d = [1 2 4];
plot3(X(:,d(1)),X(:,d(2)),X(:,d(3)),'k.');
axis equal
grid on