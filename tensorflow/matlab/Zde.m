clear all

net_rep;

X = load('../../data/abb_samples_noJL_short.db');
N = size(X,1);

Z = encoder(X, W, b, x_max, x_min);

% Z = zeros(N, 6);
% for i = 1:N
%     Z(i,:) = encoder(X(i,:));
% end



%% 
% Zrand = [0.3 0.6 0.5 0.5 0.95 0.5];%rand(1,6);
% Zrand = [0.5 0.6 0.05 0.05 0.05 0.9];
Zrand = rand(1000,6);

[inx, D] = knnsearch(Z, Zrand, 'K', 1);

Y = zeros(size(Zrand,1),1);
for i = 1:size(Zrand,1)
    Y(i) = any(D(i,:) < 0.25);    
end


%%
ij = [1 2 3; 1 2 4; 1 2 5; 1 2 6];

figure(1)
for i = 1:4
    subplot(2,2,i)
    plot3(Z(:,ij(i,1)),Z(:,ij(i,2)),Z(:,ij(i,3)),'.b');
    hold on
    for j = 1:length(Y)
        if Y(j)
            plot3(Zrand(j,ij(i,1)),Zrand(j,ij(i,2)),Zrand(j,ij(i,3)),'ok','markerfacecolor','g');
        else
            plot3(Zrand(j,ij(i,1)),Zrand(j,ij(i,2)),Zrand(j,ij(i,3)),'ok','markerfacecolor','r');
        end
%         plot3(Z(inx(j),ij(i,1)),Z(inx(j),ij(i,2)),Z(inx(j),ij(i,3)),'ok','markerfacecolor','m');
    end
    hold off
end

