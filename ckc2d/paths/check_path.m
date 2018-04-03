clear all
clc

D = dlmread('./path.txt',' ',1,0);
D = D(:,1:end-1);

DM = dlmread('./path_milestones.txt',' ',0,0);
DM = DM(:,1:end-1);
[k,d] = knnsearch(D,DM);

%%

figure(1)
clf
hold on
for i = 1:size(D,2)
    %     plot(D(:,1),D(:,i),'.-k');
    plot(rad2deg(D(:,i)),'.-k');
    
    plot(k, rad2deg(DM),'or','markerfacecolor','r');
    
    %     plot(Do(:,i),'x--r');
    plot(xlim,180*[1 1],':k','linewidth',1.5);
    plot(xlim,-180*[1 1],':k','linewidth',1.5);
    
    grid on
    %     pause(1)
end
hold off
ylabel('angles [^o]');
title(['Local connection distance: ' num2str( norm(D(1,:)-D(end,:)) ) ]);

%%
% for i = 2:size(D,1)
%     d(i-1) = norm(D(i,:)-D(i-1,:));
%     m(i-1) = max(abs(D(i,:)-D(i-1,:)));
% end
% figure(2)
% plot(d);
% hold on
% plot(m);
% hold off
% legend('d','m');
