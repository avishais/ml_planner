clear all
I = load('../paths/path_info.txt');
% I = load('C:\Users\Avishai\Dropbox\UIUC\backup\ckc2d\path_info.txt');
n = I(1);
m = I(2);
L = I(3);
b = I(4:5);
qminmax = rad2deg(I(6));
if length(I)>6
    C = reshape(I(7:end),3,[])';
end

% L = [1 1.55 1 1];
% L = 0.2632*ones(1,19);
% L(2) = 0.51;

vid = 0;

withObs = false;
%%
D = dlmread('../paths/path.txt',' ',1,0);
D = D(:,1:end-1);
% n = size(D,2);
% D = load('C:\Users\Avishai\Dropbox\UIUC\backup\ckc2d\path.txt');
qs = D(1,:);
qg = D(end,:);

if vid
    writerObj = VideoWriter('ckc2.avi');%,'MPEG-4');
    open(writerObj);
end

figure(1)
clf
for j = 1:size(D,1)
    disp(j);
    clf;
    
    %{
    hold on
    xp = 0; yp = 0;
    for i = 1:n-1
        phi = sum(qs(1:i));
        
        x = xp + L*cos(phi);
        y = yp + L*sin(phi);
        
        %plot([xp x],[yp y],'k');
        %plot(x, y, 'ok','MarkerFaceColor','k');
        plotLink([xp yp],[x y],250*[1 1 1]/255);
        xp = x;
        yp = y;
    end
    hold off
    %}
    % Plot goal conf.
    %     hold on
    %     xp = 0; yp = 0;
    %     for i = 1:n-1
    %         phi = sum(qg(1:i));
    %
    %         x = xp + L*cos(phi);
    %         y = yp + L*sin(phi);
    %
    %         %plot([xp x],[yp y],'k');
    %         %plot(x, y, 'ok','MarkerFaceColor','k');
    %         plotLink([xp yp],[x y],254*[1 1 1]/255);
    %         xp = x;
    %         yp = y;
    %     end
    %     hold off
    
    
    q = D(j,:);
    xp = 0; yp = 0;
    for i = 1:n-1
        phi = sum(q(1:i));
        
        x = xp + L*cos(phi);
        y = yp + L*sin(phi);
        
        %plot([xp x],[yp y],'k');
        %plot(x, y, 'ok','MarkerFaceColor','k');
        plotLink([xp yp],[x y],'r',0.1);
        xp = x;
        yp = y;
    end
    hold on
    plot(0, 0,'o','MarkerFaceColor','k','MarkerSize',10);
    plot(b(1), b(2),'o','MarkerFaceColor','k','MarkerSize',10);
    hold off
    
    if length(I)>6 && withObs
        % Obstacles
        hold on
        phi = 0:0.1:2*pi;
        circ = [cos(phi)' sin(phi)'];
        for i = 1:size(C,1)
            patch(C(i,3)*circ(:,1)+C(i,1),C(i,3)*circ(:,2)+C(i,2), 'b');
        end
        hold off
    end
    
    axis equal
    if withObs
        axis([-5 9 -5 9]);
    else
        axis([-b(1)*1 b(1)*2 -2 2]);        
    end
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    
    %text(-2.5708, 8.3754, 'start conf.','FontSize',16);
    %text(-0.4093, -4, 'goal conf.','FontSize',16);
    
    drawnow;
    %             pause();
    
    if vid
        %set(h, 'Position', [1 1 ax])
        Im = getframe;
        %I.cdata = imcrop(I.cdata,[1 1 ax]);
        writeVideo(writerObj, Im);
    end
    
end
if vid
    %     for i = 1:10
    %         writeVideo(writerObj, Im);
    %     end
    close(writerObj);
end
% print startI.png -dpng -r200
%%
% figure(2)
% plot(rad2deg(D(:,:)),'.-');
% hold on
% plot(xlim, qminmax*[1 1], '--');
% plot(xlim, -qminmax*[1 1], '--');
% hold off