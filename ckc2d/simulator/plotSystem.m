function plotSystem(q)

fig = 1;

if nargin < 1
    % D = load('/home/avishai/Downloads/omplapp/ompl/Workspace/ckc2d/path.txt');
    % q = D(1,:);
%     q = [-0.166233 0.33943 0.953414 -1.24087 -0.806106 2.22124 ];
    q = [-0.955006 1.40412 0.213556 -1.30293 1.01319 -2.76867+2*pi];
end

if size(q,2) < size(q,1)
    q = q';
end

% disp(q);
n = size(q,2);

PhyPro;

figure(fig)
clf
% subplot(121)
hold on
xp = o(1); yp = o(2);
for i = 1:n-1
    phi = sum(q(1:i));
    text(xp+0.3, yp, num2str(i),'FontSize',18);
    
    x = xp + L(i)*cos(phi);
    y = yp + L(i)*sin(phi);
    
    %plot([xp x],[yp y],'k');
    %plot(x, y, 'ok','MarkerFaceColor','k');
    plotLink([xp yp],[x y],'r');
    xp = x;
    yp = y;
end
hold on
plot(0, 0,'o','MarkerFaceColor','k','MarkerSize',7);
plot(b(1), b(2),'o','MarkerFaceColor','k','MarkerSize',7);
hold off

% Obstacles
hold on
phi = 0:0.1:2*pi;
circ = [cos(phi)' sin(phi)'];
C = [-3.2000    4.0000    1.0000
    2.6000    6.8000    1.0000
    4.2000   -2.9000    1.5000
    8.0000    2.0000    0.8000];
for i = 1:size(C,1)
    patch(C(i,3)*circ(:,1)+C(i,1),C(i,3)*circ(:,2)+C(i,2), 'b');
end
hold off

axis equal
grid

% subplot(122)
% hold on
% xp = b(1); yp = b(2);
% for i = n:-1:2
%     if i==n
%         phi = q(n);
%     else
%         phi = q(n)-sum(q(n-1:-1:i));
%     end
%         
%     x = xp + L(i-1)*cos(phi);
%     y = yp + L(i-1)*sin(phi);
%     
%     %plot([xp x],[yp y],'k');
%     %plot(x, y, 'ok','MarkerFaceColor','k');
%     plotLink([xp yp],[x y],'r');
%     xp = x;
%     yp = y;
% end
% hold on
% plot(0, 0,'o','MarkerFaceColor','k','MarkerSize',7);
% plot(b(1), b(2),'o','MarkerFaceColor','k','MarkerSize',7);
% hold off
% axis equal
% grid


end

