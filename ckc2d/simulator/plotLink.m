function plotLink(p1, p2, color,b)

if nargin < 4
    b = 0.01;
end
EdgeColor = 'k';

phi = 0:0.1:2*pi;
circ = [b*cos(phi') b*sin(phi')];
hold on
v = p2'-p1';
vt = rotz(90)*[v; 0]; vt = vt(1:2)/norm(vt(1:2));
v = [p2(1)+vt(1)*b, p2(2)+vt(2)*b;
    p2(1)-vt(1)*b, p2(2)-vt(2)*b;
    p1(1)-vt(1)*b, p1(2)-vt(2)*b;
    p1(1)+vt(1)*b, p1(2)+vt(2)*b];
patch(v(:,1), v(:,2),color,'EdgeColor',EdgeColor);

patch(circ(:,1)+p1(1), circ(:,2)+p1(2),color,'EdgeColor',EdgeColor);
patch(circ(:,1)+p2(1), circ(:,2)+p2(2),color,'EdgeColor',EdgeColor);
% plot(p1(1), p1(2), 'ok','MarkerFaceColor','k');
% plot(p2(1), p2(2), 'ok','MarkerFaceColor','k');

hold off
% axis equal

end

function R = rotz(ang)

R = [cosd(ang) -sind(ang) 0; sind(ang) cosd(ang) 0; 0 0 1];

end



