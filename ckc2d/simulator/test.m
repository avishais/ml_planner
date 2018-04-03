% test
n = 20;
PhyPro;

if 1
    % Start state
    q = [95 10 10 10  -2 -10 -10 -30 -40 -30 -50 -10 5 20 10 10 10];
    ik_sol = 2;
else
    % Goal state
    %q = [-122 20 30 30 40 30 20 20 -20 -23 -35 -30 35 40 40 30 20];
    q = [-122 20 30 30 40 35 35 -10 -45 -30 -20 30 45 45 15 25 -10];
    ik_sol = 1;
end
q = [q zeros(1, n-numel(q))];

q = deg2rad(q);
Tl = FK(q(1:n-3), 1, 1);
Tl = Tl * [-1 0 0; 0 -1 0; 0 0 1];
T = [1 0 b(1); 0 1 b(2); 0 0 1];

qr = IKp(T^-1*Tl, ik_sol);
q(n-2:n) = flipud(qr.*[1;-1;-1]);

if  q(end) < 0
    q(end) = q(end) + 2*pi;
end

plotSystem(q);
identify_IK(q')

rad2deg(max(abs(q)))

str = [];
for i = 1:n
str = [str num2str(q(i)) ', '];
end
str