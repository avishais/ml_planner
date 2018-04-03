function phi = IKp(T, ik_index, chain_num)
% PhyPro;

Lp = [1 1 1];

theta = atan2(T(2,1),T(1,1));

p3 = T(1:2,3);
p2 = p3 - [Lp(3)/2*cos(theta); Lp(3)/2*sin(theta)];

r = norm(p2);
S = (r^2-Lp(1)^2-Lp(2)^2)/(-2*Lp(1)*Lp(2));
Ss = (1-S^2);

if ik_index == 1
    sign = 1;
else
    sign = -1;
end

if Ss<=1e-8
    phi = -1;
    return;
end

phi = zeros(3,1);

Phi = atan2(sign*Ss^0.5, S);
phi(2) = pi+Phi;%-(pi - Phi);
if phi(2) < -pi
    phi(2) = phi(2) + 2*pi;
end
if phi(2) > pi
    phi(2) = phi(2) - 2*pi;
end

phi(1) = atan2(Lp(2)*sin(Phi), Lp(1)-Lp(2)*cos(Phi)) + atan2(p2(2), p2(1));
if phi(1) < -pi
    phi(1) = phi(1) + 2*pi;
end
if phi(1) > pi
    phi(1) = phi(1) - 2*pi;
end

phi(3) = theta - (phi(1)+phi(2));
if phi(3) < -pi
    phi(3) = phi(3) + 2*pi;
end
if phi(3) > pi
    phi(3) = phi(3) - 2*pi;
end
