function R = FK(q, LeftOrRight, half)
% l - Open chain from left base - LeftOrRight==1
% r - Open chain from right base - LeftOrRight==2

if nargin < 3
    half = 0;
end

n = length(q);
m = length(q);
PhyPro;

R = eye(3);
if LeftOrRight==1    
    for i = m:-1:1
        if half && i == m
            l = L(i)/2;
        else
            l = L(i);
        end
        R = Rz(q(i))*Rt([l;0])*R;
    end
else % LeftOrRight==2
    L = L(end-m+1:end);
    for i = 1:m
        if i==m
            angle = q(i);
        else
            angle = -q(i);
        end
        if i==1
            l = L(i)/2;
        else
            l = L(i);
        end
        R = Rz(angle)*Rt([l;0])*R;
    end
    R = Rt(b)*R;
end