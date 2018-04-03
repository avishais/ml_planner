function IK = identify_IK(q)

n = length(q);
PhyPro;

for i = 1:n-3
    if i==1
        Tl = eye(3);
    else
        Tl = FK(q(1:i-1), 1);
    end
    Tr = FK(q(i+3:n), 2);
    Tr = Tr * [-1 0 0; 0 -1 0; 0 0 1];
    
    Tp = Tl^-1*Tr; % Tp = FK3R(q(i:i+2));
    IK(i) = solve_all_IK(Tp, q(i:i+2), i); 
end

% Last passive chain based on the right joint
Tl = FK(q(1:n-3), 1, 1);
Tl = Tl * [-1 0 0; 0 -1 0; 0 0 1];
T = [1 0 b(1); 0 1 b(2); 0 0 1];
Tp = T^-1*Tl;
% Tr = FK(q(n-2:n), 2);
qo = flipud(q(n-2:n).*[-1; -1; 1]);
IK(n-2) = solve_all_IK(Tp, qo, n-3); 

end

function ik_num = solve_all_IK(T, phi_q, chain_num)

for i = 1:2
    phi = IKp(T, i, chain_num);
    if norm(phi_q-phi) < 3e-2
        ik_num = i;
        return;
    end
end

ik_num = 0; % No solution found
return;

end

