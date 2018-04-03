function q = sample(n, IK_sol)
PhyPro;

if nargin==1
    IK_sol = -1;
end

c = 1; max_c = 1000;
while c < max_c
    q = rand_joints(n);
    
    % chain_num == 1
    R = eye(3);
    for i = 4:n
        if i==n
            angle = q(i);
        else
            angle = -q(i);
        end
        if i==4
            l = L(i-1)/2;
        else
            l = L(i-1);
        end
        R = Rz(angle)*Rt([l;0])*R;
    end
    R = Rt(b)*R;
    
    T = R * [-1 0 0; 0 -1 0; 0 0 1];
    
    if IK_sol == -1
        phi = IKp(T, randi(2), 1);
    else
        phi = IKp(T, IK_sol, 1);
    end
    
    if length(phi)>1 %&& phi~=-1
        %T1 = FK3R(phi);
        %disp([T T1]);
        q(1:3) = phi;
        break;
    end
    c = c + 1;
end

if c == max_c
    disp('No sample found');
end

end

function q = rand_joints(n)
% Generate random joint angles in [-pi,pi]

% q = deg2rad([68 -35 -32 -35 -30 100]);
% q(6) = pi + sum(q(1:n-1));


q = rand(n,1)*2*pi - pi;

end

