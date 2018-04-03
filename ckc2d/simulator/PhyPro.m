L = 1*ones(n,1);

b = [7,4];
o = [0,0];

Rz = @(x) [cos(x) -sin(x) 0; sin(x) cos(x) 0; 0 0 1];
Rt = @(t) [1 0 t(1); 0 1 t(2); 0 0 1];