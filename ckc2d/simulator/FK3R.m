function T = FK3R(phi, n)

PhyPro;

T = Rz(phi(1)) * Rt([L(1); 0]) * Rz(phi(2)) * Rt([L(2); 0]) * Rz(phi(3)) * Rt([L(3)/2; 0]);