clear

syms x y t rho K

rho = 1;
K = 1;

A = [0,0,-1/rho;
     0,0,0;
     -K,0,0];
 
B = [0,0,0;
    0,0,-1/rho;
    0,-K,0];

nn = 3;
mm = 4;

u = -nn*cos(nn*pi*x).*sin(mm*pi*y).*sin(pi*sqrt(nn*nn + mm*mm)*t)/sqrt(nn*nn + mm*mm);
v = -mm*sin(nn*pi*x).*cos(mm*pi*y).*sin(pi*sqrt(nn*nn + mm*mm)*t)/sqrt(nn*nn + mm*mm);
p = sin(pi*nn*x).*sin(pi*mm*y).*cos(pi*sqrt(nn*nn + mm*mm)*t);

w = [u;v;p];

wt = diff(w,t);
wx = diff(w,x);
wy = diff(w,y);

res = wt - A*wx - B*wy

subs(p,'y',-1)