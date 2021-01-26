clear
close all

m = 21;
xvec = linspace(-1,1,m)';
yvec = linspace(-1,1,m)';

k = 2;
Ik = eye(k);
e1 = [1,0]';
e2 = [0,1]';

F = @(x,t) [sin(x);
    cos(x)];

A = kr(sin(xvec),cos(yvec),e1) + kr(cos(xvec),sin(yvec),e2);

B = zeros(m*k,1);
for xidx = 0:m-1
    for yidx = 0:m-1
        B(k*(m*xidx + yidx) + 0 + 1) = sin(xvec(xidx+1))*cos(yvec(yidx+1));
        B(k*(m*xidx + yidx) + 1 + 1) = cos(xvec(xidx+1))*sin(yvec(yidx+1));
    end
end

max(abs(B - A))