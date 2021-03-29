clear

nlevels = 4;

nxprocs = [1,2,4,6,8];

% 1x1 = 1 
% 
% 2x2 = 4
% 2x4 = 8
% 2x6 = 12
% 2x8 = 16
% 4x2 = 8
% 4x4 = 16
% 4x6 = 24
% 4x8 = 32
% 6x6 = 36
% 6x8 = 48
% 
% 2x2x2 = 8
% 2x2x4 = 16
% 2x2x6 = 24
% 2x2x8 = 32
% 2x4x2 = 16
% 2x4x4 = 32
% 2x4x6 = 48
% 2x4x8 = 64
% 2x6x2 = 24
% 2x6x4 = 48
% 2x6x6 = 72
% 2x6x8 = 96
% 2x8x2 = 32
% 2x8x4 = 64
% 2x
% 
% 

% muls = 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 128 

% % 

divs = nxprocs*2^(nlevels-1);

N = 400;
mat = divs.*(1:N)';

vals = [];

for idx1 = 1:N
%     for idx2 = 1:numel(nxprocs)-1
%         if any(mat(idx1,end) == mat(:,idx2))
    if all(any(mat(idx1,end) == mat,1))
            vals = [vals,mat(idx1,end)];
    end
%     end
end
vals = sort(unique(vals))