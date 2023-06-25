function [H, d] = hessian_and_gradient(S, y, x, z, sy, sx, sz)
%   [H,d] = hessian_and_gradient(S, y, x, z, sy, sx, sz);
%     - S        = tensore tridimensionale, rappresenta un campo scalare.
%     - y, x, z  = matrici di uguali dimensioni, rappresentanti le tre
%                  componenti di indici in S.
%     - sy,sx,sz = scalari, rappresentanti il passo del campionamento di S
%                  lungo le tre dimensioni.
%
%     - H        = tensore 3x3x(size(y)), rappresenta la matrice hessiana
%                  di S nei punti indicati da y,x,z.
%     - d        = tensore 3x1x(size(y)), rappresenta il gradiente
%                  di S nei punti indicati da y,x,z.

% offset dal punto base in cui valutare il segnale.
dy = [0, -1, 1,  0, 0,  0, 0, -1,  1, -1, 1, -1,  1, -1, 1,  0,  0,  0, 0];
dx = [0,  0, 0, -1, 1,  0, 0, -1, -1,  1, 1,  0,  0,  0, 0, -1,  1, -1, 1];
dz = [0,  0, 0,  0, 0, -1, 1,  0,  0,  0, 0, -1, -1,  1, 1, -1, -1,  1, 1];

Si = interpn(S, y(:)+dy, x(:)+dx, z(:)+dz, "nearest");

% gradiente
d = zeros([3, 1, numel(y)]);
d(1,1,:) = (Si(:,3) - Si(:,2)) ./ (2*sy);
d(2,1,:) = (Si(:,5) - Si(:,4)) ./ (2*sx);
d(3,1,:) = (Si(:,7) - Si(:,6)) ./ (2*sz);

H = zeros([3, 3, numel(y)]);
% hessiana, componenti omogenee
H(1,1,:) = (Si(:,3) + Si(:,2) - 2*Si(:,1)) ./ (sy^2);
H(2,2,:) = (Si(:,5) + Si(:,4) - 2*Si(:,1)) ./ (sx^2);
H(3,3,:) = (Si(:,7) + Si(:,6) - 2*Si(:,1)) ./ (sz^2);

% hessiana, componenti miste
H(1,2,:) = H(2,1,:) = ((Si(:, 8) + Si(:,11)) - (Si(:, 9) + Si(:,10))) ./ (4*sy*sx);
H(1,3,:) = H(3,1,:) = ((Si(:,12) + Si(:,15)) - (Si(:,13) + Si(:,14))) ./ (4*sy*sz);
H(2,3,:) = H(3,2,:) = ((Si(:,16) + Si(:,19)) - (Si(:,17) + Si(:,18))) ./ (4*sx*sz);

H = reshape(H, [3, 3, size(y)]);
d = reshape(d, [3, 1, size(y)]);
