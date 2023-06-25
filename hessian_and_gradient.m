function [H, d] = hessian_and_gradient(S, y, x, z, sy, sx, sz)
%   [H,d] = hessian_and_gradient(S, y, x, z, sy, sx, sz);
%     - S        = 3d tensor representing a scalar field.
%     - y, x, z  = equally sized matrices representing the three components of indices in S.
%     - sy,sx,sz = scalars representing the sampling step of S along each dimension.
%
%     - H        = 3x3x(size(y)) tensor, representing the hessian matrix of S at each point in y,x,z.
%     - d        = 3x1x(size(y)) tensor, representing the gradient of S at each point in y,x,z.

% offsets of the samples relative to the base points.
dy = [0, -1, 1,  0, 0,  0, 0, -1,  1, -1, 1, -1,  1, -1, 1,  0,  0,  0, 0];
dx = [0,  0, 0, -1, 1,  0, 0, -1, -1,  1, 1,  0,  0,  0, 0, -1,  1, -1, 1];
dz = [0,  0, 0,  0, 0, -1, 1,  0,  0,  0, 0, -1, -1,  1, 1, -1, -1,  1, 1];

Si = interpn(S, y(:)+dy, x(:)+dx, z(:)+dz, "nearest");

% gradient
d = zeros([3, 1, numel(y)]);
d(1,1,:) = (Si(:,3) - Si(:,2)) ./ (2*sy);
d(2,1,:) = (Si(:,5) - Si(:,4)) ./ (2*sx);
d(3,1,:) = (Si(:,7) - Si(:,6)) ./ (2*sz);

H = zeros([3, 3, numel(y)]);
% homogenous components of the hessian
H(1,1,:) = (Si(:,3) + Si(:,2) - 2*Si(:,1)) ./ (sy^2);
H(2,2,:) = (Si(:,5) + Si(:,4) - 2*Si(:,1)) ./ (sx^2);
H(3,3,:) = (Si(:,7) + Si(:,6) - 2*Si(:,1)) ./ (sz^2);

% heterogeneous components of the hessian
H(1,2,:) = H(2,1,:) = ((Si(:, 8) + Si(:,11)) - (Si(:, 9) + Si(:,10))) ./ (4*sy*sx);
H(1,3,:) = H(3,1,:) = ((Si(:,12) + Si(:,15)) - (Si(:,13) + Si(:,14))) ./ (4*sy*sz);
H(2,3,:) = H(3,2,:) = ((Si(:,16) + Si(:,19)) - (Si(:,17) + Si(:,18))) ./ (4*sx*sz);

H = reshape(H, [3, 3, size(y)]);
d = reshape(d, [3, 1, size(y)]);
