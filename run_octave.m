function [P, D] = run_octave(J, pad, o)
%	[P, D] = run_octave(J, pad, o);
%
%	  - J = integral image of size [M+2*pad, N+2*pad].
%	  - o = octave to search for descriptors in.
%
%	  - P = positions of keypoints as a 4xn matrix: [y;x;L;theta]
%	        - y, x  = absolute position in image space
%	        - L     = scale of the keypoint
%	        - theta = gradient angle estimated around the keypoint
%	  - D = SURF descriptors for keypoints as a 64xn matrix.
%           Each column is a normalized vector.

M = size(J, 1) - 2*pad;
N = size(J, 2) - 2*pad;
printf("begin octave %g\n", o);
ns = 4;
p = 2.^(o-1);
scales = 2.^o .* [1, 2, 3, 4] + 1;

% we calculate Dx, Dy and DoH operators.
% - Dy, Dx are first partial derivatives of the signal
% - DoH stands for determinant of hessian and is computed from
%   the second partial derivatives of the signal.
DoH = zeros(ceil(M ./ p), ceil(N ./ p), ns, "int64");
Dx  = zeros(ceil(M ./ p), ceil(N ./ p), ns, "int64");
Dy  = zeros(ceil(M ./ p), ceil(N ./ p), ns, "int64");

printf(" compute local operators on J ...\n");
for s = 1:ns
	L = scales(s);
	[DoH(:,:,s), Dy(:,:,s), Dx(:,:,s)] = integral_image_operators(J, L, p, pad);
end

printf(" search maxima of DoH ...\n");
% compute the 3x3-windowed maxima of the DoH map.
MaxMap = DoH;
MaxMap = max(MaxMap(1:end-2,:,:), max(MaxMap(2:end-1,:,:), MaxMap(3:end,:,:)));
MaxMap = max(MaxMap(:,1:end-2,:), max(MaxMap(:,2:end-1,:), MaxMap(:,3:end,:)));
MaxMap = max(MaxMap(:,:,1:end-2), max(MaxMap(:,:,2:end-1), MaxMap(:,:,3:end)));


% select points where the DoH operator is locally maximised.
% we discard the very edge of the image, as a descriptor built with information
% that is inferred from boundary conditions is unlikely to be very good for matching.
DoH_crop = DoH(2:end-1,2:end-1,2:end-1);

tH = 1e3;
raw_maxima = find((MaxMap == DoH_crop) & DoH_crop >= tH)';

printf(" compute derivatives of DoH ...\n");
% partial derivatives of DoH, used to refine keypoint positions.
% H: 3x3x(numel(raw_maxima)) matrix. the (:,:,i) component is the hessian at point  (p*(1+y0(i)), p*(1+x0(i)),2*p*(1+s0(i)))
% d: 3x1x(numel(raw_maxima)) matrix. the (:,:,i) component is the gradient at point (p*(1+y0(i)), p*(1+x0(i)),2*p*(1+s0(i)))
[y0, x0, s0] = ind2sub(size(MaxMap), raw_maxima);
[H, d] = hessian_and_gradient(DoH, y0+1, x0+1, s0+1, p, p, 2*p);

printf(" refine and filter %d points ...\n", numel(raw_maxima));
% raffina la posizione dei massimi imponendo il gradiente di DoH uguale a 0.
% refine the position of the maxima solving the linear system grad(DoH(y,x,L)) == 0.
xi = zeros([3, numel(raw_maxima)]);
for i = 1:numel(raw_maxima)
	% xi is the sub-pixel offset of the inferred true maximum from the sample of maximum value.
	xi(:,i) = -H(:,:,i)\d(:,:,i);
end

% coordinates [y0;x0;s0] were in the index space of DoH_crop, which is subsampled compared to the
% original image. We rescale them to be coordinates in image space.
yxL = [p*y0; p*x0; scales(1+s0)] + xi;

% discard the points where the required correction is too large, and the inferred maximum would
% fall beyond a different sample point (which was found to be lower than the actual maximum sample).
non_degen = max(abs(xi ./ [1; 1; 2]), [], 1) < p;

% as observed before, remove the points where a descriptor would be built with data inferred from
% boundary conditions.
in_bounds = min(min(yxL([1,2],:), [M;N]-yxL([1,2],:)), [], 1) > 2*yxL(3,:);

raw_maxima = raw_maxima(non_degen & in_bounds);
yxL = yxL(:,non_degen & in_bounds);

printf(" compute gradient angles for %d points ...\n", numel(raw_maxima));

% offset indices for a 13x13 grid around the keypoints for gradient angle estimation.
[Bv, Bu, Bs] = ndgrid([-6:6],[-6:6],[0]);
Bv = Bv(:);
Bu = Bu(:);
Bs = Bs(:);

% select indices to a circular neighbourhood of the keypoints
select = Bu.^2 + Bv.^2 <= 36;
Bv = Bv(select);
Bu = Bu(select);
Bs = Bs(select);

% samples will be weighted with a gaussian multiplier based on the distance from the keypoint.
GB = exp(-((Bv./2).^2+(Bu./2).^2)./2)(:);

sigma = round(0.4 .* yxL(3,:));
% create indices in the Dx, Dy tensors by offsetting each of the keypoints with each of the values in Bv,Bu,Bs.
% the results are all the points at which Dx, Dy will need to be sampled.
% each of the index matrices is of size [numel(Bv), size(yxL,2)].
% each column represents the coordinates of the sample points in the neighbourhood of a keypoint.
y_indices = 1 + (yxL(1,:) + sigma .* Bv) ./ p;
x_indices = 1 + (yxL(2,:) + sigma .* Bu) ./ p;
s_indices = 1 + (yxL(3,:) - 2.^o - 1) ./ (2*p) + Bs;

% sample the Dx,Dy tensors at the above coordiantes.
% the results are matrices of size [numel(Bv), size(yxL,2)].
% it's fine to use zero-padding here because the angle calculation weighs the samples by their modulus,
% and a null gradient counts as a void vote.
phi_y = interpn(Dy, y_indices, x_indices, s_indices, "linear", 0) .* GB;
phi_x = interpn(Dx, y_indices, x_indices, s_indices, "linear", 0) .* GB;

% compute angles for the gradients
angles = atan2(phi_y, phi_x);

% 40 angles to compare the samples to
test_thetas = [0:39].*pi./20;

% compare each sampled angle with each of the test angle.
% angle alpha is in the range [theta - pi/6, theta + pi/6] mod 2*pi
%     iff alpha - theta + pi/6 is in range [0, pi/3] mod 2*pi
%     iff mod(alpha - theta + pi/6, 2*pi) <= pi/3.
angle_diffs = reshape(angles(:)' - test_thetas(:) + pi./6, [40, size(angles)]);
% select is a boolean valued tensor of size [40, 113, n].
%   - the first dimension enumerates the test angles.
%   - the second dimension enumerates the samples of each keypoint's neighbourhood.
%   - the third dimension enumerates the indices of the keypoints.
select = mod(angle_diffs, 2.*pi) <= pi./3;

% somma dei gradienti il cui angolo è entro pi/6 da ciascun angolo di test.
% Phi_y_theta and Phi_x_theta are 40xn matrices, representing the weighted sum of gradients
% whose angle is within pi/6 of each of the test angles, for each of the n keypoints.
Phi_y_theta = reshape(sum(shiftdim(phi_y, -1) .* select, 2), 40, []);
Phi_x_theta = reshape(sum(shiftdim(phi_x, -1) .* select, 2), 40, []);

% we take the index of the test angle that maximises the modulus of this quantity.
[~, best_theta_index] = max(Phi_y_theta.^2+Phi_x_theta.^2, [], 1);

% compute the angle of the average gradient, among the ones selected by the best test angle.
theta = atan2(
	interpn(Phi_y_theta, best_theta_index, [1:size(Phi_y_theta, 2)], "nearest"),
	interpn(Phi_x_theta, best_theta_index, [1:size(Phi_x_theta, 2)], "nearest")
);

printf(" compute descriptors for %d points ...\n", size(yxL, 2));
% indices for the 20x20 neighbourhoods of the keypoints.
% logic is exactly the same as the sampling for angle estimation.
[Rv Ru Rs] = ndgrid([-9.5:9.5],[-9.5:9.5],[0]);
Rv = Rv(:);
Ru = Ru(:);
Rs = Rs(:);
GR = exp(-((Rv./3.3).^2 + (Ru./3.3).^2)./2)(:);

sth = sin(theta);
cth = cos(theta);

% change coordinates according to the gradient angle of the keypoints to guarantee rotation invariance
% of the descriptor.
y_indices = 1 + (sigma .* ( cth .* Rv + sth .* Ru) + yxL(1,:)) ./ p;
x_indices = 1 + (sigma .* (-sth .* Rv + cth .* Ru) + yxL(2,:)) ./ p;
s_indices = 1 + (yxL(3,:) - 2.^o - 1) ./ (2*p) + Rs;

% sample gradient in the neighbourhood as one 400xn matrix for each component
Dy_samples = interpn(Dy, y_indices, x_indices, s_indices, "linear", 0) .* GR;
Dx_samples = interpn(Dx, y_indices, x_indices, s_indices, "linear", 0) .* GR;

% given that the ravel order of matrices in MATLAB is column-major, we can reinterpret
% the slice (:,:,i) of the sampled data as a [5, 4, 5, 4] tensor where the
% coordinates run as shown in the diagram below.
%
%            e4
%    +========>--------+--------+--------+
%    !        |        |        |        |
%    !        |        |        |        |
%    !        |        |        |        |
%    !        |    e3  |        |        |
% e2 v--------+=====>--+--------+--------+
%    |        !     .  |        |        |
%    |        !     .                    |
%    |     e1 v . . @=du(e1,e2,e3,e4,i)  |
%    |        |                          |
%    +--------+--------+--------+--------+
%    |        |        |        |        |
%    |        |        |        |        |
%    |        |        |        |        |
%    |        |        |        |        |
%    +--------+--------+--------+--------+
%    |        |        |        |        |
%    |        |        |        |        |
%    |        |        |        |        |
%    |        |        |        |        |
%    +--------+--------+--------+--------+

% we invert the rotation to bring the gradients in the frame of reference of the estimated
% gradient angle of the keypoint, which is again necessary for rotation invariance.
dv = reshape(cth .* Dy_samples - sth .* Dx_samples, 5, 4, 5, 4, []);
du = reshape(sth .* Dy_samples + cth .* Dx_samples, 5, 4, 5, 4, []);

% compute the SURF descriptors as sums of the gradient's
% components and their modulus in the 5x5 regions displayed above.
mu = vertcat(
	reshape(sum(sum(dv,3),1),      16, []),
	reshape(sum(sum(du,3),1),      16, []),
	reshape(sum(sum(abs(dv),3),1), 16, []),
	reshape(sum(sum(abs(du),3),1), 16, [])
);

P = [yxL; theta];
% normalize the descriptors (this is necessary for robustness against changes of contrast).
D = mu ./ sqrt(sumsq(mu, 1));
