function [P, D] = make_descriptors(I)
%   [P, D] = make_descriptors(I)
%     - I: image to compute the descriptors of
%     - P: 4xn matrix: [y;x;L;theta] each column represents coordinates,
%          scale and orientation of a keypoint
%     - D: 64xn matrix: each column is a normalized vector representing
%          the SURF descriptor of a keypoint.

octaves = [1, 2, 3, 4];
maxScale = 2.^max(octaves) * 4 + 1;

pad = ceil(3/2*maxScale);
[M, N] = size(I);

% impose reflected boundary conditions.
I = [I(:,pad+1:-1:2), I, I(:,end-1:-1:end-pad-1)];
I = [I(pad+1:-1:2,:); I; I(end-1:-1:end-pad-1,:)];

% compute summed area table (integral image)
J = zeros(size(I) + [1, 1], "int64");
J(2:end, 2:end) = cumsum(cumsum(uint64(I)), 2);

P = [];
D = [];

% compute keypoints and descriptors for different octaves.
for o = octaves
	[new_points, new_descriptors] = run_octave(J, pad, o);
	P = [P,new_points];
	D = [D,new_descriptors];
end
