function [matches, ratios] = match_descriptors(points1, descriptors1, points2, descriptors2)
% -- [matches, ratios] = match_descriptors(points1, descriptors1, points2, descriptors2)
%      matches keypoints of two images based on descriptors.
%      - points{1,2}: 4xn matrices [y;x;L;theta], positions of the keypoints.
%      - descriptors{1,2}: 64xn matrices, descriptors of the keypoints.
%      - matches: 2xm matrix, each column contains the indices of two matching points.
%      - ratios: 1xm matrix, score of each match (ratio of loss of the best match and loss of the second best match).

error_ratio_threshold = 0.7;
square_error_ratio_threshold = error_ratio_threshold^2;

matches = [];
ratios = [];
for i = 1:size(descriptors1,2)
	% square distance of i-th descriptor in d1 from each descriptor in d2.
	norm_diffs = sumsq(descriptors2 - descriptors1(:,i), 1);

	% minimum and second-minimum error.
	[min_error,min_index] = min(norm_diffs);
	norm_diffs(min_index) = NaN;
	second_min_error      = min(norm_diffs);
	square_error_ratio = min_error ./ second_min_error;

	if square_error_ratio < square_error_ratio_threshold
		ratios(end+1) = sqrt(square_error_ratio);
		matches(:,end+1) = [i, min_index];
	end
end
