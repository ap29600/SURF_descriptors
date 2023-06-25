function demo(I1, I2)

% showcase the feature matching algorithm.
% I1, I2 are grayscale images with values 0..255.
% computes the keypoints and descriptors for both images, then tries to match them.
% some statistics and visualizations are showed, under the assumption that the images
% can be overlapped via an affine transformation (e.g. if one is obtained from the other
% by some combination of rototranslation, scaling, and additive noise).

M = size(I1,1);
N = size(I1,2);
L = size(I2,1);
K = size(I2,2);

% make a joint image for visualization purposes.
disp = zeros([max(M,L),N+K]);
disp(1:M,1:N) = I1;
disp(1:L,N+1:N+K) = I2;

% compute keypoints and descriptors
[p1,d1] = make_descriptors(I1);
[p2,d2] = make_descriptors(I2);

% ==== matching keypoint visualization:
% shows the images side by side with lines connecting the matching descriptors.
[matches,ratios] = match_descriptors(p1,d1,p2,d2);
figure();
image(disp);
colormap(gray(256));

yx1 =         p1([1,2],matches(1,:));
yx2 = [0;N] + p2([1,2],matches(2,:));

l = line([yx1(2,:);yx2(2,:)],[yx1(1,:);yx2(1,:)], "Marker", "o");

% ==== keypoint visualization and scale distribution
% shows keypoints of I1 and their scales, as well as a histogram of the scales.
figure();
subplot(1,2,1);
image(I1);
colormap(gray(256));
plot_points_of_interest(p1);

subplot(1,2,2);
hist([p1,p2](3,:),100);

% ==== find the best (least squares) affine transformation satisfying the matching points.
X1 = [p1([1,2], matches(1,:)); ones([1, size(matches,2)])]';
Y1 = [p2([1,2], matches(2,:)); ones([1, size(matches,2)])]';

A1t = X1\Y1;

% the least squares method is vulnerable to outliers, so we select
% the 50% of matches with the lowest residue and repeat on those.
affine_matching_outlier_threshold = 0.5;
[~, ind] = sort(sumsq(X1 * A1t - Y1, 2)');
X2 = X1(ind(1:floor(affine_matching_outlier_threshold * size(X1,1))),:);
Y2 = Y1(ind(1:floor(affine_matching_outlier_threshold * size(Y1,1))),:);

printf("best affine transformation:\n");
A2 = (X2\Y2)'

% === plot the distribution of residues and matching ratios.
% We expect these to be positively correlated (lower ratio is better).
residuals = sqrt(sumsq(X1 * A2' - Y1, 2)');
figure();
subplot(3,1,1);
scatter(ratios, residuals);

subplot(3,1,2);
hist(ratios, 100);

subplot(3,1,3);
hist(residuals, 100);
