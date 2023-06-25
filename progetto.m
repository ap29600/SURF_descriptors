function progetto(I1, I2)

M = size(I1,1);
N = size(I1,2);
L = size(I2,1);
K = size(I2,2);

disp = zeros([max(M,L),N+K]);
disp(1:M,1:N) = I1;
disp(1:L,N+1:N+K) = I2;

[p1,d1] = make_descriptors(I1);
[p2,d2] = make_descriptors(I2);

% ==== matching dei descrittori tra le due immagini
[matches,ratios] = match_descriptors(p1,d1,p2,d2);
figure();
image(disp);
colormap(gray(256));

yx1 =         p1([1,2],matches(1,:));
yx2 = [0;N] + p2([1,2],matches(2,:));

l = line([yx1(2,:);yx2(2,:)],[yx1(1,:);yx2(1,:)], "Marker", "o");

% ==== visualizzazione punti di interesse
figure();
subplot(1,2,1);
image(I1);
colormap(gray(256));
plot_points_of_interest(p1);

subplot(1,2,2);
hist([p1,p2](3,:),100);

% ==== matching della migliore trasformazione affine
X1 = [p1([1,2], matches(1,:)); ones([1, size(matches,2)])]';
Y1 = [p2([1,2], matches(2,:)); ones([1, size(matches,2)])]';

% trasformazione affine ai minimi quadrati che sposti i punti X in Y.
A1t = X1\Y1;

% per rimuovere gli outliers seleziono i punti con residuo minore e ripeto su quelli
affine_matching_outlier_threshold = 0.5;
[~, ind] = sort(sumsq(X1 * A1t - Y1, 2)');
X2 = X1(ind(1:floor(affine_matching_outlier_threshold * size(X1,1))),:);
Y2 = Y1(ind(1:floor(affine_matching_outlier_threshold * size(Y1,1))),:);

A2 = (X2\Y2)';

printf("miglior trasformazione affine:\n");
A2

% === plot della correlazione tra residui e score dei match
residuals = sqrt(sumsq(X1 * A2' - Y1, 2)');
figure();
subplot(3,1,1);
scatter(ratios, residuals);

subplot(3,1,2);
hist(ratios, 100);

subplot(3,1,3);
hist(residuals, 100);
