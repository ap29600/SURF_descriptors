function [matches, ratios] = match_descriptors(points1, descriptors1, points2, descriptors2)
% -- [matches, ratios] = match_descriptors(points1, descriptors1, points2, descriptors2)
%      accoppia punti di interesse tra due immagini sulla base dei descrittori
%      - points{1,2}: matrici 4xn [y;x;L;theta], posizioni dei punti di interesse.
%      - descriptors{1,2}: matrici 64xn, descrittori dei punti di interesse.
%      - matches: matrice 2xm, ogni colonna contiene indici di due punti corrispondenti tra le matrici di input.
%      - ratios: matrice 1xm, score di ogni corrispondenza (rapporto tra errore minimo e secondo errore minimo).

error_ratio_threshold = 0.7;
square_error_ratio_threshold = error_ratio_threshold^2;

matches = [];
ratios = [];
for i = 1:size(descriptors1,2)
	% distanza al quadrato del descrittore i-esimo da ciascun altro descrittore
	norm_diffs = sumsq(descriptors2 - descriptors1(:,i), 1);

	% trova l'errore minimo e il secondo errore minimo.
	[min_error,min_index] = min(norm_diffs);
	norm_diffs(min_index) = NaN;
	second_min_error      = min(norm_diffs);
	square_error_ratio = min_error ./ second_min_error;

	if square_error_ratio < square_error_ratio_threshold
		ratios(end+1) = sqrt(square_error_ratio);
		matches(:,end+1) = [i, min_index];
	end
end
