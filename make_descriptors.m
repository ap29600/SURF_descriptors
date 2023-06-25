function [P, D] = make_descriptors(I)
%   [P, D] = make_descriptors(I)
%     - I: immagine di cui calcolare i descrittori
%     - P: matrice 4xn: [y;x;L;theta] ogni colonna rappresenta le coordinate,
%          la scala e l'orientazione di un punto di interesse.
%     - D: matrice 64xn: ogni colonna è un vettore normalizzato che rappresenta
%          il descrittore SURF di un punto di interesse.

octaves = [1, 2, 3, 4];
maxScale = 2.^max(octaves) * 4 + 1;

pad = ceil(3/2*maxScale);
[M, N] = size(I);

% impongo condizioni di riflessione al contorno.
I = [I(:,pad+1:-1:2), I, I(:,end-1:-1:end-pad-1)];
I = [I(pad+1:-1:2,:); I; I(end-1:-1:end-pad-1,:)];

% calcolo l'immagine integrale.
J = zeros(size(I) + [1, 1], "int64");
J(2:end, 2:end) = cumsum(cumsum(uint64(I)), 2);

P = [];
D = [];

% calcolo dei descrittori per le diverse ottave.
for o = octaves
	[new_points, new_descriptors] = run_octave(J, pad, o);
	P = [P,new_points];
	D = [D,new_descriptors];
end
