function [P, D] = run_octave(J, pad, o)
%	[P, D] = run_octave(J, pad, o);
%
%	  - J = immagine integrale di dimensioni [M+2*pad, N+2*pad]
%	  - o = ottava in cui eseguire la ricerca dei descrittori
%
%	  - P = posizioni dei punti di interesse come matrice 4xn: [y; x; L; theta]
%	        - y, x  = posizione assoluta nell'immagine
%	        - L     = scala del punto di interesse
%	        - theta = angolo del gradiente stimato nell'intorno del punto di interesse
%	  - D = descrittori SURF per i punti di interesse come matrice 64xn.
%	        ogni colonna è un vettore normalizzato.

M = size(J, 1) - 2*pad;
N = size(J, 2) - 2*pad;
printf("inizio ottava %g\n", o);
ns = 4;
p = 2.^(o-1);
scales = 2.^o .* [1, 2, 3, 4] + 1;

DoH = zeros(ceil(M ./ p), ceil(N ./ p), ns, "int64");
Dx  = zeros(ceil(M ./ p), ceil(N ./ p), ns, "int64");
Dy  = zeros(ceil(M ./ p), ceil(N ./ p), ns, "int64");

printf(" calcolo operatori locali di J ...\n");
for s = 1:ns
	L = scales(s);
	[DoH(:,:,s), Dy(:,:,s), Dx(:,:,s)] = integral_image_operators(J, L, p, pad);
end

printf(" ricerca massimi di DoH ...\n");
% calcola il massimo nel 9-intorno di ogni punto.
MaxMap = DoH;
% MaxMap(1:end-1,:,:) = max(MaxMap(1:end-1,:,:), MaxMap(2:end,:,:));
% MaxMap(2:end,  :,:) = max(MaxMap(1:end-1,:,:), MaxMap(2:end,:,:));
% MaxMap(:,1:end-1,:) = max(MaxMap(:,1:end-1,:), MaxMap(:,2:end,:));
% MaxMap(:,2:end,  :) = max(MaxMap(:,1:end-1,:), MaxMap(:,2:end,:));
% MaxMap(:,:,1:end-1) = max(MaxMap(:,:,1:end-1), MaxMap(:,:,2:end));
% MaxMap(:,:,2:end  ) = max(MaxMap(:,:,1:end-1), MaxMap(:,:,2:end));
MaxMap = max(MaxMap(1:end-2,:,:), max(MaxMap(2:end-1,:,:), MaxMap(3:end,:,:)));
MaxMap = max(MaxMap(:,1:end-2,:), max(MaxMap(:,2:end-1,:), MaxMap(:,3:end,:)));
MaxMap = max(MaxMap(:,:,1:end-2), max(MaxMap(:,:,2:end-1), MaxMap(:,:,3:end)));


% seleziono solo punti non troppo vicini al bordo: il calcolo dei descrittori
% richiede molte informazioni dall'intorno; si ritiene rischioso usare per matching
% un descrittore costruito con pixel su cui non si hanno informazioni concrete.
DoH_crop = DoH(2:end-1,2:end-1,2:end-1);
% MaxMap = MaxMap(2:end-1,2:end-1,2:end-1);

% soglia di selezione.
% scelgo pixel in cui il valore di DoH corrisponde al massimo nel suo 9-intorno.
tH = 1e3;
raw_maxima = find((MaxMap == DoH_crop) & DoH_crop >= tH)';

printf(" calcolo derivate di DoH ...\n");
% derivate parziali di DoH, necessarie per il raffinamento dei punti.
% H: matrice 3x3xnumel(raw_maxima). la componente (:,:,i) è l'hessiana nel punto   (p*(1+y0(i)),p*(1+x0(i)),2p*(1+s0(i))).
% d: matrice 3x1xnumel(raw_maxima). la componente (:,:,i) è il gradiente nel punto (p*(1+y0(i)),p*(1+x0(i)),2p*(1+s0(i))).
[y0, x0, s0] = ind2sub(size(MaxMap), raw_maxima);
[H, d] = hessian_and_gradient(DoH, y0+1, x0+1, s0+1, p, p, 2*p);

printf(" raffino e filtro %d punti ...\n", numel(raw_maxima));
% raffina la posizione dei massimi imponendo il gradiente di DoH uguale a 0.
xi = zeros([3, numel(raw_maxima)]);
for i = 1:numel(raw_maxima)
	xi(:,i) = -H(:,:,i)\d(:,:,i);
end

% le coordinate y0, x0, s0 sono nello spazio degli indici, yxL nello spazio dell'immagine originale
yxL = [p*y0; p*x0; scales(1+s0)] + xi;

% rimuovo tutti i punti per cui la correzione necessaria è troppo grande.
non_degen = max(abs(xi ./ [1; 1; 2]), [], 1) < p;

% rimuovo tutti i punti per cui la finestra del calcolo del descrittore
% esce dall'immagine in modo significativo
in_bounds = min(min(yxL([1,2],:), [M;N]-yxL([1,2],:)), [], 1) > 2*yxL(3,:);

raw_maxima = raw_maxima(non_degen & in_bounds);
yxL = yxL(:,non_degen & in_bounds);

printf(" calcolo angoli per %d punti ...\n", numel(raw_maxima));

% offset degli indici per i punti della griglia 13x13 per il calcolo degli angoli
[Bv, Bu, Bs] = ndgrid([-6:6],[-6:6],[0]);
Bv = Bv(:);
Bu = Bu(:);
Bs = Bs(:);

% restringo agli indici in un intorno circolare
select = Bu.^2 + Bv.^2 <= 36;
Bv = Bv(select);
Bu = Bu(select);
Bs = Bs(select);

% moltiplicatore gaussiano per smorzare i valori del gradiente
GB = exp(-((Bv./2).^2+(Bu./2).^2)./2)(:);

sigma = round(0.4 .* yxL(3,:));
% creo indici nei tensori Dx e Dy: yxL sono indici nello spazio dell'immagine originale.
% Bv, Bu, Bs sono vettori colonna, quindi i risultati sono matrici in cui ogni
% colonna contiene gli indici dei punti che formano un intorno di un'entrata di yxL.
y_indices = 1 + (yxL(1,:) + sigma .* Bv) ./ p;
x_indices = 1 + (yxL(2,:) + sigma .* Bu) ./ p;
s_indices = 1 + (yxL(3,:) - 2.^o - 1) ./ (2*p) + Bs;

% campioni delle derivate in un intorno dei punti di interesse, pesate con un moltiplicatore gaussiano.
% Qui si è deciso di fare zero-padding, in quanto questi valori servono solo a "votare"
% sull'angolo da scegliere e un gradiente nullo equivale a un voto scartato.
phi_y = interpn(Dy, y_indices, x_indices, s_indices, "linear", 0) .* GB;
phi_x = interpn(Dx, y_indices, x_indices, s_indices, "linear", 0) .* GB;

% gli angoli dei gradienti
angles = atan2(phi_y, phi_x);

% 40 angoli equispaziati con cui confrontarli
test_thetas = [0:39].*pi./20;

% confronto tutti gli angoli theta con tutti gli angoli dei campioni del gradiente.
% l'angolo alpha è nel range [theta - pi/6, theta + pi/6] mod 2pi
%     iff alpha - theta + pi/6 è nel range [0, pi/3] mod 2pi
%     iff mod(alpha - theta + pi/6, 2pi) <= pi/3.
angle_diffs = reshape(angles(:)' - test_thetas(:) + pi./6, [40, size(angles)]);
% select è un tensore 40x113xn con valori booleani:
%   - la prima dimensione enumera gli angoli di prova
%   - la seconda dimensione enumera i campioni del gradiente nell'intorno
%   - la terza dimensione enumera l'indice del punto in cui è centrato l'intorno
select = mod(angle_diffs, 2.*pi) <= pi./3;

% somma dei gradienti il cui angolo è entro pi/6 da ciascun angolo di test.
Phi_y_theta = reshape(sum(shiftdim(phi_y, -1) .* select, 2), 40, []);
Phi_x_theta = reshape(sum(shiftdim(phi_x, -1) .* select, 2), 40, []);

% prendo l'indice dell'angolo che massimizza la norma della somma
[~, best_theta_index] = max(Phi_y_theta.^2+Phi_x_theta.^2, [], 1);

% calcolo l'angolo del vettore che rappresenta la somma dei gradienti selezionati dal
% valore ottimale di theta.
theta = atan2(
	interpn(Phi_y_theta, best_theta_index, [1:size(Phi_y_theta, 2)], "nearest"),
	interpn(Phi_x_theta, best_theta_index, [1:size(Phi_x_theta, 2)], "nearest")
);

printf(" calcolo i descrittori per %d punti ...\n", size(yxL, 2));
% indici per gli intorni 20x20 dei punti di interesse, la logica è identica agli intorni
% per il calcolo degli angoli.
[Rv Ru Rs] = ndgrid([-9.5:9.5],[-9.5:9.5],[0]);
Rv = Rv(:);
Ru = Ru(:);
Rs = Rs(:);
GR = exp(-((Rv./3.3).^2 + (Ru./3.3).^2)./2)(:);

sth = sin(theta);
cth = cos(theta);

% cambio coordinate da v,u,s a y,x,L ruotando con l'angolo theta trovato in precedenza.
% riscalo con il fattore p per ottenere indici nei tensori Dx, Dy, che hanno un campionamento
% sparso.
y_indices = 1 + (sigma .* ( cth .* Rv + sth .* Ru) + yxL(1,:)) ./ p;
x_indices = 1 + (sigma .* (-sth .* Rv + cth .* Ru) + yxL(2,:)) ./ p;
s_indices = 1 + (yxL(3,:) - 2.^o - 1) ./ (2*p) + Rs;

% valori campionati, come matrici 400xn.
Dy_samples = interpn(Dy, y_indices, x_indices, s_indices, "linear", 0) .* GR;
Dx_samples = interpn(Dx, y_indices, x_indices, s_indices, "linear", 0) .* GR;

% dato l'ordine degli indici, posso reinterpretarli come tensori 5x4x5x4xn,
% dove la seconda e la quarta dimensione indicano il quadrante di appartenenza,
% la prima e terza indicano la posizione nel quadrante, come segue:
% Layout di una "fetta" del tensore dv relativa al punto di interesse n-esimo:
% l'ordine degli elementi in questo tensore emerge dall'ordine naturale degli
% elementi nelle matrici di indici Ru,Rv,Rs fornite da ndgrid.
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
%    |     e1 v . . @=du(e1,e2,e3,e4,n)  |
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

% dv,du sono i gradienti riportati nel sistema di coordinate orientato all'angolo
% scelto per il punto di interesse, per garantire invarianza per rotazioni del descrittore.
dv = reshape(cth .* Dy_samples - sth .* Dx_samples, 5, 4, 5, 4, []);
du = reshape(sth .* Dy_samples + cth .* Dx_samples, 5, 4, 5, 4, []);

% calcolo i descrittori come somme o somme dei moduli lungo la prima e terza dimensione.
mu = vertcat(
	reshape(sum(sum(dv,3),1),      16, []),
	reshape(sum(sum(du,3),1),      16, []),
	reshape(sum(sum(abs(dv),3),1), 16, []),
	reshape(sum(sum(abs(du),3),1), 16, [])
);

P = [yxL; theta];
% normalizzo le colonne dei descrittori.
D = mu ./ sqrt(sumsq(mu, 1));
