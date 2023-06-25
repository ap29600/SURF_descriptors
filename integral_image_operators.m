function [DoH, Dy, Dx] = integral_image_operators(J, L, p, pad)

%   [DoH, Dx, Dy] = integral_image_operators(J, L, p, pad)
%     - J: matrice (M + 2*pad)x(N + 2*pad) di interi, rappresenta l'immagine integrale.
%     - L: fattore scala dell'operatore locale da applicare.
%     - p: passo di campionamento dell'operatore locale.
%
%     - DoH: matrice ceil(M/p) x ceil(N/p). risultato dell'operatore "determinante dell'hessiana" sull'immagine.
%     - Dx,Dy: matrici ceil(M/p) x ceil(N/p). risultati degli operatori di derivata parziale sull'immagine.

M = size(J, 1) - 2*pad;
N = size(J, 2) - 2*pad;
l   = round(0.8 * L);
L2  = floor(L/2);
LL2 = L+L2;
w   = sqrt((2*L-1)./(2*L));

% integrale dell'immagine originale sul rettangolo delimitato dai vertici opposti:
% {(x+a, y+c), (x+b, y+c)} dove (x,y) varia nella regione [pad+1:pad+M]x[pad+1:pad+N].
sl = @(y, x) J(pad+y+1:p:pad+M+y, pad+x+1:p:pad+N+x);
rect = @(a,b,c,d) sl(a-1,c-1)+sl(b,d)-sl(a-1,d)-sl(b,c-1);

% operatori di derivata parziale.
Dx  = rect(-l,l, 1,l) - rect(-l, l,-l,-1);
Dy  = rect( 1,l,-l,l) - rect(-l,-1,-l, l);
Dxy = rect(-L,-1,-L,-1) + rect(1,L,1,L) - rect(-L,-1,1,L) - rect(1,L,-L,-1);
Dyy = rect(-LL2,LL2,-L,L) - 3*rect(-L2,L2,-L,L);
Dxx = rect(-L,L,-LL2,LL2) - 3*rect(-L,L,-L2,L2);

% determinante dell'hessiana.
DoH  = (Dxx .* Dyy - (w * Dxy) .^ 2) ./ L .^ 4;
