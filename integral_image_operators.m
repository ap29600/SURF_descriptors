function [DoH, Dy, Dx] = integral_image_operators(J, L, p, pad)

%   [DoH, Dx, Dy] = integral_image_operators(J, L, p, pad)
%     - J: (M + 2*pad)x(N + 2*pad) integer matrix, representing the integral image
%     - L: scale of the local operator
%     - p: sampling stride of the operator
%
%     - DoH: ceil(M/p) x ceil(N/p) matrix. result of the "determinant of hessian" operator on the image.
%     - Dx,Dy: ceil(M/p) x ceil(N/p) matrices. results of the partial derivative operators on the image.

M = size(J, 1) - 2*pad;
N = size(J, 2) - 2*pad;
l   = round(0.8 * L);
L2  = floor(L/2);
LL2 = L+L2;
w   = sqrt((2*L-1)./(2*L));

% integral over the original (non-integral) image on a rectangle bounded by opposite vertices
% {(x+a, y+c), (x+b, y+c)} as (x,y) varies in the region [pad+1:pad+M]x[pad+1:pad:N].
sl = @(y, x) J(pad+y+1:p:pad+M+y, pad+x+1:p:pad+N+x);
rect = @(a,b,c,d) sl(a-1,c-1)+sl(b,d)-sl(a-1,d)-sl(b,c-1);

% partial derivative operators.
Dx  = rect(-l,l, 1,l) - rect(-l, l,-l,-1);
Dy  = rect( 1,l,-l,l) - rect(-l,-1,-l, l);
Dxy = rect(-L,-1,-L,-1) + rect(1,L,1,L) - rect(-L,-1,1,L) - rect(1,L,-L,-1);
Dyy = rect(-LL2,LL2,-L,L) - 3*rect(-L2,L2,-L,L);
Dxx = rect(-L,L,-LL2,LL2) - 3*rect(-L,L,-L2,L2);

% determinant of hessian.
% the w parameter is necessary to preserve the split of energy associated with each component
% of the hessian in the continuous operator that DoH is a discretization of.
DoH  = (Dxx .* Dyy - (w * Dxy) .^ 2) ./ L .^ 4;
