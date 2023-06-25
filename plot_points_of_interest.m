function plot_points_of_interest(points, col='blue')
% -- plot_points_of_interest(points)
% -- plot_points_of_interest(points, col)
%      traccia quadrati orientati in posizione dei punti di interesse dell'immagine.
%      - points: matrice 4xn [y;x;L;theta]: coordinate dei punti di interesse.
%      - col: colore del grafico

y = points(1,:)';
x = points(2,:)';
L = points(3,:)';
thetas = points(4,:)';
side = L;

cy = [0.0, 0.0, -0.5, -0.5,  0.5,  0.5, 0.0];
cx = [0.0, 0.5,  0.5, -0.5, -0.5,  0.5, 0.5];

oy =  cos(thetas) .* cy + sin(thetas) .* cx;
ox = -sin(thetas) .* cy + cos(thetas) .* cx;

Py = y+oy.*side;
Px = x+ox.*side;

set(gca, 'YDir', 'reverse');
line(Px', Py', 'Color', col);

