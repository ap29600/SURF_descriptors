# SURF in GNU octave/MATLAB

The original SURF paper: `DOI:10.1007/11744023_32`

## Demo:

```MATLAB
im1 = imread(...);
im2 = imnoise(imrotate(im1, 25), "gaussian", 0, 0.001);

demo(im1, im2);
```

## Usage:

```MATLAB
im1 = imread(...);
im2 = imread(...);

[p1,d1] = make_descriptors(im1);
[p2,d2] = make_descriptors(im2);

[matches, ratios] = match_descriptors(p1,d1,p2,d2);
```
