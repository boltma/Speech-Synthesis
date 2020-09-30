clear; clc; close all;

a = [1, -1.3789, 0.9506];

% rise peak 150Hz
a_new = peak_rise(a, 150, 8000);
fprintf('a1 = %f, a2 = %f\n', -a_new(2), -a_new(3));

% Draw zero and pole points
figure;
zplane(1, a_new);

% Draw amplitude and phase with regard to frequency
figure;
freqz(1, a_new);

% Draw impulse response
figure;
impz(1, a_new, 101);
