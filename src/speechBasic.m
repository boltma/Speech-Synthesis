clear; clc; close all;

a = [1, -1.3789, 0.9506];

% Compute peak frequency
p = roots(a);
fprintf("Frequency: %f = %f * pi\n", angle(p(1)), angle(p(1)) / pi);

% Draw zero and pole points
figure;
zplane(1, a);

% Draw amplitude and phase with regard to frequency
figure;
freqz(1, a);

% Draw impulse response
figure;
% Use filter to generate impulse response
subplot(2, 1, 1);
n = (0:100)';
x = (n == 0);
y = filter(1, a, x);
stem(n, y, 'filled');
title('Impulse Response: filter');
xlabel('n (samples)');
ylabel('Amplitude');
% Use impz
subplot(2, 1, 2);
impz(1, a, 101);
title('Impulse Response: impz');
