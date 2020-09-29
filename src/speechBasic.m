clear; clc; close all;

figure;
A = [1, -1.3789, 0.9506];
zplane(1, A);

figure;
freqz(1, A);

figure;
subplot(2, 1, 1);
n = (0:100)';
x = (n == 0);
y = filter(1, A, x);
stem(n, y, 'filled');
subplot(2, 1, 2);
impz(1, A, 101);
