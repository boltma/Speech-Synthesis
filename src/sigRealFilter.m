clear; clc; close all;

Fs = 8000;
dur = 1;
t_seg = 10;

NS = round(Fs * dur);
e = zeros(NS, 1);
p = 1;
while p <= NS
    e(p) = 1;
    m = ceil(p / (t_seg * Fs / 1000));
    PT = 80 + 5 * mod(m, 50);
    p = p + PT;
end

figure;
subplot(2, 1, 1);
plot((0:7999)/8000, e);
subplot(2, 1, 2);
fft_singleband_plot(e, 8000);

A = [1, -1.3789, 0.9506];
s = filter(1, A, e);
sound([e; s], 8000);

figure;
subplot(2, 1, 1);
plot((0:7999)/8000, s);
subplot(2, 1, 2);
fft_singleband_plot(s, 8000);
