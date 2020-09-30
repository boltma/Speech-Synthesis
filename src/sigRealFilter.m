clear; clc; close all;

Fs = 8000;              % sampling rate
dur = 1;                % duration
t_seg = 10;             % time segment (ms)

% generate each pulse with while loop
NS = round(Fs * dur);
e = zeros(NS, 1);
p = 1;
while p <= NS
    e(p) = 1;
    m = ceil(p / (t_seg * Fs / 1000));
    PT = 80 + 5 * mod(m, 50);
    p = p + PT;
end

% plot signal
figure;
subplot(2, 1, 1);
plot((0:Fs-1)/Fs, e);
subplot(2, 1, 2);
fft_singleside_plot(e, Fs);

% go through filter
a = [1, -1.3789, 0.9506];
s = filter(1, a, e);

% sound two signals
sound([e; s/max(abs(s))], Fs);

% plot response
figure;
subplot(2, 1, 1);
plot((0:Fs-1)/Fs, s);
subplot(2, 1, 2);
fft_singleside_plot(s, Fs);

% plot part of signal
figure;
plot((0:799)/Fs, s(1:800));
xlim([0, 799/Fs]);
