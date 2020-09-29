function fft_singleband_plot(x, Fs)
%FFT_SINGLEBAND Plots single-banded fft
%   fft_singleband_plot(x, Fs)
    L = length(x);
    f = (0:L/2) * Fs / L;
    y = abs(fft(x) / L);
    y = y(1:L/2+1);
    y(2:end-1) = 2 * y(2:end-1);
    plot(f, y);
    ylim([0 0.02])
end

