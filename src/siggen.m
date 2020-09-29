function sig = siggen(f, Fs, dur)
%SIGGEN Generates impulse train with frequency f, sampling rate Fs and
%duration dur
%   sig = siggen(f, Fs, dur)
    NS = round(Fs * dur);
    N = Fs / f;
    sig = zeros(NS, 1);
    k = floor((NS - 1) / N);
    p = round((0:k)*N) + 1;
    sig(p) = 1;
end

