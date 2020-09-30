function a_new = peak_rise(a, f, Fs)
%PEAK_RISE Rises formant of poles polynomial A with frequency f and sampling rate Fs
%   A_new = peak_rise(A, f, Fs)
    theta = 2 * pi * f / Fs;
    p = roots(a);       % find all poles of polynomial a
    p_new = arrayfun(@(p) p * exp(theta * sign(imag(p)) * 1j), p);  % rotate poles with angle theta depending on sign of imag(p)
    a_new = poly(p_new) * a(1); % times a(1) to get a_new(1) = a(1)
end

