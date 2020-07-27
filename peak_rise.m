function A_new = peak_rise(A, f, Fs)
%PEAK_RISE Rises formant of system A with frequency f and sampling rate Fs
%   A_new = peak_rise(A, f, Fs)
    theta = 2 * pi * f / Fs;
    p = roots(A);
    p_new = arrayfun(@(p) rotate_pole(p, theta), p);
    A_new = poly(p_new) * A(1);
end

function p_new = rotate_pole(p, theta)
%ROTATE_POLE Tests and rotate pole p with angle theta
%   p_new = rotate_pole(p, theta)
    if imag(p) == 0
        p_new = p;
    else
        p_new = abs(p) * exp(mod(abs(angle(p)) + theta, pi) * sign(imag(p)) * 1j);
    end
end

