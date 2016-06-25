function [S, U] = nanoscat_compute (sig, psi, phi, M)
U = {};
S = {};
%initialise U(1,1) (U0 in paper) with signal before starting the loop
U{1}{1}= sig;
log2N = log2 (length (psi{1}{1})); % maximal length
J = numel(psi);

for m = 1:M+1
    %this is the index of lambda (or j) for the wavelet filters 
    hindex = 1;
    for s = 1:numel(U{m})
        sigf = fft (U{m}{s});
        %res here always calculates to 1
        res = (log2N - (log2 (length(sigf)))) + 1;
        %Convolution with \psi_\lambda wavelet filters
        if m<=M
            for j = s : numel(psi{res})
                ds = 2^(j-s);
                c = abs(ifft(sigf .* psi{res}{j}));
                U{m+1}{hindex} = c;
                hindex = hindex + 1;
            end
        end
        %Convolution with \phi_J (low pass) at various resolutions res
        ds = (J - res)^2;
        c = abs (ifft(sigf .* phi{res}));
        if ds > 1
            c = c(1:ds:end);
        end
        S{m}{s} = c;
    end
end
end