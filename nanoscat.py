import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
#import numpy.fft as fft_module
import scipy.fftpack as fft_module
from scipy import signal
def interpft(x,n):
    """
    Fourier interpolation of signal x to n points.
    """
    return signal.resample(x,n)
#    return np.fft.irfft(np.fft.rfft(x), n)
    
def nanoscat_load(filename):
    """
    Loading input audio sample
    """
    (sr, y) = scipy.io.wavfile.read(filename);
    leny = len(y);
    N = int(np.power(2, np.floor(np.log2(leny)) + 1));
    sig = np.zeros(shape=(N));
    sig[0:leny] = y;
    return (sig, N, leny, sr)


def nanoscat_make_filters(N, J, shape='gaussian'):
    """ 
    calculate bandpass filters \psi for size N and J different center frequencies and low pass filter \phi 
    and resolutions res < log2(N)
    """
    nResolutions = int(1 + np.floor(np.log2(N)))
    psi = {}  
    phi = {}  

    for res in range(0, nResolutions):
        N0 = round(N / 2 ** res)
        if N0 <= N / (2 ** J):
            break

        # each entry is again a dict
        psi[res + 1] = {}

        for j in range(0, J):
#            print('Why add this 0.8 in the sz size variable ? Further on this is not dyadic ratio')
            sz = int(np.floor(0.8 * N0 / 2 ** j))
            if sz <= N / 2 ** J:
                break

            if shape == 'hanning':
                v = np.zeros(shape=(N0))
                v = (1 - np.cos(2 * np.pi * np.arange(0, sz - 2) / sz))[0:N0]
            else:  # 'gaussian'
                xi = 0.4 * 2 ** (-j)
                v = 2 * np.exp(- np.square(np.arange(0, N0, dtype=float) / N - xi) * 10 * np.log(2) / xi ** 2).transpose()


            psi[res + 1][res + j + 1] = v

            if (res + j == J - 1):
                if shape == 'hanning':
                    print('FIXME : length issue here')
                    f = np.zeros(shape=(N0))
                    half = np.floor(sz / 2)
                    f[-1 - half + 1:-1] = v[0:half] * .5
                    f[0:half] = v[half + 1:half + half] * .5
                    phi[res] = f
                else:  # 'gaussian' This is for Q = 1
                    bw = 0.4 * 2 ** (-1 + J)
                    phi[res + 1] = np.exp(-np.square(np.arange(0, N0,dtype=float)) * 10 * np.log(2) / bw**2 ).transpose()

    # Calculate little-wood Paley function
    lp = np.zeros(shape=psi[1][1].shape)
    for i in range(1, len(psi[1]) + 1):
        lp = lp + 0.5 * np.square(np.abs(psi[1][i]))
    lp = lp + np.square(np.abs(phi[1]))
    temp = lp[0]
    lp = (lp + lp[::-1]) * .5
    lp[0] = temp
    return (psi, phi, lp)


def nanoscat_display_filters(psi, phi, lp):
    """
    Function displays filters psi[res] for all lambdas and phi[res] for all resolutions
    psi : wavelet filters
    phi : low pass filters at different resolutions
    lp : Littlewood Paley response (for now not plotted)
    """
    res = 1
    plt.figure()
    for j in psi[res]:
        plt.plot(psi[res][j], 'r')
    plt.plot(lp,'k')
    for res in phi:
        plt.plot(phi[res], 'b')
    # plt.legend('\psi_\lambda', '\phi','LP')
    plt.title('PSI/PHI  at higher resolution')
    plt.show()
    return


def nanoscat_compute(sig, psi, phi, M):
    """
    Function computes scattering coefficientrs for order M and Q = 1 (dyadic case)
    sig : input signal
    psi : wavelet filters dictionary
    phi : low pass filters dictionary
    M : scattering order (>= 1)
    """
    
    U = {}
    U[1] = {}
    # initialize the signal at U_0 which will be recursively updated with x * \psi_\lambda
    U[1][1] = sig
    S = {}
    # maximal length
    log2N = np.log2(len(psi[1][1]))
    # number of lambdas
    nResolutions = 1 + int(np.log2(len(sig)))
    
    for m in range(1, M + 2):
        lambda_idx = 1
        #create new dictionary : m+1 th order U signals and mth order S signals
        S[m] = {}
        U[m + 1] = {}
        
        for s in range(1, len(U[m]) + 1):
            
            sigf = fft_module.fft(U[m][s])
            #res always calculates to 1 here
            #scatnet/scatnetligt does : ds = round(log2(2*pi/phi_bw)) - j0 - options.oversampling;
            res = int((log2N - (np.log2(len(sigf)))) + 1)
            
            if m <= M:
                
                for j in range(s, len(psi[res]) + 1):
                    print(res, len(psi[res]))
                    # subsample rate is different between the resolution and the bandpass critical frequency support j
                    ds = 2 ** (j - s)
                    c = np.abs(fft_module.ifft(np.multiply(sigf, psi[res][j])))
                    U[m + 1][lambda_idx] = c[0::ds]
                    lambda_idx = lambda_idx + 1
                    
            #why is subsampling fixed to this value here ? and not low_pass_bw/sampling_fs
            ds = (nResolutions - res)**2
            c = np.abs(fft_module.ifft(np.multiply(sigf, phi[res])))
#            print('Order = ' + repr(m) + ' ,res = 2' + repr(res) + ', siglen = ' + repr(sigf.shape) + ', ds_phi = ' + repr(ds)) #', psi_j_len = ' + repr(len(psi[1][1]))            
            if ds > 1:
                c = c[0::ds]
            S[m][s] = c
    return (S, U)


def nanoscat_format(S, M):

    t = {}
    maxlen = len(S[1][1])
    coeff = np.zeros(shape=(1, maxlen))
    for m in range(1, M+2):
        maxlen = len(S[m][1])
        nlambdas = len(S[m])

        c = np.zeros(shape=(nlambdas, maxlen))

        for j in S[m]:
            itp = interpft(S[m][j], maxlen)
            c[j-1, :] = itp #S[m][j] 
        t[m] = c
    
    for m in t:
        coeff = np.vstack((coeff, t[m]))
    
    return coeff[1::,:]


## nanoscat demo starts here
## Q = 1 (dyadic wavelets)
M = 2  # orders
J = 12  # maximal scale
plot_flag = 1

## load and zero pad audio
(sig, N, lensig, sr) = nanoscat_load('samples\drum1_90.wav')
sig = sig / np.linalg.norm(sig)  # normalization

if(plot_flag):
    plt.figure()
    plt.plot(sig)
    plt.title('Input signal')
    plt.xlabel('Time(s)')
    plt.ylabel('Normalized Amplitude')
    plt.show()

#validate that J is bounded within Log2(N) 
assert (J < np.log2(N))

#compute filters
(psi, phi, lp) = nanoscat_make_filters(N, J, 'gaussian')

if(plot_flag):
    nanoscat_display_filters(psi, phi, lp)

#compute scattering
(S, U) = nanoscat_compute(sig, psi, phi, M)

#format and plot S coefficients
scat = nanoscat_format(S, M)  # creates a matrix with all coefficients

if(plot_flag):
    plt.figure()
    plt.imshow(scat, cmap = 'jet')
    plt.title('Scattering coefficients (all orders)')
    plt.show()