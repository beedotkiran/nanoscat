% nanoscat demo, (c) 2015 carmine e. cella
%
% basic pedagogical implementation of scattering transform for 1D signals
% (dyadic wavelets only)
%

savepath = 'C:\Users\ravikiran\Documents\RaviDocs\ThalesMallatProjet\PythonScatteringToolbox\Sx_outputs\';
Sx_prefix = 'nanoscat';

addpath ('lib');

%% params
M = 2; % orders
J = 12; % maximal scale

%% load and zero pad audio
[sig, N, len] = nanoscat_load ('samples/drum1_90.wav');
sig = sig / norm(sig); % normalization

assert (J < log2(N));

%% compute filters
[psi, phi, lp] = nanoscat_make_filters (N, J, 'gaussian');

%% plot filters
nanoscat_display_filters (psi, phi, lp);

%% compute scattering
[S, U] = nanoscat_compute (sig, psi, phi, M);
save([savepath Sx_prefix '_Sx.mat'],'S')
%% format and plot S coefficients
scat = nanoscat_format (S, [1:M+1]); % creates a matrix with all coefficients
size(scat)
figure,
imagesc(scat);
title ('Scattering coefficients (all orders)');

% eof

