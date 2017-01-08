% compare DR, ChPo, HPE, CoPe, GFB on single signal processing task
%
%	 min_{x} 1/2 ||Y - PhiWx||^2 + ||x||_{1,2}^{B}
% 
% Y is the observation, Phi is a blurring operator or a mask operator, W is a wavelet tight frame (redundant, its adjoint is its inverse to the left), x are the wavelet coefficients of the recovered image, ||.||_{1,2}^{B} is the l1/l2 norm by blocks (here, B is the union of all the n = S^2 overlapping grids over the whole image)
%
% Hugo Raguet 2011
				%--------------%
				%  Parameters  %
				%--------------%

rand( 'seed', 0 ), randn( 'seed', 0 )
verbose = true; % diaplay of iterations
doReport = true; % compute energy along iterations - set to false to compare execution time
imFile = 'images/LaBouteVsLena256.png'; % absolute path to the image file
N = 256; % a power of 2, image is N-by-N

task = 'deblurring';
sig_K = 2; % width of convolution kernel, in pixels
% task = 'inpainting';
% rho_M = .7; % ratio of randomly degraded pixels

wvltFltr = compute_wavelet_filter( 'Daubechies', 4 );
J0 = 4; % number of levels of wavelet decomposition
sig_w = .025; % noise level
mu = .0013; % l1/l2-norm penalization 
S = 2; % size of blocks

nIter = 100; % number of iterations per experiments
nIterInf = 1000; % number of iterations to reach minimum energy

			%-----------------------------%
			%  Create Operators and Data  %
			%-----------------------------%

% create data generating operator
if strcmp( task, 'deblurring' )
	[X,Y] = meshgrid( [0:N/2-1 -N/2:-1] );
	K = exp( - (X.^2+Y.^2) / (2*sig_K^2) ); % gaussian kernel
	clear X Y
	K = K./sum(abs(K(:))); % ||K||_1 = 1
	FK = real( fft2(K) );
	FK2 = FK.^2;
	Phi = @(I) real( ifft2( fft2(I).*FK ) );
	Phis = Phi; % convolution with centered Kernel is self-adjoint
	normPhi2 = 1; % ||K*f||_2 <= ||K||_1 ||f||_2
	invIgaPhiPhis = @(I,ga) real( ifft2( fft2(I)./(ga*FK2+1) ) ); % (Id + ga*Phi Phis)^-1; diagonal operator in fourier domain
elseif strcmp( task, 'inpainting' )
	mask = rand(N) > rho_M; % mask
	Phi = @(I) mask.*I;
	Phis = Phi; % orthogonal projector is self-adjoint
	normPhi2 = 1;
	invIgaPhiPhis = @(I,ga) I./(ga*mask+1); % (Id + ga*Phi Phis)^-1; diagonal operator in pixels domain
end

% create observation
Y0 = rescale( double( imread( imFile ) ) );
Y = Phi(Y0)+sig_w*randn(N);

% create representation operator
Jmin = log2(N) - J0;
W = @(x)fwt2Dti( x, Jmin, -2, wvltFltr );
Ws = @(x)fwt2Dti( x, Jmin, +2, wvltFltr );
normW2 = 1;
J = 3*J0+1;
xSize = [N N J];
mu_ = [2^(-J0); kron( 2.^(-J0:-1)', [1; 1; 1] )]*mu;
mu_ = reshape( mu_, 1, 1, [] );

% create block structure, penalization and associated proximal operators
n = S^2;
proxGi = cell( 1, n );
proxGis = cell( 1, n );
proxIdx = 0;
for shifti = 0:S-1, for shiftj = 0:S-1
	proxIdx = proxIdx + 1;
	proxGi{proxIdx} = @(x,ga)blockGridThresholding( x, ga*mu_, 2, [S S], [shifti shiftj] );
	proxGis{proxIdx} = compute_dual_prox( proxGi{proxIdx} );
end, end
G = @(x)normAllGridBlock( x, 2, [S S], mu_ );

% create functions and gradients, set other optimization variables
F = @(x) 1/2*sum( sum( abs( Y - Phi(W(x)) ).^2 ) );
gradF = @(x) Ws(Phis( Phi(W(x)) - Y ));
be_F = 1/(normPhi2*normW2); % F is 1/be-Lipschitz continuous
be_Fn = 1/(normPhi2*normW2+sqrt(n));
WsPhisY = Ws(Phis(Y));
proxF = @(x,ga)x+ga*WsPhisY - ga*Ws(Phis( invIgaPhiPhis( Phi(W(x+ga*WsPhisY)), ga ) )); % suppose that WWs = Id

if doReport, report = {@(x)F(x)+G(x)}; else, report = {}; end

				%--------------%
				%  Processing  %
				%--------------%

%{
%}
%%%  GFB  %%%
ga_gfb = 1.8*be_F;
la_gfb = 1;
tic
[xGFB, eGFB] = GeneralizedForwardBackward( gradF, proxGi, zeros( [xSize n] ), nIter, ga_gfb, la_gfb, verbose, report{:} );
tGFB = toc;

%%%  CoPe  %%%
ga_cope = .9*be_Fn;
tic
[xCoPe, eCoPe] = CombettesPesquet( gradF, proxGis, zeros( xSize ), zeros( [xSize n] ), nIter, ga_cope, verbose, report{:} );
tCoPe = toc;

%%%  HPE  %%%
si = .9;
ga = 2*be_F*si^2/(1+sqrt(1+4*si^2*be_F^2));
tic
[xHPE, eHPE] = BD_HPE( gradF, proxGi, zeros( [xSize n] ), zeros( [xSize n] ), nIter, ga, verbose, report{:} );
tHPE = toc;

%%%  DR  %%%
ga_pr = 1/(n+1);
la_pr = 1;
tic
[xDR, eDR] = GeneralizedForwardBackward( [], [{proxF} proxGi], zeros( [xSize n+1] ), nIter, ga_pr, la_pr, verbose, report{:} );
tDR = toc;

%%%  ChPo  %%%
proxGs = @(x,ga)proxGi_n(x,ga,proxGis,4);
sig = 1;
tau = .9/(sig*(normW2*normPhi2+n));
the = 1;
tic
[xChPo, eChPo] = ChambollePock( ...
			@(x)cat( 4, padarray((Phi(W(x))),[0 0 J-1],'post'), repmat(x,[1 1 1 S^2]) ), ...
			@(x)Ws(Phis(x(:,:,1,1)))+sum(x(:,:,:,2:end),4), ...
			@(x,ga)cat( 4, padarray((x(:,:,1,1)-ga*Y)/(1+ga),[0 0 J-1],'post'), proxGs(x(:,:,:,2:end),ga) ), [], ...
			zeros(xSize), nIter, sig, tau, the, verbose, report{:} );
tChPo = toc;
%{
%}

if doReport
e = [eChPo eDR eHPE eCoPe eGFB];
%%%  PsiMin  %%%
PsiMinFile = sprintf( 'Results/PsiMin_%s.mat', task );
if nIterInf > nIter
	if exist( PsiMinFile ) % be sure that it corresponds to the right parameters
		load( PsiMinFile, 'PsiMin' );
	else
		xTMP = GeneralizedForwardBackward( gradF, proxGi, zeros( [xSize n] ), nIterInf, ga_gfb, la_gfb, verbose );
		PsiMin = F(xTMP)+G(xTMP); clear xTMP
		save( PsiMinFile, 'PsiMin' )
	end
	plotIt = nIter;
else
	PsiMin = min( e(:) );
	plotIt = .8*nIter;
end
end %endif doReport

			 %----------------------------%
			 %  Display and Save Results  %
			 %----------------------------%

recov = W(xGFB);
imgprint( recov, 'Results/%s_l1l%d_%d_%d_recov.png', task, 2, S, nIter )
imgprint( Y, 'Results/%s_l1l%d_%d_%d_obs.png', task, 2, S, nIter )

clf
subplot( 2, 2, 1 )
if doReport
	plot( log10( e(1:plotIt,:) - PsiMin ) )
	sxlabel( 'iteration #' )
	sylabel( 'log_{10}(\\Psi-\\Psi_{min})' )
	legend( {'ChPo' 'DR' 'HPE' 'CoPe' 'GFB'} );
	axis tight
else
	plot( [tChPo tDR tHPE tCoPe tGFB], '*' )
	stitle( 't_{ChPo}: %ds; t_{DR}: %ds; t_{HPE}: %ds; t_{CoPe}: %ds; t_{GFB}: %ds', round(tChPo), round(tDR), round(tHPE), round(tCoPe), round(tGFB) )
end
subplot( 2, 2, 2 ), imshow(Y0), stitle( 'N: %d', N )
subplot( 2, 2, 3 ), imshow(Y),
if strcmp( task, 'deblurring' )
	stitle( '\\sigma_{w}: %.3f; \\sigma_{K}: %d; SNR: %.2fdB', sig_w, sig_K, SNRdB( Y0, Y ) )
else
	stitle( '\\sigma_{w}: %.3f; \\rho_{M}: %.1f; SNR: %.2fdB', sig_w, rho_M, SNRdB( Y0, Y ) )
end
subplot( 2, 2, 4 ), imshow(recov), stitle( '\\mu_{l1/l%d}: %.1e; S: %d;\n it. #%d; SNR: %.2fdB', 2, mu, S, nIter, SNRdB( Y0, recov ) )
drawnow

if doReport
	epsprint( 'Results/%s_l1l%d_%d_%d.eps', task, 2, S, nIter );
else
	epsprint( 'Results/%s_l1l%d_%d_%d_time.eps', task, 2, S, nIter );
end
