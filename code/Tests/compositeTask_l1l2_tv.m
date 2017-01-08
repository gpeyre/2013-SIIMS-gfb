% compare DR, ChPo, HPE, CoPe, GFB on composite signal processing task and composite regularization
%
%	 min_{x} 1/2 ||Y - PhiWx||^2 + ||x||_{1,2}^{B} + ||Wx||_{TV}
% 
% Y is the observation, Phi is the composition of a blurring operator and a mask operator, W is a wavelet tight frame (redundant, its adjoint is its inverse to the left), x are the wavelet coefficients of the recovered image, ||.||_{1,2}^{B} is the l1/l2 norm by blocks (here, B is the union of all the n = S^2 overlapping grids over the whole image), ||.||_{TV} is the TV pseudo-norm implemented with filters
% H = (-1 1) and V = (-1 0) with center pixel at the upper-left corner
%     ( 0 0)         ( 1 0)
%
% Hugo Raguet 2011

				%--------------%
				%  Parameters  %
				%--------------%

rand( 'seed', 0 ), randn( 'seed', 0 )
verbose = true; % diaplay of iterations
doReport = false; % compute energy along iterations
imFile = 'images/LaBouteVsLena256.png'; % absolute path to the image file
N = 256; % a power of 2, image is N-by-N
rho_M = .4; % ratio of randomly degraded pixels
sig_K = 2; % width of convolution kernel, in pixels
wvltFltr = compute_wavelet_filter( 'Daubechies', 4 );
J0 = 4; % number of levels of wavelet decomposition
sig_w = .025; % sig_w level
mu = .0005; % l1/l2-norm penalization 
nu = .005; % TV-pseudo-norm penalization
S = 4; % side length of the blocks

nIter = 100; % number of iterations per experiments
nIterInf = 1000; % number of iterations to reach minimum energy

			%-----------------------------%
			%  Create Operators and Data  %
			%-----------------------------%

% create data generating operator
mask = rand(N) > rho_M; % mask
M = @(I) mask.*I;
Ms = M; % orthogonal projector is self-adjoint
normM2 = 1;
invIgaMsM = @(I,ga) I./(ga*mask+1); % (Id + ga*Ms M)^-1; diagonal operator in pixels domain

[X,Y] = meshgrid( [0:N/2-1 -N/2:-1] );
K = exp( - (X.^2+Y.^2) / (2*sig_K^2) );
clear X Y
K = K./sum(abs(K(:))); % ||K||_1 = 1
FK = real( fft2(K) );
FK2 = FK.^2;
K = @(I) real( ifft2( fft2(I).*FK ) );
Ks = K; % convolution with centered Kernel is self-adjoint
normK2 = 1; % ||K*f||_2 <= ||K||_1 ||f||_2
invIKKs = @(I) real( ifft2( fft2(I)./(FK2+1) ) ); % (Id + K Ks)^-1; diagonal operator in fourier domain

Phi = @(I)M(K(I));
Phis = @(I)Ks(Phi(I));

% create observation
Y0 = rescale( double( imread( 'images/LaBouteVsLena256.png' ) ) );
Y = Phi(Y0)+sig_w*randn( size(Y0) );

% create representation operator
Jmin = log2(N) - J0;
W = @(x)fwt2Dti( x, Jmin, -2, wvltFltr );
Ws = @(x)fwt2Dti( x, Jmin, +2, wvltFltr );
normW2 = 1;
J = 3*J0+1;
xSize = [N N J];
mu_ = [2^(-J0); kron( 2.^(-J0:-1)', [1; 1; 1] )]*mu;
mu_ = reshape( mu_, 1, 1, [] );

% create block structure, l1/l2 penalization, total variation penalization and associated proximal operators
nB = S^2;
nTV = 2;
proxGi = cell( 1, nB );
proxGis = cell( 1, nB );
proxIdx = 0;
for shifti = 0:S-1, for shiftj = 0:S-1
	proxIdx = proxIdx + 1;
	proxGi{proxIdx} = @(x,ga)blockGridThresholding( ...
			x, ga*mu_, 2, [S S], [shifti shiftj] );
	proxGis{proxIdx} = compute_dual_prox( proxGi{proxIdx} );
end, end
% some pre-computation for (Id + Grads Grad)^-1
Kv = padarray( [-1; zeros(2*N-2,1); 1], [0 2*N-1], 'post' );
Kvs = padarray( [-1; 1; zeros(2*N-2,1)], [0 2*N-1], 'post' );
Kh = padarray( [-1, zeros(1,2*N-2), 1], [2*N-1 0], 'post' );
Khs = padarray( [-1, 1, zeros(1,2*N-2)], [2*N-1 0], 'post' );
FKv = fft2( Kv ); FKvs = fft2( Kvs );
FKh = fft2( Kh ); FKhs = fft2( Khs );
Kinv = FKvs.*FKv+FKhs.*FKh+1;
clear FKh FKv Kh Kv FKhs FKvs Khs Kvs
G = @(x)normAllGridBlock( x, 2, [S S], mu_ ) + normByBlock( gradientOp(W(x)), 2, nu );

% create functions and gradients, set other optimization variables
F = @(x) 1/2*sum( sum( abs( Y - Phi(W(x)) ).^2 ) );
gradF = @(x) Ws(Phis( Phi(W(x)) - Y ));
be_F = 1/(normM2*normK2*normW2); % F is 1/be-Lipschitz continuous
be_Fn = 1/(normM2*normK2*normW2+sqrt(nB+8));
KsY = Ks(Y);

				%--------------%
				%  Processing  %
				%--------------%

%{
%}
%%%  GFB  %%%
ga_gfb = 1.8*be_F;
la_gfb = 1;
% direct implementation reduces the number of auxiliary variables
tic
eGFB = [];
x = zeros( xSize );
Nx = numel( x );
zi = zeros( size( x ) );
z = zeros( [xSize nB+1] ); 
y = zeros(N,N,2);
y1 = y;
y2 = y;
if la_gfb~=1, x0 = x; y0 = y; end
for it=1:nIter
	if verbose, progressbar(it,nIter); end
	if doReport, eGFB(it,:) = F(x)+G(x); end
	forward = ga_gfb*gradF(x);
	for i=1:nB
		idx = (i-1)*Nx+1:i*Nx;
		zi(:) = z(idx);	
		z(idx) = zi + la_gfb*( proxGi{i}( 2*x - zi - forward, (nB+2)*ga_gfb ) - x );
	end
	y1 = y1 + la_gfb*( blockThresholding( 2*y - y1, (nB+2)*ga_gfb*nu, 2 ) - y );
	idx = nB*Nx+1:(nB+1)*Nx;
	zi(:) = z(idx);
	xg = projGradFrame(cat(3,2*x - zi - forward,2*y - y2),Kinv,W,Ws); 
	z(idx) = zi + la_gfb*( xg(:,:,1:J) - x );
	y2 = y2 + la_gfb*( xg(:,:,end-1:end) - y );
	if la_gfb~=1;
		x0 = x0 + la_gfb*( x - x0 );
		y0 = y0 + la_gfb*( y - y0 );
		x = (sum( z, 4 ) + x0)/(nB+2);
		y = (y1+y2+nB*y0)/(nB+2);
	else
		x = (sum( z, 4 ) + x)/(nB+2);
		y = (y1+y2+nB*y)/(nB+2);
	end
end
tGFB = toc;
xGFB = x;

%{
%%%  CoPe  %%%
ga_cope = .9*be_Fn;
% direct implementation reduces the number of auxiliary variables
tic
eCoPe = [];
x = zeros( xSize );
Nx = numel( x );
zi = zeros( xSize );
vB = zeros( [xSize nB] ); 
vTV = zeros( [N N 2] );
for it=1:nIter
	if verbose, progressbar(it,nIter); end
	if doReport, eCoPe(it,:) = F(x)+G(x); end
	y = x - ga_cope*( gradF( x ) + sum( vB, 4 ) + Ws(gradientAdj(vTV)) );
	sumProx = 0;
	for i=1:nB
		idx = (i-1)*Nx+1:i*Nx;
		zi(:) = vB(idx) + ga_cope*x(:)';
		pB = proxGis{i}( zi, ga_cope );
		sumProx = sumProx + pB;
		vB(idx) = vB(idx) - zi(:)' + pB(:)' + ga_cope*y(:)';
	end
	zTV = vTV + ga_cope*gradientOp(W(x));
	pTV = zTV - ga_cope*blockThresholding( zTV/ga_cope, nu/ga_cope, 2 );
	vTV = vTV - zTV + pTV + ga_cope*gradientOp(W(y));
	x = x - ga_cope*( gradF( y )+sumProx+Ws(gradientAdj(pTV)) );
end
tCoPe = toc;
xCoPe = x;

%%%  BD-HPE  %%%
si = .9;
ga_hpe = 2*be_F*si^2/(1+sqrt(1+4*si^2*be_F^2));
% direct implementation reduces the number of auxiliary variables
tic
eHPE = [];
x = zeros( xSize );
u = zeros( xSize );
Nx = numel( x );
zi = zeros( size( x ) );
z = zeros( [xSize nB+1] ); 
vi = zeros( size( x ) );
v = zeros( [xSize nB+1] ); 
yx = zeros(N,N,2);
yx1 = yx;
yx2 = yx;
yu = yx;
yu1 = yx;
yu2 = yx;
for it=1:nIter
	if verbose, progressbar(it,nIter); end
	if doReport, eHPE(it,:) = F(x)+G(x); end
	forward = ga_hpe*gradF(x);
	for i=1:nB
		idx = (i-1)*Nx+1:i*Nx;
		zi(:) = z(idx);	
		vi(:) = v(idx);
		z(idx) = proxGi{i}( ga_hpe^2*x + (1-ga_hpe^2)*zi + ga_hpe*(vi-u) - forward, (nB+2)*ga_hpe );
	end
	yx1 = blockThresholding( ga_hpe^2*yx + (1-ga_hpe^2)*yx1 + ga_hpe*(yu1-yu), (nB+2)*ga_hpe*nu, 2 );
	idx = nB*Nx+1:(nB+1)*Nx;
	zi(:) = z(idx);
	vi(:) = v(idx);	
	xg = projGradFrame(cat(3,ga_hpe^2*x + (1-ga_hpe^2)*zi + ga_hpe*(vi-u) - forward,ga_hpe^2*yx + (1-ga_hpe^2)*yx2 + ga_hpe*(yu2-yu)),Kinv,W,Ws); 
	z(idx) = xg(:,:,1:J);
	yx2 = xg(:,:,end-1:end);
	x = (sum( z, 4 ) + x)/(nB+2);
	yx = (yx1+yx2+nB*yx)/(nB+2);
	v = v - ga_hpe*z + repmat( ga_hpe*x, [ones(1, 4-1) nB+1] );
	yu1 = yu1 - ga_hpe*yx1 + ga_hpe*yx;
	yu2 = yu2 - ga_hpe*yx2 + ga_hpe*yx;
	u = (sum( v, 4 ) + u)/(nB+2);
	yu = (yu1+yu2+nB*yu)/(nB+2);
end
tHPE = toc;
xHPE = x;

%%%  DR  %%%
ga_dr = 1/(nB+4);
la_dr = 1;
% direct implementation reduces the number of auxiliary variables
tic
eDR = [];
x = zeros( xSize );
Nx = numel( x );
zi = zeros( size( x ) );
z = zeros( [xSize nB+2] ); 
y1 = zeros(N,N,2);
y11 = y1;
y12 = y1;
y2 = zeros(N);
y21 = y2;
y22 = y2;
if la_dr~=1, x0 = x; y10 = y1; y20 = y2; end
for it=1:nIter
	if verbose, progressbar(it,nIter); end
	if doReport, eDR(it,:) = F(x)+G(x); end
	for i=1:nB
		idx = (i-1)*Nx+1:i*Nx;
		zi(:) = z(idx);	
		z(idx) = zi + la_dr*( proxGi{i}( 2*x - zi, (nB+4)*ga_dr ) - x );
	end
	y11 = y11 + la_dr*( blockThresholding( 2*y1 - y11, (nB+4)*ga_dr*nu, 2 ) - y1 );
	idx = nB*Nx+1:(nB+1)*Nx;
	zi(:) = z(idx);
	xg = projGradFrame(cat(3,2*x - zi,2*y1 - y12),Kinv,W,Ws); 
	z(idx) = zi + la_dr*( xg(:,:,1:J) - x );
	y12 = y12 + la_dr*( xg(:,:,end-1:end) - y1 );
	y21 = y21 + la_dr*( invIgaMsM( 2*y2 - y21 + (nB+4)*ga_dr*KsY, (nB+4)*ga_dr ) - y2 );
	idx = (nB+1)*Nx+1:(nB+2)*Nx;
	zi(:) = z(idx);
	invLLs_L = invIKKs( 2*y2 - y22 - K(W( 2*x - zi )) );
	z(idx) = zi + la_dr*( x - zi + Ws(Ks( invLLs_L )) );
	y22 = y22 + la_dr*( y2 - y22 - invLLs_L );
	if la_dr~=1;
		x0 = x0 + la_dr*( x - x0 );
		y10 = y10+ la_dr*( y - y10 );
		y20 = y20 + la_dr*( y - y20 );
		x = (sum( z, 4 ) + 2*x0)/(nB+4);
		y1 = (y11+y12+(nB+2)*y10)/(nB+4);
		y2 = (y11+y22+(nB+2)*y20)/(nB+4);
	else
		x = (sum( z, 4 ) + 2*x)/(nB+4);
		y1 = (y11+y12+(nB+2)*y1)/(nB+4);
		y2 = (y21+y22+(nB+2)*y2)/(nB+4);
	end
end
tDR = toc;
xDR = x;

%%%  ChPo  %%%
sig = 1;
tau = .9/(sig*(normW2*normM2*normK2+nB+8));
the = 1;
% direct implementation reduces the number of auxiliary variables
tic
eChPo = [];
x = zeros( xSize );
xRec = Phi(W(x));
xBlock = repmat(x,[1 1 1 S^2]);
xGrad = gradientOp(W(x));
xx = x;
proxFs = @(x,ga) (x-ga*Y)/(1+ga);
proxBks = @(x,ga)proxGi_n(x,ga,proxGis,4);
proxGrs = compute_dual_prox( @(x,ga)blockThresholding(x,ga*nu,2) );
for it=1:nIter
	if verbose, progressbar(it,nIter); end
	if doReport, eChPo(it,:) = F(x)+G(x); end
	xRec = proxFs( xRec+sig*Phi(W(xx)), sig );
	xBlock = proxBks( xBlock+sig*repmat(xx,[1 1 1 S^2]), sig );
	xGrad = proxGrs( xGrad+sig*gradientOp(W(xx)), sig );
	xtmp = x;
	x = x - tau*( Ws(Phis(xRec)) + sum(xBlock,4) + Ws(gradientAdj(xGrad)) );
	xx = x + the*(x-xtmp);
end
tChPo = toc;
xChPo = x;
%}
tGFB
tChPo = 358;
tDR = 294;
tHPE = 409;
tCoPe = 441;
tGFB = 286;

if doReport
e = [eChPo eDR eHPE eCoPe eGFB];
%%%  PsiMin  %%%
PsiMinFile = 'Results/PsiMin_composite_tv.mat';
if nIterInf > nIter
	if exist( PsiMinFile )
		load( PsiMinFile, 'PsiMin' ); % be sure that it corresponds to the right parameters
	else
		x = zeros( xSize );
		Nx = numel( x );
		zi = zeros( size( x ) );
		z = zeros( [xSize nB+1] ); 
		y = zeros(N,N,2);
		y1 = y;
		y2 = y;
		if la_gfb~=1, x0 = x; y0 = y; end
		for it=1:nIterInf
			if verbose, progressbar(it,nIterInf); end
			forward = ga_gfb*gradF(x);
			for i=1:nB
				idx = (i-1)*Nx+1:i*Nx;
				zi(:) = z(idx);	
				z(idx) = zi + la_gfb*( proxGi{i}( 2*x - zi - forward, (nB+2)*ga_gfb ) - x );
			end
			y1 = y1 + la_gfb*( blockThresholding( 2*y - y1, (nB+2)*ga_gfb*nu, 2 ) - y );
			idx = nB*Nx+1:(nB+1)*Nx;
			zi(:) = z(idx);
			xg = projGradFrame(cat(3,2*x - zi - forward,2*y - y2),Kinv,W,Ws); 
			z(idx) = zi + la_gfb*( xg(:,:,1:J) - x );
			y2 = y2 + la_gfb*( xg(:,:,end-1:end) - y );
			if la_gfb~=1;
				x0 = x0 + la_gfb*( x - x0 );
				y0 = y0 + la_gfb*( y - y0 );
				x = (sum( z, 4 ) + x0)/(nB+2);
				y = (y1+y2+nB*y0)/(nB+2);
			else
				x = (sum( z, 4 ) + x)/(nB+2);
				y = (y1+y2+nB*y)/(nB+2);
			end
		end
		PsiMin = F(x)+G(x);
		save( PsiMinFile, 'PsiMin' )
	end
	plotIt = nIter;
else
	PsiMin = min( e(:) );
	plotIt = round( .8*nIter );
end
end%endif doReport

			 %----------------------------%
			 %  Display and Save Results  %
			 %----------------------------%

recov = W(xGFB);
imgprint( recov, 'Results/composite_l1l%d_tv_%d_%d_recov.png', 2, S, nIter )
imgprint( Y, 'Results/composite_l1l%d_tv_%d_%d_obs.png', 2, S, nIter )

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
subplot( 2, 2, 3 ), imshow(Y), stitle( '\\sigma_{n}: %.3f; \\rho_{d}: %.1f; \\sigma_{b}: %d; SNR: %.2fdB', sig_w, rho_M, sig_K, SNRdB( Y0, Y ) )
subplot( 2, 2, 4 ), imshow(recov), stitle( '\\mu_{l1/l%d}: %.1e; S: %d; \\nu_{TV}: %.1e;\n it. #%d; SNR: %.2fdB', 2, mu, S, nu, nIter, SNRdB( Y0, recov ) )
drawnow

if doReport
	epsprint( 'Results/composite_l1l%d_tv_%d_%d.eps', 2, S, nIter  );
else
	epsprint( 'Results/composite_l1l%d_tv_%d_%d_time.eps', 2, S, nIter  );
end
