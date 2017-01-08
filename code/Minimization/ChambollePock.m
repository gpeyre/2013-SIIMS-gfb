function [x, R] = ChambollePock( K, Ks, proxFs, proxG, x, nIter, sig, tau, the, verbose, report )
%
%	 [x, R] = ChambollePock( K, Ks, proxFs, proxG, x, nIter, sig, tau, [the=1], [verbose=false], [report=None] )
%
% Solves
%		min_x F(Kx) + G(x)
% where
%	prox_{ga*Fs}(x) = argmin_y 1/2||x-y||^2 + ga*Fs(x) is computable;
% 	Fs(x) = sup_y <x|y> - F(y) is the Legendre-Frenchel transform of Gi;
%	prox_{ga*G}(x) = argmin_y 1/2||x-y||^2 + ga*G(x) is computable;
%	K is a linear operator
% with Chambolle-Pock primal-dual method (Antonin Chambolle, Thomas Pock, "A first-order primal-dual algorithm for convex problems with applications to imaging", Preprint CMAP-685)
%
% INPUT:
%	'K': function handle: class(x) -> class(Kx)
%		computes the linear operator K
%	'Ks': function handle: class(Kx) -> class(x)
%		computes the adjoint operator of K
%	'proxFs': function handle: (class(Kx),double) -> class(Kx)
%		proxFs(y,ga) computes prox_{ga*Fs}(y)
%	'proxG': function handle: (class(x),double) -> class(x)
%		proxG(x,ga) computes prox_{ga*G}(x)
%	'x': vector or N-D matrix
%		the initial variable
%	'nIter': non-negative integer
%		the number of iterations recquired
%	'sig': positive real number
%		the size of proximal step on F
%	'tau': positive real number
%		the size of proximal step on G
% 		sig*tau*||K||^2 < 1
%	'the' [default=1]: double
%		relaxation of the iterations
% 		the \in [0,1]
%	'verbose' [default=false]: logical
%		set the diaplay of iterations
%	'report' [default=None]: function handle : class(x) -> 1-by-m matrix
%		a user-defined report called at each iteration
% OUTPUT:
%	'x': vector or N-D matrix
%		the final minimizer
%	'R' [default='empty']: nIter-by-m matrix
%		the report sequence
%
% Hugo Raguet 2011

if nargin < 9, the=1; end
if nargin < 10, verbose=false; end
doReport = nargin >= 11;

y = K(x);
xx = x;
R = [];

for it=1:nIter    
	if verbose, progressbar(it,nIter); end
	if doReport, R(it,:) = report( x ); end
	y = proxFs( y+sig*K(xx), sig );
	xtmp = x;
	if isempty(proxG)
		x = x-tau*Ks(y);
	else
		x = proxG( x-tau*Ks(y), tau );
	end
	xx = x + the*(x-xtmp);
end

end %ChambollePock
