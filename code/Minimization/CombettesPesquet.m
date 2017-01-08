function [x, R, v] = CombettesPesquet( gradF, proxGis, x, v, nIter, ga, verbose, report )
%
%	 [x, R, v] = CombettesPesquet( gradF, proxGis, x, v, nIter, ga, [verbose=false], [report=None] )
%
% Solve
%		min_x F(x)+\sum_{i=1}^{n} Gi(x)
% where
%	F is differentiable and its gradient is mu-Lipschitz continuous;
%	prox_{ga*Gis}(x) = argmin_y 1/2||x-y||^2 + ga*Gis(x) is computable;
% 	Gis(x) = sup_y <x|y> - Gi(y) is the Legendre-Frenchel transform of Gi;
% with [CombettesPesquet11]
%
% INPUT:
%	'gradF': function handle: class(x) -> class(x)
%		computes the gradient of F
%	'proxGis': n-long cell array of function handles : (class(x),double) -> class(x)
%		proxGis{i}(x,ga) computes prox_{ga*Gis}(x)
%	'x': vector or ND-matrix
%		the initial guess
%	'v': [size(x)]-by-n matrix
%		the initial 'dual' auxiliary variables
%	'nIter': num
%		the number of iterations recquired
%	'ga': double
%		the size of forward step,
%		ga \in ]0,1/(mu+sqrt(n))[
%	'verbose' [default=false]: logical
%		set the diaplay of iterations
%	'report' [default=None]: function handle : class(x) -> 1-by-m matrix
%		a user-defined report called at each iteration
% OUTPUT:
%	'x': vector or ND-matrix
%		the final minimizer
%	'R' [default='empty']: nIter-by-m matrix
%		the report sequence
%	'v': [size(x)]-by-n matrix
%		the final 'dual' auxiliary variables
%
% Hugo Raguet 2011

if nargin < 7, verbose = false; end
doReport = nargin >= 8;

n = length( proxGis );
catDim = ndims( v );
N = numel( x );
vi = zeros( size( x ) );
R = [];

for it=1:nIter
	if verbose, progressbar(it,nIter); end
	if doReport, R(it,:) = report( x ); end
	forward = ga*( gradF(x) + sum( v, catDim ) );
	sumProx_y2i = 0;
	for i=1:n
		idx = (i-1)*N+1:i*N;
		vi(:) = v(idx);
		prox_y2i = proxGis{i}( vi + ga*x, ga );
		sumProx_y2i = sumProx_y2i + prox_y2i;
		v(idx) = prox_y2i - ga*forward;
	end
	x = x - ga*( gradF( x - forward ) + sumProx_y2i );
end

end %CombettesPesquet
