function [x, R, z, u] = BD_HPE( gradF, proxGi, z, v, nIter, ga, verbose, report )
%
%	 [x, R, z, u] = BD_HPE( gradF, proxGi, z, v, nIter, ga, [verbose=false], [report=None] )
%
% Solve
%		min_x F(x)+\sum_{i=1}^{n} Gi(x)
% where
%	F is differentiable and its gradient is 1/be-Lipschitz continuous,
%	prox_{ga*Gi}(x) = argmin_y 1/2||x-y||^2 + ga*Gi(x) is computable;
% with block-decomposition hybrid proximal extragradient.
% INPUT:
%	'gradF': function handle : class(x) -> class(x)
%		computes the gradient of F
%	'proxGi': n-long cell array of function handles : (class(x),double) -> class(x)
%		proxGi{i}(x,ga) computes prox_{ga*Gi}(x)
%	'z': [size(x)]-by-n matrix
%		the initial auxiliary variables
%	'v': [size(x)]-by-n matrix
%		the initial auxiliary adjoint variables
%	'nIter': num
%		the number of iterations recquired
%	'ga': double
%		size of forward step,
%		ga = 2be*si^2/( 1+sqrt(1+4 si^2 be^2) ), where si \in ]0,1].
%	'verbose' [default=false]: logical
%		set the diaplay of iterations
%	'report' [default=None]: function handle : class(x) -> 1-by-m matrix
%		a user-defined report called at each iteration
% OUTPUT:
%	'x': vector or ND-matrix
%		the final minimizer
%	'R' [default='empty']: nIter-by-m matrix
%		the report sequence
%	'z': [size(x)]-by-n matrix
%		the final auxiliary variables
%	'v': [size(x)]-by-n matrix
%		the final auxiliary adjoint variables
%
% Hugo Raguet 2011

if nargin < 7, verbose = false; end
doReport = nargin >= 8;

n = length( proxGi );
catDim = ndims( z );
x = mean( z, catDim );
zi = zeros( size( x ) );
u = mean( v, catDim );
vi = zeros( size( u ) );
N = numel( x );
R = [];

for it=1:nIter
	if verbose, progressbar(it,nIter); end
	if doReport, R(it,:) = report( x ); end
	forward = ga*gradF(x);
	for i=1:n
		idx = (i-1)*N+1:i*N;
		zi(:) = z(idx);	
		vi(:) = v(idx);	
		z(idx) = proxGi{i}( ga^2*x + (1-ga^2)*zi + ga*(vi-u) - forward, n*ga );
	end
	x = mean( z, catDim );
	v = v - ga*z + repmat( ga*x, [ones(1, catDim-1) n] );
	u = mean( v, catDim );
end

end %BD_HPE
