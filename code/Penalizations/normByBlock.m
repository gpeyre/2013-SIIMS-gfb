function n = normByBlock( x, p, mu )
%
%	 n = normByBlock( x, p, [mu=1] )
% 
% compute norm by blocks
%	||x||_{l1/lp}^{B,mu} = sum_{b \in B} mu_b ||x||_p
% where B is any blocks structure and mu is a vector weighting the contributions of the blocks in the l1/lp-norm.
% INPUT:
%	'x': N-by-B-by-L matrix
%		the input vector, constituted by B blocks of L vectors of length N.
%	'p': positive real number 
%		the order of the norm on each block
%	'mu': muN-by-muB matrix of positive terms
% 		the vector of weights on each blocks
%		muN is 1 or the number of coefficients
%		muB is 1 or the number of blocks
% OUTPUT:
%	'n': double
%		the resulting norm
%
% Hugo Raguet 2011

[N, B, L] = size( x );

switch p
case 0
	n = any( x, 3 );
case 1
	n = sum( abs( x ), 3 );
case 2
	n = sqrt( sum( abs( x ).^2, 3 ) );
case Inf
	n = max( abs( x ), [], 3 );
otherwise
	n = sum( abs( x ).^p, 3 ).^(1/p);
end
if nargin > 2
	n = bsxfun( @times, mu, n );
end
n = sum( n(:) );
n = mu(:)'*n(:);
 
end %normByBlocks
