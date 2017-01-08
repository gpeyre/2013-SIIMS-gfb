function n = normAllGridBlock( x, p, sz, mu  )
%
%	 n = normAllGridBlock( x, p, sz, [mu=1]  )
% compute
%	||x||_{1,p}^{B,mu}
% where B the set of all (overlapping) "spatial" (2-D) rectangular blocks defined over x, and mu is a vector weighting the l1/lp-norm.
%
% INPUT:
%	'x': X-by-Y-by-N matrix
%		the vector of coefficients, where the blocks are defined over the two first dimensions
%	'p': positive real number
%		the order of the norm on each block
%	'sz': non-negative integer row vector of length 2
% 		the x- and y-size of the blocks
%	'mu': muX-by-muY-by-muN matrix of positive terms
% 		the vector of weights on each blocks
%		muX is 1 or the number of blocks along first dimension
%		muY is 1 or the number of blocks along second dimension
%		muZ is 1 or N
% OUTPUT:
%	'n': non-negative real number
%		the resulting norm
%
% Hugo Raguet

if sz==[1 1], p = 1; end

switch p
case 0
	x = padarray( logical( x ), [sz(1)-1 sz(2)-1], 'both' );
	M = 0;
	for b1=1-sz(1):0, for b2=1-sz(2):0,
		M = M | circshift( x, [b1 b2] );
	end, end
	x = M;
case 1
	x = abs( x );
case 2
	x = padarray( abs( x ).^2, ceil((sz-1)/2), 'pre' );
	x = padarray( x, ceil((sz-2)/2), 'post' );
	x = sqrt( convn( x, ones(sz), 'same' ) );
case Inf
	x = padarray( abs( x ), [sz(1)-1 sz(2)-1], 'both' );
	M = 0;
	for b1=1-sz(1):0, for b2=1-sz(2):0,
		M = max( circshift( x, [b1 b2] ), M );
	end, end
	x = M;
otherwise
	x = padarray( abs( x ).^p, ceil((sz-1)/2), 'pre' );
	x = padarray( x, ceil((sz-2)/2), 'post' );
	x = ( convn( x, ones(sz), 'same' ) ).^(1/p);
end

if nargin > 3
	x = bsxfun( @times, mu, x );
end
n = sum( x(:) );
 
end %normAllGridBlock
