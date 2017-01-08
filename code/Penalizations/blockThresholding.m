function x = blockThresholding( x, T, p )
%
%	 x = blockThresholding( x, T, p )
% 
% compute
%	prox_{||.||_{l1/lp}^{B,T}}(x) = argmin_{y} 1/2*||x-y||^2 + ||y||_{l1/lp}^{B,T}
% where B is any blocks structure and T is a vector weighting the contributions of the blocks in the l1/lp-norm (see normByBlock.m).
% INPUT:
%	'x': N-by-B-by-L matrix
%		the input vector, constituted by B blocks of L vectors of length N.
%	'T': TN-by-TB matrix of positive terms
% 		the vector of weights on each blocks
%		TN is 1 or divides the number of coefficients
%		TB is 1 or divides the number of blocks
%		for p=0, T is treated as sqrt(T)
%	'p': positive real number 
%		the order of the norm on each block
%		IMPLEMENTED ONLY FOR p \in {0,1,2,Inf}
% OUTPUT:
%	'x': N-by-B-by-L matrix
%		the "thresholded" output vector
%
% Hugo Raguet 2011

[N, B, L] = size( x );
if L==1, p = 1; end
[TN, TB] = size( T );
T = repmat( T, [N/TN B/TB] );
x1 = abs( x );

switch p
case 0
	x( repmat( sum( x1.^2, 3 ) <= 2*T, [1 1 L] ) ) = 0;
case 1
	T = repmat( T, [1 1 L] );
	supp = x1 > T;
	x( supp ) = ( 1 - T( supp )./x1( supp ) ).*x( supp );
	x( ~supp ) = 0;
case 2
	x1 = sqrt( sum( x1.^2, 3 ) );
	supp = x1 > T;
	s = 1 - T( supp )./x1( supp );
	supp = repmat( supp, [1 1 L] );
	x( supp ) = repmat( s(:), [L 1] ).*x( supp );
	x( ~supp ) = 0;
case Inf % maybe it would be better to implement Id - Proj_{||.||_1 < T}
	x1sort = sort( x1, 3, 'descend' );
	x1cum = cumsum( x1sort, 3 );
	repT = repmat( T, [1 1 L] );
	repL = repmat( shiftdim( [1:L], -1 ), [N B] );
	d = x1sort.*repL - x1cum + repT;
	x1cum( d < 0 ) = 0;
	[yiMax, iMax] = max( x1cum, [], 3 );
	yiMax = repmat( max( ( yiMax - T ) ./ iMax, 0 ), [1 1 L] );
	supp = x1 > yiMax; 
	x( supp ) = ( yiMax( supp )./x1( supp ) ).*x( supp );
otherwise
	error( 'solve for all i, (T*||y||_{%d}^(%d))*|y_i|^%d + |y_i| = |x_i|', p, 1-p, p-1 )
end

end %blockThresholding
