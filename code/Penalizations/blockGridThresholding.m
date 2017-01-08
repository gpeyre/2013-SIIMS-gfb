function x = blockGridThresholding( x, T, p, sz, shift )
%
%	 x = blockGridThresholding( x, T, p, sz, shift )
% 
% compute
%	prox_{||.||_{l1/lp}^{B,T}}(x) = argmin_{y} 1/2*||x-y||^2 + ||y||_{l1/lp}^{B,T}
% where B is a non-overlapping "spatial" (2-D) decomposition of x in equal rectangular blocks, aligned in upper-left corner and shifted by 'shift' and T is a vector weighting the contributions of the blocks in the l1/lp-norm.
% INPUT:
%	'x': X-by-Y-by-N matrix
%		the vector of coefficients, where the blocks are defined over the two first dimensions
%	'T': TX-by-TY-by-TN matrix of positive terms
% 		the vector of weights on each blocks
%		TX is 1 or the number of blocks along first dimension
%		TY is 1 or the number of blocks along second dimension
%		TZ is 1 or N
%	'p': positive real number 
%		the order of the norm on each block
%		IMPLEMENTED ONLY FOR p \in {0,1,2,Inf}
%	'sz': positive integer row vector of length 2
% 		the x- and y-size of the blocks
%	'shift': non-negative integer row vector of length 2
%		the shift of the spatial grid compared to the vector or coefficients
%		smaller than 'sz' along each dimension
% OUTPUT:
%	'x': X-by-Y-by-N matrix
%		the "thresholded" vector of coefficients
%
% Hugo Raguet 2011

[X, Y, N] = size( x );
if all( sz==[1 1] )
	p = 1;
else
	pre = mod(-shift,sz);
	post = mod(shift-[X Y],sz);
	x = padarray( padarray( x, pre, 'pre' ), post, 'post' );
	[X, Y, N] = size( x );
end
X_ = X/sz(1);
Y_ = Y/sz(2);

switch p
case 0
	x2 = abs( x ).^2;
	normBlocks2 = zeros( X_, Y_, N );
	for i=1:sz(1), for j=1:sz(2)
		normBlocks2 = normBlocks2 + x2( i:sz(1):X, j:sz(2):Y, : );
	end, end
	supp = bsxfun( @gt, normBlocks2, 2*T );
	for i=1:sz(1), for j=1:sz(2)
		iIdx = i:sz(1):X;
		jIdx = j:sz(2):Y;
		xTmp = x( iIdx, jIdx, : );
		xTmp( ~supp ) = 0;
		x( iIdx, jIdx, : ) = xTmp;
	end, end
case 1
	x1 = abs( x );
	supp = bsxfun( @gt, x1, T );
	x = max( 1 - bsxfun( @rdivide, T, x1 ), 0 ).*x;
case 2
	x2 = abs( x ).^2; % for complex valued coefficients
	normBlocks = zeros( X_, Y_, N );
	for i=1:sz(1), for j=1:sz(2)
		normBlocks = normBlocks+x2( i:sz(1):X, j:sz(2):Y, : );
	end, end
	normBlocks = sqrt( normBlocks );
	supp = bsxfun( @gt, normBlocks, T );
	s = max( 1 - bsxfun( @rdivide, T, normBlocks ), 0 );
	for i=1:sz(1), for j=1:sz(2)
		iIdx = i:sz(1):X;
		jIdx = j:sz(2):Y;
		x( iIdx, jIdx, : ) = s.*x( iIdx, jIdx, : );
	end, end
case Inf % maybe it would be better to implement Id - Proj_{||.||_1 < T}
	L = prod( sz );
	blocks = zeros( X_, Y_, N, L );
	for i=1:sz(1), for j=1:sz(2)
		blocks( :, :, :, i+sz(1)*(j-1) ) = x( i:sz(1):X, j:sz(2):Y, : );
	end, end
	x1sort = sort( abs( blocks ), 4, 'descend' );
	x1cum = cumsum( x1sort, 4 );
	Ltimes = shiftdim( [1:L], -2 );
	d = bsxfun( @plus , bsxfun( @times, x1sort, Ltimes ), bsxfun( @minus, T, x1cum ) );
	x1cum( d < 0 ) = 0;
	[yiMax, iMax] = max( x1cum, [], 4 );
	yiMax = max( bsxfun( @minus, yiMax, T ) ./ iMax, 0 );
	supp = bsxfun( @gt, blocks, yiMax ); 
	blocks( supp ) = yiMax( supp );
	supp = bsxfun( @lt, blocks, - yiMax ); 
	blocks( supp ) = - yiMax( supp );
	for i=1:sz(1), for j=1:sz(2)
		x( i:sz(1):X, j:sz(2):Y, : ) = blocks( :, :, :, i+sz(1)*(j-1) );
	end, end
otherwise
	error( 'solve for all i, (T*||y||_{%d}^(%d))*|y_i|^%d + |y_i| = |x_i|', p, 1-p, p-1 )
end

if any( sz~=[1 1] )
	x = x( 1+pre(1):end-post(1), 1+pre(2):end-post(2), : );
end

end %blockGridThresholding
