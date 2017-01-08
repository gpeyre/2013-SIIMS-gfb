function XG = projGradFrame( XG, Kinv, W, Ws )
%
%	 XG = projGradFrame( XG, Kinv, W, Ws )
% 
% orthogonal projection of XG = (X,G) onto {(X,G) \ Grad(W(X)) = G}
% where X is a 3-D vector, Grad is the image gradient operator computed with finite difference filters
% H = (-1 1) and V = (-1 0) with center pixel at the upper-left corner
%     ( 0 0)         ( 1 0)
% (see gradientOp.m), and W is a Parseval tight frame (W Ws = Id, Ws being its adjoint).
% uses the fact that if LLs = Id, prox_{FoL} = Id + Ls(prox_{F} - Id)L
% so that for L = [W 0]
%                 [0 I],
%             prox_{FoL}(x,y) = (x + Ws( prox_F(Wx,y)_x - Wx ), prox_F(Wx,y)_y )
%
% INPUT:
% 	'XG': N1-by-N2-by-(N3+2) matrix
%		concatenation in the third dimension of (X,G)
%		X is N1-by-N2-N3 matrix
%		G is N1-by-N2-by-2 matrix
%	'Kinv': the filter in fourier domain allowing to compute (Id+Grads Grad)^{-1}
% 		can be pre-computed for N-by-N image by
%			Kv = padarray( [-1; zeros(2*N-2,1); 1], [0 2*N-1], 'post' );
%			Kvs = padarray( [-1; 1; zeros(2*N-2,1)], [0 2*N-1], 'post' );
%			Kh = padarray( [-1, zeros(1,2*N-2), 1], [2*N-1 0], 'post' );
%			Khs = padarray( [-1, 1, zeros(1,2*N-2)], [2*N-1 0], 'post' );
%			FKv = fft2( Kv ); FKvs = fft2( Kvs );
%			FKh = fft2( Kh ); FKhs = fft2( Khs );
%			Kinv = FKvs.*FKv+FKhs.*FKh+1;
%	'W': function handle: N1-by-N2-N3 array -> N1-by-N2 array
%		the linear operator corresponding to the parseval tight frame
%	'Ws': function handle: N1-by-N2 array -> N1-by-N2-by-N3 array
%		the adjoint operator of W
%		W Ws = Id.
% OUPTUT:
% 	'XG': N1-by-N2-by-(N3+2) matrix
%		concatenation in the third dimension of the solution
%
% Hugo Raguet 2011

X = XG(:,:,1:end-2);
WX = W(X);
PXG = projGrad( cat(3,WX,XG(:,:,end-1:end)), Kinv );
XG = cat(3,X + Ws( PXG(:,:,1) - WX ), PXG(:,:,2:3) );

end %projGradFrame
