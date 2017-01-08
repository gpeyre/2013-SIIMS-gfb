function XG = projGrad( XG, Kinv )
%
%	 XG = projGrad( XG, Kinv )
%
% orthogonal projection of XG = (X,G) onto {(X,G) \ Grad(X) = G}
% where X is an image, Grad is the gradient computed with finite difference filters
% H = (-1 1) and V = (-1 0) with center pixel at the upper-left corner.
%     ( 0 0)         ( 1 0)
% see gradientOp.m
% INPUT:
% 	'XG': N1-by-N2-by-3 matrix
%		concatenation in the third dimension of (X,G)
%		X is N1-by-N2 matrix
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
% OUPTUT:
% 	'XG': N1-by-N2-by-3 matrix
%		concatenation in the third dimension of the solution
%
% Hugo Raguet 2011

N1 = size( XG, 1 );
N2 = size( XG, 2 );
Y = XG(:,:,1) + gradientAdj( XG(:,:,2:3) );
Y = [Y Y(:,N2:-1:1); Y(N1:-1:1,:), Y(N1:-1:1,N2:-1:1)];
Y = ifft2(fft2(Y)./Kinv);
Y = Y(1:N1,1:N2);
XG = cat(3,Y,gradientOp(Y));

end %projGrad
