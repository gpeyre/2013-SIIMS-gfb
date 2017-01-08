function X = gradientAdj( G )
%
%	 X = gradientAdj( G )
%
% compute adjoint operator of the gradient implemented in gradientOp.m
%
% Hugo Raguet 2011

X = padarray( G(1:end-1,:,1), [1 0], 'pre' ) - padarray( G(1:end-1,:,1), [1 0], 'post' ) + padarray( G(:,1:end-1,2), [0 1], 'pre' ) - padarray( G(:,1:end-1,2), [0 1], 'post' );

end %gradientAdj
