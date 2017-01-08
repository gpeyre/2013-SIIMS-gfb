function G = gradientOp( X )
%
%	 G = gradientOp( X )
%
% compute gradient of the given image X with finite differences, with filters 
% H = (-1 1) and V = (-1 0) with center pixel at the upper-left corner
%     ( 0 0)         ( 1 0)
% INPUT: 'X': m-by-n array
%	original image
% OUTPUT: 'G': m-by-n-by-2 array
% 	gradient of the image, along vertical dimension for the first frame
%	                       along horizontal dimension for the second frame
% Boundary conditions: X is mirrored, so that the bottom row of G(:,:,1) and the right column of G(:,:,2) are equal to zero.
%
% Hugo Raguet 2011

G = cat( 3, padarray( X(2:end,:) - X(1:end-1,:), [1 0], 'post' ), ...
            padarray( X(:,2:end) - X(:,1:end-1), [0 1], 'post' ) );

end %gradientOp
