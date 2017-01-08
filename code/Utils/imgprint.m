function imgprint( img, varargin )
%
% 	 imgprint( img, varargin )
%
% write the given variable 'img'
% the destination is given in varargin, following sprintf syntax
% extension should be given in the destination name
% 
% Hugo Raguet 2011

imwrite( img, sprintf( varargin{:} ) )
