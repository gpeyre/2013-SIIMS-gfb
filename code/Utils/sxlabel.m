function sxlabel( varargin )
%
% 	 sxlabel( varargin )
%
% write the label of current x-axes in fontsize 14
% label given in varargin, following sprintf syntax
%
% Hugo Raguet 2011

xlabel( sprintf( varargin{:} ), 'FontSize', 14 )
