function sylabel( varargin )
%
% 	 sylabel( varargin )
%
% write the label of current y-axes in fontsize 14
% label given in varargin, following sprintf syntax
%
% Hugo Raguet 2011

ylabel( sprintf( varargin{:} ), 'FontSize', 14 )
