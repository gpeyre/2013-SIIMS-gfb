function stitle( varargin )
%
% 	 stitle( varargin )
%
% write the title of current axes in fontsize 14
% title given in varargin, following sprintf syntax
%
% Hugo Raguet 2011

title( sprintf( varargin{:} ), 'FontSize', 14 )
