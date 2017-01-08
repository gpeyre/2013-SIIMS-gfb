function epsprint( varargin )
%
% 	 epsprint( varargin )
%
% create a colored Encapsulated PostScript document from the current figure
% the destination is given in varargin, following sprintf syntax
% 
% Hugo Raguet 2011

print( gcf, '-depsc', sprintf( varargin{:} ) )
