function vprintf( verbose, varargin )
%
%	 vprintf( verbose, varargin )
%
% print to screen if verbose is true
% varargin follows the sprintf syntax
%
% Hugo Raguet 2011

if verbose, fprintf( varargin{:} ), end
