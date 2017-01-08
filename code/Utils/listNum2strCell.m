function C = listNum2strCell( L )
%
%	 C = listNum2strCell( L )
%
% Hugo Raguet 2012

[M,N] = size( L );
C = arrayfun( @(c){ num2str(c{:}) }, mat2cell( L, ones(1,M), ones(1,N) ) );
end %listNum2strCell
