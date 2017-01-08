function x = proxGi_n( x, ga, proxGi, catDim )
% 
%	 x = proxGi_n( x, ga, proxGi, catDim )
%
% Compute
%		prox_{ga*G}(x) = argmin_y 1/2||x-y||^2 + ga*G(x)
% where
%	x = (x1,...,xn) and G(x) = \sum_{i=1}^n Gi(xi)
% i.e.  prox_{ga*G}(x) = (prox_{ga*G1}(x1), ..., prox_{ga*Gn}(xn)) 
% INPUT:
% 	'x': catDim-D matrix
%		concatenation in the last dimension of (x1,...,xn)
%	'ga': double
%		ga > 0
%	'proxGi': n-long cell array of function handles : (class(xi),double) -> class(xi)
%	'catDim': positive integer
%		dimension of the concatenation
%		proxGi{i}(xi,ga) computes prox_{ga*Gi}(xi)
% OUPTUT:
%	'x': catDim-D matrix
%
% Hugo Raguet 2011

n = length( proxGi );
xSize = size( x );
switch catDim
case 1, xSize = [1 1];
case 2, xSize = [xSize(1) 1];
otherwise, xSize = xSize(1:catDim-1); end
N = prod( xSize );

for i=1:n
	idx = (i-1)*N+1:i*N;
	x(idx) = proxGi{i}( reshape( x(idx), xSize ), ga );
end

end %proxGi_n
