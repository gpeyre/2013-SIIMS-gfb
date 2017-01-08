function signals = fwt2Dti( signals, Jmin, dir, h )

% 	 signals = fwt2Dti( signals, Jmin, dir, [h=('Daubechies',4)] )
%
% Compute 2-D translation (shift) invariant wavelet transform with algorithme-à-trou
%
% WARNING: length of signals must be power of 2
% WARNING: signals must be square
% INPUT:
%	'signals': len-by-len matrix (pixels domain) or len-by-len-by-(3J+1) matrix (wavelets domain)
%		2-D signals of length len-by-len
%		2-D wavelets coefficients of size len-by-len for J different decomposition scales plus the approximation coefficients (see below)
%	'Jmin': positive integer
%		the coarsest level of decomposition 0 <= Jmin < log2(len)
%	'dir': num: -1 | +1 | -2 | +2
%		-1 for the pseudo-inverse of the transform
%		+1 for the direct transform
%		-2 for a scaled inverse transform
%		+2 for the adjoint operator of the scaled inverse transform
%	'h' [(Daubechies,4)]: vector
%		the high pass filter of the considered wavelets
% OUTPUT:
%	'signals': len-by-len matrix (pixels domain) or len-by-len-by-(3J+1) matrix (wavelets domain)
%
% CONVENTION: In the transformed domain, the len-by-len-by-(3J+1) matrix W is such that for i = 3*j+1, j > 0, W(:,:,i) (resp.  W(:,:,i+1),  W(:,:,i+2)) is the concatenation in the first 2 dimensions, for the 4^(Jmax-j+1) possible (vertical and horizontal) translations for the j-th scale, of the wavelets coefficients of the vertical (resp. horizontal, diagonal) successive filtering, and W(:,n,1) is the concatenation of the corresponding 4^(Jmax-Jmin+1) approximation coefficients.
%
% Hugo Raguet 2011

if nargin < 4, h = compute_wavelet_filter( 'Daubechies', 4 ); end

Lh = length( h );
g = [0 h(Lh:-1:2)] .* (-1).^(1:Lh);

if dir > 0
	[len, len] = size( signals ); 
	Jmax = log2( len ) - 1;
	wcoef = zeros( len, len, 3*(Jmax-Jmin+1) );
	Ch = ceil( Lh/2 );
	% scaling factor if one want the adjoint of the inverse
	if dir==2, factor=1; scaling = 2; end
	for j=Jmax:-1:Jmin
		locLen = 2^(j+1); % length of current scale
		oddIdx = 1:2:locLen;
		evenIdx = 2:2:locLen;
		% scale if adjoint of the inverse
		if dir==2, factor = factor/scaling; end
		for tV = 0:(2^(Jmax-j)-1) % all vertical translations for current scale
			subIdxV = ((tV*locLen)+1):((tV+1)*locLen);
		for tH = 0:(2^(Jmax-j)-1) % all horizontal translations for current scale
			subIdxH = ((tH*locLen)+1):((tH+1)*locLen);
			subSignals = signals( subIdxV, subIdxH, 1 );

			%%%  vertically  %%%
			coarse = zeros( locLen, locLen ); %C0
			detail = zeros( locLen, locLen ); %V0
			% circular convolution
			subSignals = circshift( subSignals, -Ch );
			for z=1:Lh
				subSignals = circshift( subSignals, 1 );
				coarse = coarse + h(z)*subSignals;
				detail = detail + g(z)*subSignals;
			end

			%%%  horizontally  %%%
			subSignals = cat( 3, coarse, detail );
			coarse = zeros( locLen, locLen, 2 ); %[C;V]
			detail = zeros( locLen, locLen, 2 ); %[H;D]
			% circular convolution
			subSignals = circshift( subSignals, [0 -Ch] );
			for z=1:Lh
				subSignals = circshift( subSignals, [0 1] );
				coarse = coarse + h(z)*subSignals;
				detail = detail + g(z)*subSignals;
			end

			% merge the coefficients and update coarse signals
			signals( subIdxV, subIdxH ) = [ coarse( oddIdx, oddIdx, 1 ), coarse( oddIdx, evenIdx, 1 ); coarse( evenIdx, oddIdx, 1 ), coarse( evenIdx, evenIdx, 1 ) ];
			if dir==+1
				wcoef( subIdxV, subIdxH, 3*(j-Jmin)+1 ) = [ coarse( oddIdx, oddIdx, 2 ), coarse( oddIdx, evenIdx, 2 ); coarse( evenIdx, oddIdx, 2 ), coarse( evenIdx, evenIdx, 2 ) ]; % V
				wcoef( subIdxV, subIdxH, 3*(j-Jmin)+2 ) = [ detail( oddIdx, oddIdx, 1 ), detail( oddIdx, evenIdx, 1 ); detail( evenIdx, oddIdx, 1 ), detail( evenIdx, evenIdx, 1 ) ]; % H
				wcoef( subIdxV, subIdxH, 3*(j-Jmin)+3 ) = [ detail( oddIdx, oddIdx, 2 ), detail( oddIdx, evenIdx, 2 ); detail( evenIdx, oddIdx, 2 ), detail( evenIdx, evenIdx, 2 ) ]; % D
			else
				wcoef( subIdxV, subIdxH, 3*(j-Jmin)+1 ) = factor*[ coarse( oddIdx, oddIdx, 2 ), coarse( oddIdx, evenIdx, 2 ); coarse( evenIdx, oddIdx, 2 ), coarse( evenIdx, evenIdx, 2 ) ]; % V
				wcoef( subIdxV, subIdxH, 3*(j-Jmin)+2 ) = factor*[ detail( oddIdx, oddIdx, 1 ), detail( oddIdx, evenIdx, 1 ); detail( evenIdx, oddIdx, 1 ), detail( evenIdx, evenIdx, 1 ) ]; % H
				wcoef( subIdxV, subIdxH, 3*(j-Jmin)+3 ) = factor*[ detail( oddIdx, oddIdx, 2 ), detail( oddIdx, evenIdx, 2 ); detail( evenIdx, oddIdx, 2 ), detail( evenIdx, evenIdx, 2 ) ]; % D
			end
		end, end %for tV,tH
	end
	% add coarse approximation
	if dir==+1
		signals = cat( 3, signals, wcoef );
	else
		signals = cat( 3, factor*signals, wcoef );
	end
else % inverse
	if dir==-1
		scaling = 4;
	else
		scaling = 2;
	end
	[len, len, Jmax] = size( signals );
	Jmax = (Jmax - 1)/3+Jmin-1;
	wcoef = signals( :, :, 2:end );
	signals = signals( :, :, 1 ); 
	Ch = floor( Lh/2 )+1;
	odd = mod( Ch, 2 ); % parity of the filter's centers index
	for j=Jmin:Jmax
		locLen = 2^(j+1); % length of current scale
		oddIdx = 1:2:locLen;
		evenIdx = 2:2:locLen;
		for tH = 0:(2^(Jmax-j)-1) % all horizontal translations for current scale
			subIdxH = ((tH*locLen)+1):((tH+1)*locLen);
			idxH1 = subIdxH(1:locLen/2);
			idxH2 = subIdxH(((locLen/2)+1):locLen);
		
		for tV = 0:(2^(Jmax-j)-1) % all vertical translations for current scale
			subIdxV = ((tV*locLen)+1):((tV+1)*locLen);
			idxV1 = subIdxV(1:locLen/2);
			idxV2 = subIdxV(((locLen/2)+1):locLen);

			C = cat( 3, signals( idxV1, idxH1 ), signals( idxV1, idxH2 ), signals( idxV2, idxH1 ), signals( idxV2, idxH2 ) );
			V = cat( 3, wcoef( idxV1, idxH1, 3*(j-Jmin)+1 ), wcoef( idxV1, idxH2, 3*(j-Jmin)+1 ), wcoef( idxV2, idxH1, 3*(j-Jmin)+1 ), wcoef( idxV2, idxH2, 3*(j-Jmin)+1 ) );
			H = cat( 3, wcoef( idxV1, idxH1, 3*(j-Jmin)+2 ), wcoef( idxV1, idxH2, 3*(j-Jmin)+2 ), wcoef( idxV2, idxH1, 3*(j-Jmin)+2 ), wcoef( idxV2, idxH2, 3*(j-Jmin)+2 ) );
			D = cat( 3, wcoef( idxV1, idxH1, 3*(j-Jmin)+3 ), wcoef( idxV1, idxH2, 3*(j-Jmin)+3 ), wcoef( idxV2, idxH1, 3*(j-Jmin)+3 ), wcoef( idxV2, idxH2, 3*(j-Jmin)+3 ) );
			W = [C H; V D];

			%%%  horizontally  %%%
			coarse = W( :, 1:locLen/2, : );
			detail = W( :, (locLen/2 +1):locLen, : );
			% simultaneous upsampling and circular convolution
			oddSignals = zeros( locLen, locLen/2, 4 );
			evenSignals = zeros( locLen, locLen/2, 4 );
			for z=(2-odd):2:Lh
				oddSignals = oddSignals + h(z)*circshift( coarse, [0 (Ch-z)/2] ) + g(z)*circshift( detail, [0 (Ch-z)/2] );
			end
			for z=(1+odd):2:Lh
				evenSignals = evenSignals + h(z)*circshift( coarse, [0 (Ch-z-1)/2] ) + g(z)*circshift( detail, [0 (Ch-z-1)/2] );
			end
			% update signals on line
			W( :, 1:2:locLen, : ) = oddSignals;
			W( :, 2:2:locLen, : ) = evenSignals;

			%%%  vertically  %%%
			coarse = W( 1:locLen/2, :, : );
			detail = W( (locLen/2 +1):locLen, :, : );
			% simultaneous upsampling and circular convolution
			oddSignals = zeros( locLen/2, locLen, 4 );
			evenSignals = zeros( locLen/2, locLen, 4 );
			for z=(2-odd):2:Lh
				oddSignals = oddSignals + h(z)*circshift( coarse, (Ch-z)/2 ) + g(z)*circshift( detail, (Ch-z)/2 );
			end
			for z=(1+odd):2:Lh
				evenSignals = evenSignals + h(z)*circshift( coarse, (Ch-z-1)/2 ) + g(z)*circshift( detail, (Ch-z-1)/2 );
			end
			% update signals on line
			W( 1:2:locLen, :, : ) = oddSignals;
			W( 2:2:locLen, :, : ) = evenSignals;

			signals( subIdxV, subIdxH ) = ( W(:,:,1)+circshift(W(:,:,2), [0 1])+circshift(W(:,:,3), [1 0])+circshift(W(:,:,4), [1 1]) )/scaling;
		end, end %for tV,tH
	end
end 

end %fwt2Dti
