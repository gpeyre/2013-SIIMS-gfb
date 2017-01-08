function snr = SNRdB( xOrig, xRecov )
%
%	 snr = SNRdB( xOrig, xRecov )
%
% Compute signal-to-noise ratio in decibel
%
%	snr = 20*log10( ||xOrig|| / ||xRecov - xOrig|| )
%
% where
%	xOrig is the original or clean signal
% 	xRecov is the recovered or noisy signal
%
% Hugo Raguet 2011

snr = 20*log10( norm( xOrig(:) ) / norm( xRecov(:) - xOrig(:) ) );

end %SNRdB
