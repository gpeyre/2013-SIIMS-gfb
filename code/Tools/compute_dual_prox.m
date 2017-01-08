function ProxFS = compute_dual_prox(ProxF)

% compute_dual_prox - compute the proximal operator of the dual
%
%   ProxFS = compute_dual_prox(ProxF);
%
%   Compute the proximity callback of the dual function given the proximity
%   callback of the primal function.
%
%   Make use of Moreau's identity:
%       x = Prox_{tau*F^*}(x) + tau*Prox_{F/tau}(x/tau)
%   which implies
%       ProxFS(x,tau) = x - tau*ProxF(x/tau,1/tau);
%
%   Copyright (c) 2010 Gabriel Peyre

ProxFS = @(y,ga)y-ga*ProxF(y/ga,1/ga);
