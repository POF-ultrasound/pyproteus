function dvec = ode45_rp_rhs(tau, vec, p)
% Rayleigh-Plesset-like right-hand side for the pyproteus ode45 oracle tests.
% Same structure as PROTEUS simple_RP (Marmottant-type, nondimensional,
% stacked [x; xdot] per bubble) but with a constant surface tension and a
% polynomial driving pulse, so that MATLAB and Python evaluate it with the
% same IEEE operations (no pchip). Mirrors tests/oracle/test_ode45.py::rp_rhs.
N = numel(p.R0);
vec = reshape(vec, [2, N]);
x = vec(1,:);
xdot = vec(2,:);
% Driving pressure: window u^2 (1-u)^2 times Chebyshev T_n(2u-1), u = tau/tau1.
u = tau / p.tau1;
if u > 0 && u < 1
    z = 2*u - 1;
    Tm = 1; Tc = z;
    for k = 2:p.n
        Tn = 2*z*Tc - Tm;
        Tm = Tc; Tc = Tn;
    end
    P = p.Pa * (16*u*u*(1-u)*(1-u)) * Tc;
else
    P = 0;
end
xdotdot = 1./(1+x).*( ...
    -3/2.*xdot.*xdot ...
    + p.P0/p.rho./(p.R0.*p.R0)*(p.T*p.T).* ...
    ( ...
    (1 + 2*p.sig./(p.R0*p.P0)).*(1+x).^(-3*p.kappa).* ...
    (1-3*p.kappa/p.c/p.T.*p.R0.*xdot) ...
    - 1 - 2*p.sig./(p.R0.*p.P0).*(1+x).^(-1) ...
    - 4*p.nu/p.P0/p.T.*xdot./(1+x) ...
    - 4*p.Ks/p.P0./p.R0/p.T.*xdot./((1+x).*(1+x)) ...
    - P/p.P0 ...
    ) ...
    );
dvec = [xdot; xdotdot];
dvec = reshape(dvec, [2*N, 1]);
end
