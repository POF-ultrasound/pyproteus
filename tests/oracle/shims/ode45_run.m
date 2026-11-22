function [t, y] = ode45_run(rhs, tspan, y0, optcell, p)
% Solve a named test problem with MATLAB ode45 and return [t, y].
% Used by tests/oracle/test_ode45.py to compare solutions within a
% tolerance (no solver internals are recorded).
if nargin < 5, p = struct(); end
switch rhs
    case 'decay',  f = @(t,y) -p.lam(:).*y;
    case 'rot',    f = @(t,y) [-p.w*y(2); p.w*y(1)];
    case 'vdp',    f = @(t,y) [y(2); p.mu*(1-y(1)^2)*y(2)-y(1)];
    case 'scales', f = @(t,y) [-1e3*(y(1)-cos(t)); 1e-6*y(1)-0.1*y(2); y(1)*y(2)*1e4];
    case 'rp',     f = @(t,y) ode45_rp_rhs(t, y, p);
    otherwise, error('ode45_run:rhs', 'unknown rhs %s', rhs);
end
[t, y] = ode45(f, tspan, y0, odeset(optcell{:}));
end
