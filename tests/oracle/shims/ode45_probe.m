function out = ode45_probe(rhs, tspan, y0, optcell, p)
% Run MATLAB ode45 on a named test right-hand side and log every call.
% out.calls_t/calls_y/calls_f: every odefun call (t, y, f) in call order;
% out.t, out.y: [t, y] = ode45(...); out.sol_x/sol_y/stats: from sol = ode45(...).
% Used by tests/oracle/test_ode45.py (black-box probing, no solver source).
if nargin < 5, p = struct(); end
switch rhs
    case 'decay',  f = @(t,y) -p.lam(:).*y;
    case 'rot',    f = @(t,y) [-p.w*y(2); p.w*y(1)];
    case 'vdp',    f = @(t,y) [y(2); p.mu*(1-y(1)*y(1))*y(2)-y(1)];
    case 'scales', f = @(t,y) [-1e3*(y(1)-1+t*t); 1e-6*y(1)-0.1*y(2); y(1)*y(2)*1e4];
    case 'rp',     f = @(t,y) ode45_rp_rhs(t, y, p);
    otherwise, error('ode45_probe:rhs', 'unknown rhs %s', rhs);
end
opts = odeset(optcell{:});
log_t = zeros(1, 0); log_y = zeros(numel(y0), 0); log_f = zeros(numel(y0), 0);
    function dy = logged(t, y)
        dy = f(t, y);
        log_t(end+1) = t;
        log_y(:, end+1) = y;
        log_f(:, end+1) = dy;
    end
[t, y] = ode45(@logged, tspan, y0, opts);
out.t = t; out.y = y;
out.calls_t = log_t(:); out.calls_y = log_y.'; out.calls_f = log_f.';
sol = ode45(f, tspan, y0, opts);
out.sol_x = sol.x(:); out.sol_y = sol.y.';
out.stats = [sol.stats.nsteps, sol.stats.nfailed, sol.stats.nfevals];
end
