function [flags, times, sizes, t, y] = ode45_outputfcn_probe(tspan, y0, optcell, stop_at)
% Record MATLAB's OutputFcn call pattern for ode45 on y' = -y (componentwise).
% flags{k}, times{k} (row), sizes(k,:) = size(y) of the k-th OutputFcn call.
% The function returns true (stop) on its stop_at-th call with flag ''
% (0 = never). Used by tests/oracle/test_ode45.py (black-box probing).
flags = {}; times = {}; sizes = zeros(0, 2);
ncalls = 0;
    function status = ofcn(tt, yy, flag)
        flags{end+1} = flag;
        times{end+1} = reshape(tt, 1, []);
        sizes(end+1, :) = size(yy);
        status = false;
        if isempty(flag)
            ncalls = ncalls + 1;
            status = (stop_at > 0 && ncalls == stop_at);
        end
    end
opts = odeset(optcell{:}, 'OutputFcn', @ofcn);
[t, y] = ode45(@(t,y) -y, tspan, y0, opts);
end
