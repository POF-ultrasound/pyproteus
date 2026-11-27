function out = full_pieces_oracle(c, what)
% Test-only driver for tests/oracle/test_full_pieces.py: builds the MATLAB
% inputs (kWaveGrid, sparse weights, Grid/medium structs) from plain arrays
% and calls the oracle PROTEUS functions unchanged.
%
% what = 'update_source' : update_source.m only
%        'run_full'      : update_source.m, then run_simulation.m (k-Wave)
%        'self_sense'    : compute_self_sense_pressure.m (k-Wave per bubble)
% Indices in c are 1-based (MATLAB).

medium.sound_speed = c.sound_speed;
if isfield(c, 'density')
    medium.density     = c.density;
    medium.BonA        = c.BonA;
    medium.alpha_coeff = c.alpha_coeff;
    medium.alpha_power = c.alpha_power;
end

Grid.Nx = c.Nx; Grid.Ny = c.Ny; Grid.Nz = c.Nz;
Grid.dx = c.dx; Grid.dy = c.dx; Grid.dz = c.dx;

switch what
    case {'update_source', 'run_full'}
        W = sparse(c.wi, c.wj, c.wv, c.n_new, c.n_mb);
        source = [];
        if isfield(c, 'p_mask')
            source.p_mask = c.p_mask;
            source.p      = c.p;
        end
        if isfield(c, 'u_mask')
            source.u_mask = c.u_mask;
            source.ux     = c.ux;
        end
        source = update_source(source, c.mass_source, W, c.idx_new(:), ...
            Grid, medium);
        if strcmp(what, 'update_source')
            out.p_mask = double(source.p_mask);
            out.p      = source.p;
            out.p_class = class(source.p);
            return
        end
        kgrid = kWaveGrid(c.Nx, c.dx, c.Ny, c.dx, c.Nz, c.dx);
        kgrid.setTime(c.Nt, c.dt);
        sensor.mask   = logical(c.sensor_mask);
        sensor.record = {'p'};
        sensor_data = run_simulation(c.run_param, kgrid, medium, source, sensor);
        out.p = double(sensor_data.p);

    case 'self_sense'
        Grid.dt = c.dt;
        Grid.x = c.x(:)'; Grid.y = c.y(:)'; Grid.z = c.z(:)';
        Grid.full_size = c.full_size(:);
        Grid.sensor_on_grid = c.sensor_on_grid;
        kgrid = kWaveGrid(c.Nx, c.dx, c.Ny, c.dx, c.Nz, c.dx);
        kgrid.setTime(c.Nt, c.dt);
        out.p = compute_self_sense_pressure(kgrid, Grid, c.MB_idx(:), ...
            c.MB_points, c.mass_source, medium, c.N_sup, c.run_param);
        out.p_class = class(out.p);

    otherwise
        error('unknown case %s', what)
end
end
