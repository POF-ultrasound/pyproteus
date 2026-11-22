function out = run_kwave_matlab(c, binary_dir, data_dir, data_name)
% Run one tracer-bullet case through MATLAB k-Wave's kspaceFirstOrder3DC
% (the real C++ binary), with the input options of PROTEUS' hybrid path
% (acoustic-module/run_simulation.m). Called from Python via the engine.
%
% c          struct from tools/tracer_bullet/kwave_tracer.py (arrays [ix,iy,iz];
%            source rows in find(mask) order, as PROTEUS builds them)
% binary_dir folder holding the binary (or the wrapper) kspaceFirstOrder-OMP
% data_dir   folder for k-Wave's HDF5 input/output files (kept: DeleteData=false)
% data_name  prefix for those files
%
% Returns the fields of sensor_data (c.record; rows in find(sensor.mask)
% order) as double.

kgrid = kWaveGrid(c.Nx, c.dx, c.Ny, c.dx, c.Nz, c.dx);
kgrid.dt = c.dt;
kgrid.Nt = c.Nt;

medium.sound_speed = c.sound_speed;
medium.density     = c.density;
medium.BonA        = c.BonA;
medium.alpha_coeff = c.alpha_coeff;
medium.alpha_power = c.alpha_power;
if isfield(c, 'alpha_mode')
    medium.alpha_mode = c.alpha_mode;  % as define_medium.m for |y - 1| < 0.1
end

% Pressure and/or velocity sources (full_simulator.m can have both: bubble
% pressure sources added by update_source.m to a dipole transducer source).
if isfield(c, 'p_mask')
    source.p_mask = logical(c.p_mask);
    source.p      = c.p;
end
if isfield(c, 'u_mask')
    source.u_mask = logical(c.u_mask);
    for f = {'ux', 'uy', 'uz'}
        if isfield(c, f{1})
            source.(f{1}) = c.(f{1});
        end
    end
end

sensor.mask   = logical(c.sensor_mask);
sensor.record = cellstr(c.record);

input_args = {...
    'PMLInside', false, ...
    'PMLAlpha',  c.pml_alpha, ...
    'PMLSize',   double(c.pml_size(:)'), ...
    'DataCast',  'single', ...
    'Smooth',    false, ...
    'BinaryPath', binary_dir, ...
    'DataPath',  data_dir, ...
    'DataName',  data_name, ...
    'DeleteData', false};
if isfield(c, 'num_threads')
    input_args = [input_args {'NumThreads', c.num_threads}];
end

sensor_data = kspaceFirstOrder3DC(kgrid, medium, source, sensor, input_args{:});
out = struct();
for f = fieldnames(sensor_data)'
    out.(f{1}) = double(sensor_data.(f{1}));
end
end
