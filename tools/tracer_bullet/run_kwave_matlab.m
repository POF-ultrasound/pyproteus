function p = run_kwave_matlab(c, binary_dir, data_dir, data_name)
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
% Returns sensor_data.p (rows in find(sensor.mask) order) as double.

kgrid = kWaveGrid(c.Nx, c.dx, c.Ny, c.dx, c.Nz, c.dx);
kgrid.dt = c.dt;
kgrid.Nt = c.Nt;

medium.sound_speed = c.sound_speed;
medium.density     = c.density;
medium.BonA        = c.BonA;
medium.alpha_coeff = c.alpha_coeff;
medium.alpha_power = c.alpha_power;

if isfield(c, 'p_mask')
    source.p_mask = logical(c.p_mask);
    source.p      = c.p;
else
    source.u_mask = logical(c.u_mask);
    source.ux     = c.ux;
    source.uy     = c.uy;
end

sensor.mask   = logical(c.sensor_mask);
sensor.record = {'p'};

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

sensor_data = kspaceFirstOrder3DC(kgrid, medium, source, sensor, input_args{:});
p = double(sensor_data.p);
end
