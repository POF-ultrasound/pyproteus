function RF = transmit_only_rf(settingsfile, seed)
% Test-only driver for tests/oracle/test_main_rf.py: the element RF of the
% transducer's own transmit, without bubbles, for each pulse of the sequence.
% Same steps as main_RF.m up to the first k-Wave run (oracle functions,
% unchanged), with the transducer as the only sensor and the round-trip
% record length of the hybrid path. rng(seed,'twister') before
% define_medium, as before main_RF.

load(settingsfile,'Acquisition','Geometry','Medium', ...
    'SimulationParameters', 'Transducer', 'Transmit')
run_param = sim_setup(SimulationParameters);
if isfield(SimulationParameters,'TransducerOnGrid')
    Transducer.OnGrid = SimulationParameters.TransducerOnGrid;
else
    Transducer.OnGrid = false;
end
if isfield(SimulationParameters,'IntegrationDensity')
    Transducer.IntegrationDensity = SimulationParameters.IntegrationDensity;
else
    Transducer.IntegrationDensity = 1;
end
Geometry.GeometriesPath = run_param.GeometriesPath;

Grid = define_grid(SimulationParameters, Geometry);
kgrid = kWaveGrid(Grid.Nx, Grid.dx, Grid.Ny, Grid.dy, Grid.Nz, Grid.dz);
kgrid.dt = Grid.dt;
rng(seed, 'twister');
medium = define_medium(Grid, Medium, Geometry, Transducer);

Transducer = get_transducer_integration_points(Transducer, Grid);
Transducer = get_transducer_integration_delays(Transducer, Medium);
run_param = compute_travel_times(run_param, Geometry,Medium,Transducer,Transmit);
run_param.PML = Grid.PML;
kgrid.Nt = floor(run_param.tr(1) / kgrid.dt) + 1;
Transmit = preprocess_transmit(Transmit,Medium,kgrid);

switch Acquisition.PulsingScheme
    case 'Amplitude modulation'
        sequence = {'odd' 'even' 'all'};
    case 'Pulse inversion'
        sequence = {'plus' 'minus'};
    case 'Standard'
        sequence = {'pulse'};
    case 'Amplitude modulation with pulse inversion'
        sequence = {'odd' 'even' 'minus'};
end

[sensor_transducer, sensor_weights] = define_sensor_transducer(Transducer, Grid);
mask_idx_trans = find(logical(sensor_transducer.mask));
kgrid.Nt = floor(run_param.tr(3) / kgrid.dt) + 1;

RF = cell(1, numel(sequence));
for k = 1:numel(sequence)
    Transmit.SeqPulse = sequence{k};
    source = define_source_transducer(Transducer, Transmit, Medium, Grid, ...
        transpose(sensor_weights), mask_idx_trans);
    sensor_data = run_simulation(run_param, kgrid, medium, source, sensor_transducer);
    RF{k} = compute_RF_data(Transducer, sensor_data, sensor_weights, Grid, run_param);
end
end
