function R = q100_point_scatterer(oracle_copy, kwave_path, binary_dir, cases)
%Q100_POINT_SCATTERER Q-100 investigation: where does MATLAB PROTEUS put a
%point scatterer in time, relative to the DAS time axis?
%
% Runs main_RF (hybrid path, k-Wave C++ binary per D-10) on a COPY of the
% oracle PROTEUS (oracle_copy; never the oracle itself) for one microbubble
% at a known position, then compares, element by element, the envelope
% peak of the RF data on the DAS time axis (initialize_das_parameters:
% time to peak I/Fs, lens and k-Wave corrections) with the geometric
% two-way travel time. Residual 0 = the DAS time axis is right; residual
% -1/Transmit.SamplingRate = the 1-based time to peak is an off-by-one.
%
% The bubble is small (resonance far above the transmit band) and driven
% at low pressure, so it scatters linearly and nearly in phase (radiated
% pressure ~ second time derivative of a symmetric pulse: envelope peak
% unchanged). Medium: homogeneous, no inhomogeneity, no absorption (the
% k-space scheme is then free of numerical dispersion).
%
% cases: struct array with fields name, depth [m], radius [m], hybrid,
% fs_transmit [Hz] (Transmit/Transducer rate), ppw.
%
% Investigation only (tools/investigations/); not part of the package.

setenv('PROTEUS_KWAVE_PATH', kwave_path);
addpath(fullfile(oracle_copy,'tests'));
env = ProteusTestEnvironment;
env.setup();
cleanup = onCleanup(@() env.teardown()); %#ok<NASGU>
P = env.PATHS;

% Use the real k-Wave C++ solver: drop the test double, point BinaryPath
% at the wrapper of k-wave-python's v1.4.1 OMP binary.
shim = P.kWavePath;
delete(fullfile(shim,'kspaceFirstOrder3DC.m'));
copyfile(fullfile(binary_dir,'kspaceFirstOrder-OMP'), P.BinaryPath);
fileattrib(fullfile(P.BinaryPath,'kspaceFirstOrder-OMP'),'+x');
rehash path;
assert(contains(which('kspaceFirstOrder3DC'), kwave_path), ...
    'kspaceFirstOrder3DC must be the real k-Wave function');

% A box geometry deep enough for the scatterer:
gfolder = fullfile(P.GeometriesPath,'q100_box');
mkdir(gfolder);
BoundingBox = struct('Xmin',0,'Xmax',8e-3,'Ymin',-1.5e-3,'Ymax',1.5e-3, ...
    'Zmin',-1e-3,'Zmax',1e-3); %#ok<NASGU>
EmbedVessel = false; Rotation = eye(3); STLfile = 'none.stl'; %#ok<NASGU>
STLunit = 1; Visualization = struct('Image','none.png','Fraction',1); %#ok<NASGU>
save(fullfile(gfolder,'GeometryProperties.mat'),'BoundingBox', ...
    'EmbedVessel','Rotation','STLfile','STLunit','Visualization');

R = struct([]);
for k = 1:numel(cases)
    C = cases(k);
    S = make_default_settings(P, 'q100_box', 'reference');
    S.Geometry.startDepth = 1e-3;
    S.Transducer.NumberOfElements = 16;
    S.Transducer.SamplingRate = C.fs_transmit;
    S.Medium.Inhomogeneity = 0;
    S.Medium.AttenuationA  = 0;
    S.Medium = compute_sound_speed_extrema(S.Medium);
    S.Transmit = reset_transmit(S.Transducer);
    S.Transmit.SamplingRate = C.fs_transmit;
    S.Transmit.AcousticPressure = 10e3;
    S.Transmit = get_voltage_signal(S.Transmit);
    S.Transmit = get_pressure_signal(S.Transmit, S.Transducer);
    S.SimulationParameters.HybridSimulation = C.hybrid;
    S.SimulationParameters.PointsPerWavelength = C.ppw;
    S.Microbubble.Number = 1;
    S.Acquisition.NumberOfFrames = 1;
    S.Acquisition.PulsingScheme = 'Standard';
    S.Acquisition.NumberOfPulses = 1;
    S = update_derived_settings(S);

    % One bubble on the array axis at the requested depth (transducer
    % coordinates; load_microbubbles maps Points - BB.Center + Center):
    target = [C.depth 0 0];
    BB = S.Geometry.BoundingBox;
    gt = fullfile(P.GroundTruthPath, C.name);
    mkdir(gt);
    FlowSimulationParameters = struct('TimeBtwPulse', ...
        S.Acquisition.TimeBetweenPulses, 'FrameRate', S.Acquisition.FrameRate, ...
        'NBPulses', 1, 'NMicrobubbles', 1, 'NumberOfFrames', 1, ...
        'Microbubble', S.Microbubble); %#ok<NASGU>
    save(fullfile(gt,'FlowSimulationParameters.mat'),'FlowSimulationParameters');
    Frame.Pulse1.Points = target - transpose(S.Geometry.Center) + transpose(BB.Center);
    Frame.Pulse1.Velocity = [0 0 0];
    Frame.Pulse1.Radius = C.radius;
    Frame.Pulse1.StreamNumber = 1; %#ok<STRNU>
    save(fullfile(gt,'Frame_1.mat'),'Frame');

    settingsFile = fullfile(P.SettingsPath,[C.name '.mat']);
    write_settings(S, P, settingsFile);
    rng(0,'twister');
    main_RF(settingsFile, C.name, C.name);
    F = load(fullfile(P.ResultsPath, C.name, 'Frame_1.mat'),'RF');
    RF = double(F.RF{1});                       % Nelem-by-Nt
    [Nelem, Nt] = size(RF);
    Sx = load(settingsFile);                    % settings as DAS reads them
    [recon, tr] = initialize_das_parameters(Sx.Medium, ...
        Sx.SimulationParameters, Sx.Transducer, Sx.Transmit, Sx.Geometry, ...
        Nelem, Nt);
    c = recon.c; Fs = recon.Fs;

    % Envelope peak per element, parabolic sub-sample refinement:
    env_ = abs(hilbert(RF.'));                  % Nt-by-Nelem
    t_peak = zeros(1,Nelem);
    for e = 1:Nelem
        [~,i] = max(env_(:,e));
        i = min(max(i,2),Nt-1);
        y = env_(i-1:i+1,e);
        d = 0.5*(y(1)-y(3))/(y(1)-2*y(2)+y(3));
        t_peak(e) = recon.t(1) + (i-1+d)/Fs;
    end
    % Geometric two-way time (plane wave, no steering: d_in = depth):
    t_geom = (target(1) + hypot(target(1) - tr.x_el, target(2) - tr.y_el))/c;

    V = conv(Sx.Transmit.PressureSignal, Sx.Transducer.ReceiveImpulseResponse);
    [~,I] = max(abs(hilbert(V)));
    R(k).name = C.name;
    R(k).hybrid = C.hybrid;
    R(k).depth = target(1);
    R(k).radius = C.radius;
    R(k).fs_transmit = Sx.Transmit.SamplingRate;
    R(k).fs_transducer = Sx.Transducer.SamplingRate;
    R(k).fs_sim = Fs;
    R(k).dx = Sx.SimulationParameters.GridSize;
    R(k).I = I;
    R(k).residual = t_peak - t_geom;            % [s], per element
    R(k).residual_mean = mean(R(k).residual);
    R(k).residual_std = std(R(k).residual);
    R(k).in_tx_samples = R(k).residual_mean*Sx.Transmit.SamplingRate;
    R(k).in_sim_samples = R(k).residual_mean*Fs;
    fprintf('%s: residual %.3g ns (std %.2g ns) = %.2f transmit samples = %.3f sim samples\n', ...
        C.name, 1e9*R(k).residual_mean, 1e9*R(k).residual_std, ...
        R(k).in_tx_samples, R(k).in_sim_samples);
end
end
