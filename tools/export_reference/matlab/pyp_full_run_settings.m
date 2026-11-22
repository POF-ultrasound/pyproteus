function S = pyp_full_run_settings(caseName, PATHS)
%PYP_FULL_RUN_SETTINGS Settings of the full-run reference cases.
%
% S = PYP_FULL_RUN_SETTINGS(caseName, PATHS) returns the seven PROTEUS
% settings structs (fields Microbubble, Medium, Acquisition, Transducer,
% Transmit, SimulationParameters, Geometry) for one case, built headless
% with the oracle's own test helpers (PROTEUS tests/helpers, on the path
% during a full-run export). Part of pyproteus (MIT).
%
%   'tiny'    = PROTEUS regression case rf_hybrid_standard
%               (make_tiny_settings: 'tiny_box' geometry, 8 elements,
%               4 points per wavelength, 4 bubbles, 2 frames). Seconds.
%   'small'   tiny_box, 16 elements, 6 points per wavelength, 12 bubbles,
%             3 frames, pulse inversion (2 pulses). Minutes on a laptop.
%   'medium'  default MainGUI settings on the shipped 'renal_tree'
%             geometry, 2 frames. GPU workstation (M3 "medium case on GPU").

switch caseName
    case 'tiny'
        S = make_tiny_settings(PATHS);
    case 'small'
        S = make_tiny_settings(PATHS);
        S.Transducer.NumberOfElements = 16;
        S.Transmit = reset_transmit(S.Transducer);
        S.SimulationParameters.PointsPerWavelength = 6;
        S.Microbubble.Number = 12;
        S.Acquisition.NumberOfFrames = 3;
        S.Acquisition.PulsingScheme  = 'Pulse inversion';
        S.Acquisition.NumberOfPulses = 2;
        S = update_derived_settings(S);
    case 'medium'
        S = make_default_settings(PATHS, 'renal_tree');
        S.Acquisition.NumberOfFrames = 2;
        S = update_derived_settings(S);
    otherwise
        error('pyp_full_run_settings:unknownCase', 'Unknown case %s.', caseName)
end
end
