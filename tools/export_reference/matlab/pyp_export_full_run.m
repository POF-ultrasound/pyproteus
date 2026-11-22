function [out, extra, settings, ground_truth] = pyp_export_full_run(caseName, gtSeed, rngSeed)
%PYP_EXPORT_FULL_RUN Run one full-run reference case through PROTEUS main_RF.
%
% [out, extra, settings, ground_truth] = PYP_EXPORT_FULL_RUN(caseName,
% gtSeed, rngSeed) builds the case settings (pyp_full_run_settings),
% writes them and synthetic ground-truth frames (seed gtSeed, PROTEUS
% write_synthetic_ground_truth) into the folders of path_setup() (scratch
% folders set up by tools/export_reference), sets rng(rngSeed,'twister')
% (define_medium draws a random scatterer background) and runs main_RF
% with the solver in the settings (3DC = real k-Wave C++ binary at
% PATHS.BinaryPath). Mirrors run_rf_simulation in PROTEUS tests/helpers.
%
%   out.RF{k}, out.Frame{k}  RF and Frame of Frame_k.mat (cell per pulse)
%   out.dt                   RF sampling interval
%   extra.medium, extra.vessel_grid, extra.Grid   content of medium.mat
%   settings                 the settings file main_RF loaded
%   ground_truth             FlowSimulationParameters and Frames{k}
% Part of pyproteus (MIT).

PATHS = path_setup();
S = pyp_full_run_settings(caseName, PATHS);
name = ['pyp_' caseName];

stl = fullfile(PATHS.GeometriesPath, S.Geometry.Folder, S.Geometry.STLfile);
if S.Geometry.EmbedVessel && ~isfile(stl)
    error('pyp_export_full_run:missingGeometry', ['%s not found. The flow ' ...
        'geometry STL files are not shipped with PROTEUS (Zenodo record ' ...
        '10573662): set PYPROTEUS_GEOMETRIES_PATH to a folder holding ' ...
        '%s/%s.'], stl, S.Geometry.Folder, S.Geometry.STLfile)
end

settingsFile = fullfile(PATHS.SettingsPath, [name '.mat']);
gtFolder     = fullfile(PATHS.GroundTruthPath, name);
resultDir    = fullfile(PATHS.ResultsPath, name);
for d = {gtFolder, resultDir}
    if isfolder(d{1})
        rmdir(d{1}, 's');
    end
end

write_settings(S, PATHS, settingsFile);
write_synthetic_ground_truth(gtFolder, S, gtSeed);

settings = load(settingsFile);
ground_truth.FlowSimulationParameters = ...
    getfield(load(fullfile(gtFolder,'FlowSimulationParameters.mat')), ...
    'FlowSimulationParameters'); %#ok<GFLD>
gtFiles = sort_by_name(dir(fullfile(gtFolder,'Frame_*.mat')));
ground_truth.Frames = cell(1, numel(gtFiles));
for k = 1:numel(gtFiles)
    F = load(fullfile(gtFolder, gtFiles(k).name), 'Frame');
    ground_truth.Frames{k} = F.Frame;
end

rng(rngSeed, 'twister');
main_RF(settingsFile, name, name);

files = sort_by_name(dir(fullfile(resultDir,'Frame_*.mat')));
if isempty(files)
    error('pyp_export_full_run:noOutput', 'main_RF wrote no frames in %s.', resultDir)
end
out.RF    = cell(1, numel(files));
out.Frame = cell(1, numel(files));
for k = 1:numel(files)
    F = load(fullfile(resultDir, files(k).name), 'RF', 'dt', 'Frame');
    out.RF{k}    = F.RF;
    out.Frame{k} = F.Frame;
    out.dt       = F.dt;
end
extra = load(fullfile(resultDir, 'medium.mat'), 'medium', 'vessel_grid', 'Grid');
end

function files = sort_by_name(files)
[~, order] = sort({files.name});
files = files(order);
end
