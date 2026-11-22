function pyp_write_tiny_geometry(geometriesPath)
%PYP_WRITE_TINY_GEOMETRY Synthetic 2 mm flow geometry 'tiny_box' (no vessel, no STL).
%
% Same content as write_tiny_geometry in PROTEUS tests/ProteusTestEnvironment.m
% (oracle commit 8d37938), which the oracle's make_tiny_settings expects in
% PATHS.GeometriesPath. Ported from PROTEUS (MIT).

folder = fullfile(geometriesPath,'tiny_box');
if ~isfolder(folder)
    mkdir(folder);
end
BoundingBox = struct('Xmin',0,'Xmax',2e-3,'Ymin',-1e-3,'Ymax',1e-3, ...
    'Zmin',-1e-3,'Zmax',1e-3); %#ok<NASGU>
EmbedVessel = false; %#ok<NASGU>
Rotation = eye(3); %#ok<NASGU>
STLfile = 'none.stl'; %#ok<NASGU>
STLunit = 1; %#ok<NASGU>
Visualization = struct('Image','none.png','Fraction',1); %#ok<NASGU>
save(fullfile(folder,'GeometryProperties.mat'), 'BoundingBox', ...
    'EmbedVessel','Rotation','STLfile','STLunit','Visualization');
end
