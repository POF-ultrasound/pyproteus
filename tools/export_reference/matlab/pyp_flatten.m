function records = pyp_flatten(value)
%PYP_FLATTEN Flatten a MATLAB value into records for the reference export.
%
% records = PYP_FLATTEN(value) returns a 1xN cell of scalar structs, one
% per node of value in depth-first pre-order, with fields
%   path    1xK cell of char: field names, or element indices '0','1',...
%           (0-based, column-major linear order) of struct arrays and cells
%   class   'struct', 'cell', 'char', 'string', 'function_handle', or a
%           numeric/logical class name
%   size    size(node) (double row vector)
%   fields  1xF cell of field names (struct nodes), else {}
%   sparse  true if the node was a sparse matrix (data is then full)
%   data    numeric/logical array (MATLAB size kept), char row for text
%           nodes, [] for struct/cell nodes
%
% Every record carries its exact class and size, so the Python side
% (tools/export_reference/records.py) rebuilds values the MATLAB Engine
% would otherwise blur (1x1 arrays become Python scalars, cell and struct
% array shapes are lost). Part of pyproteus (MIT); not PROTEUS code.

records = walk(value, {}, {});
end


function records = walk(v, path, records)
r = struct('path', {path}, 'class', class(v), 'size', size(v), ...
    'fields', {{}}, 'sparse', false, 'data', []);

if isstruct(v)
    r.class  = 'struct';
    r.fields = reshape(fieldnames(v), 1, []);
    records{end+1} = r;
    if isequal(size(v), [1 1])
        for f = r.fields
            records = walk(v.(f{1}), [path f], records);
        end
    else
        for k = 1:numel(v)
            records = walk(v(k), [path {sprintf('%d', k-1)}], records);
        end
    end
elseif iscell(v)
    records{end+1} = r;
    for k = 1:numel(v)
        records = walk(v{k}, [path {sprintf('%d', k-1)}], records);
    end
elseif ischar(v)
    if ~(isempty(v) || isrow(v))
        error('pyp_flatten:charMatrix', ...
            'Char matrix at %s: only char row vectors are supported (use cellstr).', ...
            strjoin(path, '/'))
    end
    r.data = reshape(v, 1, []);
    records{end+1} = r;
elseif isstring(v)
    if ~isscalar(v) || ismissing(v)
        error('pyp_flatten:stringArray', ...
            'String array or missing string at %s: only string scalars are supported.', ...
            strjoin(path, '/'))
    end
    r.data = char(v);
    records{end+1} = r;
elseif isa(v, 'function_handle')
    r.data = func2str(v);
    records{end+1} = r;
elseif isnumeric(v) || islogical(v)
    if issparse(v)
        r.sparse = true;
        v = full(v);
    end
    r.data = v;
    records{end+1} = r;
else
    error('pyp_flatten:unsupported', ...
        'Cannot export a value of class %s at %s; convert it in the export spec.', ...
        class(v), strjoin(path, '/'))
end
end
