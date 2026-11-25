function y = boolean(x)
% Test-only shim: PROTEUS compress_bubble.m calls boolean(), which needs a
% toolbox that is not installed here; boolean(x) is logical(x) (documented).
y = logical(x);
end
