function [LatentDataHandler] = clearLatentsDB(LatentDataHandler)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Information about what is stored/loaded is all in LatentDataHandler! Keep this object around.
%
% clears all blocks of latents


fileList = dir(LatentDataHandler.DataFolder);

for ii = 1:length(fileList)
    if(~fileList(ii).isdir)
        delete([LatentDataHandler.DataFolder '/' fileList(ii).name]);
    end
end

LatentDataHandler = struct();