function [] = clearLatentsDB()
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Warning: These functions use global variables (not a smart choice)
%
% clears all blocks of latents

global DataFolder;

fileList = dir(DataFolder);

for ii = 1:length(fileList)
    if(~fileList(ii).isdir)
        delete([DataFolder '/' fileList(ii).name]);
    end
end

clear -global DataRowLength DataRowsPerBlock DataBlockNumber DataBlock DataChanged DataValidRows;