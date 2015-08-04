function [] = loadLatentFileDB(blockNum)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Warning: These functions use global variables (not a smart choice)
%
% loads up one block of latents from a file
global DataRowLength DataRowsPerBlock DataBlockNumber DataBlock DataChanged DataValidRows DataFolder;

%checks data setup
if(blockNum <= 0)
    error('Invalid data block');
end
if(DataRowLength <= 0)
    error('Data storage not initialized.');
end

if(blockNum == DataBlockNumber)
    %no need to reload current block
    return;
else
    if(DataChanged > 0)
        %save current block
        save([DataFolder '/dataBlock' num2str(DataBlockNumber) '.mat'],'-v7.3','DataBlock','DataValidRows');
    end

    DataBlockNumber = blockNum;
    if(~exist([DataFolder '/dataBlock' num2str(blockNum) '.mat'],'file'))
        %create new block
        DataBlock = zeros(DataRowLength, DataRowsPerBlock);
        DataValidRows = zeros(DataRowsPerBlock,1);
        DataChanged = 1;
    else
        %load old block
        load([DataFolder '/dataBlock' num2str(DataBlockNumber) '.mat'],'DataBlock','DataValidRows');
        DataChanged = 0;
    end

end