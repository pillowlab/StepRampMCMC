function [LatentDataHandler] = loadLatentFileDB(blockNum,LatentDataHandler)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Information about what is stored/loaded is all in LatentDataHandler! Keep this object around.
%
% loads up one block of latents from a file

%checks data setup
if(blockNum <= 0)
    error('Invalid data block');
end
if(LatentDataHandler.DataRowLength <= 0)
    error('Data storage not initialized.');
end

if(blockNum == LatentDataHandler.DataBlockNumber)
    %no need to reload current block
    return;
else
    if(LatentDataHandler.DataChanged > 0)
        %save current block
        DataBlock     = LatentDataHandler.DataBlock; %#ok<NASGU>
        DataValidRows = LatentDataHandler.DataValidRows; %#ok<NASGU>
        save([LatentDataHandler.DataFolder '/dataBlock' num2str(LatentDataHandler.DataBlockNumber) '.mat'],'-v7.3','DataBlock','DataValidRows');
    end

    LatentDataHandler.DataBlockNumber = blockNum;
    if(~exist([LatentDataHandler.DataFolder '/dataBlock' num2str(blockNum) '.mat'],'file'))
        %create new block
        LatentDataHandler.DataBlock = zeros(LatentDataHandler.DataRowLength, LatentDataHandler.DataRowsPerBlock);
        LatentDataHandler.DataValidRows = zeros(LatentDataHandler.DataRowsPerBlock,1);
        LatentDataHandler.DataChanged = 1;
    else
        %load old block
        DB = load([LatentDataHandler.DataFolder '/dataBlock' num2str(LatentDataHandler.DataBlockNumber) '.mat'],'DataBlock','DataValidRows');
        LatentDataHandler.DataChanged = 0;
        LatentDataHandler.DataBlock     = DB.DataBlock; 
        LatentDataHandler.DataValidRows = DB.DataValidRows;
    end

end