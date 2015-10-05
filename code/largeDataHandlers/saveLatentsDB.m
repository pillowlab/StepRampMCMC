function [LatentDataHandler] = saveLatentsDB(latentBlock,indices,LatentDataHandler)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Information about what is stored/loaded is all in LatentDataHandler! Keep this object around.


%checks data setup
if(LatentDataHandler.DataRowLength <= 0)
    LatentDataHandler = resetLatentsDB(size(latentBlock,1),LatentDataHandler);
elseif(nargin > 1 && LatentDataHandler.DataRowLength ~= size(latentBlock,1))
    error('Data size mismatch.');
end

if(nargin == 0)
    DataBlock     = LatentDataHandler.DataBlock; %#ok<NASGU>
    DataValidRows = LatentDataHandler.DataValidRows; %#ok<NASGU>
    save([LatentDataHandler.DataFolder '/dataBlock' num2str(LatentDataHandler.DataBlockNumber) '.mat'],'-v7.3','DataBlock','DataValidRows');
    LatentDataHandler.DataChanged = 0;
    return;
end


% get block number for each index
blockIdx = floor((indices-1)/LatentDataHandler.DataRowsPerBlock) + 1;

% save each block, starting with any rows in current block
blockIdxs = unique(blockIdx);

blockOrder = [blockIdxs(blockIdxs == LatentDataHandler.DataBlockNumber) blockIdxs(blockIdxs ~= LatentDataHandler.DataBlockNumber)];
for ii = blockOrder
    
    LatentDataHandler = loadLatentFileDB(ii,LatentDataHandler);
    indicesInBlock = find(blockIdx == ii);
    offset = (ii-1)*LatentDataHandler.DataRowsPerBlock;
        
    LatentDataHandler.DataBlock(:,indices(indicesInBlock)-offset) = latentBlock(:,indicesInBlock);
    LatentDataHandler.DataValidRows(indices(indicesInBlock)-offset) = 1;

    LatentDataHandler.DataChanged = 1;
end
