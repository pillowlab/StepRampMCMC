function [latentBlock,LatentDataHandler] = loadLatentsDB(indices,LatentDataHandler)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Information about what is stored/loaded is all in LatentDataHandler! Keep this object around.
%
% loads up one block of latents

%checks data setup
if(LatentDataHandler.DataBlockNumber <= 0)
    error('Invalid data block');
end
if(LatentDataHandler.DataRowLength <= 0)
    error('Data storage not initialized.');
end

latentBlock = zeros(LatentDataHandler.DataRowLength,length(indices));


% get block number for each index
blockIdx = floor((indices-1)/LatentDataHandler.DataRowsPerBlock) + 1;

% save each block, starting with any rows in current block
blockIdxs = unique(blockIdx);

blockOrder = [blockIdxs(blockIdxs == LatentDataHandler.DataBlockNumber) blockIdxs(blockIdxs ~= LatentDataHandler.DataBlockNumber)];
for ii = blockOrder
    
    LatentDataHandler = loadLatentFileDB(ii,LatentDataHandler);
    
    indicesInBlock = find(blockIdx == ii);
    offset = (ii-1)*LatentDataHandler.DataRowsPerBlock;
    
    if(max(indices(indicesInBlock)-offset) > size(LatentDataHandler.DataBlock,2))
        error('Accessing data outside pre-initialized block');
    else
        latentBlock(:,indicesInBlock) = LatentDataHandler.DataBlock(:,indices(indicesInBlock)-offset);
        if(sum(LatentDataHandler.DataValidRows(indices(indicesInBlock)-offset) < 1) > 0)
            warning('Loading data from unitialized array index');
        end
    end
end
