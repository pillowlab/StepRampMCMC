function [latentMean,LatentDataHandler] = meanLatentsDB(indices,LatentDataHandler)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Information about what is stored/loaded is all in LatentDataHandler! Keep this object around.
%
% takes the mean of all latents


%checks data setup
if(LatentDataHandler.DataBlockNumber <= 0)
    error('Invalid data block');
end
if(LatentDataHandler.DataRowLength <= 0)
    error('Data storage not initialized.');
end



% get block number for each index
blockIdx = floor((indices-1)/LatentDataHandler.DataRowsPerBlock) + 1;

% save each block, starting with any rows in current block
blockIdxs = unique(blockIdx);


latentBlock = zeros(LatentDataHandler.DataRowLength,length(blockIdxs));
Ns = zeros(length(blockIdxs),1);

blockOrderToRun = [blockIdxs(blockIdxs == LatentDataHandler.DataBlockNumber) blockIdxs(blockIdxs ~= LatentDataHandler.DataBlockNumber)];
for ii = blockOrderToRun
    
    LatentDataHandler = loadLatentFileDB(ii,LatentDataHandler);
    
    indicesInBlock = find(blockIdx == ii);
    offset = (ii-1)*LatentDataHandler.DataRowsPerBlock;
     
    
    if(max(indices(indicesInBlock)-offset) > size(LatentDataHandler.DataBlock,2))
        error('Accessing data outside pre-initialized block');
    else
        latentBlock(:,blockIdxs == ii) = mean(LatentDataHandler.DataBlock(:,indices(indicesInBlock)-offset),2);
        Ns(blockIdxs == ii) = length(indicesInBlock);

        if(sum(LatentDataHandler.DataValidRows(indices(indicesInBlock)-offset) < 1) > 0)
            warning('Loading data from unitialized array index');
        end
    end
end


latentMean = latentBlock * (Ns./sum(Ns));