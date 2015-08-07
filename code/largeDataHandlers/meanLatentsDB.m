function [latentMean] = meanLatentsDB(indices)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Warning: These functions use global variables (not a smart choice)
%
% takes the mean of all latents

global DataRowLength DataRowsPerBlock DataBlockNumber DataBlock DataValidRows;

%checks data setup
if(DataBlockNumber <= 0)
    error('Invalid data block');
end
if(DataRowLength <= 0)
    error('Data storage not initialized.');
end



% get block number for each index
blockIdx = floor((indices-1)/DataRowsPerBlock) + 1;

% save each block, starting with any rows in current block
blockIdxs = unique(blockIdx);


latentBlock = zeros(DataRowLength,length(blockIdxs));
Ns = zeros(length(blockIdxs),1);

blockOrderToRun = [blockIdxs(blockIdxs == DataBlockNumber) blockIdxs(blockIdxs ~= DataBlockNumber)];
for ii = blockOrderToRun
    
    loadLatentFileDB(ii);
    
    indicesInBlock = find(blockIdx == ii);
    offset = (ii-1)*DataRowsPerBlock;
     
    
    if(max(indices(indicesInBlock)-offset) > size(DataBlock,2))
        error('Accessing data outside pre-initialized block');
    else
        latentBlock(:,blockIdxs == ii) = mean(DataBlock(:,indices(indicesInBlock)-offset),2);
        Ns(blockIdxs == ii) = length(indicesInBlock);

        if(sum(DataValidRows(indices(indicesInBlock)-offset) < 1) > 0)
            warning('Loading data from unitialized array index');
        end
    end
end


latentMean = latentBlock * (Ns./sum(Ns));