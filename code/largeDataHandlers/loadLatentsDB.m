function [latentBlock] = loadLatentsDB(indices)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Warning: These functions use global variables (not a smart choice)
%
% loads up one block of latents
global DataRowLength DataRowsPerBlock DataBlockNumber DataBlock DataValidRows;

%checks data setup
if(DataBlockNumber <= 0)
    error('Invalid data block');
end
if(DataRowLength <= 0)
    error('Data storage not initialized.');
end

latentBlock = zeros(DataRowLength,length(indices));


% get block number for each index
blockIdx = floor((indices-1)/DataRowsPerBlock) + 1;

% save each block, starting with any rows in current block
blockIdxs = unique(blockIdx);

for ii = [blockIdxs(blockIdxs == DataBlockNumber) blockIdxs(blockIdxs ~= DataBlockNumber)]
    
    loadLatentFileDB(ii);
    
    indicesInBlock = find(blockIdx == ii);
    offset = (ii-1)*DataRowsPerBlock;
    
    if(max(indices(indicesInBlock)-offset) > size(DataBlock,2))
        error('Accessing data outside pre-initialized block');
    else
        latentBlock(:,indicesInBlock) = DataBlock(:,indices(indicesInBlock)-offset);
        if(sum(DataValidRows(indices(indicesInBlock)-offset) < 1) > 0)
            warning('Loading data from unitialized array index');
        end
    end
end
