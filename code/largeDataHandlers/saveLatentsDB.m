function [] = saveLatentsDB(latentBlock,indices)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Warning: These functions use global variables (not a smart choice)
%
% saves a block of latents to the drive (so that another block can be brought into memory)

global DataRowLength DataRowsPerBlock DataBlockNumber DataBlock DataChanged DataValidRows DataFolder;

%checks data setup
if(DataRowLength <= 0)
    resetLatentsDB(size(latentBlock,1));
elseif(nargin > 1 && DataRowLength ~= size(latentBlock,1))
    error('Data size mismatch.');
end

if(nargin == 0)
    save([DataFolder '/dataBlock' num2str(DataBlockNumber) '.mat'],'-v7.3','DataBlock','DataValidRows');
    DataChanged = 0;
    return;
end


% get block number for each index
blockIdx = floor((indices-1)/DataRowsPerBlock) + 1;

% save each block, starting with any rows in current block
blockIdxs = unique(blockIdx);

for ii = [blockIdxs(blockIdxs == DataBlockNumber) blockIdxs(blockIdxs ~= DataBlockNumber)]
    
    loadLatentFileDB(ii);
    indicesInBlock = find(blockIdx == ii);
    offset = (ii-1)*DataRowsPerBlock;
        
    DataBlock(:,indices(indicesInBlock)-offset) = latentBlock(:,indicesInBlock);
    DataValidRows(indices(indicesInBlock)-offset) = 1;

    DataChanged = 1;
end
