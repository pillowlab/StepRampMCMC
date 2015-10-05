function [LatentDataHandler] = resetLatentsDB(rowLength,rowsToInitialize,LatentDataHandler)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Information about what is stored/loaded is all in LatentDataHandler! Keep this object around.
%
% resets the blocks of latents

if(nargin < 3)
    LatentDataHandler = struct();
end

if(~isfield(LatentDataHandler,'DataFolder') || isempty(LatentDataHandler.DataFolder))
    LatentDataHandler.DataFolder = './temp/';
end

if(~exist(LatentDataHandler.DataFolder,'file'))
    mkdir(LatentDataHandler.DataFolder);
end

fileList = dir(LatentDataHandler.DataFolder);

for ii = 1:length(fileList)
    if(~fileList(ii).isdir)
        delete([ LatentDataHandler.DataFolder '/' fileList(ii).name]);
    end
end

%clear DataRowLength DataRowsPerBlock DataBlockNumber DataBlock DataChanged DataValidRows;

if(nargin > 0)
    
    maxSizePerBlock      = 1e9;
    
    if(nargin >= 2 && rowsToInitialize >= 1)
        sizeToStartBlocking  = 20e9; %Maximum amount of memory to take up with latent vars
        bytesNeeded = rowLength*rowsToInitialize*8;
        if(sizeToStartBlocking >= bytesNeeded) 
            maxSizePerBlock = sizeToStartBlocking;
        end
    end
    
    LatentDataHandler.DataRowLength    = rowLength;
    LatentDataHandler.DataRowsPerBlock = max(1,floor(maxSizePerBlock/(rowLength*8)));
    
    LatentDataHandler.DataValidRows    = zeros(LatentDataHandler.DataRowsPerBlock,1);
    LatentDataHandler.DataBlockNumber    = 1;
    
    
    dataBlockInitialized = false;
    if(nargin >= 2 && rowsToInitialize >= 1)
        nBlocks = ceil(rowsToInitialize/LatentDataHandler.DataRowsPerBlock);
        
        for ii = nBlocks:-1:1
            LatentDataHandler.DataValidRows = zeros(LatentDataHandler.DataRowsPerBlock,1);
            trNums = (1:LatentDataHandler.DataRowsPerBlock) + (ii-1)*LatentDataHandler.DataRowsPerBlock;
            LatentDataHandler.DataValidRows(trNums <= rowsToInitialize) = 1;
            
            if(sum(trNums <= rowsToInitialize) < LatentDataHandler.DataRowsPerBlock)
                LatentDataHandler.DataBlock = zeros(LatentDataHandler.DataRowLength, sum(trNums <= rowsToInitialize));
                if(ii > 1)
                    DataBlock     = LatentDataHandler.DataBlock; %#ok<NASGU>
                    DataValidRows = LatentDataHandler.DataValidRows; %#ok<NASGU>
                    save([LatentDataHandler.DataFolder '/dataBlock' num2str(ii) '.mat'],'DataBlock','DataValidRows','-v7.3');
                    if(size(LatentDataHandler.DataBlock,2) < LatentDataHandler.DataRowsPerBlock)
                        LatentDataHandler.DataBlock          = zeros(LatentDataHandler.DataRowLength, LatentDataHandler.DataRowsPerBlock);
                        dataBlockInitialized = true;
                    end
                end
            elseif(ii > 1)
                if(~dataBlockInitialized)
                    LatentDataHandler.DataBlock          = zeros(LatentDataHandler.DataRowLength, LatentDataHandler.DataRowsPerBlock);
                    dataBlockInitialized = true;
                end
                DataBlock     = LatentDataHandler.DataBlock; %#ok<NASGU>
                DataValidRows = LatentDataHandler.DataValidRows; %#ok<NASGU>
                save([LatentDataHandler.DataFolder '/dataBlock' num2str(ii) '.mat'],'DataBlock','DataValidRows','-v7.3');
            end
           
            
        end
    else
        LatentDataHandler.DataBlock          = zeros(LatentDataHandler.DataRowLength, LatentDataHandler.DataRowsPerBlock);
        %dataBlockInitialized = true;
    end
    LatentDataHandler.DataChanged        = 1;
else
    LatentDataHandler.DataRowLength    = 0;
    LatentDataHandler.DataRowsPerBlock = 0;
    LatentDataHandler.DataBlock          = [];
    LatentDataHandler.DataBlockNumber    = -1;
    LatentDataHandler.DataChanged        = 0;
end