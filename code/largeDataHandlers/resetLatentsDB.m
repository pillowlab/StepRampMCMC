function [] = resetLatentsDB(rowLength,rowsToInitialize)
%functions to keep a matrix of all the latent variables sampled for the ramping model without taking too much RAM
%Warning: These functions use global variables (not a smart choice)
%
% resets the blocks of latents

global DataRowLength DataRowsPerBlock DataBlockNumber DataBlock DataChanged DataValidRows DataFolder;


fileList = dir(DataFolder);

for ii = 1:length(fileList)
    if(~fileList(ii).isdir)
        delete([ DataFolder '/' fileList(ii).name]);
    end
end

%clear DataRowLength DataRowsPerBlock DataBlockNumber DataBlock DataChanged DataValidRows;

if(nargin > 0)
    
    maxSizePerBlock      = 1e9;
    
    if(nargin >= 2 && rowsToInitialize >= 1)
        sizeToStartBlocking  = 20e9;
        bytesNeeded = rowLength*rowsToInitialize*8;
        if(sizeToStartBlocking >= bytesNeeded) 
            maxSizePerBlock = sizeToStartBlocking;
        end
    end
    
    DataRowLength    = rowLength;
    DataRowsPerBlock = max(1,floor(maxSizePerBlock/(rowLength*8)));
    
    DataValidRows    = zeros(DataRowsPerBlock,1);
    DataBlockNumber    = 1;
    
    
    dataBlockInitialized = false;
    if(nargin >= 2 && rowsToInitialize >= 1)
        nBlocks = ceil(rowsToInitialize/DataRowsPerBlock);
        
        for ii = nBlocks:-1:1
            DataValidRows = zeros(DataRowsPerBlock,1);
            trNums = (1:DataRowsPerBlock) + (ii-1)*DataRowsPerBlock;
            DataValidRows(trNums <= rowsToInitialize) = 1;
            
            if(sum(trNums <= rowsToInitialize) < DataRowsPerBlock)
                DataBlock = zeros(DataRowLength, sum(trNums <= rowsToInitialize));
                if(ii > 1)
                    save([DataFolder '/dataBlock' num2str(ii) '.mat'],'DataBlock','DataValidRows','-v7.3');
                    if(size(DataBlock,2) < DataRowsPerBlock)
                        DataBlock          = zeros(DataRowLength, DataRowsPerBlock);
                        dataBlockInitialized = true;
                    end
                end
            elseif(ii > 1)
                if(~dataBlockInitialized)
                    DataBlock          = zeros(DataRowLength, DataRowsPerBlock);
                    dataBlockInitialized = true;
                end
                save([DataFolder '/dataBlock' num2str(ii) '.mat'],'DataBlock','DataValidRows','-v7.3');
            end
           
            
        end
    else
        DataBlock          = zeros(DataRowLength, DataRowsPerBlock);
        dataBlockInitialized = true;
    end
    DataChanged        = 1;
else
    DataRowLength    = 0;
    DataRowsPerBlock = 0;
    DataBlock          = [];
    DataBlockNumber    = -1;
    DataChanged        = 0;
end