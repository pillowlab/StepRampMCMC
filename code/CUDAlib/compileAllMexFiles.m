% This script attempts to compile all the necessary CUDA functions into MEX
% files. This function should be run from the base directory of the project
%
%
% Before running, set the paths in code/CUDAlib/myPaths.m to match your CUDA
% and MATLAB installation folders.


fprintf('Make sure to set paths in code/CUDAlib/myPaths.m\n');

%% adds path to some CUDA files
addpath code/CUDAlib/

%% gets list of files to compile

fNames = compileCUDAmex();

%% compiles and links these files

for ii = 1:length(fNames);
    fprintf('Compiling file %s (%d / %d)...\n',fNames{ii},ii,length(fNames));
    objFile = compileCUDAmex(fNames{ii});

    if(~exist(objFile,'file'))
      error(sprintf('No object file found (%s). Cannot link mex file.',objFile));
    end
    fprintf('Linking file %s (%d / %d)...\n',fNames{ii},ii,length(fNames));
    mexFile = linkCUDAmex(fNames{ii});
    if(~exist(mexFile,'file'))
      error(sprintf('No mex file created (%s).',mexFile));
    end
end

fprintf('Finished compiling and linking. All mex files exist. If you stumble upon any CUDA/MEX errors, check through the output for any errors that were missed by this script.\n');
