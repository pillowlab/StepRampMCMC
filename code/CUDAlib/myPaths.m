function [ projectHome, CUDAdirectory, MATLABdirectory ] = myPaths(  )
%MYPATHS this function contains path information to the CUDA and MATLAB folders.
%   This is used for compiling CUDA files into mex files.

%aboslute path to the base directory for this project
projectHome =  '/home/latimerk/gitCode/LIPStateSpaceRelease/';

%directory of the CUDA installation
CUDAdirectory   = '/usr/local/cuda-7.0/';

%directory of the MATLAB installation
MATLABdirectory = '/usr/local/MATLAB/R2013a/';

end

