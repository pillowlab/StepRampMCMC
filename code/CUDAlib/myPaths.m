function [ projectHome, CUDAdirectory, CUDASamplesdirectory, MATLABdirectory ] = myPaths(  )
%MYPATHS this function contains path information to the CUDA and MATLAB folders.
%   This is used for compiling CUDA files into mex files.


%% 1. Set absolute path to the base directory for this project
projectHome = which('myPaths.m');
projectHome = projectHome(1:end-22);

% check if directory exists
if ~isdir(projectHome)
    warning(['ERROR: projectHome directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],projectHome);
end

% IF this fails for some reason, specify absolute path to directory containing this function, e.g.:
% projectHome =  '/home/USERNAME/gitCode/StepRampMCMC/';

%% 2. Set absolute path for directory where CUDA installation lives:
CUDAdirectory   = '/usr/local/cuda-7.0/';
CUDASamplesdirectory = [CUDAdirectory '/samples/']; %samples that come with the CUDA sdk - includes some convenient error checking functions


% check if directory exists
if ~isdir(CUDAdirectory)
    warning(['ERROR: CUDAdirectory directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],CUDAdirectory);
end
if ~isdir(CUDASamplesdirectory)
    warning(['ERROR: CUDASamplesdirectory directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],CUDASamplesdirectory);
end


%% 3. Directory of the MATLAB installation. 
MATLABdirectory = matlabroot;  % this *shouldn't* need adjusting
MATLABdirectory = [MATLABdirectory, '/'];
