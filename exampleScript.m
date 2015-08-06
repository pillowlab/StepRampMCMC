%% 
%  This script shows the basics of how to run the model comparison for stepping/ramping dynamics
%  given in
%
%    Kenneth W. Latimer, Jacob L. Yates, Miriam L. R. Meister, Alexander C. Huk, 
%    & Jonathan W. Pillow (2015). Single-trial spike trains in parietal cortex reveal 
%    discrete steps during decision-making. Science, 349(6244):184-187.
%
%
%  Simulates stepping or ramping trials, and runs the model fitting/comparison
%  on the simulated spikes.
%   
%  The CUDA files must be compiled before running this function.
%  See code/CUDAlib/compileAllMexFiles.m to run the compiler.

%% sets up necessary paths to code
addAllPaths;

%% gets a simulated stepping neuron
timeSeries = simulateRampingModel();
% timeSeries = simulateSteppingModel();

%% gets parameters
params = setupMCMCParams();

%% sets up GPU
kcSetDevice(0); %in case you have more than 1 GPU, this sets which device to use

%% set files for output
resultsFiles.ramp = './Results/RampFit_Sim.mat';
resultsFiles.step = './Results/StepFit_Sim.mat';

samplesFiles.ramp = './Samples/RampFit_Sim.mat';
samplesFiles.step = './Samples/StepFit_Sim.mat';

%% runs model comparison

[DICs, StepFit, RampFit] = runModelComparison( timeSeries, params, resultsFiles, samplesFiles);
