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

%% gets spikes from a simulated neuron
% timeSeries = simulateRampingModel();
timeSeries = simulateSteppingModel();

%% gets parameters
params = setupMCMCParams();

%% set files for output
resultsFiles.ramp = './Results/RampFit_Sim.mat';
resultsFiles.step = './Results/StepFit_Sim.mat';

if(~exist('./Results/','file'))
    mkdir('Results');
end

samplesFiles.ramp = './Samples/RampFit_Sim.mat';
samplesFiles.step = './Samples/StepFit_Sim.mat';

if(~exist('./Samples/','file'))
    mkdir('Samples');
end

%% sets which GPU to use
kcSetDevice(0); %in case you have more than one GPU and want to select (0 is the default)

%% runs model comparison

[DICs, StepFit, RampFit] = runModelComparison( timeSeries, params, resultsFiles, samplesFiles);
