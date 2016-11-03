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
timeSeries = simulateRampingModel();
% timeSeries = simulateSteppingModel();

%% gets parameters
params = setupMCMCParams();
params.tempDataFolder = './temp/'; %temporary folder to store samples of the latent states of the ramping model
                                   %if number of samples/trials is large (saves on RAM)

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
kcSetDevice(1); %in case you have more than one GPU and want to select (0 is the default)

%% sets a temporary folder to use for saving latent variables without hogging all the memory (this part of the code uses an ugly global variable)
global DataFolder 
DataFolder = './temp/';
%The dependence on this variable should be gone at this point - I haven't
%tested the script without this line to be absolutely sure.


%% runs model comparison

[DICs, StepFit, RampFit, StepModelComp, RampModelComp] = runModelComparison( timeSeries, params, resultsFiles, samplesFiles);
