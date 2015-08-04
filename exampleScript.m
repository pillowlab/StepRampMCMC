%% sets up necessary paths
addpath ./code/
addpath ./code/CUDAlib/mex/


%% gets a simulated stepping neuron
timeSeries = simulateSteppingModel();

%% gets parameters
params = setupMCMCParams();

%% set files for output
resultsFiles.ramp = './Results/RampFit_Sim.mat';
resultsFiles.step = './Results/StepFit_Sim.mat';

samplesFiles.ramp = './Samples/RampFit_Sim.mat';
samplesFiles.step = './Samples/StepFit_Sim.mat';

%% runs model comparison

[DICs, StepFit, RampFit] = runModelComparison( timeSeries, params, resultsFiles, samplesFiles);