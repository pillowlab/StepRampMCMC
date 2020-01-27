%% 
%  This script shows the basics of how to run the model comparison for the 
%   extended stepping/ramping dynamics
%   given in
%
%    David M. Zoltowski, Kenneth W. Latimer, Jacob L. Yates, Alexander C. Huk, 
%    & Jonathan W. Pillow (2019). Discrete stepping and nonlinear ramping 
%    dynamics underlie spiking responses of LIP neurons during decision-making. 
%    Neuron, 2019.
%
%
%  Simulates stepping or ramping trials, and runs the model fitting/comparison
%  on the simulated spikes.
%   
%  The CUDA files must be compiled before running this function.
%  See code/CUDAlib/compileAllMexFiles.m to run the compiler.

%% sets up necessary paths to code
addAllPaths;

%% setup model parameters
% Ramping model
ramping_model.nlin = "softplus"; % original "softplus" nonlinearity lambda = log(1 + exp(x*gamma))
ramping_model.bias = true; % fit bias parameter 
ramping_model.history = false; % no history dependence (set true for history dependence)
ramping_model.power = 1; % power for nonlinearity if ramping_model.nlin = "power"

% Stepping model
stepping_model.history = false; % no history dependence (set true for history dependence)
stepping_model.MR = true; % reparameterize step time distribution

%% setup MCMC params
ramp_params = setupExtendedMCMCParams(ramping_model);
step_params = setupExtendedMCMCParams(stepping_model);

%% gets spikes from a simulated neuron
timeSeries = simulateExtendedRampingModel(ramping_model);
% timeSeries = simulateExtendedSteppingModel();

%% gets parameters
params = setupMCMCParams();
params.tempDataFolder = './temp/'; %temporary folder to store samples of the latent states of the ramping model
                                   %if number of samples/trials is large (saves on RAM)

%% set files for output
resultsFiles.ramp = './Results/ExtendedRampFit_Sim.mat';
resultsFiles.step = './Results/ExtendedStepFit_Sim.mat';

if(~exist('./Results/','file'))
    mkdir('Results');
end

samplesFiles.ramp = './Samples/ExtendedRampFit_Sim.mat';
samplesFiles.step = './Samples/ExtendedStepFit_Sim.mat';

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

%% fit models
[RampFit, RampSamples, LatentDataHandler] = fitExtendedRampingModel(timeSeries,ramp_params,ramping_model);
[StepFit, StepSamples ] = fitExtendedSteppingModel(timeSeries,step_params,stepping_model);

%% get model comparison statistics
RampingModelComp = getExtendedRampingModelComp(RampSamples,ramp_params,RampFit,timeSeries,ramping_model);
SteppingModelComp = getExtendedSteppingModelComp(StepSamples,step_params,StepFit,timeSeries,stepping_model);

%% Compare results
waic_difference = RampModelComp.WAIC - SteppingModelComp.WAIC;
