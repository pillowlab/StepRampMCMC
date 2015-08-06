%% Computes the probability of the spikes timeSeries.y given a fixed setting of the ramp model parameters
% marginalized over diffusion paths
% parameters used: 
% modelFit.beta.mean
% modelFit.l_0.mean
% modelFit.w2.mean
% modelFit.gamma.mean
%
% This value is approximated using Monte Carlo simulations (params.DIC.meanLikelihoodSamples per trial)
function [l_like, trial_likelihood] = getSteppingLogLikelihood(timeSeries,params,modelFit)

timeSeries = setupTrialIndexStructure(timeSeries);

betas  = modelFit.beta.mean';
w2s    = modelFit.w2.mean';
l_0    = modelFit.l_0.mean;
gamma  = modelFit.gamma.mean;


%% Setup the GPU variables
NT  = size(timeSeries.trialIndex,1);
TT  = size(timeSeries.y,1);
trIndex = zeros(NT+1,1);

betaVector = zeros(TT+1,1);
maxTrLength = 0;
for tr = 1:NT
    T1 = timeSeries.trialIndex(tr,1);
    T2 = timeSeries.trialIndex(tr,2);
    T = T2 - T1 + 1;
    maxTrLength = max(T,maxTrLength);
    
    trIndex(tr+1) = trIndex(tr) + T;
    
    betaVector(T1:T2) = timeSeries.trCoh(tr)-1; 
end

gpu_y                = kcArrayToGPU( timeSeries.y);
gpu_trIndex          = kcArrayToGPUint(int32(trIndex));       
gpu_trBetaIndex      = kcArrayToGPUint(int32(betaVector));  


%% compute likelihood
[l_like,trial_likelihood] = kcRampLikelihood(gpu_y, gpu_trIndex, gpu_trBetaIndex, betas, w2s, l_0, gamma, params.delta_t, params.DIC.meanLikelihoodSamples);

%% free up GPU variables

kcFreeGPUArray(gpu_y);
kcFreeGPUArray(gpu_trIndex);      
kcFreeGPUArray(gpu_trBetaIndex); 
