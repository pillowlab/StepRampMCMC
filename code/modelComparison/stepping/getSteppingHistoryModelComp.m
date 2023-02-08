function [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] = getSteppingHistoryModelComp(StepSamples,params,modelFit,timeSeries)

timeSeries = setupTrialIndexStructure(timeSeries);

%% get log likelihood
NT  = size(timeSeries.trialIndex,1);
TT  = size(timeSeries.y,1);
trIndex = zeros(NT+1,1);
NH = length(modelFit.hs.mean);
SpikeHistory = zeros(NT*NH,1);

betaVector = zeros(TT+1,1);
maxTrLength = 0;
for tr = 1:NT
    T1 = timeSeries.trialIndex(tr,1);
    T2 = timeSeries.trialIndex(tr,2);
    T = T2 - T1 + 1;
    maxTrLength = max(T,maxTrLength);
    
    trIndex(tr+1) = trIndex(tr) + T;
    
    betaVector(T1:T2) = timeSeries.trCoh(tr)-1; 

    % place the spikes before each trial into one long concatenated array
    SpikeHistory((tr-1)*NH+1:(tr*NH)) = timeSeries.SpikeHistory(tr,:);

end

gpu_y                = kcArrayToGPU( timeSeries.y);
gpu_trIndex          = kcArrayToGPUint(int32(trIndex));       
gpu_spikeHistory     = kcArrayToGPU(SpikeHistory);

gpu_spe              = kcGetSpikeHistoryEffect(gpu_y,gpu_trIndex,gpu_spikeHistory,modelFit.hs.mean);
spe = kcArrayToHost(gpu_spe);

[l_like,trial_l_like] = getSteppingHistoryLogLikelihood(timeSeries,params,modelFit,spe);

kcFreeGPUArray(gpu_spe);

%% calculates DIC and posterior predictive
sample_likelihoods = zeros(params.MCMC.nSamples,1);
sample_trial_likelihoods = zeros(NT,params.MCMC.nSamples);

for ss = 1:params.MCMC.thinRate:params.MCMC.nSamples
    if(ss == 1 || mod(ss-1,250) == 0 || ss == params.MCMC.nSamples)
        fprintf('  Stepping model DIC calculations %d / %d\n',ss,params.MCMC.nSamples);
    end
    sampleModelFit.alpha.mean = StepSamples.alpha(:,ss+params.MCMC.burnIn);
    sampleModelFit.r.mean     = StepSamples.r(ss+params.MCMC.burnIn);
    sampleModelFit.phi.mean   = StepSamples.phi(:,ss+params.MCMC.burnIn);
    sampleModelFit.p.mean     = StepSamples.p(:,ss+params.MCMC.burnIn);
    gpu_spe = kcGetSpikeHistoryEffect(gpu_y,gpu_trIndex,gpu_spikeHistory,StepSamples.hs(ss+params.MCMC.burnIn,:));
    spe = kcArrayToHost(gpu_spe);
    [sample_likelihoods(ss), sample_trial_likelihoods(:,ss)] = getSteppingHistoryLogLikelihood(timeSeries, params,sampleModelFit, spe);
    kcFreeGPUArray(gpu_spe);
end

sample_likelihoods = sample_likelihoods(1:params.MCMC.thinRate:end);
sample_trial_likelihoods = sample_trial_likelihoods(:,1:params.MCMC.thinRate:end);

termsToDrop = sum(isnan(sample_likelihoods) | isinf(sample_likelihoods));
if(termsToDrop > 0)
  fprintf('Dropping %d terms from calculation! Potential numerical errors!\n',termsToDrop);
end

sample_likelihoods = sample_likelihoods(~isnan(sample_likelihoods) & ~isinf(sample_likelihoods));
if(~isempty(sample_likelihoods))
  expectedLikelihood = mean(sample_likelihoods);
 else
   error('Numerical error calculating expected likelihood (DIC) in ramping model');
end

DIC = 2*l_like-4*expectedLikelihood;
trial_DIC = 2*trial_l_like-4*mean(sample_trial_likelihoods,2);

numSamples = size(sample_trial_likelihoods,2);
ll_max = max(sample_trial_likelihoods,[],2);
log_point_pred_density_trial = -log(numSamples)+log(sum(exp(sample_trial_likelihoods-repmat(ll_max,[1 numSamples])),2))+ll_max;
log_point_pred_density = sum(log_point_pred_density_trial);
pwaic_trial = var(sample_trial_likelihoods,[],2);
pwaic = sum(pwaic_trial);

WAIC = -2*(log_point_pred_density-pwaic);
trial_WAIC = -2*(log_point_pred_density_trial-pwaic_trial);

%% free up GPU variables

kcFreeGPUArray(gpu_y);
kcFreeGPUArray(gpu_trIndex);

fprintf('Stepping model DIC computation complete.\n');

end
