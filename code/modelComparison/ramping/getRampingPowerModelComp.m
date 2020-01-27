function [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] = getRampingTrialCompLog1PFlex(RampSamples,params,modelFit,timeSeries,model_spec)

timeSeries = setupTrialIndexStructure(timeSeries);

betas  = modelFit.beta.mean';
w2s    = modelFit.w2.mean';
l_0    = modelFit.l_0.mean;
gamma  = modelFit.gamma.mean;

% default values -> log1p, p = 1, no spe
modelInd = 1;
speInd = 0;

% try to set modelInd and speInd to input values
try 
    modelInd = model_spec.modelInd;
end
try
    speInd = model_spec.speInd;
end
modelInd

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

[l_like,trial_l_like] = kcRampLikelihoodFlex(gpu_y, gpu_trIndex, gpu_trBetaIndex, betas, w2s, l_0, gamma, timeSeries.delta_t, params.DIC.meanLikelihoodSamples,modelInd);


%% get log prior of fit
% prior = 0;
% 
% prior = prior + sum(- 1/2*log(2*pi*params.rampPrior.beta_sigma.^2) - 1./(2*params.rampPrior.beta_sigma.^2) .* (betas - params.rampPrior.beta_mu).^2  );
% prior = prior + sum(- 1/2*log(2*pi*params.rampPrior.l0_sigma^2) - 1./(2*params.rampPrior.l0_sigma.^2) .* (l_0 - params.rampPrior.l0_mu).^2  );
% prior = prior + sum(params.rampPrior.w2_shape.*log(params.rampPrior.w2_scale) - gammaln(params.rampPrior.w2_shape) + (-params.rampPrior.w2_shape-1).*log(w2s) - params.rampPrior.w2_scale./w2s  );
% prior = prior + params.rampPrior.gammaAlphalog(params.rampPrior.gammaBeta) - gammaln(params.rampPrior.gammaAlpha) + (params.rampPrior.gammaAlpha1-1)*log(gamma)   - params.rampPrior.gammaBeta*gamma;



%% DIC estimates
thinRate = params.MCMC.thinRate;
burnIn = params.MCMC.burnIn;
sample_likelihoods = zeros(params.MCMC.nSamples,1);
sample_trial_likelihoods = zeros(NT,params.MCMC.nSamples);

for ss = 1:thinRate:params.MCMC.nSamples
    if(ss == 1 || mod(ss-1,50) == 0 || ss == params.MCMC.nSamples)
        fprintf('  Ramping model DIC calculations %d / %d\n',ss,params.MCMC.nSamples);
    end
    
    [sample_likelihoods(ss), sample_trial_likelihoods(:,ss)] = kcRampLikelihoodFlex(gpu_y, gpu_trIndex, gpu_trBetaIndex, RampSamples.betas(ss+burnIn,:), RampSamples.w2s(ss+burnIn,1), RampSamples.l_0(ss+burnIn,:), RampSamples.gammas(ss+burnIn), timeSeries.delta_t, params.DIC.likelihoodSamples,modelInd);
end

sample_likelihoods = sample_likelihoods(1:thinRate:end);
sample_trial_likelihoods = sample_trial_likelihoods(:,1:thinRate:end);

termsToDrop = sum(isnan(sample_likelihoods) | isinf(sample_likelihoods));
if(termsToDrop > 0)
  fprintf('Dropping %d terms from DIC calculation! Potential numerical errors!\n',termsToDrop);
end

sample_likelihoods = sample_likelihoods(~isnan(sample_likelihoods) & ~isinf(sample_likelihoods));
if(~isempty(sample_likelihoods))
  expectedLikelihood = mean(sample_likelihoods);
 else
   error('Numerical error calculating expected likelihood (DIC) in ramping model');
end

DIC = 2*l_like-4*expectedLikelihood;
trial_DIC = 2*trial_l_like-4*mean(sample_trial_likelihoods,2);

% WAIC = -2 * Sum_trials ( log mean of sample likelihoods - sample variance of log sample likelihoods )
% WAIC = -2 * Sum ( log E[p(y|theta^s] - V[log p(y|theta^s)] )

numSamples = size(sample_trial_likelihoods,2);
ll_max = max(sample_trial_likelihoods,[],2); % maximum log p(y|theta^s)

% compute log of mean of sample likelihoods for each trial 
log_point_pred_density_trial = -log(numSamples)+log(sum(exp(sample_trial_likelihoods-repmat(ll_max,[1 numSamples])),2))+ll_max;

% sum log mean of sample likelihoods across trials
log_point_pred_density = sum(log_point_pred_density_trial);

% compute penalty, which is variance of log sample likelihoods for each
% trial
pwaic_trial = var(sample_trial_likelihoods,[],2);

% sum penalty across trials
pwaic = sum(pwaic_trial);

% compute WAIC and trial WAIC 
WAIC = -2*(log_point_pred_density-pwaic);
trial_WAIC = -2*(log_point_pred_density_trial-pwaic_trial);

%% free up GPU variables

kcFreeGPUArray(gpu_y);
kcFreeGPUArray(gpu_trIndex);
kcFreeGPUArray(gpu_trBetaIndex);

fprintf('Ramping model DIC computation complete.\n');

end
