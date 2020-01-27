function [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] = getSteppingTrialComp(StepSamples,params,modelFit,timeSeries)

timeSeries = setupTrialIndexStructure(timeSeries);

%% get log likelihood

NT = size(timeSeries.trialIndex,1);
[l_like,trial_l_like] = getSteppingLogLikelihood(timeSeries,params,modelFit);

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
    [sample_likelihoods(ss), sample_trial_likelihoods(:,ss)] = getSteppingLogLikelihood(timeSeries, params,sampleModelFit);
end

sample_likelihoods = sample_likelihoods(1:params.MCMC.thinRate:end);
sample_trial_likelihoods = sample_trial_likelihoods(:,1:params.MCMC.thinRate:end);

termsToDrop = sum(isnan(sample_likelihoods) | isinf(sample_likelihoods));
if(termsToDrop > 0)
  fprintf('Dropping %d terms from DIC calculation! Potential numerical errors!\n',DICtermsToDrop);
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

fprintf('Stepping model DIC computation complete.\n');

end