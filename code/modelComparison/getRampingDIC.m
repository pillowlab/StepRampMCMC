function [DIC, l_like,DIClikelihoods ] = getRampingDIC(RampSamples,params,modelFit,timeSeries)

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

l_like = kcRampLikelihood(gpu_y, gpu_trIndex, gpu_trBetaIndex, betas, w2s, l_0, gamma, timeSeries.delta_t, params.DIC.meanLikelihoodSamples);


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
DIClikelihoods = zeros(params.MCMC.nSamples,1);
for ss = 1:thinRate:params.MCMC.nSamples
    if(ss == 1 || mod(ss-1,50) == 0 || ss == params.MCMC.nSamples)
        fprintf('  Ramping model DIC calculations %d / %d\n',ss,params.MCMC.nSamples);
    end
    
    DIClikelihoods(ss) = kcRampLikelihood(gpu_y, gpu_trIndex, gpu_trBetaIndex, RampSamples.betas(ss+burnIn,:), RampSamples.w2s(ss+burnIn,1), RampSamples.l_0(ss+burnIn,:), RampSamples.gammas(ss+burnIn), timeSeries.delta_t, params.DIC.likelihoodSamples);
end


DIClikelihoods = DIClikelihoods(1:thinRate:end);

DICtermsToDrop = sum(isnan(DIClikelihoods) | isinf(DIClikelihoods));
if(DICtermsToDrop > 0)
    fprintf('Dropping %d terms from DIC calculation! Potential numerical errors!\n',DICtermsToDrop);
end

DIClikelihoods = DIClikelihoods(~isnan(DIClikelihoods) & ~isinf(DIClikelihoods));
if(~isempty(DIClikelihoods))
    expectedLikelihood = mean(DIClikelihoods);
else
    error('Numerical error calculating expected likelihood (DIC) in stepping model');
end
DIC = 2*l_like-4*expectedLikelihood;

%% free up GPU variables

kcFreeGPUArray(gpu_y);
kcFreeGPUArray(gpu_trIndex);      
kcFreeGPUArray(gpu_trBetaIndex); 



fprintf('Ramping model DIC computation complete.\n');
