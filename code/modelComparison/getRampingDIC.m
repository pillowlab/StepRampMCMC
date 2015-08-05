function [DIC, l_like,DIClikelihoods ] = getRampingDIC(StepSamples,params,modelFit,timeSeries)


betas  = modelFit.beta.mean';
w2s    = modelFit.w2.mean';
l_0    = modelFit.l_0.mean;
gamma  = modelFit.gamma.mean;


%% --SETUP THE GPU VARS---------------
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
    
    betaVector(T1:T2) = timeSeries.trB(tr)-1; 
end

gpu_y                = kcArrayToGPU( timeSeries.y);
gpu_trIndex          = kcArrayToGPUint(int32(trIndex));       
gpu_trBetaIndex      = kcArrayToGPUint(int32(betaVector));  

fprintf('Diffusion-to-Bound Model: estimating p(y|theta_bar) ( with %d MC samples)... ', lambda_sets);
[ll, ll_pred] = kcSimGaussianBound(gpu_y, gpu_trIndex, gpu_trBetaIndex, gpu_y_test, gpu_trIndex_test, gpu_trBetaIndex_test, betas, w2s, l_0, gamma, params.delta_t, lambda_sets);
if(nargin >= 8 && ~isempty(trueParams))
    betas_true  = trueParams.meanBeta';
    w2s_true    = trueParams.meanW2';
    l_0_true    = trueParams.meanL0;
    gamma_true  = trueParams.meanGamma;

    [ll_true, ll_true_pred] = kcSimGaussianBound(gpu_y, gpu_trIndex, gpu_trBetaIndex, gpu_y_test, gpu_trIndex_test, gpu_trBetaIndex_test, betas_true, w2s_true, l_0_true, gamma_true, params.delta_t, lambda_sets);
    
    fprintf('Done (LL = %f, PLL=%f, true LL=%f).\n',ll,ll_pred,ll_true);
else
    ll_true = nan;
    ll_true_pred = nan;
    fprintf('Done (LL = %f, PLL=%f).\n',ll,ll_pred);
end



    
prior = 0;

prior = prior + sum(- 1/2*log(2*pi*params.beta_sigma.^2) - 1./(2*params.beta_sigma.^2) .* (betas - params.beta_mu).^2  );

prior = prior + sum(- 1/2*log(2*pi*params.l0_sigma_bound.^2) - 1./(2*params.l0_sigma_bound.^2) .* (l_0 - params.l0_mu_bound).^2  );
prior = prior + sum(params.w2_p1_bound.*log(params.w2_p2_bound) - gammaln(params.w2_p1_bound) + (-params.w2_p1_bound-1).*log(w2s) - params.w2_p2_bound./w2s  );
prior = prior + params.gamma_1*log(params.gamma_2) - gammaln(params.gamma_1) + (params.gamma_1-1)*log(gamma)   - params.gamma_2*gamma;




    

%% DIC estimates
thinRate = params.MCMC.thinRate;
burnIn = params.MCMC.burnIn;
log_p = zeros(samples.numSamples-burnIn,1);
for s = burnIn+1:thinRate:samples.numSamples
    log_p(s-burnIn) = kcRampLikelihood(gpu_y, gpu_trIndex, gpu_trBetaIndex, gpu_y_test, gpu_trIndex_test, gpu_trBetaIndex_test, samples.betas(s,:), samples.w2s(s,1), samples.l_0(s,:), samples.gammas(s), params.delta_t, DIClambdaSets);
    if(isnan(log_p_pred(s-burnIn)) || isinf(log_p_pred(s-burnIn))) 
        display(['bayesFactorsGaussianBound log_p_pred = ' num2str(log_p_pred(s-burnIn)) ' on sample ' num2str(s) ' . Throwing out.']);
    end
    if(isnan(log_p(s-burnIn)) || isinf(log_p(s-burnIn))) 
        display(['bayesFactorsGaussianBound log_p = ' num2str(log_p(s-burnIn)) ' on sample ' num2str(s) ' . Throwing out.']);
    end
    if(mod(s-1,100) == 0 || s==samples.numSamples) 
        display(['Gaussian Bound DIC sample ' num2str(s-burnIn) '/' num2str(samples.numSamples-burnIn)]);
    end
    
    if(mod((s-1)/thinRate,params.GPUresetTime) == 0)
        
        display('Reloading GPU values...');
        kcFreeGPUArray(gpu_y);
        kcFreeGPUArray(gpu_y_test);
        kcFreeGPUArray(gpu_trIndex);
        kcFreeGPUArray(gpu_trIndex_test);
        kcFreeGPUArray(gpu_trBetaIndex);
        kcFreeGPUArray(gpu_trBetaIndex_test);
        kcResetDevice;
        pause(1);
        
        
        gpu_y                = kcArrayToGPU( timeSeries.y);
        gpu_y_test           = kcArrayToGPU( timeSeriesTest.y);
        gpu_trIndex          = kcArrayToGPUint(int32(trIndex));
        gpu_trIndex_test     = kcArrayToGPUint(int32(trIndex_test));
        gpu_trBetaIndex      = kcArrayToGPUint(int32(betaVector));
        gpu_trBetaIndex_test = kcArrayToGPUint(int32(betaVector_test));
        pause(1);
        display('Done.');
    end
    if(mod((s-1)/thinRate,params.GPUpauseTime) == 0)
        pause(params.GPUpauseLength);
    end



    % get priors
    priorProbs(s-burnIn) = priorProbs(s-burnIn) + sum(- 1/2*log(2*pi*params.beta_sigma.^2) - 1./(2*params.beta_sigma.^2) .* (samples.betas(s,:) - params.beta_mu).^2  );

    priorProbs(s-burnIn) = priorProbs(s-burnIn) + sum(- 1/2*log(2*pi*params.l0_sigma_bound.^2) - 1./(2*params.l0_sigma_bound.^2) .* (samples.l_0(s,:) - params.l0_mu_bound).^2  );
    priorProbs(s-burnIn) = priorProbs(s-burnIn) + sum(params.w2_p1_bound.*log(params.w2_p2_bound) - gammaln(params.w2_p1_bound) + (-params.w2_p1_bound-1).*log(samples.w2s(s,1)) - params.w2_p2_bound./samples.w2s(s,1)  );
    priorProbs(s-burnIn) = priorProbs(s-burnIn) + params.gamma_1*log(params.gamma_2) - gammaln(params.gamma_1) + (params.gamma_1-1)*log(samples.gammas(s))   - params.gamma_2*samples.gammas(s);

end

kcFreeGPUArray(gpu_y);
kcFreeGPUArray(gpu_trIndex);      
kcFreeGPUArray(gpu_trBetaIndex); 
kcFreeGPUArray(gpu_y_test);
kcFreeGPUArray(gpu_trIndex_test);      
kcFreeGPUArray(gpu_trBetaIndex_test); 
%kcFreeGPUArray(gpu_lambdas);

log_p_pred = log_p_pred(1:thinRate:end);
log_p = log_p(1:thinRate:end);
priorProbs = priorProbs(1:thinRate:end);
postProbs = priorProbs + log_p;
postProbs(isnan(postProbs)) = min(postProbs);
likelihoods =log_p;
likelihoods(isnan(likelihoods)) = min(log_p);

ss2 = find(likelihoods == max(likelihoods),1);
ss = (ss2-1)*thinRate + 1;

sampleModelFit.meanBeta   = samples.betas(ss+burnIn,:);
sampleModelFit.meanL0     = samples.l_0(ss+burnIn);
sampleModelFit.meanW2     = samples.w2s(ss+burnIn);
sampleModelFit.meanGamma  = samples.gammas(ss+burnIn);
MAP_ML_stuff.ML_est.vals = sampleModelFit; 
MAP_ML_stuff.ML_est.ll   = likelihoods(ss2);
MAP_ML_stuff.ML_est.prior  = priorProbs(ss2);
MAP_ML_stuff.ML_est.ss   = ss;

ss2 = find(postProbs == max(postProbs),1);
ss = (ss2-1)*thinRate + 1;

sampleModelFit.meanBeta   = samples.betas(ss+burnIn,:);
sampleModelFit.meanL0     = samples.l_0(ss+burnIn);
sampleModelFit.meanW2     = samples.w2s(ss+burnIn);
sampleModelFit.meanGamma  = samples.gammas(ss+burnIn);
MAP_ML_stuff.MAP_est.ll   = likelihoods(ss2);
MAP_ML_stuff.MAP_est.prior  = priorProbs(ss2);
MAP_ML_stuff.MAP_est.vals = sampleModelFit; 
MAP_ML_stuff.MAP_est.ss   = ss;


MAP_ML_stuff.true_ll = ll_true;
MAP_ML_stuff.true_ll_pred = ll_true_pred;

display(['Diff-to-Bound ML : ' num2str(MAP_ML_stuff.ML_est.ll)]);
display(['Diff-to-Bound MAP: ' num2str(MAP_ML_stuff.MAP_est.ll + MAP_ML_stuff.MAP_est.prior)]);

if(sum(isinf(log_p_pred) | isnan(log_p_pred)) > 0)
    display(['Tossing out ' num2str(sum(isinf(log_p) | isnan(log_p))) ' bad values for DIC expectation.']);
    log_p = log_p(~isinf(log_p) & ~isnan(log_p)); 
end
if(~isempty(log_p))
    D = mean(log_p);
else
    D = 0;
    display('Errors computing DIC bayesFactorsGaussianBound!');
end

DIC = 2*l_like-4*D;


MAP_ML_stuff.MAP_est.DIC = 2*MAP_ML_stuff.MAP_est.ll -4*D;
MAP_ML_stuff.ML_est.DIC  = 2*MAP_ML_stuff.ML_est.ll -4*D;

if(sum(isinf(log_p_pred) | isnan(log_p_pred)) > 0)
    display(['Tossing out ' num2str(sum(isinf(log_p_pred) | isnan(log_p_pred))) ' bad values on log post predictive distribution.']);
end
log_p_pred = log_p_pred(~isinf(log_p_pred) & ~isnan(log_p_pred)); 
if(~isempty(log_p_pred))
    [log_p_predictive, m_log_p_predictive, numerror] = meanL(log_p_pred);
    if(numerror > 0)
        display('Numerical errors in calculating log p y predictive bound');
        log_p_predictive = m_log_p_predictive;
    end
else
    log_p_predictive = 0;
    display('Errors computing predictive dist bayesFactorsGaussianBound!');
end


BIC = -2*l_like + length(theta_mu)*log(TT);