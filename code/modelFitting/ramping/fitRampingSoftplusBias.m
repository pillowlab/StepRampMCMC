%% Fit a ramping model to the observed spiking trials by sampling over the 
%  posterior distribution p(\Theta,lambdas|y) where
%   y = observed spikes
%   \Theta  = model parameters
%   lambdas = latent variables (diffusion-to-bound paths)
%  by taking only the samples of \Theta, we obtain a sampled estimate of
%         p(\Theta|y)
%
% model params
%  beta  - drift slope. 1 value per stimulus/coherence level.. Takes values (-\infty, \infty)
%  l_0   - initial diffusion position. 1 value. Takes values (-\infty, 1)
%  w2    - diffusion variance. 1 value. Takes values (0, \infty)
%  gamma - bound height parameter. 1 value. Takes values (0, \infty)
%
% Model of a trial (trial number j)
%
%   lambda(1) = l_0 + randn*sqrt(w2);    (randn*sqrt(w2) gives zero mean Gaussian noise with variance w2)
%   
%   for t = 2...T  (T=trial length)
%     lambda(t) = lambda(t-1) + beta(timeSeries.trCoh(j)) + randn*sqrt(w2)
%   end 
%   auxThreshold(j) = find first t such that lambda(t) > 1 (if doesn't exist set to T+1)
%
%   y(t|t <  auxThreshold(j))      ~ Poisson(log(1+ exp(lambda(t)*gamma))* timeSeries.delta_t)
%   y(t|t >= auxThreshold(j))      ~ Poisson(log(1+ exp(          gamma))* timeSeries.delta_t)
%
% Model fiting outputs
%   RampSamples.betas  = drift slopes (numSamples,numCoherences)
%   RampSamples.l_0    = initial drift position (numSamples,1)
%   RampSamples.w2s    = drift variance (numSamples,1)
%   RampSamples.gammas = bound height (or diffusion path scaling) parameter (numSamples,1)
%
%   RampSamples.auxThreshold = auxiliary variable to say when (if) bound was hit on each trial for each sample of lambda 
%                              (NT,numSamples)
%                              if RampSamples.auxThreshold is < 0 or greater than the trial length, then bound was not hit on the sample
%
%
%
%   RampSamples includes the burnin period samples 
%
%   RampFit contains sample mean of each parameter (after throwing out burnin and thinning according to params.MCMC.thin)
%           and a 95% credible interval. (This structure summarizes the RampSamples)
%        e.g., RampFit.beta.mean   contains the posterior mean over the drift slope rate parameters
% 
%
%
% timeSeries - holds all trial information (NT = number of trials)
%   timeSeries.y          = spikes at each time (one long vector) 
%   timeSeries.trialIndex = NT x 2 matrix, each row holds the start and end
%                           indices for each trial (with respect to timeSeries.y)
%   timeSeries.trCoh        = coherence for each trial
%
%
%   model priors
%       beta(i)  ~ normal(params.rampPrior.beta_mu, params.rampPrior.beta_sigma^2)
%       l_0      ~ normal(params.rampPrior.l0_mu, params.rampPrior.l0_sigma^2)
%       w2       ~ inverse-gamma(params.rampPrior.w2_shape,params.rampPrior.w2_scale)
%       gamma    ~ gamma(params.rampPrior.gammaAlpha,params.rampPrior.gammaBeta)
%


function [ RampFit, RampSamples, LatentDataHandler] = fitRampingSoftplusBias(timeSeries,params)


totalSamples = params.MCMC.nSamples+params.MCMC.burnIn;
timeSeries = setupTrialIndexStructure(timeSeries);
TT = size(timeSeries.y,1);
NT = size(timeSeries.trialIndex,1);
NC = max(timeSeries.trCoh);


%% max firing rate (bound) initialization ------------------------------
firingRateFunc    = @(X) log(1+exp(X))*timeSeries.delta_t;
firingRateFuncInv = @(X) log(exp(X/timeSeries.delta_t)-1);
timeIndices = timeSeries.trialIndex(: ,1);
startFR = firingRateFuncInv(  max(mean( timeSeries.y(timeIndices )), 1e-20));

timeIndices = timeSeries.trialIndex(timeSeries.choice == 1 ,2);
timeIndices = [timeIndices;timeIndices-1;timeIndices-2]; 
endFR1 = firingRateFuncInv(  max(mean( timeSeries.y(timeIndices )), 1e-20));

timeIndices = timeSeries.trialIndex(timeSeries.choice == 2 ,2);
timeIndices = [timeIndices;timeIndices-1;timeIndices-2]; 
endFR2 = firingRateFuncInv(  max(mean( timeSeries.y(timeIndices )), 1e-20));

initialGamma = max([10, startFR,endFR1,endFR2]); %initial gamma is the max of: beginning firing rate, end trial firing rate for choice 1, or end trial firing rate for choice 2 trials 
initialGamma = min(initialGamma,160); %keep initial gamma within some bounds


%% Sets up space for sampling --------------------------------------
RampSamples.betas        = zeros(totalSamples,NC);
RampSamples.w2s          = zeros(totalSamples,1);
RampSamples.auxThreshold = zeros(NT,totalSamples); %auxiliary variable to say when (if) bound was hit on each trial for each sample of lambda


RampSamples.l_0      = zeros(totalSamples,1);
RampSamples.gammas   = zeros(totalSamples,1);
RampSamples.biases   = zeros(totalSamples,1);

acceptanceCount.g  = 0;
acceptanceCount.sample = zeros(totalSamples,1);

%special functions that save temp files to keep latent variables from taking over too much RAM
LatentDataHandler.DataFolder = params.tempDataFolder;
LatentDataHandler = resetLatentsDB(length(timeSeries.y), totalSamples,LatentDataHandler);
%LatentDataHandler = saveLatentsDB(RampingFit.lambdas,1,LatentDataHandler);

%% initial values
RampSamples.betas(1,:) = sort(mvnrnd(zeros(NC,1),0.001*eye(NC)),'ascend'); % 0
RampSamples.w2s(1,:)   = 5e-4 + 0.0045*rand(1,1); % 0.005
RampSamples.l_0(1)     = max(0.1,min(0.9, normrnd(startFR/initialGamma,0.01)));%0.5;
RampSamples.biases(1)  = max(0.5,normrnd(1,0.1));
RampSamples.gammas(1)  = max(1e-20,normrnd(initialGamma-RampSamples.biases(1),1));

RampSamples.rb.sig = zeros([NC+1,totalSamples]); %keeps around variables for potential Rao-Blackwell estimates over betas (I don't use these)
RampSamples.rb.mu  = zeros([NC+1,totalSamples]);

RampSamples.latent_sum = zeros(TT,1);
RampSamples.latent_sum_sqr = zeros(TT,1);
RampSamples.latent_total = 0;

%% prior parameters setup
%makes prior param structures fit the number of params
%  -the same prior might be used for several param values

beta_mu = params.rampPrior.beta_mu;
if(NC > 1 && length(beta_mu) == 1)
    beta_mu = repmat(beta_mu,NC,1);
end
beta_sigma = params.rampPrior.beta_sigma;
if(NC > 1 && length(beta_sigma) == 1)
    beta_sigma = repmat(beta_sigma,NC,1);
end


p_init  = zeros(NC+1,NC+1);
p = zeros(size(p_init));
c_init  = zeros(NC+1,1);
c = zeros(size(c_init));

for b = 1:NC
    p_init(b,b) = 1/beta_sigma(b).^2;
    c_init(b)     = beta_mu(b) / beta_sigma(b).^2;
end
c_init(end)     = params.rampPrior.l0_mu/params.rampPrior.l0_sigma^2;
p_init(end,end) = 1/params.rampPrior.l0_sigma^2;


%% Setting up the GPU variables
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

lambdaBlockSize = 50; %how often to pull samples back from GPU

lambdaCounter  = 0;
lambdaBlockNum = 0;
[LB,LatentDataHandler] = loadLatentsDB(1:min(lambdaBlockSize,totalSamples),LatentDataHandler);
gpu_lambda       = kcArrayToGPU( LB); %latent variables are loaded/unloaded in blocks to the GPU
gpu_auxThreshold = kcArrayToGPUint( int32(RampSamples.auxThreshold(:,1:min(lambdaBlockSize,totalSamples))));
gpu_y            = kcArrayToGPU( timeSeries.y);
gpu_trIndex      = kcArrayToGPUint(int32(trIndex));      
gpu_trBetaIndex  = kcArrayToGPUint(int32(betaVector)); 

%% run the sampler

fprintf('Starting Ramping MCMC sampler...\n');

for ss = 2:totalSamples
    if(mod(ss,50) == 0 || ss == totalSamples)
        fprintf('  Ramping MCMC sample %d / %d \n', ss, totalSamples);
    end
    
    %% sample latent states
    c(1:end)       = c_init;
    p(1:end,1:end) = p_init;
    gpu_lambdaN       = kcArrayGetColumn(gpu_lambda,mod(lambdaCounter+1,lambdaBlockSize));
    gpu_auxThresholdN = kcArrayGetColumnInt(gpu_auxThreshold,mod(lambdaCounter+1,lambdaBlockSize));

    kcRampPathSamplerBias(gpu_lambdaN,gpu_auxThresholdN,gpu_y,gpu_trIndex,gpu_trBetaIndex,RampSamples.betas(ss-1,:),RampSamples.w2s(ss-1),RampSamples.l_0(ss-1),RampSamples.gammas(ss-1),timeSeries.delta_t, params.rampSampler.numParticles, params.rampSampler.minNumParticles,params.rampSampler.sigMult,maxTrLength, c, p, RampSamples.biases(ss-1));
    
    % if new sample is to be kept, add the summary statistics of the latent path
    if ss > params.MCMC.burnIn && mod(ss-params.MCMC.burnIn-1,params.MCMC.thinRate)==0;
        lambdaN = kcArrayToHost(gpu_lambdaN);
        RampSamples.latent_sum = RampSamples.latent_sum+lambdaN(:);
        RampSamples.latent_sum_sqr = RampSamples.latent_sum_sqr+(lambdaN(:).^2);
        RampSamples.latent_total = RampSamples.latent_total+1;
        clear lambdaN
    end
    
    lambdaCounter = mod(lambdaCounter+1,lambdaBlockSize);
    if(lambdaCounter == lambdaBlockSize-1) 
        LatentDataHandler = saveLatentsDB(kcArrayToHost(gpu_lambda),(1:lambdaBlockSize) + lambdaBlockNum*lambdaBlockSize,LatentDataHandler);
        RampSamples.auxThreshold(:,(1:lambdaBlockSize) + lambdaBlockNum*lambdaBlockSize) = kcArrayToHostint(gpu_auxThreshold);
        lambdaBlockNum = lambdaBlockNum + 1;
    end   
   
    %% Sample betas, l_0
    mu = p\c;
    sig = inv(p);
    RampSamples.rb.sig(:,ss) = diag(sig);
    RampSamples.rb.mu(:,ss)  = mu;
    
    maxSample = 100; %samples, and resamples l_0 until a value below 1 is found (truncating the multivariate normal)
    for sampleAttempt = 1:maxSample
        driftSample = mvnrnd(mu,sig);
        if(ss < 500 && ss < params.MCMC.burnIn)

            if(driftSample(end) < 0.95)
                break;
            elseif(maxSample == sampleAttempt)
                fprintf('Warning: l_0 going too high! Attemping to correct...\n');
                driftSample(end) = 0.95;
            end
        else
            if(driftSample(end) < 1)
                break;
            elseif(maxSample == sampleAttempt)
                fprintf('Warning: l_0 going too high! Attemping to correct...\n');
                driftSample(end) = 1 - 1e-4;
            end
        end
    end
        
    RampSamples.betas(ss,:) = driftSample(1:end-1);

    RampSamples.l_0(ss) = driftSample(end);
    
    if(sum(isnan( driftSample))>0)
        RampSamples.l_0(ss)     = RampSamples.l_0(ss-1);
        RampSamples.betas(ss,:) = RampSamples.betas(ss-1,:);
        warning('Unknown problem with sampling drift rates (most likely numerical error). Keeping previous sample.');
    end
     
    
    %% Sample w^2
    [w1_c, w2_c] = kcRampVarianceSampler(gpu_lambdaN,gpu_auxThresholdN,gpu_trIndex,gpu_trBetaIndex,RampSamples.betas(ss,:),RampSamples.l_0(ss));
    w2s_1 = sum(w1_c);
    w2s_2 = sum(w2_c);
    
    RampSamples.w2s(ss) = 1./gamrnd(params.rampPrior.w2_shape + w2s_1,1./(params.rampPrior.w2_scale + w2s_2)); %gamrnd does not use (alpha,beta) param, uses the one with theta on the wikipedia page for gamma dist
    if(isnan( RampSamples.w2s(ss) ) )
        RampSamples.w2s(ss)     = RampSamples.w2s(ss-1);
        warning('Unknown problem with sampling drift variance (most likely numerical error). Keeping previous sample.');
        if(params.rampPrior.w2_shape <= 1)
            warning('Note: the current prior on the drift variance does not have a mean (shape <= 1). Suggested alternate values for a more constraining prior are given in setupMCMCParams.m');
            
        end
    end    
    
    
    %% Step size setup for MALA on parameters
    if(params.rampSampler.learnStepSize && ss < params.MCMC.burnIn)
        if(ss <= 2)
            g_delta = params.rampSampler.epsilon_init;
            fprintf('Starting Langevin step size at %f\n',g_delta);
        elseif(mod(ss-1, params.rampSampler.MALAadjust) == 0)
            acceptPercent = mean(acceptanceCount.sample(ss-params.rampSampler.MALAadjust:ss-1));
            if(g_delta > params.rampSampler.epsilon_min && (acceptPercent < params.rampSampler.accept_min))
                g_delta = max(g_delta/params.rampSampler.adjustRate,params.rampSampler.epsilon_min);
                fprintf('Adjusting Langevin step size down to %f\n',g_delta);
            elseif(g_delta < params.rampSampler.epsilon_max && (acceptPercent > params.rampSampler.accept_max))
                g_delta = min(g_delta*params.rampSampler.adjustRate,params.rampSampler.epsilon_max);
                fprintf('Adjusting Langevin step size up to %f\n',g_delta);
            end
        end
    elseif(ss < params.MCMC.burnIn)
    	g_delta = params.rampSampler.epsilon_init;
    end
    if(ss == params.MCMC.burnIn + 1 && ~isnan(params.rampSampler.epsilon_fixed))
        g_delta = params.rampSampler.epsilon_fixed;
        fprintf('Fixing Langevin step size to %f\n',g_delta);
    end
    
    %% MALA (Metropolis-Adjusted Langevin Algorithm) sample gamma
    % this MALA proposal is conditioned using the fisher information of gamma
    gamma_a = params.rampPrior.gammaAlpha;
    gamma_b = params.rampPrior.gammaBeta;
    bias_a = 1;
    bias_b = 0.01;
    
    G_prior = (gamma_a-1)/RampSamples.gammas(ss-1)^2;
    der_log_prior = (gamma_a-1)/RampSamples.gammas(ss-1) - gamma_b;
    bias_g = (bias_a-1)/RampSamples.biases(ss-1)^2;
    bias_l = (bias_a-1)/RampSamples.biases(ss-1) - bias_b;
    
[log_p_lambda, der_log_p_y, G_log_p_y, der_log_p_y_bias, G_log_p_y_bias, G_off_diag] = kcRampBoundHeightSamplerBias(gpu_lambdaN,gpu_auxThresholdN,gpu_y,gpu_trIndex,RampSamples.gammas(ss-1),timeSeries.delta_t,G_prior,der_log_prior,RampSamples.biases(ss-1),bias_g,bias_l);
    dL = [der_log_p_y; der_log_p_y_bias];
G = [G_log_p_y G_off_diag; G_off_diag G_log_p_y_bias];
    
    p_mu = [RampSamples.gammas(ss-1); RampSamples.biases(ss-1)] + 1/2*g_delta^2*(G \ dL);
p_sig = (g_delta^2)*inv(G);
    
    params_star = mvnrnd(p_mu,p_sig)'; 
    gamma_star = params_star(1);
    bias_star = params_star(2);

    log_q_star = -2/2*log(2*pi) - 1/2*log(det(p_sig)) - 1/2*((params_star-p_mu)'/p_sig*(params_star-p_mu));

    G_prior_star = (gamma_a-1)/gamma_star^2;
    der_log_prior_star = (gamma_a-1)/gamma_star - gamma_b; 
    bias_g_star = (bias_a-1)/bias_star^2;
    bias_l_star = (bias_a-1)/bias_star - bias_b;
    
[log_p_lambda_star, der_log_p_y_star, G_log_p_y_star, der_log_p_y_bias_star, G_log_p_y_bias_star, G_off_diag_star] = kcRampBoundHeightSamplerBias(gpu_lambdaN,gpu_auxThresholdN,gpu_y,gpu_trIndex,gamma_star,timeSeries.delta_t,G_prior_star,der_log_prior_star,bias_star,bias_g_star,bias_l_star);
    dL_star = [der_log_p_y_star; der_log_p_y_bias_star];
G_star = [G_log_p_y_star G_off_diag_star; G_off_diag_star G_log_p_y_bias_star];

    p_mu_star = params_star + 1/2*g_delta^2*(G_star \ dL_star);
p_sig_star = (g_delta^2)*inv(G_star);
    
    log_q = -2/2*log(2*pi) - 1/2*log(det(p_sig_star)) - 1/2*(([RampSamples.gammas(ss-1); RampSamples.biases(ss-1)]-p_mu_star)'/p_sig_star*([RampSamples.gammas(ss-1); RampSamples.biases(ss-1)]-p_mu_star));
    
    log_p_bias = bias_a*log(bias_b) - gammaln(bias_a) + (bias_a-1) * log(RampSamples.biases(ss-1))    - bias_b*RampSamples.biases(ss-1);
    log_p_bias_star = bias_a*log(bias_b) - gammaln(bias_a) + (bias_a-1) * log(bias_star)    - bias_b*bias_star;
    
    if(gamma_a > 0 && gamma_b > 0)
        log_p      = log_p_lambda      + gamma_a*log(gamma_b) - gammaln(gamma_a) + (gamma_a-1) * log(RampSamples.gammas(ss-1))    - gamma_b*RampSamples.gammas(ss-1);
        log_p_star = log_p_lambda_star + gamma_a*log(gamma_b) - gammaln(gamma_a) + (gamma_a-1) * log(gamma_star) - gamma_b*gamma_star;
    else
        log_p      = log_p_lambda      - log(RampSamples.gammas(ss-1));
        log_p_star = log_p_lambda_star - log(gamma_star);
    end
    log_p = log_p + log_p_bias;
    log_p_star = log_p_star + log_p_bias_star;
    
    log_a = log_p_star + log_q - log_p - log_q_star;
    lrand = log(rand);
    if(gamma_star > 0 && bias_star > 0 && lrand < log_a)
        RampSamples.gammas(ss) = gamma_star;
        RampSamples.biases(ss) = bias_star;
        acceptanceCount.g = acceptanceCount.g+1;
        acceptanceCount.sample(ss) = 1;
    else
        RampSamples.gammas(ss) = RampSamples.gammas(ss-1);
        RampSamples.biases(ss) = RampSamples.biases(ss-1);
        acceptanceCount.sample(ss) = 0;
    end
    
end


%% finish up---------------------------------------
thinRate = params.MCMC.thinRate;

%get sampling stats for path
try
    [RampFit.lambdas.mean,LatentDataHandler]   = meanLatentsDB((params.MCMC.burnIn+1):thinRate:totalSamples,LatentDataHandler);
catch exc %#ok<NASGU>
    RampFit.lambdas.mean   = [];
end
RampFit.auxThreshold.mean   = mean(RampSamples.auxThreshold(:,params.MCMC.burnIn+1:thinRate:end),2);
RampFit.auxThreshold.median = median(RampSamples.auxThreshold(:,params.MCMC.burnIn+1:thinRate:end),2);
RampFit.auxThreshold.std    = std(RampSamples.auxThreshold(:,params.MCMC.burnIn+1:thinRate:end),[],2);

RampFit.beta.mean  = mean(RampSamples.betas(params.MCMC.burnIn+1:thinRate:end,:))';
RampFit.w2.mean    = mean(RampSamples.w2s(params.MCMC.burnIn+1:thinRate:end))';
RampFit.gamma.mean = mean(RampSamples.gammas(params.MCMC.burnIn+1:thinRate:end))';
RampFit.l_0.mean   = mean(RampSamples.l_0(params.MCMC.burnIn+1:thinRate:end))';
RampFit.bias.mean = mean(RampSamples.biases(params.MCMC.burnIn+1:thinRate:end))';
RampFit.latent_path.mean = (RampSamples.latent_sum)/RampSamples.latent_total;
RampFit.latent_path.var = (RampSamples.latent_sum_sqr - (RampSamples.latent_sum.*RampSamples.latent_sum) / RampSamples.latent_total) / (RampSamples.latent_total-1);

RampFit.beta.interval  = prctile(RampSamples.betas(((params.MCMC.burnIn+1):thinRate:end),:),[2.5 97.5],1);
RampFit.w2.interval    = prctile(RampSamples.w2s((params.MCMC.burnIn+1):thinRate:end,:),[2.5 97.5],1);
RampFit.l_0.interval   = prctile(RampSamples.l_0((params.MCMC.burnIn+1):thinRate:end),[2.5 97.5],1);
RampFit.gamma.interval = prctile(RampSamples.gammas((params.MCMC.burnIn+1):thinRate:end),[2.5 97.5],1);
RampFit.bias.interval = prctile(RampSamples.biases((params.MCMC.burnIn+1):thinRate:end),[2.5 97.5],1);

try 
    kcFreeGPUArray(gpu_y);
    kcFreeGPUArray(gpu_lambda);
    kcFreeGPUArray(gpu_auxThreshold);
    kcFreeGPUArray(gpu_trIndex);
    kcFreeGPUArray(gpu_trBetaIndex);
catch e
    fprintf('Error clearing cuda memory: %s\n',e);
end

fprintf('Ramping model sampler complete.\n');
end
