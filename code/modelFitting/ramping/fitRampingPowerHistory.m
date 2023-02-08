%% Fit a ramping model with additive spike history to the observed spiking 
% trials by sampling over the posterior distribution p(\Theta,lambdas|y) where
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
%  h     - spike history filters. NH values for previous time bins. Takes values (-\infty, \infty)
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
%   y(t|t <  auxThreshold(j))      ~ Poisson(log(1+ exp(lambda(t)*gamma + spe(t)))* timeSeries.delta_t)
%   y(t|t >= auxThreshold(j))      ~ Poisson(log(1+ exp(          gamma + spe(t)))* timeSeries.delta_t)
%   
%   where spe(t) is the weighted sum of the past NH spike bins.
%
% Model fiting outputs
%   RampSamples.betas  = drift slopes (numSamples,numCoherences)
%   RampSamples.l_0    = initial drift position (numSamples,1)
%   RampSamples.w2s    = drift variance (numSamples,1)
%   RampSamples.gammas = bound height (or diffusion path scaling) parameter (numSamples,1)
%   RampSamples.hs     = spike history filter weights
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
%   timeSeries.SpikeHistory = NT * NH matrix, which holds the spiking
%                           history in the NH spike bins before the beginning of each trial. If
%                            this does not exist, the spike history will be set to zeros(NT,NH) with
%                             NH = 10
%
%   model priors
%       beta(i)  ~ normal(params.rampPrior.beta_mu, params.rampPrior.beta_sigma^2)
%       l_0      ~ normal(params.rampPrior.l0_mu, params.rampPrior.l0_sigma^2)
%       w2       ~ inverse-gamma(params.rampPrior.w2_shape,params.rampPrior.w2_scale)
%       gamma    ~ gamma(params.rampPrior.gammaAlpha,params.rampPrior.gammaBeta)
%       hs       ~ normal(params.spikeHistoryPrior.hMu,params.spikeHistoryPrior.hSig2)

function [ RampFit, RampSamples, LatentDataHandler] = fitRampingPowerHistory(timeSeries,params,pow)


totalSamples = params.MCMC.nSamples+params.MCMC.burnIn;
timeSeries = setupTrialIndexStructure(timeSeries);
TT = size(timeSeries.y,1);
NT = size(timeSeries.trialIndex,1);
NC = max(timeSeries.trCoh);

% if the SpikeHistory structure is present, get the number of spike history
% bins. If not, set the number of spike history bins to 10 and create a
% structure of zeros for the spike history before the trials.

try 
    NH = size(timeSeries.SpikeHistory,2);
catch
    NH = 10;
    timeSeries.SpikeHistory = zeros(NT,NH);
end

%% max firing rate (bound) initialization ------------------------------
firingRateFunc    = @(X) log(1+exp(X)).^(pow)*timeSeries.delta_t;
firingRateFuncInv = @(X) log(exp((X/timeSeries.delta_t).^(1/pow))-1);
timeIndices = timeSeries.trialIndex(: ,1);
startFR = firingRateFuncInv(  max(mean( timeSeries.y(timeIndices )), 1e-20));

timeIndices = timeSeries.trialIndex(timeSeries.choice == 1 ,2);
timeIndices = [timeIndices;timeIndices-1;timeIndices-2]; 
endFR1 = firingRateFuncInv(  max(mean( timeSeries.y(timeIndices )), 1e-20));

timeIndices = timeSeries.trialIndex(timeSeries.choice == 2 ,2);
timeIndices = [timeIndices;timeIndices-1;timeIndices-2]; 
endFR2 = firingRateFuncInv(  max(mean( timeSeries.y(timeIndices )), 1e-20));

initialGamma = max([2, startFR,endFR1,endFR2]); %initial gamma is the max of: beginning firing rate, end trial firing rate for choice 1, or end trial firing rate for choice 2 trials 

if pow == 2                                                                                                                
   initialGamma = min(initialGamma,15); %keep initial gamma within some bounds                                                  
end                                                                                                                             
initialGamma = min(initialGamma,10000);

%% Sets up space for sampling --------------------------------------
RampSamples.betas        = zeros(totalSamples,NC);
RampSamples.w2s          = zeros(totalSamples,1);
RampSamples.auxThreshold = zeros(NT,totalSamples); %auxiliary variable to say when (if) bound was hit on each trial for each sample of lambda

RampSamples.l_0      = zeros(totalSamples,1);
RampSamples.gammas   = zeros(totalSamples,1);
RampSamples.hs = zeros(totalSamples,NH);

RampSamples.latent_sum = zeros(TT,1);
RampSamples.latent_sum_sqr = zeros(TT,1);
RampSamples.latent_total = 0;

acceptanceCount.g  = 0;
acceptanceCount.sample = zeros(totalSamples,1);

%special functions that save temp files to keep latent variables from taking over too much RAM
LatentDataHandler.DataFolder = params.tempDataFolder;
LatentDataHandler = resetLatentsDB(length(timeSeries.y), totalSamples,LatentDataHandler);
%LatentDataHandler = saveLatentsDB(RampingFit.lambdas,1,LatentDataHandler);

%% initial values
RampSamples.betas(1,:) = sort(mvnrnd(zeros(NC,1),1e-4*eye(NC)),'ascend'); % originally 0
RampSamples.w2s(1,:)   = 5e-4 + 0.0045*rand(1,1); % 0.005
RampSamples.l_0(1)     = max(0.1,min(0.9, normrnd(startFR/initialGamma,0.01)));%0.5;
RampSamples.gammas(1)  = max(1e-20,normrnd(initialGamma,1));
RampSamples.hs(1,:)    = mvnrnd(zeros(NH,1),(0.1)^2*eye(NH));

RampSamples.rb.sig = zeros([NC+1,totalSamples]); %keeps around variables for potential Rao-Blackwell estimates over betas (I don't use these)
RampSamples.rb.mu  = zeros([NC+1,totalSamples]);


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
SpikeHistory = zeros(NT*NH,1);
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

lambdaBlockSize = 50; %how often to pull samples back from GPU

lambdaCounter  = 0;
lambdaBlockNum = 0;
[LB,LatentDataHandler] = loadLatentsDB(1:min(lambdaBlockSize,totalSamples),LatentDataHandler);
gpu_lambda       = kcArrayToGPU( LB); %latent variables are loaded/unloaded in blocks to the GPU
gpu_auxThreshold = kcArrayToGPUint( int32(RampSamples.auxThreshold(:,1:min(lambdaBlockSize,totalSamples))));
gpu_y            = kcArrayToGPU( timeSeries.y);
gpu_trIndex      = kcArrayToGPUint(int32(trIndex));      
gpu_trBetaIndex  = kcArrayToGPUint(int32(betaVector)); 
gpu_spikeHistory = kcArrayToGPU(SpikeHistory);

%% set up spike history effect - spe for first spike history filter samples
gpu_spe = kcGetSpikeHistoryEffect(gpu_y,gpu_trIndex,gpu_spikeHistory,RampSamples.hs(1,:));

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

    kcRampPathSamplerLog1PPowerMult(gpu_lambdaN,gpu_auxThresholdN,gpu_y,gpu_trIndex,gpu_trBetaIndex,RampSamples.betas(ss-1,:),RampSamples.w2s(ss-1),RampSamples.l_0(ss-1),RampSamples.gammas(ss-1),timeSeries.delta_t, params.rampSampler.numParticles, params.rampSampler.minNumParticles,params.rampSampler.sigMult,maxTrLength, c, p, gpu_spe, pow);
    
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
%         g_delta = params.rampSampler.epsilon_fixed;
        fprintf('Fixing Langevin step size to %f\n',g_delta);
    end
    
    %% MALA (Metropolis-Adjusted Langevin Algorithm) sample gamma and hs
    % prior parameter values on gamma and hs
    gamma_a = params.rampPrior.gammaAlpha;
    gamma_b = params.rampPrior.gammaBeta;
    h_mu = params.spikeHistoryPrior.hMu;
    h_sig2 = params.spikeHistoryPrior.hSig2;

    % derivative and Fisher information of log priors of current hs and gamma
    dL_prior = [-(RampSamples.hs(ss-1,:)'-h_mu)/h_sig2; (gamma_a-1)/RampSamples.gammas(ss-1)-gamma_b];
    H_prior = diag([1/h_sig2*ones(NH,1); (gamma_a-1)/RampSamples.gammas(ss-1)^2]);
    
    % compute log prob, gradients, hessian for current sampled values
    [log_p_lambda_h, der_log_p_y_h, H_log_p_y] = kcGammaSpikeHistorySamplerLog1PPowerMult(gpu_lambdaN,gpu_auxThresholdN,gpu_y,gpu_trIndex,RampSamples.gammas(ss-1),gpu_spe,timeSeries.delta_t,reshape(H_prior',[(NH+1)^2 1]),RampSamples.hs(ss-1,:),gpu_spikeHistory,dL_prior,pow);                                                                                                            
    
    % Only the lower diagonal of the Fisher Information matrix is computed,
    % so place lower diag (besides main daig) in upper diag of matrix
    H_log_p_y = tril(H_log_p_y)+triu(H_log_p_y',1);
    
    % compute mean and covariance of multivariate proposal distribution
    hp_mu = [RampSamples.hs(ss-1,:)'; RampSamples.gammas(ss-1)] + 1/2*g_delta.^2.*( (H_log_p_y) \ (der_log_p_y_h) );
    hp_sig = (g_delta(1))^2 * inv(H_log_p_y);

    % sample new values
    hs_star = mvnrnd(hp_mu,hp_sig); hs_star = hs_star(:);
        
    % compute log probability of sampled values under the proposal
    % distribution
    log_qh_star = -(NH+1)/2*log(2*pi)-1/2*log(det(hp_sig))-1/2*((hs_star(:)-hp_mu(:))'/hp_sig*(hs_star(:)-hp_mu(:)));
    
    % compute new spike history effect of sampled hs
    gpu_spe_star = kcGetSpikeHistoryEffect(gpu_y,gpu_trIndex,gpu_spikeHistory,hs_star(1:NH));
    if hs_star(NH+1) > 0

    % derivatives of log priors of sampled hs and gamma
    dL_prior_star = [-(hs_star(1:NH)-h_mu)/h_sig2; (gamma_a-1)/hs_star(NH+1)-gamma_b]; 
    H_prior_star = diag([1/h_sig2*ones(NH,1); (gamma_a-1)/hs_star(NH+1)^2]);

    % compute log prob, gradients, hessian for new sampled values
    [log_p_lambda_hstar, der_log_p_y_hstar, H_log_p_y_star] = kcGammaSpikeHistorySamplerLog1PPowerMult(gpu_lambdaN,gpu_auxThresholdN,gpu_y,gpu_trIndex,hs_star(NH+1),gpu_spe_star,timeSeries.delta_t,reshape(H_prior_star',[(NH+1)^2 1]),hs_star(1:NH),gpu_spikeHistory,dL_prior_star,pow);
    
    % Only the lower diagonal of the Fisher Information matrix is computed,
    % so place lower diag (besides main daig) in upper diag of matrix
    H_log_p_y_star = tril(H_log_p_y_star)+triu(H_log_p_y_star',1); 
    
    % compute proposal distribution of new sampled value
    hp_mu_star = hs_star + 1/2*g_delta.^2.*( (H_log_p_y_star) \ (der_log_p_y_hstar) );
    hp_sig_star = (g_delta(1))^2*inv(H_log_p_y_star);
    
    % compute log probability of original values under the sampled proposal
    % distribution
    log_qh = -(NH+1)/2*log(2*pi)-1/2*log(det(hp_sig_star))-1/2*(([RampSamples.hs(ss-1,:)'; RampSamples.gammas(ss-1)] - hp_mu_star(:))'/hp_sig_star*([RampSamples.hs(ss-1,:)'; RampSamples.gammas(ss-1)] -hp_mu_star(:)));
    
    % compute probabilities 
    log_ph = log_p_lambda_h + sum(-1/2*log(2*pi) - 1/2*log(h_sig2) - ( (RampSamples.hs(ss-1,:)'-h_mu).^2 ./ (2*h_sig2) ))+ gamma_a*log(gamma_b) - gammaln(gamma_a) + (gamma_a-1) * log(RampSamples.gammas(ss-1))    - gamma_b*RampSamples.gammas(ss-1);
    log_ph_star = log_p_lambda_hstar + sum(-1/2*log(2*pi) - 1/2*log(h_sig2) - ( (hs_star(1:NH) - h_mu).^2 ./ (2*h_sig2) ))+gamma_a*log(gamma_b) - gammaln(gamma_a) + (gamma_a-1) * log(hs_star(NH+1)) - gamma_b*hs_star(NH+1);

    % compute acceptance probability and random value
    log_a = log_ph_star + log_qh - log_ph - log_qh_star;
    
    else
        log_a = -Inf;
    end
    lrand = log(rand(1,1));
    
    % if random value is less than acceptance probability, the spike
    % history filter weights are real, and gamma is > 0, accept
    if(sum(imag(hs_star))==0 && hs_star(NH+1) > 0 && lrand < log_a)
        RampSamples.hs(ss,:) = hs_star(1:NH);
        RampSamples.gammas(ss) = hs_star(NH+1);
        acceptanceCount.g = acceptanceCount.g+1;
        acceptanceCount.sample(ss) = 1;
        try % if accept -> free old spike history and assign new spike history to old spike history variable
            kcFreeGPUArray(gpu_spe)
        catch e
            fprintf('Error clearing spike history memory: %s\n',e);
        end
        gpu_spe = gpu_spe_star;
    else
        RampSamples.hs(ss,:) = RampSamples.hs(ss-1,:);
        RampSamples.gammas(ss) = RampSamples.gammas(ss-1);
        acceptanceCount.sample(ss) = 0;
        try
            kcFreeGPUArray(gpu_spe_star)
        catch e
            fprintf('Error clearing spike history memory: %s\n',e);
        end
    end
    
    % if new sample is to be kept, add the summary statistics of the latent rate
    if ss > params.MCMC.burnIn && mod(ss-params.MCMC.burnIn-1,params.MCMC.thinRate)==0;
        lambdaN = kcArrayToHost(gpu_lambdaN);
        spe = kcArrayToHost(gpu_spe);
        fr = (log1p(exp(min(lambdaN,1)*RampSamples.gammas(ss))).^pow).*exp(spe);
        RampSamples.latent_sum = RampSamples.latent_sum+fr(:);
        RampSamples.latent_sum_sqr = RampSamples.latent_sum_sqr+(fr(:).^2);
        RampSamples.latent_total = RampSamples.latent_total+1;
        clear lambdaN
        clear spe
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
RampFit.hs.mean    = mean(RampSamples.hs(params.MCMC.burnIn+1:thinRate:end,:))';
RampFit.latent_path.mean = (RampSamples.latent_sum)/RampSamples.latent_total;
RampFit.latent_path.var = (RampSamples.latent_sum_sqr - (RampSamples.latent_sum.*RampSamples.latent_sum) / RampSamples.latent_total) / (RampSamples.latent_total-1);

RampFit.beta.interval  = prctile(RampSamples.betas(((params.MCMC.burnIn+1):thinRate:end),:),[2.5 97.5],1);
RampFit.w2.interval    = prctile(RampSamples.w2s((params.MCMC.burnIn+1):thinRate:end,:),[2.5 97.5],1);
RampFit.l_0.interval   = prctile(RampSamples.l_0((params.MCMC.burnIn+1):thinRate:end),[2.5 97.5],1);
RampFit.gamma.interval = prctile(RampSamples.gammas((params.MCMC.burnIn+1):thinRate:end),[2.5 97.5],1);
RampFit.hs.interval    = prctile(RampSamples.hs((params.MCMC.burnIn+1):thinRate:end,:),[2.5 97.5],1);

try 
    kcFreeGPUArray(gpu_y);
    kcFreeGPUArray(gpu_lambda);
    kcFreeGPUArray(gpu_auxThreshold);
    kcFreeGPUArray(gpu_trIndex);
    kcFreeGPUArray(gpu_trBetaIndex);
    kcFreeGPUArray(gpu_spikeHistory);
    kcFreeGPUArray(gpu_spe);
catch e
    fprintf('Error clearing cuda memory: %s\n',e);
end

fprintf('Ramping model sampler complete.\n');
end
