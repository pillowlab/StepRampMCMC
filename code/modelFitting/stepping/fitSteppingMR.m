%% Fit a stepping model to the observed spiking trials by sampling over the 
%  posterior distribution p(\Theta,z,s|y) where
%   y = observed spikes
%   \Theta = model parameters
%   z,s    = latent variables (step times and step directions)
%  by taking only the samples of \Theta, we obtain a sampled estimate of
%         p(\Theta|y)
%
% model params
%  alpha - firing rates. 3 values (one for each init, up, and down states). Takes values [0, \infty)
%  p     - step time parameter. 1 value per stimulus/coherence level. Takes values [0, 1]
%  r     - step time parameter. 1 value. Takes values (0, \infty)
%  phi   - step up/down probability. 1 value per stimulus/coherence level. Takes values [0, 1]
%
% Model of a trial (trial number j)
%   z ~ negative binomial(p(timeSeries.trCoh(j)),r)     p is stimulus coherence dependent, r is shared across all trials
%   s ~ bernoulli(phi(timeSeries.trCoh(j)))   valued on set {2,3} instead of {0,1} for indexing (P(s=3) = phi),   phi is stimulus coherence dependent
%
%
%   for t = 1...T  (T=trial length)
%   (y(t) | t<z)      ~ Poisson(alpha(1) * timeSeries.delta_t)
%   (y(t) | t>=z,s=2) ~ Poisson(alpha(2) * timeSeries.delta_t)
%   (y(t) | t>=z,s=3) ~ Poisson(alpha(3) * timeSeries.delta_t)
%
% Model fiting outputs
%   StepSamples.alpha = firing rates per each state (3,numSamples)
%   StepSamples.r     = negative binomial failure number parameter (1,numSamples)
%   StepSamples.p     = negative binomial success probability parameter (numCoherences,numSamples)
%   StepSamples.phi   = probability of switching to up or down state from initial state (numCoherences,numSamples)
%   StepSamples.z     = sampled switch times per trial (NT,numSamples)
%   StepSamples.s     = state switched to in each trial (NT,numSamples);
%
%   StepSamples.spikeStats = summary of spike count (column 1) and number of observations (column 2) for each state
%                            Used for sampling
%
%
%   StepSamples includes the burnin period samples 
%
%   StepFit contains sample mean of each parameter (after throwing out burnin and thinning according to params.MCMC.thin)
%           and a 95% credible interval. (This structure summarizes the StepSamples)
%        e.g., StepFit.alpha.mean   contains the posterior mean over the firing rate parameters
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
%       alpha(i) ~ gamma(params.stepPrior.alpha.shape,params.stepPrior.alpha.rate)
%         alpha(3)>alpha(2)
%       phi      ~ beta(params.stepPrior.switchto.alpha,params.stepPrior.switchto.beta)
%       p        ~ beta(params.stepPrior.negBin.succAlpha,params.stepPrior.negBin.succBeta)
%       r        ~ gamma(params.stepPrior.negBin.failShape,params.stepPrior.negBin.failRate)
%


function [StepFit, StepSamples] = fitSteppingModelMR(timeSeries,params)           

totalSamples = params.MCMC.nSamples+params.MCMC.burnIn;
timeSeries = setupTrialIndexStructure(timeSeries);

%% get trial info

NT = size(timeSeries.trialIndex,1);
TT = length(timeSeries.y);
NC = max(timeSeries.trCoh);
NTC = zeros(NC,1);
trialsOfCoh = zeros(NT,NC);
for cc = 1:NC
    trialsOfCoh(:,cc) = timeSeries.trCoh == cc;
    NTC(cc) = sum(trialsOfCoh(:,cc));
end
trialsOfCoh = logical(trialsOfCoh);

%% setup the sampler space
StepSamples.alpha = zeros(3,totalSamples);
StepSamples.m     = zeros(NC,totalSamples);
StepSamples.p     = zeros(NC,totalSamples);
StepSamples.phi   = zeros(NC,totalSamples);
StepSamples.r      = zeros(1,totalSamples);

StepSamples.z      = zeros(NT,totalSamples);
StepSamples.s      = zeros(NT,totalSamples);
acceptanceCount.sample = zeros(totalSamples,1);

StepSamples.spikeStats = zeros(3,2,totalSamples); %summary of spike count (column 1) 
           %and number of observations (column 2) for each state
           %Used for sampling the alphas

%saves extra info for Rao-Blackwellized estimates of alpha,p, and phi (these estimates do not exist for r)
StepSamples.rb.alpha = zeros(3,totalSamples);
StepSamples.rb.p     = zeros(NC,totalSamples);
StepSamples.rb.phi   = zeros(NC,totalSamples);


%% initial parameters for stepping
StepSamples.m(:,1) = normrnd(30,5,[NC 1]);
StepSamples.r(1) = max(0.5,normrnd(1,0.05)); % 1

StepSamples.phi(:,1) = mvnrnd(0.5*ones(NC,1),(0.05)^2*eye(NC)); %0.5 %initial probability of stepping up for all stimulus levels

%% initializes firing rates - state 0 as average rate in bin 1, and states 2,3 as firing rates in last 3 bins for in/out choices
StepSamples.alpha(1,1) = mean(timeSeries.y(timeSeries.trialIndex(:,1)))/timeSeries.delta_t;

timeIndices = timeSeries.trialIndex(timeSeries.choice == 1 ,2);
timeIndices = [timeIndices;timeIndices-1;timeIndices-2]; 
endFR1 =  max(mean( timeSeries.y(timeIndices )), 1e-20)/timeSeries.delta_t;

timeIndices = timeSeries.trialIndex(timeSeries.choice == 2 ,2);
timeIndices = [timeIndices;timeIndices-1;timeIndices-2]; 
endFR2 =  max(mean( timeSeries.y(timeIndices )), 1e-20)/timeSeries.delta_t;

StepSamples.alpha(2,1) = min(endFR1,endFR2);
StepSamples.alpha(3,1) = max(endFR1,endFR2);

StepSamples.alpha(1,1) = max(1e-20,normrnd(StepSamples.alpha(1,1),1));
StepSamples.alpha(2,1) = max(1e-20,normrnd(StepSamples.alpha(2,1),1));
StepSamples.alpha(3,1) = max(1e-20,normrnd(StepSamples.alpha(3,1),1));

%% MALA step-size info for r param
StepSamples.acceptanceCount.r = zeros(totalSamples,1);

if(params.stepSampler.learnStepSize)
    epsilon = params.stepSampler.epsilon_init(1);
else
    epsilon = params.stepSampler.epsilon_init(1);
end

%% setup GPU variables to run the latent state sampler in parallel for all trials

gpu_y = kcArrayToGPU(timeSeries.y);
trIndex = zeros(NT+1,1);
trCoh = timeSeries.trCoh-1;
trIndex(1:NT) = timeSeries.trialIndex(:,1)-1;
trIndex(end) = TT;
gpu_trIndex = kcArrayToGPUint(int32(trIndex));
gpu_trCoh = kcArrayToGPUint(int32(trCoh));

fprintf('Starting Stepping MCMC sampler...\n');

%% run the sampler
for ss = 2:totalSamples
    if(mod(ss,250) == 0 || ss == totalSamples)
        fprintf('  Stepping MCMC sample %d / %d \n', ss, totalSamples);
    end
    %% sample the latent states (z and s)
    
    %output from GPU
    % z = sampled switch times per trial
    % s = sampled state that was switched to per trial
    % spikeStats = (3,2) matrix, each row contains observed spike count statistics per each hidden state
    %     column 1 = total spike count observed while in each state
    %     column 2 = num observations made in each state
    
    %precalculate some negative binomial stats for the sampler (this is so much easier in MATLAB than in c)
    nbPDF = zeros(params.stepSampler.maxJumpTime,NC);
    for cc = 1:NC
        x = (0:(params.stepSampler.maxJumpTime-1))';
        r_c = StepSamples.r(ss-1);
        p_c = StepSamples.m(cc,ss-1)/(StepSamples.m(cc,ss-1)+r_c);
        nbPDF(:,cc) = gammaln(x+r_c) - (gammaln(x+1)+gammaln(r_c)) + r_c*log(1-p_c) + x.*log(p_c);
    end
    
    [StepSamples.z(:,ss), StepSamples.s(:,ss), StepSamples.spikeStats(:,:,ss)] = kcStepTimeSampler(gpu_y,gpu_trIndex,gpu_trCoh,StepSamples.alpha(:,ss-1),StepSamples.phi(:,ss-1),nbPDF,timeSeries.delta_t);
    
    
    %% sample the firing rates, switch-to-state probabilities and p from the negative binomial (all independent given r,y,and latent states)
    
    %firing rate sampler
    improperPrior = (params.stepPrior.alpha.shape == 0) | (params.stepPrior.alpha.rate == 0); %detects if using an improper prior over firing rates
    improperPriorFixAlpha = 1;
    improperPriorFixBeta = 0.5; %in case an improper prior fails to work, uses these prior params instead to keep from crashing
                                %(i.e., no trial steps into one of the 2 "decision" states, which can happen early in sampling)
                                %NOTE: Default priors are proper - this variable will never be used in that case
    
    if(StepSamples.spikeStats(1,1,ss) == 0 && improperPrior)
        StepSamples.alpha(1,ss) = gamrnd(improperPriorFixAlpha, improperPriorFixBeta);
        fprintf('Improper prior error on alpha 1! Sampling from different gamma to partially correct this.\n');
    else
        StepSamples.alpha(1,ss) = gamrnd(params.stepPrior.alpha.shape + StepSamples.spikeStats(1,2,ss), 1/(params.stepPrior.alpha.rate + StepSamples.spikeStats(1,1,ss)));
    end
    
    if(StepSamples.spikeStats(2,1,ss) == 0 && improperPrior)
        StepSamples.alpha(2,ss) = gamrnd(improperPriorFixAlpha, improperPriorFixBeta);
        fprintf('Improper prior error on alpha 2! Sampling from different gamma to partially correct this.\n');
    else
        StepSamples.alpha(2,ss) = gamrnd(params.stepPrior.alpha.shape + StepSamples.spikeStats(2,2,ss), 1/(params.stepPrior.alpha.rate + StepSamples.spikeStats(2,1,ss)));
    end
    if(min(StepSamples.phi(:,ss-1) ) == 1)
        StepSamples.alpha(2,ss) = 0;
    end
    
    post_alpha3_1 = params.stepPrior.alpha.shape+ StepSamples.spikeStats(3,2,ss);
    post_alpha3_2 = params.stepPrior.alpha.rate + StepSamples.spikeStats(3,1,ss);
    if(StepSamples.spikeStats(3,1,ss) == 0 && improperPrior)
        post_alpha3_1 = improperPriorFixAlpha;
        post_alpha3_2 = improperPriorFixBeta;
        fprintf('Improper prior error on alpha 3! Sampling from different gamma to partially correct this.\n');
    end
    gamMax = gamcdf(StepSamples.alpha(2,ss),post_alpha3_1,1/post_alpha3_2);
    
    StepSamples.rb.alpha(1,ss) = (params.stepPrior.alpha.shape + StepSamples.spikeStats(1,2,ss))/(params.stepPrior.alpha.rate + StepSamples.spikeStats(1,1,ss));
    StepSamples.rb.alpha(2,ss) = (params.stepPrior.alpha.shape + StepSamples.spikeStats(2,2,ss))/(params.stepPrior.alpha.rate + StepSamples.spikeStats(2,1,ss));
    StepSamples.rb.alpha(3,ss) = (params.stepPrior.alpha.shape + StepSamples.spikeStats(3,2,ss))/(params.stepPrior.alpha.rate + StepSamples.spikeStats(3,1,ss));
    
    rNum   = min(gamMax + (1-gamMax)*rand,1-1e-3);
    StepSamples.alpha(3,ss) = gaminv(rNum,post_alpha3_1,1/post_alpha3_2);
    
    if(isinf(StepSamples.alpha(3,ss)) || isnan(StepSamples.alpha(3,ss)) || StepSamples.alpha(3,ss) > 20 || StepSamples.alpha(3,ss) < StepSamples.alpha(2,ss))
        StepSamples.alpha(3,ss) = StepSamples.alpha(2,ss)+1e-3;
    end
    
    StepSamples.alpha(:,ss) = StepSamples.alpha(:,ss)./timeSeries.delta_t; %scale new parameters with time bin size - makes alpha in terms of spikes/second
    
    %switch to state sampler
    for cc = 1:NC
        switchesToState3 = sum(StepSamples.s(trialsOfCoh(:,cc),ss) == 3);
        post_alpha = switchesToState3 + params.stepPrior.switchto.alpha;
        post_beta = NTC(cc) - switchesToState3 + params.stepPrior.switchto.beta;
        
        StepSamples.phi(cc,ss) = betarnd(post_alpha,post_beta);
        
        StepSamples.rb.phi(cc,ss) = post_alpha/(post_alpha+post_beta); 
    end

    StepSamples.phi(StepSamples.phi(:,1) == 1,ss) = 1;
    
    %% Setup MALA for m sampling
    if(params.rampSampler.learnStepSize && ss < params.MCMC.burnIn)
        if(ss <= 2)
            eps = params.rampSampler.epsilon_init;
            fprintf('Starting Langevin step size at %f\n',eps);
        elseif(mod(ss-1, params.rampSampler.MALAadjust) == 0)
            acceptPercent = mean(acceptanceCount.sample(ss-params.rampSampler.MALAadjust:ss-1));
            if(eps > params.rampSampler.epsilon_min && (acceptPercent < params.rampSampler.accept_min))
                eps = max(eps/params.rampSampler.adjustRate,params.rampSampler.epsilon_min);
                fprintf('Adjusting Langevin step size down to %f\n',eps);
            elseif(eps < params.rampSampler.epsilon_max && (acceptPercent > params.rampSampler.accept_max))
                eps = min(eps*params.rampSampler.adjustRate,params.rampSampler.epsilon_max);
                fprintf('Adjusting Langevin step size up to %f\n',eps);
            end
        end
    elseif(ss < params.MCMC.burnIn)
    	eps = params.rampSampler.epsilon_init;
    end
    if(ss == params.MCMC.burnIn + 1 && ~isnan(params.rampSampler.epsilon_fixed))
        eps = params.rampSampler.epsilon_fixed;
        fprintf('Fixing Langevin step size to %f\n',eps);
    end
    
    %% sample individual m's using a MALA step
    
	       alpha_m = 2; %1;
beta_m = 0.02; % 1e-3;
    
    m = StepSamples.m(:,ss-1); % p = m/(m+r);
    r = StepSamples.r(:,ss-1);    

    trM = m(timeSeries.trCoh);
    der_log_prior_m = (alpha_m-1)./m-beta_m;
    G_log_prior_m = (alpha_m-1)./(m.^2);

    for cc = 1:NC
        
        log_p_m = sum(gammaln(StepSamples.z(timeSeries.trCoh==cc,ss)+r)+StepSamples.z(timeSeries.trCoh==cc,ss).*log(trM(timeSeries.trCoh==cc)./(trM(timeSeries.trCoh==cc)+r)) + r*log(1-trM(timeSeries.trCoh==cc)./(trM(timeSeries.trCoh==cc)+r)) - gammaln(r)) + (alpha_m-1)*log(m(cc))-beta_m*m(cc);
        der_log_p_m = sum( (StepSamples.z(timeSeries.trCoh==cc,ss)*r) ./ (trM(timeSeries.trCoh==cc).^2 + trM(timeSeries.trCoh==cc)*r) - r./(trM(timeSeries.trCoh==cc)+r) ) + der_log_prior_m(cc);
        G = sum( r./ (trM(timeSeries.trCoh==cc).^2 + trM(timeSeries.trCoh==cc)*r) ) + G_log_prior_m(cc);
        
        p_mu = m(cc)+eps^2*1/2*( der_log_p_m ./ G );
        p_sig = eps^2*1./G;
        
        m_star = p_mu + sqrt(p_sig)*randn;

        der_log_prior_m_star = (alpha_m-1)./m_star-beta_m;
        G_log_prior_m_star = (alpha_m-1)./(m_star.^2);
        
        log_q_star = -(1)/2*log(2*pi)-1/2*log(det(p_sig))-1/2*((m_star-p_mu)'/p_sig*(m_star-p_mu));
        
        log_p_m_star = sum(gammaln(StepSamples.z(timeSeries.trCoh==cc,ss)+r)+StepSamples.z(timeSeries.trCoh==cc,ss).*log(m_star./(m_star+r)) + r*log(1-m_star./(m_star+r)) - gammaln(r)) + (alpha_m-1)*log(m_star)-beta_m*m_star;
        der_log_p_m_star = sum( (StepSamples.z(timeSeries.trCoh==cc,ss)*r) ./ (m_star.^2 + m_star*r) - r./(m_star+r) ) + der_log_prior_m_star;
        G_star = NTC(cc)*( r./ (m_star.^2 + m_star*r) ) + G_log_prior_m_star;

        p_mu_star = m_star+eps^2*1/2*( der_log_p_m_star ./ G_star );
        p_sig_star = eps^2*1./G_star;
        
        log_q = -(1)/2*log(2*pi)-1/2*log(det(p_sig_star))-1/2*((m(cc)-p_mu_star)'/p_sig_star*(m(cc)-p_mu_star));
        
        log_a = log_p_m_star - log_p_m + log_q - log_q_star;

        log_rand = log(rand);
        if(log_rand < log_a && m_star > 0)
            StepSamples.m(cc,ss) = m_star;
            acceptanceCount.sample(ss) = 1;
        else
            StepSamples.m(cc,ss) = StepSamples.m(cc,ss-1);
        end
        
    end
        
    %% sample r using a MALA step
    
    m = StepSamples.m(:,ss); % p = m/(m+r);
    r = StepSamples.r(:,ss-1);
    trM = m(timeSeries.trCoh);

    der_log_prior_r = (params.stepPrior.negBin.failShape-1)/r - params.stepPrior.negBin.failRate;
    
    log_p_r = sum(gammaln(StepSamples.z(:,ss)+r)+StepSamples.z(:,ss).*log(trM./(trM+r)) + r*log(1-trM./(trM+r))) - NT*gammaln(r) + (params.stepPrior.negBin.failShape-1)*log(r) - params.stepPrior.negBin.failRate*r;
    der_log_p_r = sum(psi(StepSamples.z(:,ss)+r) - StepSamples.z(:,ss)./(trM+r) + log(r./(r+trM)) + trM./(trM+r) ) - NT* psi(r) + der_log_prior_r;
    
    p_mu = r+epsilon^2*1/2*( der_log_p_r );
    p_sig = epsilon^2;
    
    r_star = p_mu + sqrt(p_sig)*randn;
            
    der_log_prior_r_star = (params.stepPrior.negBin.failShape-1)/r_star - params.stepPrior.negBin.failRate;
    
    if r_star > 0
        
        log_q_star = -(1)/2*log(2*pi)-1/2*log(det(p_sig))-1/2*((r_star-p_mu)'/p_sig*(r_star-p_mu));
        
        log_p_r_star = sum(gammaln(StepSamples.z(:,ss)+r_star)+StepSamples.z(:,ss).*log(trM./(trM+r_star)) + r_star*log(1-trM./(trM+r_star))) - NT*gammaln(r_star) + (params.stepPrior.negBin.failShape-1)*log(r_star) - params.stepPrior.negBin.failRate*r_star;
        der_log_p_r_star = sum(psi(StepSamples.z(:,ss)+r_star) - StepSamples.z(:,ss)./(trM+r_star) + log(r_star./(r_star+trM)) + trM./(trM+r_star) ) - NT* psi(r_star) + der_log_prior_r_star;
    
        p_mu_star = r_star+epsilon^2*1/2*( der_log_p_r_star );
        p_sig_star = epsilon^2;
        
        log_q = -(1)/2*log(2*pi)-1/2*log(det(p_sig_star))-1/2*((r-p_mu_star)'/p_sig_star*(r-p_mu_star));
        
        log_a = log_p_r_star - log_p_r + log_q - log_q_star;
    
    else
        
        log_a = -Inf;

    end
    
    log_rand = log(rand);
    if(log_rand < log_a && r_star > 0)
        StepSamples.r(ss) = r_star;
        StepSamples.acceptanceCount.r(ss) = 1;
    else
        StepSamples.r(ss) = StepSamples.r(ss-1);
    end
    
    StepSamples.p(:,ss) = StepSamples.m(:,ss)./(StepSamples.m(:,ss)+StepSamples.r(ss));

    %% adjust the epsilon for the MALA sampler on r - only can change before burn-in is finished
    if(params.stepSampler.learnStepSize && ss < params.MCMC.burnIn && mod(ss,params.stepSampler.MALAadjust) == 0)
        acceptancePercent = mean(StepSamples.acceptanceCount.r(max(1,ss-params.stepSampler.MALAadjust+1):ss));
        if(acceptancePercent < params.stepSampler.accept_min && epsilon > params.stepSampler.epsilon_min)
            epsilon = max(epsilon/params.stepSampler.adjustRate,params.stepSampler.epsilon_min);
        elseif(acceptancePercent > params.stepSampler.accept_max && epsilon < params.stepSampler.epsilon_max)
            epsilon = min(epsilon*params.stepSampler.adjustRate,params.stepSampler.epsilon_max);
        end
    elseif(~params.stepSampler.learnStepSize)
        if(ss < params.MCMC.burnIn)
            epsilon = params.stepSampler.epsilon_init(1);
        else
            epsilon = max(params.stepSampler.epsilon_init);
        end
    end
   
    
end


%% dump GPU variables
kcFreeGPUArray(gpu_y);
kcFreeGPUArray(gpu_trIndex);
kcFreeGPUArray(gpu_trCoh);

%% finalize sampler

StepFit.alpha.mean = mean(StepSamples.alpha(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.phi.mean   = mean(StepSamples.phi(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.m.mean           = mean(StepSamples.m(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.p.mean           = mean(StepSamples.p(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.r.mean           = mean(StepSamples.r(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.mean    = mean(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.mode    = mode(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.median  = median(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.std     = std(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[],2);
StepFit.stepDir.mean     = mean(StepSamples.s(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepDir.mode     = mode(StepSamples.s(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);

StepFit.p.interval     = prctile(StepSamples.p(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.m.interval     = prctile(StepSamples.m(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.phi.interval   = prctile(StepSamples.phi(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.alpha.interval = prctile(StepSamples.alpha(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.r.interval     = prctile(StepSamples.r(params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5]);


StepSamples.finalEpsilon = epsilon;

fprintf('Stepping model sampler complete.\n');
end
