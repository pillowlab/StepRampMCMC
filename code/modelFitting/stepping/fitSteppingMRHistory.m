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


function [StepFit, StepSamples] = fitSteppingModelMult(timeSeries,params)           

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

% if the SpikeHistory structure is present, get the number of spike history
% bins. If not, set the number of spike history bins to 10 and create a
% structure of zeros for the spike history before the trials.
try 
    NH = size(timeSeries.SpikeHistory,2);
catch
    NH = 10;
    timeSeries.SpikeHistory = zeros(NT,NH);
end

%% setup the sampler space
StepSamples.alpha = zeros(3,totalSamples);
StepSamples.p     = zeros(NC,totalSamples);
StepSamples.phi   = zeros(NC,totalSamples);
StepSamples.r      = zeros(1,totalSamples);
StepSamples.m     = zeros(NC,totalSamples);

StepSamples.z      = zeros(NT,totalSamples);
StepSamples.s      = zeros(NT,totalSamples);
StepSamples.hs     = zeros(totalSamples,NH);

acceptanceCount.sample = zeros(totalSamples,1);
acceptanceCount2.sample = zeros(totalSamples,1);

StepSamples.spikeStats = zeros(3,2,totalSamples); %summary of spike count (column 1) 
           %and number of observations (column 2) for each state
           %Used for sampling the alphas

%saves extra info for Rao-Blackwellized estimates of alpha,p, and phi (these estimates do not exist for r)
StepSamples.rb.alpha = zeros(3,totalSamples);
StepSamples.rb.p     = zeros(NC,totalSamples);
StepSamples.rb.phi   = zeros(NC,totalSamples);

%% initial parameters for stepping
StepSamples.r(1) = max(0.5,normrnd(1,0.05)); % 1
%mu = 30; %average initial step time in terms of bins
StepSamples.m(:,1) = normrnd(30,5,[NC 1]);
StepSamples.p(:,1) = StepSamples.m(:,1)./(StepSamples.m(:,1)+StepSamples.r(1));

StepSamples.phi(:,1) = mvnrnd(0.5*ones(NC,1),(0.05)^2*eye(NC)); %0.5 %initial probability of stepping up for all stimulus levels
StepSamples.hs(1,:) = mvnrnd(zeros(NH,1),(0.1)^2*eye(NH));

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

if StepSamples.alpha(2,1) > StepSamples.alpha(3,1)
temp = StepSamples.alpha(3,1);
StepSamples.alpha(3,1) = StepSamples.alpha(2,1);
StepSamples.alpha(2,1) = temp;
end

%% Priors

h_mu = params.spikeHistoryPrior.hMu;
h_sig2 = params.spikeHistoryPrior.hSig2;

alpha_a = params.stepPrior.alpha.shape;
alpha_b = params.stepPrior.alpha.rate;

%% MALA step-size info for r param
StepSamples.acceptanceCount.r = zeros(totalSamples,1);

if(params.stepSampler.learnStepSize)
    epsilon = params.stepSampler.epsilon_init(1);
else
    epsilon = params.stepSampler.epsilon_init(1);
end

%% setup GPU variables to run the latent state sampler in parallel for all trials
SpikeHistory = zeros(NT*NH,1);
for tr = 1:NT
    SpikeHistory((tr-1)*NH+1:(tr*NH)) = timeSeries.SpikeHistory(tr,:);
end

gpu_y = kcArrayToGPU(timeSeries.y);
trIndex = zeros(NT+1,1);
trCoh = timeSeries.trCoh-1;
trIndex(1:NT) = timeSeries.trialIndex(:,1)-1;
trIndex(end) = TT;
gpu_trIndex = kcArrayToGPUint(int32(trIndex));
gpu_trCoh = kcArrayToGPUint(int32(trCoh));
gpu_spikeHistory = kcArrayToGPU(SpikeHistory);

%% get spike history effect for initial filter weights
gpu_spe = kcGetSpikeHistoryEffect(gpu_y,gpu_trIndex,gpu_spikeHistory,StepSamples.hs(1,:));
    
fprintf('Starting Stepping MCMC sampler...\n');

%% run the sampler
for ss = 2:totalSamples
    if(mod(ss,250) == 0 || ss == totalSamples)
        fprintf('  Stepping MCMC sample %d / %d \n', ss, totalSamples);
    end
    
    %% sample the latent states (z and s)
    
    %output from GPU - this may be wrong?
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
    
    % modified such that the firing rate is alpha*exp(h*yh)
    [StepSamples.z(:,ss), StepSamples.s(:,ss), StepSamples.spikeStats(:,:,ss)] = kcStepTimeSamplerMult(gpu_y,gpu_trIndex,gpu_trCoh,StepSamples.alpha(:,ss-1),StepSamples.phi(:,ss-1),nbPDF,timeSeries.delta_t,gpu_spe);
    
    %% Setup MALA for alpha and spike history sampling
    % uses ramp sampler params for sampling gamma and spike history
    
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
    
    %% MH sample alphas and history filters

    alphas = StepSamples.alpha(:,ss-1);
    hs = StepSamples.hs(ss-1,:);
    
    % prior gradient and fisher information
    der_log_prior_alphah = [(alpha_a-1)./alphas - alpha_b; -(hs(:)-h_mu)./h_sig2];
    G_prior_alphah = diag([(alpha_a-1)./(alphas.^2); 1/h_sig2*ones(NH,1)]);
    
    % compute log probability and gradient information for new samples
    [log_p_alphah, der_log_alphah, G_alphah] = kcAlphaSpikeHistorySamplerMult(gpu_y,gpu_trIndex,StepSamples.z(:,ss),StepSamples.s(:,ss),timeSeries.delta_t,gpu_spe,StepSamples.alpha(:,ss-1),StepSamples.hs(ss-1,:),der_log_prior_alphah,reshape(G_prior_alphah',[(NH+3)^2 1]),gpu_spikeHistory);
    G_alphah = tril(G_alphah)+triu(G_alphah',1);  

    % if the latent steps never step to one of the states across all                                                                                                          
    % trials, and the prior alpha_a = 1, the Fisher information matrix                                                                                                        
    % will be singular - if a state is never stepped to, replace its                                                                                                          
    % diagonal element in the fisher information matrix with 1                                                                                                                
    if G_alphah(1,1) == 0, G_alphah(1,1) = 1; fprintf('Fixing Alpha FIM %d \n', ss); end                                                                                      
    if G_alphah(2,2) == 0, G_alphah(2,2) = 1; fprintf('Fixing Alpha FIM %d \n', ss); end                                                                                      
    if G_alphah(3,3) == 0, G_alphah(3,3) = 1; fprintf('Fixing Alpha FIM %d \n', ss); end   
    
    % proposal distribution
    hp_mu = [alphas(:); hs(:)] + (1/2)*eps^2*(G_alphah \ der_log_alphah);
    hp_sig = eps^2 * inv(G_alphah);
    
    % sample new parameters 
    params_star = mvnrnd(hp_mu,hp_sig); params_star = params_star(:);
    alphas_star = params_star(1:3);
    hs_star = params_star(4:end);
    
    % probability of sampled parameters under proposal
    log_q_star = -(NH+3)/2*log(2*pi)-1/2*log(det(hp_sig))-1/2*((params_star(:)-hp_mu(:))'/hp_sig*(params_star(:)-hp_mu(:)));

    % get new spike history effect
    gpu_spe_star = kcGetSpikeHistoryEffect(gpu_y,gpu_trIndex,gpu_spikeHistory,hs_star);

    % prior gradient and fisher information of new sample
    der_log_prior_alphah_star = [(alpha_a-1)./alphas_star - alpha_b; -(hs_star(:)-h_mu)./h_sig2];
    G_prior_alphah_star = diag([(alpha_a-1)./(alphas_star.^2); 1/h_sig2*ones(NH,1)]);
    
    % compute log probability and gradient information for new samples
    [log_p_alphah_star, der_log_alphah_star, G_alphah_star] = kcAlphaSpikeHistorySamplerMult(gpu_y,gpu_trIndex,StepSamples.z(:,ss),StepSamples.s(:,ss),timeSeries.delta_t,gpu_spe_star,alphas_star,hs_star,der_log_prior_alphah_star,reshape(G_prior_alphah_star',[(NH+3)^2 1]),gpu_spikeHistory);
    G_alphah_star = tril(G_alphah_star)+triu(G_alphah_star',1);

    % if the latent steps never step to one of the states across all                                                                                                          
    % trials, and the prior alpha_a = 1, the Fisher information matrix
    % will be singular - if a state is never stepped to, replace its                 
    % diagonal element in the fisher information matrix with 1                                                                                                                
    if G_alphah_star(1,1) == 0, G_alphah_star(1,1) = 1; fprintf('Fixing Alpha Star FIM %d \n', ss); end                                                                      
    if G_alphah_star(2,2) == 0, G_alphah_star(2,2) = 1; fprintf('Fixing Alpha Star FIM %d \n', ss); end                                                                       
    if G_alphah_star(3,3) == 0, G_alphah_star(3,3) = 1; fprintf('Fixing Alpha Star FIM %d \n', ss); end
    
    % new proposal distribution
    hp_mu_star = [alphas_star; hs_star] + (1/2)*eps^2*(G_alphah_star \ der_log_alphah_star);
    hp_sig_star = eps^2 * inv(G_alphah_star);    

    % probability of previous samples under new proposal
    log_q = -(NH+3)/2*log(2*pi)-1/2*log(det(hp_sig_star))-1/2*(([alphas; hs(:)] - hp_mu_star(:))'/hp_sig_star*([alphas; hs(:)] -hp_mu_star(:)));

    % priors on alphas and hs                                                                                                                                                                                                                                
    if (alpha_a > 0 & alpha_b > 0)
        log_prior_alpha = sum(alpha_a*log(alpha_b) - gammaln(alpha_a) + (alpha_a-1) * log(alphas)    - alpha_b*alphas);
        log_prior_alpha_star = sum(alpha_a*log(alpha_b) - gammaln(alpha_a) + (alpha_a-1)*log(alphas_star) - alpha_b*alphas_star);
    else
        warning('prior on alpha is undefined')
    end

    log_prior_h = sum(-1/2*log(2*pi)-1/2*log(h_sig2)-(hs-h_mu).^2./(2*h_sig2));
    log_prior_h_star = sum(-1/2*log(2*pi)-1/2*log(h_sig2)-(hs_star-h_mu).^2./(2*h_sig2));

    log_p      = log_p_alphah      + log_prior_alpha       + log_prior_h;
    log_p_star = log_p_alphah_star + log_prior_alpha_star  + log_prior_h_star;
    
    % accept samples?
    log_a = log_p_star - log_p + log_q - log_q_star;
    lrand = log(rand);
    if( (alphas_star(1) > 0) && (alphas_star(2) > 0) && (alphas_star(3) > alphas_star(2)) && (lrand < log_a))
        StepSamples.alpha(:,ss) = alphas_star;
        StepSamples.hs(ss,:) = hs_star;
        acceptanceCount.sample(ss) = 1;
        try % if accept -> free old spike history and assign new spike history to old spike history variable
            kcFreeGPUArray(gpu_spe)
        catch e
            fprintf('Error clearing spike history memory: %s\n',e);
        end
        gpu_spe = gpu_spe_star;
    else
        StepSamples.alpha(:,ss) = StepSamples.alpha(:,ss-1);
        StepSamples.hs(ss,:) = StepSamples.hs(ss-1,:);
        acceptanceCount.sample(ss) = 0;
        try
            kcFreeGPUArray(gpu_spe_star)
        catch e
            fprintf('Error clearing spike history memory: %s\n',e);
        end
    end
        
    %% sample the switch-to-state probabilities from the negative binomial (all independent given r,y,and latent states)
    
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
            eps2 = params.rampSampler.epsilon_init;
            fprintf('Starting Langevin step size at %f\n',eps2);
        elseif(mod(ss-1, params.rampSampler.MALAadjust) == 0)
            acceptPercent2 = mean(acceptanceCount2.sample(ss-params.rampSampler.MALAadjust:ss-1));
            if(eps2 > params.rampSampler.epsilon_min && (acceptPercent2 < params.rampSampler.accept_min))
                eps2 = max(eps2/params.rampSampler.adjustRate,params.rampSampler.epsilon_min);
                fprintf('Adjusting Langevin step size down to %f\n',eps2);
            elseif(eps2 < params.rampSampler.epsilon_max && (acceptPercent2 > params.rampSampler.accept_max))
                eps2 = min(eps2*params.rampSampler.adjustRate,params.rampSampler.epsilon_max);
                fprintf('Adjusting Langevin step size up to %f\n',eps2);
            end
        end
    elseif(ss < params.MCMC.burnIn)
    	eps2 = params.rampSampler.epsilon_init;
    end
    if(ss == params.MCMC.burnIn + 1 && ~isnan(params.rampSampler.epsilon_fixed))
        eps2 = params.rampSampler.epsilon_fixed;
        fprintf('Fixing Langevin step size to %f\n',eps2);
    end
    
    %% sample individual m's using a MALA step
    
    alpha_m = 2;
    beta_m = 0.02; 
    
    m = StepSamples.m(:,ss-1); % p = m/(m+r);
    r = StepSamples.r(ss-1);    

    trM = m(timeSeries.trCoh);
    der_log_prior_m = (alpha_m-1)./m-beta_m;
    G_log_prior_m = (alpha_m-1)./(m.^2);

    for cc = 1:NC
        
        log_p_m = sum(gammaln(StepSamples.z(timeSeries.trCoh==cc,ss)+r)+StepSamples.z(timeSeries.trCoh==cc,ss).*log(trM(timeSeries.trCoh==cc)./(trM(timeSeries.trCoh==cc)+r)) + r*log(1-trM(timeSeries.trCoh==cc)./(trM(timeSeries.trCoh==cc)+r)) - gammaln(r)) + (alpha_m-1)*log(m(cc))-beta_m*m(cc);
        der_log_p_m = sum( (StepSamples.z(timeSeries.trCoh==cc,ss)*r) ./ (trM(timeSeries.trCoh==cc).^2 + trM(timeSeries.trCoh==cc)*r) - r./(trM(timeSeries.trCoh==cc)+r) ) + der_log_prior_m(cc);
        G = sum( r./ (trM(timeSeries.trCoh==cc).^2 + trM(timeSeries.trCoh==cc)*r) ) + G_log_prior_m(cc);
        
        p_mu = m(cc)+eps2^2*1/2*( der_log_p_m ./ G );
        p_sig = eps2^2*1./G;
        
        m_star = p_mu + sqrt(p_sig)*randn;

        der_log_prior_m_star = (alpha_m-1)./m_star-beta_m;
        G_log_prior_m_star = (alpha_m-1)./(m_star.^2);
        
        log_q_star = -(1)/2*log(2*pi)-1/2*log(det(p_sig))-1/2*((m_star-p_mu)'/p_sig*(m_star-p_mu));
        
        log_p_m_star = sum(gammaln(StepSamples.z(timeSeries.trCoh==cc,ss)+r)+StepSamples.z(timeSeries.trCoh==cc,ss).*log(m_star./(m_star+r)) + r*log(1-m_star./(m_star+r)) - gammaln(r)) + (alpha_m-1)*log(m_star)-beta_m*m_star;
        der_log_p_m_star = sum( (StepSamples.z(timeSeries.trCoh==cc,ss)*r) ./ (m_star.^2 + m_star*r) - r./(m_star+r) ) + der_log_prior_m_star;
        G_star = NTC(cc)*( r./ (m_star.^2 + m_star*r) ) + G_log_prior_m_star;

        p_mu_star = m_star+eps2^2*1/2*( der_log_p_m_star ./ G_star );
        p_sig_star = eps2^2*1./G_star;
        
        log_q = -(1)/2*log(2*pi)-1/2*log(det(p_sig_star))-1/2*((m(cc)-p_mu_star)'/p_sig_star*(m(cc)-p_mu_star));
        
        log_a = log_p_m_star - log_p_m + log_q - log_q_star;

        log_rand = log(rand);
        if(log_rand < log_a && m_star > 0)
            StepSamples.m(cc,ss) = m_star;
            acceptanceCount2.sample(ss) = 1;
        else
            StepSamples.m(cc,ss) = StepSamples.m(cc,ss-1);
        end
        
    end
        
    %% sample r using a MALA step
    
    m = StepSamples.m(:,ss); % p = m/(m+r);
    r = StepSamples.r(ss-1);
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

try
    kcFreeGPUArray(gpu_y);
    kcFreeGPUArray(gpu_trIndex);
    kcFreeGPUArray(gpu_trCoh);
    kcFreeGPUArray(gpu_spe)
    kcFreeGPUArray(gpu_spikeHistory);
catch e
    fprintf('Error clearing cuda memory: %s\n',e);
end

%% finalize sampler

StepFit.alpha.mean = mean(StepSamples.alpha(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.phi.mean   = mean(StepSamples.phi(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.r.mean           = mean(StepSamples.r(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.p.mean           = mean(StepSamples.p(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.m.mean           = mean(StepSamples.m(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.mean    = mean(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.mode    = mode(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.median  = median(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.std     = std(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[],2);
StepFit.stepDir.mean     = mean(StepSamples.s(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepDir.mode     = mode(StepSamples.s(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.hs.mean          = mean(StepSamples.hs(params.MCMC.burnIn+1:params.MCMC.thinRate:end,:))';

StepFit.p.interval     = prctile(StepSamples.p(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.m.interval     = prctile(StepSamples.m(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.phi.interval   = prctile(StepSamples.phi(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.alpha.interval = prctile(StepSamples.alpha(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.r.interval     = prctile(StepSamples.r(params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5]);
StepFit.hs.interval    = prctile(StepSamples.hs((params.MCMC.burnIn+1):params.MCMC.thinRate:end,:),[2.5 97.5],1);


StepSamples.finalEpsilon = epsilon;

fprintf('Stepping model sampler complete.\n');
end
