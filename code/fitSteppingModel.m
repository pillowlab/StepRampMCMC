%% Fit a stepping model to the observed spiking trials
% Model of a trial
%   z ~ negative binomial(p,r)     p is stimulus coherence dependent, r is shared across all trials
%   s ~ bernoulli(phi)   valued on set {2,3} instead of {0,1} for indexing (P(s=3) = phi),   phi is stimulus coherence dependent
%
%
%   for t = 1...T  (T=trial length)
%   (y(t) | t<z)      ~ Poisson(alpha(1) dt)
%   (y(t) | t>=z,s=2) ~ Poisson(alpha(2) dt)
%   (y(t) | t>=z,s=3) ~ Poisson(alpha(3) dt)
%
% Model outputs
%   StepSamples.alpha = firing rates per each state (3,numSamples)
%   StepSamples.r     = negative binomial failure number parameter (1,numSamples)
%   StepSamples.p     = negative binomial success probability parameter (numCoherences,numSamples)
%   StepSamples.phi   = probability of switching to up or down state from initial state (numCoherences,numSamples)
%   StepSamples.z     = sampled switch times per trial (NT,numSamples)
%   StepSamples.s     = state switched to in each trial (NT,numSamples);
%
%   StepSamples.spikeStats = summary of spike count (column 1) and number of observations (column 2) per each hidden state as sampled in z,s
%
%
%   StepSamples includes the burnin period samples 
%
%   StepFit contains sample mean of each parameter (after throwing out burnin)
% 
%
% samples - num samples to get POST burn in
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


function [StepFit, StepSamples] = fitSteppingModel(timeSeries,params)           

totalSamples = params.MCMC.nSamples+params.MCMC.burnIn;

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
StepSamples.p     = zeros(NC,totalSamples);
StepSamples.phi   = zeros(NC,totalSamples);
StepSamples.r      = zeros(1,totalSamples);

StepSamples.z      = zeros(NT,totalSamples);
StepSamples.s      = zeros(NT,totalSamples);

StepSamples.spikeStats = zeros(3,2,totalSamples);

%saves extra info for Rao-Blackwellized estimates of alpha,p, and phi (these estimates do not exist for r)
StepSamples.rb.alpha = zeros(3,totalSamples);
StepSamples.rb.p     = zeros(NC,totalSamples);
StepSamples.rb.phi   = zeros(NC,totalSamples);


%% initial parameters for stepping
StepSamples.r(1) = 1;
mu = 30; %average initial step time in terms of bins
StepSamples.p(:,1) = 1/(StepSamples.r(1)/mu + 1);%0.83;

StepSamples.phi(:,1) = 0.5; %initial probability of stepping up for all stimulus levels

%% initializes firing rates - state 0 as average rate in bin 1, and states 2,3 as firing rates in last 3 bins for in/out choices
StepSamples.alpha(1,1) = mean(timeSeries.y(timeSeries.trialIndex(:,1)))/params.delta_t;

timeIndices = timeSeries.trialIndex(timeSeries.choice == 1 ,2);
timeIndices = [timeIndices;timeIndices-1;timeIndices-2]; 
endFR1 =  max(mean( timeSeries.y(timeIndices )), 1e-20)/params.delta_t;

timeIndices = timeSeries.trialIndex(timeSeries.choice == 2 ,2);
timeIndices = [timeIndices;timeIndices-1;timeIndices-2]; 
endFR2 =  max(mean( timeSeries.y(timeIndices )), 1e-20)/params.delta_t;

StepSamples.alpha(2,1) = min(endFR1,endFR2);
StepSamples.alpha(3,1) = max(endFR1,endFR2);


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
        p_c = StepSamples.p(cc,ss-1);
        nbPDF(:,cc) = gammaln(x+r_c) - (gammaln(x+1)+gammaln(r_c)) + r_c*log(1-p_c) + x.*log(p_c);
    end
    
    [StepSamples.z(:,ss), StepSamples.s(:,ss), StepSamples.spikeStats(:,:,ss)] = kcStepTimeSampler(gpu_y,gpu_trIndex,gpu_trCoh,StepSamples.alpha(:,ss-1),StepSamples.phi(:,ss-1),nbPDF,params.delta_t);
    
    
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
    
    StepSamples.alpha(:,ss) = StepSamples.alpha(:,ss)./params.delta_t; %scale new parameters with time bin size - makes alpha in terms of spikes/second
    
    %negative binomial switch time parameter p sampler
    r_c = StepSamples.r(ss-1);
    for cc = 1:NC
        post1 = params.stepPrior.negBin.succAlpha+sum(StepSamples.z(trialsOfCoh(:,cc),ss));
        post2 = params.stepPrior.negBin.succBeta+r_c*NTC(cc);
        
        StepSamples.p(cc,ss) = betarnd(post1,post2);
        StepSamples.rb.p(cc,ss) = post1/(post1+post2);
    end
    
    %switch to state sampler
    for cc = 1:NC
        switchesToState3 = sum(StepSamples.s(trialsOfCoh(:,cc),ss) == 3);
        post_alpha = switchesToState3 + params.stepPrior.switchto.alpha;
        post_beta = NTC(cc) - switchesToState3 + params.stepPrior.switchto.beta;
        
        StepSamples.phi(cc,ss) = betarnd(post_alpha,post_beta);
        
        StepSamples.rb.phi(cc,ss) = post_alpha/(post_alpha+post_beta); 
    end

    StepSamples.phi(StepSamples.phi(:,1) == 1,ss) = 1;
    %% sample r using a MALA step
    
    log_p_r = sum(gammaln(StepSamples.z(:,ss)+r_c)) - NT*gammaln(r_c) + r_c*(NTC'*log(1-StepSamples.p(:,ss))) + (params.stepPrior.negBin.failShape-1)*log(r_c) - params.stepPrior.negBin.failRate*r_c;
    der_log_p_r = sum(psi(StepSamples.z(:,ss)+r_c)) - NT* psi(r_c) + (NTC'*log(1-StepSamples.p(:,ss)))        + (params.stepPrior.negBin.failShape-1)/r_c - params.stepPrior.negBin.failRate;
    
    prop_mu = max(0,r_c + 1/2*epsilon^2*der_log_p_r);
    minr = max(normcdf(0,prop_mu,epsilon),1e-20);
    r_star = max(1e-20, norminv(rand*(1-minr) + minr,prop_mu,epsilon));
    
    log_p_r_star = sum(gammaln(StepSamples.z(:,ss)+r_star)) - NT*gammaln(r_star) + r_star*(NTC'*log(1-StepSamples.p(:,ss))) + (params.stepPrior.negBin.failShape-1)*log(r_star) - params.stepPrior.negBin.failRate*r_star;
    der_log_p_r_star = sum(psi(StepSamples.z(:,ss)+r_star)) - NT* psi(r_star) + (NTC'*log(1-StepSamples.p(:,ss)))           + (params.stepPrior.negBin.failShape-1)/r_star - params.stepPrior.negBin.failRate;
    prop_mu_star = max(0,r_star + 1/2*epsilon^2*der_log_p_r_star);
    minr_star = max(normcdf(0,prop_mu_star,epsilon),1e-20);
    
    log_q_tostar = -log(1-minr)     -1/(2*epsilon^2)*(r_star - prop_mu)^2;
    log_q_toc    = -log(1-minr_star)-1/(2*epsilon^2)*(r_c    - prop_mu_star)^2;
    
    log_a = log_p_r_star - log_p_r + log_q_toc - log_q_tostar;
    
    log_rand = log(rand);
    if(log_rand < log_a)
        StepSamples.r(ss) = r_star;
        StepSamples.acceptanceCount.r(ss) = 1;
    else
        StepSamples.r(ss) = r_c;
    end
    
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
    
    %% plot progress to show mixing and if things are working
    if(mod(ss,500) == 0 || ss == totalSamples)
        sfigure(100);
        clf
        
        startMean = max(1,ss-250);
        if(ss > params.MCMC.burnIn + 100)
            startMean = params.MCMC.burnIn+1;
        end
        
        subplot(NC+2,1,1)
        hold on
        plot(1:ss,StepSamples.alpha(:,1:ss))
        meanAlpha = mean(StepSamples.alpha(:,startMean:ss),2);
        plot([1 totalSamples],[meanAlpha meanAlpha],':');
        title(['Firing rates: ' num2str(meanAlpha')]);
        xlim([1 totalSamples]);
        ylim([0 max(max(StepSamples.alpha(:,1:ss)))*1.05]);
        hold off
        
        subplot(NC+2,1,2)
        hold on
        plot(1:ss,StepSamples.r(:,1:ss))
        meanR = mean(StepSamples.r(startMean:ss));
        plot([1 totalSamples],[meanR meanR],':k');
        rtitle = sprintf('Negative binomial r, acceptance rate = %f, MALA epsilon = %f, mean r = %f', mean(StepSamples.acceptanceCount.r(2:ss)),epsilon, meanR);
        title(rtitle);
        xlim([1 totalSamples]);
        ylim([0 max(max(StepSamples.r(:,1:ss)))*1.05]);
        hold off
        
        for cc = 1:NC
            subplot(NC+2,1,2+cc)
            hold on
            plot(1:ss,StepSamples.p(cc,1:ss),'Color',[0 0 1])
            plot(1:ss,StepSamples.phi(cc,1:ss),'Color',[0 0 0])
            meanPhi = mean(StepSamples.phi(cc,startMean:ss));
            plot([1 totalSamples],[meanPhi meanPhi],':','Color',[0 0 0]);
            meanP = mean(StepSamples.p(cc,startMean:ss));
            plot([1 totalSamples],[meanP meanP],':','Color',[0 0 1]);
            title(['Coh ' num2str(cc),' Mean p = ' num2str(meanP) ', mean phi = ' num2str(meanPhi)]);
            xlim([1 totalSamples]);
            ylim([0 1]);
            hold off
        end
        drawnow;
    end
    
end


%% dump GPU variables
kcFreeGPUArray(gpu_y);
kcFreeGPUArray(gpu_trIndex);
kcFreeGPUArray(gpu_trCoh);

%% finalize sampler

StepFit.alpha.mean = mean(StepSamples.alpha(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.phi.mean   = mean(StepSamples.phi(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.r.mean           = mean(StepSamples.r(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.p.mean           = mean(StepSamples.p(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.mean    = mean(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.mode    = mode(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.median  = median(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepTime.std     = std(StepSamples.z(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[],2);
StepFit.stepDir.mean     = mean(StepSamples.s(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);
StepFit.stepDir.mode     = mode(StepSamples.s(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),2);

StepFit.p.interval     = prctile(StepSamples.p(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.phi.interval   = prctile(StepSamples.phi(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.alpha.interval = prctile(StepSamples.alpha(:,params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5],2);
StepFit.r.interval     = prctile(StepSamples.r(params.MCMC.burnIn+1:params.MCMC.thinRate:end),[2.5 97.5]);


StepSamples.finalEpsilon = epsilon;

fprintf('Stepping model sampler complete.\n');
end