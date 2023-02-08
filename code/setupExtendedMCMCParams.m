function params = setupExtendedMCMCParams(model)

if ~isfield(model,'bias'), model.bias = false; end
if ~isfield(model,'history'), model.history = false; end
if ~isfield(model,'nlin'), model.nlin = "softplus"; end
if ~isfield(model,'pow'), model.pow = 1.0; end
if ~isfield(model,'MR'), model.MR = true; end
    
%% MCMC runs

params.MCMC.nSamples = 50e3; %number of MCMC samples to get (post burn-in)
params.MCMC.burnIn   = 10e3; %number of samples for burn in

params.MCMC.thinRate = 5;    %thin chain - take every X samples
                             %effective number of samples is then params.MCMC.nSamples/params.MCMC.thinRate
                             
params.tempDataFolder = './temp/'; %folder to store latent variables of ramping model sampler if number of samples is really big

%% model comparison

params.DIC.meanLikelihoodSamples = 25000; %num samples to use to estimate p(Data| mean parameters) for rampling model
params.DIC.likelihoodSamples     = 5000; %num samples to use to estimate p(Data| parameter_s) for rampling model - in DIC expectation



%% stepping model sampler params

params.stepSampler.maxJumpTime = 1500; %maximum allowed jump time - should be much greater than maximum trial length (in time bins)

%during burn-in, can automatically adjust step size of MALA sampler on r
params.stepSampler.learnStepSize = 1;   %to use auto-step size or not
params.stepSampler.MALAadjust    = 100; %how often to make adjustments (in number of samples) on epsilon (the step size)
params.stepSampler.epsilon_init  = 0.075; %inital value of epsilon
params.stepSampler.accept_min    = 0.2; %minimum acceptance fraction (if fraction goes below this, epsilon is adjusted down)
params.stepSampler.accept_max    = 0.8; %maximum acceptance fraction (if fraction goes above this, epsilon is adjusted up)
params.stepSampler.adjustRate    = 1.25; %epsilon multiplied by this rate to go up, and divided to go down
params.stepSampler.epsilon_min   = 0.02; %minimum allowed value of epsilon
params.stepSampler.epsilon_max   = 0.5;  %maximum allowed value of epsilon
%if params.stepSampler.learnStepSize is false, epsilon is set to epsilon_init throughout sampling


%% ramping model particle filter params

params.rampSampler.numParticles    = 200;%number of particles to simulate to sample ramping firing rates
params.rampSampler.minNumParticles = 25; %minimum "effective" number of particles before resampling
params.rampSampler.sigMult         = 1.0;%for ramping model particle filter, just keep at 1.0

%during burn-in, automatically adjusts step size of MALA sampler on gamma (similar to r above)
params.rampSampler.learnStepSize = 1;   %to use auto-step size or not
params.rampSampler.epsilon_init  = 0.05; %inital value of epislon
params.rampSampler.MALAadjust    = 100; %how often to make adjustments (in number of samples) on epsilon (the step size)
params.rampSampler.adjustRate    = 1.5; %epsilon multiplied by this rate to go up, and divided to go down
params.rampSampler.epsilon_min   = 0.02; %minimum allowed value of epsilon
params.rampSampler.epsilon_max   = 1.0;  %maximum allowed value of epsilon
params.rampSampler.accept_min    = 0.2; %minimum acceptance fraction (if fraction goes below this, epsilon is adjusted down)
params.rampSampler.accept_max    = 0.8; %maximum acceptance fraction (if fraction goes above this, epsilon is adjusted up)

%after burn in, sets epsilon to this fixed value if not nan
params.rampSampler.epsilon_fixed = 1.0; 


%% ramping model prior

%drift rate (\beta_c; coherence dependent)  normal distribution
params.rampPrior.beta_mu    = 0;   %(default: 0)
params.rampPrior.beta_sigma = 0.1; %(default: 0.1) p(\beta_c) = \frac{1}{2*beta_sigma^2} \exp(-\frac{1}{2*beta_sigma^2} (\beta_c - beta_mu))

%diffusion variance (\omega^2)   inverse gamma distribution
params.rampPrior.w2_shape = 1.1; %(default: 1.1)
params.rampPrior.w2_scale = 1e-3; %(default: 1e-3) p(\omega^2) = \frac{w2_scale^w2_shape}{\Gamma(w2_shape)} (\omega^2)^{-w2_shape-1} \exp(-\frac{w2_scale}{\omega^2})

%initial level (l_0)  truncated normal distribution
if model.bias == false
    params.rampPrior.l0_mu    = 0;  %(default: 0)
    params.rampPrior.l0_sigma = 10; %(default: 10) p(l_0) \propto \frac{1}{2*l0_sigma^2} \exp(-\frac{1}{2*l0_sigma^2} (l_0 - l0_mu)), given l_0 < 1
elseif model.bias == true
    params.rampPrior.l0_mu    = 0.5;  
    params.rampPrior.l0_sigma = 0.5;  
end

%firing rate multiplier (\gamma)  gamma distribution
if model.nlin == "softplus"
    params.rampPrior.gammaAlpha = 2;    %(default: 2)
    params.rampPrior.gammaBeta  = 0.05; %(default: 0.05) p(\gamma) = \frac{gammaBeta^{gammaAlpha} \gamma^{gammaAlpha-1} \exp(-\gamma*gammaBeta)}{\Gamma(gammaAlpha)}
elseif model.nlin == "power" && pow == 2
    params.rampPrior.gammaAlpha = 3;
    params.rampPrior.gammaBeta = 0.5;
elseif model.nlin == "power" && pow == 0.5
    params.rampPrior.gammaAlpha = 1;
    params.rampPrior.gammaBeta = 1e-4;
elseif model.nlin == "exp"
    params.rampPrior.gammaAlpha = 3;    
    params.rampPrior.gammaBeta  = 3;
else
    warning("Consider changing the prior on \gamma to match your model.")
end
    
%% stepping model prior

%firing rate (\alpha_s; same prior for each state) gamma distribution
if model.history == false
    params.stepPrior.alpha.rate  = 1; %(default: 1)
    params.stepPrior.alpha.shape = 1; %(default: 1) p(\alpha_s) = \frac{alpha.rate^{alpha.shape} \alpha_s^{alpha.shape-1} \exp(-\alpha_s*alpha.rate)}{\Gamma(alpha.shape)}
elseif model.history == true
    params.stepPrior.alpha.rate  = 0.01; 
    params.stepPrior.alpha.shape = 1;
end

%step time distribution - ``number of failures'' (r) gamma distribution
params.stepPrior.negBin.failShape  = 2; %(default: 2)
params.stepPrior.negBin.failRate   = 1; %(default: 1) p(r) = \frac{negBin.failRate^{negBin.failShape} r^{negBin.failShape-1} \exp(-r*negBin.failRate)}{\Gamma(negBin.failShape)}

%step time distribution - ``success probability''  (p_c; coherence dependent) beta distribution
params.stepPrior.negBin.succBeta   = 1; %(default: 1)
params.stepPrior.negBin.succAlpha  = 1; %(default: 1) p(p_c)  = p_c^{negBin.succAlpha-1} (1-p_c)^{negBin.succBeta-1) / B(negBin.succAlpha, negBin.succBeta)

%probability of up step (\phi_c; coherence dependent) beta distribution
params.stepPrior.switchto.alpha    = 1; %(default: 1)
params.stepPrior.switchto.beta     = 1; %(default: 1) p(\phi_c)  = \phi_c^{switchto.alpha-1} (1-\phi_c)^{switchto.beta-1) / B(switchto.alpha, switchto.beta)

%% spike history prior
params.spikeHistoryPrior.hMu = 0;
params.spikeHistoryPrior.hSig2 = 10;