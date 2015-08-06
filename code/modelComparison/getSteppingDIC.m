function [DIC, l_like,DIClikelihoods ] = getSteppingDIC(StepSamples,params,modelFit,timeSeries)

timeSeries = setupTrialIndexStructure(timeSeries);


%% get log likelihood
l_like = getSteppingLogLikelihood(timeSeries,params,modelFit);

%% get log prior of fit
% prior_alpha = 0;
% for ii = 1:3
%     p_a_cc = params.stepPrior.alpha.shape*log(params.stepPrior.alpha.rate) - gammaln(params.stepPrior.alpha.shape) + (params.stepPrior.alpha.shape-1)*log(alpha_norm(ii)) - params.stepPrior.alpha.rate*(alpha_norm(ii));
%     nc = 0;%-log(max(1e-5,1-gamcdf(alpha_norm(2),params.stepPrior.alpha.shape,1/params.stepPrior.alpha.rate)));
%     prior_alpha = prior_alpha + p_a_cc + nc;
% end
% 
% prior_phi = 0;
% for cc = 1:NC
%     if(phi(cc) < 1)
%         prior_phi = (params.stepPrior.switchto.alpha-1)*log(phi(cc)) + (params.stepPrior.switchto.beta-1)*log(1-phi(cc)) - betaln(params.stepPrior.switchto.alpha,params.stepPrior.switchto.beta) + prior_phi;
%     end
% end
% 
% prior_p = 0;
% for cc = 1:NC
%     prior_p = (params.stepPrior.negBin.succAlpha-1)*log(p(cc)) + (params.stepPrior.negBin.succBeta-1)*log(1-p(cc)) - betaln(params.stepPrior.negBin.succAlpha,params.stepPrior.negBin.succBeta) + prior_p;   
% end
% 
% prior_r = params.stepPrior.negBin.failShape*log(params.stepPrior.negBin.failRate) - gammaln(params.stepPrior.negBin.failShape) + (params.stepPrior.negBin.failShape-1)*log(r) - params.stepPrior.negBin.failRate*r;
% 
% prior = prior_alpha + prior_phi + prior_p + prior_r;

%% calculates DIC and posterior predictive
DIClikelihoods = zeros(params.MCMC.nSamples,1);

for ss = 1:params.MCMC.thinRate:params.MCMC.nSamples
    if(ss == 1 || mod(ss-1,250) == 0 || ss == params.MCMC.nSamples)
        fprintf('  Stepping model DIC calculations %d / %d\n',ss,params.MCMC.nSamples);
    end
    sampleModelFit.alpha.mean = StepSamples.alpha(:,ss+params.MCMC.burnIn);
    sampleModelFit.r.mean     = StepSamples.r(ss+params.MCMC.burnIn);
    sampleModelFit.phi.mean   = StepSamples.phi(:,ss+params.MCMC.burnIn);
    sampleModelFit.p.mean     = StepSamples.p(:,ss+params.MCMC.burnIn);
    
    DIClikelihoods(ss) = getSteppingLogLikelihood(timeSeries, params,sampleModelFit);
end

DIClikelihoods = DIClikelihoods(1:params.MCMC.thinRate:end);

DICtermsToDrop = sum(isnan(DIClikelihoods) | isinf(DIClikelihoods));
if(DICtermsToDrop > 0)
    fprintf('Dropping %d terms from DIC calculation! Potential numerical errors!\n',DICtermsToDrop);
end

DIClikelihoods = DIClikelihoods(~isnan(DIClikelihoods) & ~isinf(DIClikelihoods));
if(~isempty(DIClikelihoods))
    expectedLikelihood = mean(DIClikelihoods);
else
    except('Numerical error calculating expected likelihood (DIC) in stepping model');
end

DIC = 2*l_like-4*expectedLikelihood;

fprintf('Stepping model DIC computation complete.\n');
