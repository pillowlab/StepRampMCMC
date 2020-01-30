%% Umbrella function for getting model comparison statistics from extended stepping model fit
%
% Inputs
%   RampSamples
%   params
%   ModelFit (e.g. StepFit)
%   timeSeries
%   model - structure with details about model
%
% Default is softplus nonlinearity, no bias, no history
%
% Outputs
%   ModelComp
%       ModelComp.WAIC - WAIC value
%       ModelComp.DIC - DIC value
%       ModelComp.l_like - log_likelihood of mean parameters across trials
%       ModelComp.trial_WAIC - per trial WAIC values
%       ModelComp.trial_DIC - per trial DIC values
%       ModelComp.trial_l_like - per trial log likelihood values of mean parameters
%       ModelComp.sample_likelihoods - across trial likelihoods for each MCMC sample
%       ModelComp.sample_trial_likelihoods - likelihoods for each MCMC sample and trial
function [ModelComp] = getExtendedSteppingModelComp(StepSamples,params,modelFit,timeSeries,model)

    if model.history == false
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getSteppingModelComp(StepSamples,params,modelFit,timeSeries);
    elseif model.history == true
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getSteppingHistoryModelComp(StepSamples,params,modelFit,timeSeries);
    end
    ModelComp.WAIC = WAIC;
    ModelComp.DIC = DIC;
    ModelComp.l_like = l_like;
    ModelComp.trial_WAIC = trial_WAIC;
    ModelComp.trial_DIC = trial_DIC;
    ModelComp.trial_l_like = trial_l_like;
    ModelComp.sample_likelihoods = sample_likelihoods;
    ModelComp.sample_trial_likelihoods = sample_trial_likelihoods;

end