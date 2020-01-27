%% Umbrella function for getting model comparison statistics from extended stepping model fit
%
% Inputs
%   RampSamples
%   params
%   ModelFit (e.g. StepFit)
%   timeSeries
%   model - structure with details about model
%       history - true if include history, false if not
%       MR - true if use MR parameterization, false if not
%
% Default is softplus nonlinearity, no bias, no history
%
% Outputs
%   WAIC - WAIC value
%   DIC - DIC value
%   l_like - log_likelihood of mean parameters across trials
%   trial_WAIC - per trial WAIC values
%   trial_DIC - per trial DIC values
%   trial_l_like - per trial log likelihood values of mean parameters
%   sample_likelihoods - across trial likelihoods for each MCMC sample
%   sample_trial_likelihoods - likelihoods for each MCMC sample and trial
function [ StepFit, StepSamples ] = fitExtendedSteppingModel(timeSeries,params,model)

    if model.history == false && model.MR == true
        [StepFit, StepSamples] = fitSteppingMR(timeSeries,params);
    elseif model.history == true && model.MR == true
        [StepFit, StepSamples] = fitSteppingMRHistory(timeSeries,params);
    elseif model.history == false && model.MR == false
        [StepFit, StepSamples] = fitStepping(timeSeries,params);
    elseif model.history == true && model.MR == false
        [StepFit, StepSamples] = fitSteppingHistory(timeSeries,params);
    end

end