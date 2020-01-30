%% Umbrella function for getting model comparison statistics from extended ramping model fit
%
% Inputs
%   RampSamples
%   params
%   ModelFit (e.g. RampFit)
%   timeSeries
%   model - structure with details about model
%       model.nlin - nonlinearity is either "softplus", "power", or "exp"
%       model.bias - if true fit bias parameter, if false bias is set to 0
%       model.history - if true fit spike history weights, if false history is zero
%       model.pow - if using power nonlinearity, pow is the power log1p(exp(x))^pow
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
function [ModelComp] = getExtendedRampingModelComp(RampSamples,params,modelFit,timeSeries,model)

    if model.nlin == "softplus" && model.bias == false && model.history == false
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingSoftplusModelComp(RampSamples,params,modelFit,timeSeries);
    elseif model.nlin == "softplus" && model.bias == true && model.history == false
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingSoftplusBiasModelComp(RampSamples,params,modelFit,timeSeries);
    elseif model.nlin == "softplus" && model.bias == false && model.history == true
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingSoftplusHistoryModelComp(RampSamples,params,modelFit,timeSeries);
    elseif model.nlin == "softplus" && model.bias == true && model.history == true
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingSoftplusBiasHistoryModelComp(RampSamples,params,modelFit,timeSeries);
    elseif model.nlin == "power" && model.bias == false && model.history == false
         model_spec.modelInd = model.pow;
         model_spec.speInd = 0;
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingPowerModelComp(RampSamples,params,modelFit,timeSeries,model_spec);
    elseif model.nlin == "power" && model.bias == true && model.history == false
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingPowerBiasModelComp(RampSamples,params,modelFit,timeSeries,model.pow);
    elseif model.nlin == "power" && model.bias == false && model.history == true
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingPowerHistoryModelComp(RampSamples,params,modelFit,timeSeries,model.pow);
    elseif model.nlin == "power" && model.bias == true && model.history == true
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingPowerBiasHistoryModelComp(RampSamples,params,modelFit,timeSeries,model.pow);
    elseif model.nlin == "exp" && model.bias == false && model.history == false
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingExpModelComp(RampSamples,params,modelFit,timeSeries);
    elseif model.nlin == "exp" && model.bias == true && model.history == false
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingExpBiasModelComp(RampSamples,params,modelFit,timeSeries);
    elseif model.nlin == "exp" && model.bias == false && model.history == true
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingExpHistoryModelComp(RampSamples,params,modelFit,timeSeries);
    elseif model.nlin == "exp" && model.bias == true && model.history == true
        [WAIC, DIC, l_like, trial_WAIC, trial_DIC, trial_l_like, sample_likelihoods, sample_trial_likelihoods ] ...
            = getRampingExpBiasHistoryModelComp(RampSamples,params,modelFit,timeSeries);
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