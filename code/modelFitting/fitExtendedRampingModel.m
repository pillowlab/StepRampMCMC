%% Umbrella function for fitting the extended ramping model
%
% Inputs
%   timeSeries
%   params
%   model - structure with details about model
%       model.nlin - nonlinearity is either "softplus", "power", or "exp"
%       model.bias - if true fit bias parameter, if false bias is set to 0
%       model.history - if true fit spike history weights, if false history is zero
%       model.pow - if using power nonlinearity, pow is the power log1p(exp(x))^pow
%
% Default is softplus nonlinearity, no bias, no history
%
% Outputs
%   RampFit - model fitting structure
%   RampSamples - structure with MCMC samples for each parameter
%   LatentDataHandler - structure for handling latent variables
function [ RampFit, RampSamples, LatentDataHandler] = fitExtendedRampingModel(timeSeries,params,model)

    if model.nlin == "softplus" && model.bias == false && model.history == false
        [RampFit, RampSamples, LatentDataHandler] = fitRampingSoftplus(timeSeries, params);
    elseif model.nlin == "softplus" && model.bias == true && model.history == false
        [RampFit, RampSamples, LatentDataHandler] = fitRampingSoftplusBias(timeSeries, params);
    elseif model.nlin == "softplus" && model.bias == false && model.history == true
        [RampFit, RampSamples, LatentDataHandler] = fitRampingSoftplusHistory(timeSeries, params);
    elseif model.nlin == "softplus" && model.bias == true && model.history == true
        [RampFit, RampSamples, LatentDataHandler] = fitRampingSoftplusBiasHistory(timeSeries, params);
    elseif model.nlin == "power" && model.bias == false && model.history == false
         model_spec.modelInd = model.pow;
         model_spec.speInd = 0;
        [RampFit, RampSamples, LatentDataHandler] = fitRampingPower(timeSeries, params, model_spec);
    elseif model.nlin == "power" && model.bias == true && model.history == false
        [RampFit, RampSamples, LatentDataHandler] = fitRampingPowerBias(timeSeries, params, model.pow);
    elseif model.nlin == "power" && model.bias == false && model.history == true
        [RampFit, RampSamples, LatentDataHandler] = fitRampingPowerHistory(timeSeries, params, model.pow);
    elseif model.nlin == "power" && model.bias == true && model.history == true
        [RampFit, RampSamples, LatentDataHandler] = fitRampingPowerBiasHistory(timeSeries, params, model.pow);
    elseif model.nlin == "exp" && model.bias == false && model.history == false
        [RampFit, RampSamples, LatentDataHandler] = fitRampingExp(timeSeries, params);
    elseif model.nlin == "exp" && model.bias == true && model.history == false
        [RampFit, RampSamples, LatentDataHandler] = fitRampingExpBias(timeSeries, params);
    elseif model.nlin == "exp" && model.bias == false && model.history == true
        [RampFit, RampSamples, LatentDataHandler] = fitRampingExpHistory(timeSeries, params);
    elseif model.nlin == "exp" && model.bias == true && model.history == true
        [RampFit, RampSamples, LatentDataHandler] = fitRampingExpBiasHistory(timeSeries, params);
    end

end