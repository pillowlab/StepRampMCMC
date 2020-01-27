%% Get spike history effect for each time bin in y 
% This function creates an array containing the spike history for each time
% bin in y. The array is size TT x 1, where TT is the lengh of y.
%
% Input
%   y = observed spikes
%   trialIndex = beginning and end of each trial
%   yh = observed spikes before each trial
%   hs = spike history filter bin weights
%
% Output
%   sh = spike history effect for each bin of y

function spe = getSpikeHistoryEffect(y,trialIndex,yh,hs)
TT = length(y);
NT = size(trialIndex,1);
spe = zeros(TT,1);
NH = length(hs);
% for tr = 1:NT
%     ytrial = y(trialIndex(tr,1):trialIndex(tr,2));
%     trial_length = length(ytrial);
%     ytrial_hist = [yh(tr,:)'; ytrial];
%     spe_trial = conv(ytrial_hist,hs,'same');
%     spe(trialIndex(tr,1):trialIndex(tr,2)) = spe_trial(1:trial_length);
% end
for tr = 1:NT
    ytrial = y(trialIndex(tr,1):trialIndex(tr,2)); ytrial = ytrial(:);
    trial_length = length(ytrial);
    yh_trial = yh(tr,:); 
    y_hist = [yh_trial(:); ytrial];
    spe_trial = zeros(trial_length,1);
    for ii = 1:trial_length
        idx = ii+NH;
        spe_trial(ii) = y_hist(idx-(1:NH))'*hs(:);
    end
    spe(trialIndex(tr,1):trialIndex(tr,2)) = spe_trial;
end

end

