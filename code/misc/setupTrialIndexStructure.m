%% setupTrialTimeStructure makes sure a time series object contains a trialIndex
%  structure. The array contains start and end times for each trial, while the 
%  trialStart array only contains start times. (These two are redundant, and
%  trialStart is simpler, but I don't want to eliminate trialIndex from my code)
function timeSeries = setupTrialIndexStructure(timeSeries)

if(~isfield(timeSeries,'trialIndex'))
    %% sets up a structure that gives trial start and end time
    NT = length(timeSeries.trialStart);
    TT = length(timeSeries.y);
    timeSeries.trialIndex = zeros(NT,2);
    for ii = 1:NT-1
        timeSeries.trialIndex(ii,1) = timeSeries.trialStart(ii);
        timeSeries.trialIndex(ii,2) = timeSeries.trialStart(ii+1)-1;
    end
    timeSeries.trialIndex(NT,1) = timeSeries.trialStart(NT);
    timeSeries.trialIndex(NT,2) = TT;
elseif(~isfield(timeSeries,'trialStart'))
    timeSeries.trialStart = timeSeries.trialIndex(:,1);
end
