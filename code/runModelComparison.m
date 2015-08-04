%% runs MCMC model fitting and computes DIC for stepping and ramping models
% input 
%   timeSeries = the structure that holds on all the data
%     timeSeries.y          = spike counts in each time bin over all trials (one long, column vector)
%                             The bins should cover only the window of the trial being fit with the models
%                             Trials are indexed via timeSeries.trialStart
%     timeSeries.trialStart = vector of start times of trials (should be in ascending order)
%                             i.e., [1 99 150 ...] would say trial 1 starts at bin 1, trial 2 at bin 99, trial 3 at bin 150, ...
%                             the last bin of trial 1 is bin 98, trial 2 at bin 149...
%     timeSeries.trCoh      = stimulus value for each trial. These models consider the stimulus categorical.
%                             These values should be integers, taking values 1,2,3,...,NC-1,NC where NC is the number of categories.
%     timeSeries.choice     = choice on every trial (either 1 or 2) - only used for initializing MCMC fits and for a plot
%                             it does not really matter which choice (in-RF or out-RF) is 1 or 2 in this code
%
%   resultsFiles - full file name/path for storing main results. This is a structure that contains fields for each model's result:
%     resultsFiles.step     
%     resultsFiles.ramp 
%
%   samplesFiles - full file name/path for storing MCMC samples. This is a structure that contains fields for each model's samples:
%     samplesFiles.step     
%     samplesFiles.ramp (NOTE: may not be saving out sample firing rate trajectories, only mean, due to space)
%
%   params - parameters structure is set in the setupMCMCParams.m file
%            params include the bin size of the spike count structure

function [DICs, StepFit, RampFit] = runModelComparison( timeSeries, params, resultsFiles, samplesFiles)

DICs = [nan nan];


%% setup parameters

%sets up a structure that gives trial start and end time 
NT = length(timeSeries.trialStart);
TT = length(timeSeries.y);
timeSeries.trialIndex = zeros(NT,2);
for ii = 1:NT-1
    timeSeries.trialIndex(ii,1) = timeSeries.trialStart(ii);
    timeSeries.trialIndex(ii,2) = timeSeries.trialStart(ii+1)-1;
end
timeSeries.trialIndex(NT,1) = timeSeries.trialStart(NT);
timeSeries.trialIndex(NT,2) = TT;


%% ---------------------------
%plot out a PSTH to make sure data look reasonable/correct
maxT = max(timeSeries.trialIndex(:,2) - timeSeries.trialIndex(:,1) + 1);
PSTH  = zeros(maxT,2);
PSTH2 = zeros(maxT,max(timeSeries.trCoh));
Ns    = zeros(maxT,2);
Ns2   = zeros(maxT,max(timeSeries.trCoh));

for ii = 1:size(timeSeries.trialIndex,1)
    T1 = timeSeries.trialIndex(ii);
    T2 = timeSeries.trialIndex(ii+1)-1;
    T  = min(maxT,T2 - T1 + 1);
    T2 = T1 + T - 1;
    
    PSTH(1:T,timeSeries.choice(ii)) = PSTH(1:T,timeSeries.choice(ii)) + timeSeries.y(T1:T2);
    Ns(1:T,timeSeries.choice(ii)) = Ns(1:T,timeSeries.choice(ii)) + 1;
    PSTH2(1:T,timeSeries.trCoh(ii)) = PSTH2(1:T,timeSeries.trCoh(ii)) + timeSeries.y(T1:T2);
    Ns2(1:T,timeSeries.trCoh(ii)) = Ns2(1:T,timeSeries.trCoh(ii)) + 1;
end

figure(50);
clf
subplot(1,2,1);
hold on
plot(((1:maxT)*params.delta_t + params.timeAfterDotsOn)*1e3,(PSTH./Ns)./params.delta_t);
ylabel('spks/s');
xlabel('time after motion onset');
title('Choice sorted');
hold off
subplot(1,2,2);
hold on
plot(((1:maxT)*params.delta_t + params.timeAfterDotsOn)*1e3,(PSTH2./Ns2)./params.delta_t);
ylabel('spks/s');
xlabel('time after motion onset');
title('Coherence sorted');
hold off
drawnow;




%% ---------------------------
kcResetDevice(); %does this reset thing to make sure GPU is ready - might not be necessary



[StepFit, StepSamples] = fitSteppingModel(timeSeries,params);


[StepModelComp.DIC, StepModelComp.l_like,StepModelComp.DIClikelihoods ] = getSteppingDIC(StepSamples,params,StepFit,timeSeries);

DICs(1) = StepModelComp.DIC;

save(resultsFiles.step,'timeSeries','params','StepModelComp','StepFit','-v7.3');
save(samplesFiles.step,'StepSamples','-v7.3');
clear StepSamples

RampFit = [];


return;


%% --------------------------
kcResetDevice(); %does this reset thing to make sure GPU is ready - might not be necessary

[RampFit, RampSamples] = fitRampingModel(timeSeries,params);
[RampModelComp.DIC, RampModelComp.l_like,RampModelComp.DIClikelihoods ] = getRampingDIC(RampSamples,params,RampFit,timeSeries);

DICs(2) = RampModelComp.DIC;

save(resultsFiles.ramp,'timeSeries','params','RampModelComp','RampFit','-v7.3');
save(samplesFiles.ramp,'RampSamples','-v7.3');
clear RampSamples


%% ----------------------------
kcResetDevice(); %does this reset thing to make sure all GPU stuff sent off by the samplers is finished
