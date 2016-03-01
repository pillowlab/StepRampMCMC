function timeSeries = simulateSteppingModel()

%%
NT = 100; %number of trials per coherence level (this function always simulates equal number of trials for all coherences)
trLength = [50 100]; %trial lengths are chosen uniformly between trLength(1) and trLength(2) bins
timeSeries.delta_t = 10e-3; %bin size in seconds
timeSeries.timeAfterDotsOn = 200e-3; %saying that the spike counts here begin 200ms after stimulus onset

%% model parameters (default parameters assume 5 coherence/stimulus levels)

timeSeries.trueParams.model = 'stepping';
timeSeries.trueParams.alpha = [30 10 50]; %firing rate in initial, down, and up states (spikes/sec)
timeSeries.trueParams.phi   = [0.05 0.25 0.5 0.75 0.95]; %coherence-dependent step "up" probability
timeSeries.trueParams.p     = [0.92 0.95 0.97 0.95 0.92];  %coherence-dependent negative binomial step-time parameter (range 0-1)
timeSeries.trueParams.r     = 2; %negative binomial step-time parameter

%% sets "choice" randomly on each trial according to coherence-based probability, just to include this bit in the simulated datastructure
%  - choice in this simulation has nothing to do with behavioral implications of stepping or diffusion-to-bound

timeSeries.trueParams.choiceP = [0.05 0.25 0.5 0.75 0.95];

%% setup timeSeries space
NC = length(timeSeries.trueParams.phi); %number of coherence levels
trLengths = trLength(1)-1 + randi(trLength(2) - trLength(1),NC*NT,1); %length of each trial
timeSeries.trialStart = [1; 1+cumsum(trLengths(1:end-1))];
timeSeries.y = nan(sum(trLengths),1);
timeSeries.trueParams.x = nan(sum(trLengths),1);    %firing rate
timeSeries.trueParams.stepTimes      = nan(NC*NT,1); %at bins greater than this number, cell in "up" or "down" state
timeSeries.trueParams.stepDirections = nan(NC*NT,1); %3 = up, 2 = down
timeSeries.trCoh   = nan(NC*NT,1);

%% simulate diffusion paths
fprintf('Simulating from the stepping model (%d total trials, %d coherence levels)...\n',NC*NT,NC);
for cc = 1:NC
    for tr = 1:NT
        trNum = (cc-1)*NT + tr;
        
        timeSeries.trCoh(trNum) = cc;
        
        stepTime = nbinrnd(timeSeries.trueParams.r, 1-timeSeries.trueParams.p(cc)); %the 1-p is because matlab uses different parameterization of negbinomial dist than wikipedia
        stepDirection = (rand < timeSeries.trueParams.phi(cc)) + 2;
        
        
        fr = zeros(trLengths(trNum),1); %firing rate (sp/sec)
        
        fr(1:min(stepTime,trLengths(trNum))) = timeSeries.trueParams.alpha(1);
        fr(stepTime+1:end) = timeSeries.trueParams.alpha(stepDirection);
        
        y  = poissrnd(fr*timeSeries.delta_t); %spike counts
        
        %inserts simulation into time series
        T1 = timeSeries.trialStart(trNum);
        T2 = timeSeries.trialStart(trNum) + trLengths(trNum) - 1;
        timeSeries.y(T1:T2) = y;
        timeSeries.trueParams.x(T1:T2) = fr;
        timeSeries.trueParams.stepTimes(trNum)      = stepTime;
        timeSeries.trueParams.stepDirections(trNum) = stepDirection;
        
        %selects psuedo-choice
        timeSeries.choice(trNum) = (rand>timeSeries.trueParams.choiceP(cc)) + 1;
    end
end
fprintf('done.\n');