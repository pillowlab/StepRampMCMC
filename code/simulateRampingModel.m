function timeSeries = simulateRampingModel()

%%
NT = 10; %number of trials per coherence level
trLength = [50 100]; %trial lengths are chosen uniformly between trLength(1) and trLength(2) bins
timeSeries.trueParams.delta_t = 10e-3; %bin size in seconds

%% model parameters (default parameters assume 5 coherence/stimulus levels)

timeSeries.trueParams.model = 'ramping';
timeSeries.trueParams.beta  = [-0.02 -0.01 0 0.01 0.02]; %coherence-dependent slope
timeSeries.trueParams.l_0   = 0.5;  %inital diffusion point, must be less than one
timeSeries.trueParams.w2    = 0.005; %diffusion variance
timeSeries.trueParams.gamma = 50; %firing rate multiplier - firing rate is log(1+exp(gamma*x)) where x is the diffusion variable with a bound at 1


%% sets "choice" randomly on each trial according to coherence-based probability
%  - choice in this simulation has nothing to do with behavioral implications of diffusion-to-bound

timeSeries.trueParams.choiceP = [0.05 0.25 0.5 0.75 0.95];

%% setup timeSeries space
NC = length(timeSeries.trueParams.beta); %number of coherence levels
trLengths = trLength(1)-1 + randi(trLength(2) - trLength(1),NC*NT,1); %length of each trial
timeSeries.trialStart = [1; 1+cumsum(trLengths(1:end-1))];
timeSeries.y = nan(sum(trLengths),1);
timeSeries.trueParams.x = nan(sum(trLengths),1);
timeSeries.trCoh   = nan(NC*NT,1);

%% simulate diffusion paths
for cc = 1:NC
    for tr = 1:NT
        trNum = (cc-1)*NT + tr;
        
        timeSeries.trCoh(trNum) = cc;
        
        xs = zeros(trLengths(trNum),1); %diffusion path
        xs(1) = min(1,randn*sqrt(timeSeries.trueParams.w2) + timeSeries.trueParams.l_0);
        
        for tt = 2:trLengths(trNum)
            if(xs(tt-1) >= 1)
                xs(tt) = 1; %bound has already been hit
            else
                xs(tt) = min(1,xs(tt-1) + timeSeries.trueParams.beta(cc) + randn*sqrt(timeSeries.trueParams.w2));
            end
        end
        
        fr = log(1+exp(xs*timeSeries.trueParams.gamma)); %firing rate (sp/sec)
        
        y  = poissrnd(fr*timeSeries.trueParams.delta_t); %spike counts
        
        %inserts simulation into time series
        T1 = timeSeries.trialStart(trNum);
        T2 = timeSeries.trialStart(trNum) + trLengths(trNum) - 1;
        timeSeries.y(T1:T2) = y;
        timeSeries.trueParams.x(T1:T2) = xs;
        
        %selects psuedo-choice
        timeSeries.choice(trNum) = (rand>timeSeries.trueParams.choiceP(cc)) + 1;
    end
end