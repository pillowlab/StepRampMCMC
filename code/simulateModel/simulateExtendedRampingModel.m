function timeSeries = simulateExtendedRampingModel(model)
%
%%
NT = 100; %number of trials per coherence level (this function always simulates equal number of trials for all coherences)
trLength = [50 100]; %trial lengths are chosen uniformly between trLength(1) and trLength(2) bins
timeSeries.delta_t = 10e-3; %bin size in seconds
timeSeries.timeAfterDotsOn = 200e-3; %saying that the spike counts here begin 200ms after stimulus onset

%% model parameters (default parameters assume 5 coherence/stimulus levels)

timeSeries.trueParams.model = 'ramping';
timeSeries.trueParams.beta  = [-0.015 -0.0075 0 0.0075 0.015]; %coherence-dependent slope
timeSeries.trueParams.l_0   = 0.50;  %inital diffusion point, must be less than one
timeSeries.trueParams.w2    = 0.003; %diffusion variance

if model.nlin == "softplus"
    timeSeries.trueParams.gamma = 50; %firing rate multiplier - firing rate is log(1+exp(gamma*x)) where x is the diffusion variable with a bound at 1
elseif model.nlin == "power" && model.pow == 2
    timeSeries.trueParams.gamma = 7;
elseif model.nlin == "power" && model.pow == 0.5
    timeSeries.trueParams.gamma = 1600;
elseif model.nlin == "exp"
    timeSeries.trueParams.gamma = 4;
end

if model.history == true % history weights (zero means no history dependence) 
    timeSeries.trueParams.hs = [-0.5 -0.4 -0.2 -0.1 0 0.1 0.05 0.05 0.025 0.025]; 
else
    timeSeries.trueParams.hs = zeros(1,10); 
end

if model.bias == true % bias
    timeSeries.trueParams.bias  = 5.0; 
else
    timeSeries.trueParams.bias = 0;
end

%% sets "choice" randomly on each trial according to coherence-based probability
%  - choice in this simulation has nothing to do with behavioral implications of diffusion-to-bound
timeSeries.trueParams.choiceP = [0.05 0.25 0.5 0.75 0.95];

%% setup timeSeries space
NC = length(timeSeries.trueParams.beta); %number of coherence levels
NH = length(timeSeries.trueParams.hs);
trLengths = trLength(1)-1 + randi(trLength(2) - trLength(1),NC*NT,1); %length of each trial
timeSeries.trialStart = [1; 1+cumsum(trLengths(1:end-1))];
timeSeries.y = nan(sum(trLengths),1);
timeSeries.trueParams.x = nan(sum(trLengths),1);
timeSeries.trCoh   = nan(NC*NT,1);

% define nonlinearity
if model.nlin == "softplus"
    nlin = @(x) safeSoftplusPower(x, 1.0);
elseif model.nlin == "power"
    nlin = @(x) safeSoftplusPower(x, model.pow);
elseif model.nlin == "exp"
    nlin = @(x) exp(x);
end

timeSeries.SpikeHistory = poissrnd((nlin(timeSeries.trueParams.l_0*timeSeries.trueParams.gamma)+timeSeries.trueParams.bias)*timeSeries.delta_t,[NT*NC NH]);

%% simulate diffusion paths
fprintf('Simulating from the ramping model (%d total trials, %d coherence levels)...\n',NC*NT,NC);
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
        
        y = zeros(length(xs),1);
        y_history = [timeSeries.SpikeHistory(trNum,:)'; y];
        for xind = 1:length(xs)
            sh = timeSeries.trueParams.hs*y_history(xind+NH-(1:NH));
            fr = (nlin(xs(xind)*timeSeries.trueParams.gamma)+timeSeries.trueParams.bias)*exp(sh);
            if fr > 400
                warning('firing rate above 400, the simulation may have runaway spiking')
            end
            y_history(xind+NH) = poissrnd(fr*timeSeries.delta_t);
        end
        y = y_history(NH+1:end);
                
        %inserts simulation into time series
        T1 = timeSeries.trialStart(trNum);
        T2 = timeSeries.trialStart(trNum) + trLengths(trNum) - 1;
        timeSeries.y(T1:T2) = y;
        timeSeries.trueParams.x(T1:T2) = xs;
        
        %selects psuedo-choice
        timeSeries.choice(trNum) = (rand>timeSeries.trueParams.choiceP(cc)) + 1;
    end
end
fprintf('done.\n');
