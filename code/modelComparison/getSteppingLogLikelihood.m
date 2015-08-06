%% Computes the probability of the spikes timeSeries.y given a fixed setting of the step model parameters
% marginalized over step time and direction.
% parameters used: 
% modelFit.alpha.mean
% modelFit.p.mean
% modelFit.phi.mean
% modelFit.r.mean
function [l_like, trial_likelihood] = getSteppingLogLikelihood(timeSeries,params,modelFit)

timeSeries = setupTrialIndexStructure(timeSeries);

maxTrialLength = max(timeSeries.trialIndex(:,2) - timeSeries.trialIndex(:,1) + 1);
NC = max(timeSeries.trCoh);

nbPDF = zeros(maxTrialLength+1,NC);
nbCDF = zeros(maxTrialLength+1,NC);


mynbinPDF = @(z,r,p)gamma(z+r)./(factorial(z).*gamma(r)).*((1-p)^r.*p.^z);
% mynbinPDF = @(z,r,p)gammaln(z+r) - gammaln(z+1) - gammaln(r) + r*log(1-p) + z*log(p);
mynbinCDF = @(z,r,p)(1-betainc(p,z+1,r));


for cc = 1:NC
    x = (0:maxTrialLength)';
    nbPDF(:,cc) = mynbinPDF(x, modelFit.r.mean, modelFit.p.mean(cc));
    nbCDF(:,cc) = mynbinCDF(x, modelFit.r.mean, modelFit.p.mean(cc));
end

NT = size(timeSeries.trialIndex,1);

trial_likelihood = zeros(NT,1);

for tr = 1:NT
    T1 = timeSeries.trialIndex(tr,1);
    T2 = timeSeries.trialIndex(tr,2);
    cc = timeSeries.trCoh(tr);
    
    T = T2-T1+1;
    
    ps1 = -params.delta_t*modelFit.alpha.mean(1)+timeSeries.y(T1:T2)   .*(log(modelFit.alpha.mean(1)) + log(params.delta_t)) - gammaln(timeSeries.y(T1:T2)+1);
    ps2 = -params.delta_t*modelFit.alpha.mean(2)+timeSeries.y(T2:-1:T1).*(log(modelFit.alpha.mean(2)) + log(params.delta_t)) - gammaln(timeSeries.y(T2:-1:T1)+1);
    ps3 = -params.delta_t*modelFit.alpha.mean(3)+timeSeries.y(T2:-1:T1).*(log(modelFit.alpha.mean(3)) + log(params.delta_t)) - gammaln(timeSeries.y(T2:-1:T1)+1);
    
    
    ps1 = cumsum(ps1);
    ps2 = cumsum(ps2);
    ps3 = cumsum(ps3);
    
    like_norm = 0;%sum(timeSeries.y(T1:T2).*log(params.delta_t) - gammaln(timeSeries.y(T1:T2)+1));
    
    sp2 = zeros(T,1);
    sp3 = zeros(T,1);
    sp2(1) = ps2(T);
    sp3(1) = ps3(T);
    
    for tt = 2:T
        sp2(tt) = ps1(tt-1)+ps2(T-tt+1);
        sp3(tt) = ps1(tt-1)+ps3(T-tt+1);
    end
    
    sp2 = exp(sp2);
    sp3 = exp(sp3);
    
    nbp = nbPDF(1:T,cc);
    p2 = (1-modelFit.phi.mean(cc))*nbp'*sp2;
    p3 = (modelFit.phi.mean(cc))  *nbp'*sp3;
    
    totalP = (p2 + p3 + (1-sum(nbp))*exp(ps1(end)));

    
   
    %% finalize
    trial_likelihood(tr) = log(totalP) + like_norm;
    
end

l_like = sum(trial_likelihood);
