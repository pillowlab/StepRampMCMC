function [fName] = compileCUDAmex(fName)

if(nargin > 0)

    [myCodeHome, CUDAdirectory, CUDASamplesdirectory, MATLABdirectory, GPUArchitecture] = myPaths(); 
    sourceDir = [myCodeHome '/code/CUDAlib/src'];
    objDir    = [myCodeHome '/code/CUDAlib/obj'];
    
    extraArgs = '-Xcompiler -fpic';

    compileCUDAlibMex = @(fName) system(['cd ' objDir '; ' CUDAdirectory '/bin/nvcc -c -shared -m64 --gpu-architecture ' GPUArchitecture ' ' extraArgs ' -I' sourceDir ' -I' CUDASamplesdirectory '/common/inc/  -I' MATLABdirectory 'extern/include ' sourceDir '/' fName '.cu' ]); %mv ' fName '.o ' objDir '/' fName '.o'


    if(~isdir(objDir))
        mkdir(objDir);
    end

    compileCUDAlibMex(fName);
        
    fName = [objDir '/'  fName '.o'];
else
    %if no argument, just outputs the list of CUDA files used in the project
    fName = {};
    fName = {fName{:}, 'kcResetDevice'}; %#ok<*CCAT>
    fName = {fName{:}, 'kcSetDevice'};
    
    fName = {fName{:},'kcArrayGetColumn'};
    fName = {fName{:},'kcArrayGetColumnInt'};
    fName = {fName{:},'kcArrayToGPU'};
    fName = {fName{:},'kcArrayToHost'};
    fName = {fName{:},'kcArrayToHostint'};
    fName = {fName{:},'kcArrayToGPUint'};
    fName = {fName{:},'kcFreeGPUArray'};
    
    fName = {fName{:},'kcStepTimeSampler'};
    fName = {fName{:},'kcStepTimeSampler2'};
    fName = {fName{:},'kcStepTimeSamplerAdd'};
    fName = {fName{:},'kcStepTimeSamplerMult'};
    
    fName = {fName{:},'kcAlphaSpikeHistorySampler'};
    fName = {fName{:},'kcAlphaSpikeHistorySamplerMult'};
    
    fName = {fName{:},'kcGetSpikeHistoryEffect'};
    
    fName = {fName{:},'kcRampVarianceSampler'};
    
    fName = {fName{:},'kcRampPathSamplerExp'};
    fName = {fName{:},'kcRampPathSampler'};
    fName = {fName{:},'kcRampPathSampler2'};
    fName = {fName{:},'kcRampPathSamplerHist'};
    fName = {fName{:},'kcRampPathSamplerExpHist'};
    fName = {fName{:},'kcRampPathSamplerExpHistBias'};
    fName = {fName{:},'kcRampPathSamplerBias'};
    fName = {fName{:},'kcRampPathSamplerLog1PMultBias'};   
    fName = {fName{:},'kcRampPathSamplerLog1PBiasMult'};
    fName = {fName{:},'kcRampPathSamplerBound'};
    fName = {fName{:},'kcRampBoundHeightSamplerFlex'};

    fName = {fName{:},'kcRampBoundHeightSampler'};
    fName = {fName{:},'kcRampBoundHeightSamplerBias'};
    fName = {fName{:},'kcRampBoundHeightSamplerExp'};
    fName = {fName{:},'kcRampBoundHeightSamplerFlex'};
    fName = {fName{:},'kcGammaSpikeHistorySampler'};
    fName = {fName{:},'kcGammaSpikeHistorySamplerExp'};
    fName = {fName{:},'kcGammaSpikeHistorySamplerExpBias'};
    fName = {fName{:},'kcGammaSpikeHistorySamplerLog1PMultBias'};
    fName = {fName{:},'kcGammaSpikeHistorySamplerLog1PBiasMult'};   
    fName = {fName{:},'kcRampBoundHeightSpikeHistorySamplerFlex'};
    
    fName = {fName{:},'kcRampLikelihood'};
    fName = {fName{:},'kcRampLikelihood2'};
    fName = {fName{:},'kcRampLikelihoodBias'};
    fName = {fName{:},'kcRampLikelihoodExp'};
    fName = {fName{:},'kcRampLikelihoodExpGLM'};
    fName = {fName{:},'kcRampLikelihoodExpGLMBias'};
    fName = {fName{:},'kcRampLikelihoodLog1PMultBias'};
    fName = {fName{:},'kcRampLikelihoodLog1PBiasMult'};    
    fName = {fName{:},'kcRampLikelihoodGLM'};
    fName = {fName{:},'kcRampLikelihoodGrid'};
    
end
