function [fName] = compileCUDAmex(fName)

if(nargin > 0)

    [myCodeHome, CUDAdirectory, MATLABdirectory] = myPaths(); 
    sourceDir = [myCodeHome '/code/CUDAlib/src'];
    objDir    = [myCodeHome '/code/CUDAlib/obj'];
    
    extraArgs = '-Xcompiler -fpic';

    compileCUDAlibMex = @(fName) system(['cd ' objDir '; ' CUDAdirectory '/bin/nvcc -c -shared -m64 --gpu-architecture sm_35 ' extraArgs ' -I' sourceDir ' -I' CUDAdirectory '/samples/common/inc/  -I' MATLABdirectory 'extern/include ' sourceDir '/' fName '.cu' ]); %mv ' fName '.o ' objDir '/' fName '.o'


    if(~isdir(objDir))
        mkdir(objDir);
    end

    compileCUDAlibMex(fName);
    display('NOTE: if receveing warning about /usr/local/cuda-7.0/samples/common/inc/exception.h')
    display(' everything is fine. There is something in a CUDA lib that the compiler isn''t happy about, but it will run anyway');

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
    
    fName = {fName{:},'kcRampPathSampler'};
    fName = {fName{:},'kcRampVarianceSampler'};
    fName = {fName{:},'kcRampBoundHeightSampler'};
    fName = {fName{:},'kcRampLikelihood'};
end