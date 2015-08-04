function [fName] = compileCUDAmex(fName)

if(nargin > 0)

    myCodeHome =  '/home/latimerk/gitCode/LIPStateSpaceRelease/code'; %absolute path!
    sourceDir = [myCodeHome '/CUDAlib/src'];
    objDir    = [myCodeHome '/CUDAlib/obj'];

    extraArgs = '-Xcompiler -fpic';

    compileCUDAlibMex = @(fName) system(['cd ' objDir '; /usr/local/cuda-7.0/bin/nvcc -c -shared -m64 --gpu-architecture sm_35 ' extraArgs ' -I' sourceDir ' -I/usr/local/cuda-7.0/samples/common/inc/  -I/usr/local/MATLAB/R2013a/extern/include ' sourceDir '/' fName '.cu' ]); %mv ' fName '.o ' objDir '/' fName '.o'


    if(~isdir(objDir))
        mkdir(objDir);
    end

    compileCUDAlibMex(fName);
    display('NOTE: if receveing warning about /usr/local/cuda-7.0/samples/common/inc/exception.h')
    display(' everything is fine. There is something in a CUDA lib that the compiler isn''t happy about, but it will run anyway');

else
    fName = {};
    fName = {fName{:}, 'kcResetDevice'}; %#ok<*CCAT>

    fName = {fName{:},'kcArrayGetColumn'};
    fName = {fName{:},'kcArrayGetColumnInt'};
    fName = {fName{:},'kcArrayToGPU'};
    fName = {fName{:},'kcArrayToHost'};
    fName = {fName{:},'kcArrayToHostint'};
    fName = {fName{:},'kcArrayToGPUint'};
    fName = {fName{:},'kcFreeGPUArray'};
    
    fName = {fName{:},'kcStepTimeSampler'};
    
    fName = {fName{:},'kcRampPathSampler'};
end