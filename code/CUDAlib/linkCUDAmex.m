function [fName] = linkCUDAmex(fName)

if(nargin > 0)

    [myCodeHome, CUDAdirectory] = myPaths();
    objDir     = [myCodeHome '/code/CUDAlib/obj'];
    mexDir     = [myCodeHome '/code/CUDAlib/mex'];
    
    %linkCUDAlibMex    = @(fName) mex('-cxx','-o', [mexDir '/' fName],'-v','-L', [CUDAdirectory '/lib64/'], '-l','cuda', '-l', 'cudart', '-l','cusparse', '-l','cublas', '-l','curand', [objDir '/' fName '.o']);
    linkCUDAlibMex    = @(fName) mex('-cxx','-output', [mexDir '/' fName],'-v', '-lstdc++',['-L' CUDAdirectory '/lib64/'], '-lcuda', '-lcudart', '-lnppc', '-lnpps', '-lnppi', '-lcusparse', '-lcublas','-lcurand' , '-lmwblas',[objDir '/' fName '.o']);

    if(~isdir(mexDir))
        mkdir(mexDir);
        addpath(mexDir);
    end
    linkCUDAlibMex(fName);

    fName = [mexDir '/' fName '.' mexext];
else
    fName = compileCUDAmex();
end
