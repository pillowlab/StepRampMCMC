function [fName] = linkCUDAmex(fName)

if(nargin > 0)

    [myCodeHome, CUDAdirectory] = myPaths();
    objDir     = [myCodeHome '/CUDAlib/obj'];
    mexDir     = [myCodeHome '/CUDAlib/mex'];
    
    linkCUDAlibMex    = @(fName) mex('-cxx','-o', [mexDir '/' fName],'-v','-L', [CUDAdirectory '/lib64/'], '-l','cuda', '-l', 'cudart', '-l','cusparse', '-l','cublas', '-l','curand', [objDir '/' fName '.o']);


    if(~isdir(mexDir))
        mkdir(mexDir);
        addpath(mexDir);
    end
    linkCUDAlibMex(fName);
else
    fName = compileCUDAmex();
end