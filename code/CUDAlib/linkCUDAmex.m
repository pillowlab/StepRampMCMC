function [fName] = linkCUDAmex(fName)

if(nargin > 0)

    myCodeHome2 =  '/home/latimerk/gitCode/LIPStateSpaceRelease/code'; %absolute path!
    objDir2     = [myCodeHome2 '/CUDAlib/obj'];
    mexDir2     = [myCodeHome2 '/CUDAlib/mex'];

    linkCUDAlibMex    = @(fName) mex('-cxx','-o', [mexDir2 '/' fName],'-v','-L', '/usr/local/cuda-7.0/lib64/', '-l','cuda', '-l', 'cudart', '-l','cusparse', '-l','cublas', '-l','curand', [objDir2 '/' fName '.o']);


    if(~isdir(mexDir2))
        mkdir(mexDir2);
        addpath(mexDir2);
    end
    linkCUDAlibMex(fName);
else
    fName = compileCUDAmex();
end