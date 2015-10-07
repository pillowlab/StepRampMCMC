#include "mex.h"
#include <cuda_runtime.h>
#include "kcDefs.h" //see for info on anything starting with KC_
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    int currentDevice, newDevice;
    cudaError_t ce;
    cudaGetDevice(&currentDevice);
    
    mexPrintf("Current GPU device: %d\n",currentDevice);
    
    if(nrhs == 0) {
        ce = cudaSetDevice(KC_GPU_DEVICE);
    }
    else {
        ce = cudaSetDevice((int)mxGetScalar(prhs[0]));
    }
    if(ce != cudaSuccess) {
        mexPrintf("Error selecting device ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }

    
    cudaGetDevice(&newDevice);
    mexPrintf("Changed to GPU device: %d\n",newDevice);

}
