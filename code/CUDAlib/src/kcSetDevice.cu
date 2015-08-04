#include "mex.h"
#include <cuda.h>
#include "kcDefs.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    int currentDevice, newDevice;
    cudaError_t ce;
    cudaGetDevice(&currentDevice);
    
    printf("Current GPU device: %d\n",currentDevice);
    
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
    printf("Changed to GPU device: %d\n",newDevice);

}
