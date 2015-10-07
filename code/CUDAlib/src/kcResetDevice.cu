#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>




#include <helper_functions.h>
#include <helper_cuda.h>


#include "mex.h"

#include "kcDefs.h" //see for info on anything starting with KC_
#include "kcArrayFunctions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    cudaError_t ce;
    /*ce = cudaSetDevice(0);
    if(ce != cudaSuccess) {
        mexPrintf("Error selecting device: %d\n", (int)ce);
    }
    else {*/
    cudaGetLastError();
        ce = cudaDeviceReset();
        if(ce != cudaSuccess) {
            mexPrintf("Error reseting device: %d\n", (int)ce);
        }
        else {
            mexPrintf("Device reset.\n");
        }
    //}
}
