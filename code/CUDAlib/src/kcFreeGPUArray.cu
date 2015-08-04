//#include "cuda/cuda.h"
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include <helper_functions.h>
#include <helper_cuda.h>

#include "mex.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    cudaError_t cudaFreeStatus;
	void * d_a = kcGetArrayDataVoid(prhs[0],true);
    cudaFreeStatus = cudaFree(d_a);
   
    

    if(cudaFreeStatus == cudaErrorInvalidDevicePointer) {
        mexPrintf("Free memory, invalid device ptr.\n");
    }
    else if(cudaFreeStatus == cudaErrorInitializationError) {
        mexPrintf("Free memory, init error.\n");
    }
 
    else if(cudaFreeStatus != cudaSuccess) {
        mexPrintf("Free memory failed.\n");
    }
    else {
        unsigned KC_PTR_SIZE int * out = (unsigned KC_PTR_SIZE int *) mxGetPr(mxGetField(prhs[0],0,KC_ARRAY_PTR));
        *out = (unsigned KC_PTR_SIZE int)0;
    }    
}
