//#include "cuda/cuda.h"
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "mex.h"

#include "kcDefs.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    if(nrhs != 1) {
        mexPrintf("Incorrect RHS args: expected 1 and received %d (kcArrayToHost) ", nrhs);
        mexErrMsgTxt("CUDA errors");
    }
    if(nlhs != 1) {
        mexPrintf("Incorrect LHS args: expected 1 and received %d (kcArrayToHost) ", nlhs);
        mexErrMsgTxt("CUDA errors");
    }

	//init data crap
    unsigned int mSize = (unsigned int)mxGetScalar(mxGetField(prhs[0],0,KC_ARRAY_NUMEL));
    unsigned int ndims = (unsigned int)mxGetScalar(mxGetField(prhs[0],0,KC_ARRAY_NDIM));
    const mwSize* size = (const mwSize*)mxGetPr(mxGetField(prhs[0],0,KC_ARRAY_SIZE));


    unsigned int memSize = mSize*sizeof(KC_FP_TYPE);
    KC_FP_TYPE *d_a;
    d_a = (KC_FP_TYPE *)(unsigned KC_PTR_SIZE int)mxGetScalar(mxGetField(prhs[0],0,KC_ARRAY_PTR));
    if(d_a == KC_NULL_ARRAY) {
        mexPrintf("Invalid GPU array\n");
        return;
    }

	
    plhs[0] = mxCreateNumericArray(ndims,size,KC_FP_TYPE_MATLAB,mxREAL);
    KC_FP_TYPE* ans = (KC_FP_TYPE*)mxGetData(plhs[0]);
    cudaError_t copyResult = cudaMemcpy(ans,d_a,memSize,cudaMemcpyDeviceToHost);
    
    if(copyResult == cudaErrorInvalidValue) {
        mexPrintf("cudaErrorInvalidValue\n");
    }
    else if(copyResult == cudaErrorInvalidDevicePointer) {
        mexPrintf("cudaErrorInvalidDevicePointer\n");
    }
    else if(copyResult == cudaErrorInvalidMemcpyDirection) {
        mexPrintf("cudaErrorInvalidMemcpyDirection\n");
    }



}
