//#include <cuda.h>
#include "mex.h"
#include "matrix.h"
 

#include <helper_cuda.h>

#include "kcDefs.h" //see for info on anything starting with KC_
#include "kcArrayFunctions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs != 1) {
        mexPrintf("Incorrect RHS args: expected 1 and received %d (kcArrayToGPU) ", nrhs);
        
        mexErrMsgTxt("CUDA errors");
    }
    if(nlhs != 1) {
        mexPrintf("Incorrect LHS args: expected 1 and received %d (kcArrayToGPU) ", nlhs);
        
        mexErrMsgTxt("CUDA errors");
    }
    
    cudaError_t ce;
    KC_FP_TYPE * idata;
    // get idata(uint32) 
    if((mxIsDouble(prhs[0]) && sizeof(KC_FP_TYPE) == 8) || (mxIsSingle(prhs[0]) && sizeof(KC_FP_TYPE) == 4)) {
        idata = (KC_FP_TYPE *)mxGetPr(prhs[0]);
    }
    else if(mxIsDouble(prhs[0]) && sizeof(KC_FP_TYPE) == 4) {
        //converts the data to to the right FP type

    }
    else {
        if(sizeof(KC_FP_TYPE) == 4) {
            mexPrintf("Data type error: input must be a single-precision floating point matrix (kcArrayToGPU) ");
        }
        else {
            mexPrintf("Data type error: input must be a floating point matrix (kcArrayToGPU) ");
        }
        mexErrMsgTxt("CUDA errors");
    }

    plhs[0] = kcSetupEmptyArray(mxGetNumberOfDimensions(prhs[0]),mxGetDimensions(prhs[0]));


    // get number of elements 
    size_t numElements=mxGetNumberOfElements(prhs[0]);
    // memory size
    unsigned int memSize = sizeof(KC_FP_TYPE) * numElements;

    

    // allocate memory in GPU
    KC_FP_TYPE *gdata;
    ce =  cudaMalloc( (void**) &gdata, memSize);
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating array (kcArrayToGPU) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

    // copy to GPU
    ce =  cudaMemcpy( gdata, idata, memSize, cudaMemcpyHostToDevice) ;
    if(ce != cudaSuccess) {
        mexPrintf("Error copying array (kcArrayToGPU) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }


    // create MATLAB ref to GPU pointer
    /*mxArray* ptrVal=mxCreateNumericMatrix(1,1,KC_PTR_SIZE_MATLAB,mxREAL);
    unsigned KC_PTR_SIZE  int * out = (unsigned KC_PTR_SIZE int *)mxGetPr(ptrVal);
    *out = (unsigned KC_PTR_SIZE int)gdata;
    mxSetField(plhs[0],0,KC_ARRAY_PTR, ptrVal);*/
    unsigned KC_PTR_SIZE int * ptr = (unsigned KC_PTR_SIZE int*)mxGetPr(mxGetField(plhs[0],0,KC_ARRAY_PTR));
    *ptr = (unsigned KC_PTR_SIZE int)gdata;


    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error finalizing new array allocation (kcArrayToGPU) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

}
