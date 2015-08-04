//#include <cuda.h>
#include "mex.h"
#include "matrix.h"
 

#include <helper_cuda.h>

#include "kcDefs.h"
#include "kcArrayFunctions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs != 1) {
        mexPrintf("Incorrect RHS args: expected 1 and received %d (kcArrayToGPUint) ", nrhs);
        mexErrMsgTxt("CUDA errors");
    }
    if(nlhs != 1) {
        mexPrintf("Incorrect LHS args: expected 1 and received %d (kcArrayToGPUint) ", nlhs);
        mexErrMsgTxt("CUDA errors");
    }
    
    cudaError_t ce;
/*    ce = cudaSetDevice(KC_GPU_DEVICE);
    if(ce != cudaSuccess) {
        mexPrintf("Error setting device (kcArrayToGPUint) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }
*/
    // get idata(uint32) 
    if(!mxIsInt32(prhs[0])) {
        mexPrintf("Data type error: input must be a int32 matrix (kcArrayToGPUint) ");
        mexErrMsgTxt("CUDA errors");
    }
    
    int *idata= (int *)mxGetPr(prhs[0]);
    


    plhs[0] = kcSetupEmptyArray(mxGetNumberOfDimensions(prhs[0]),mxGetDimensions(prhs[0]));


    // get number of elements 
    size_t numElements=mxGetNumberOfElements(prhs[0]);
    // memory size
    unsigned int memSize = sizeof(int) * numElements;

    

    // allocate memory in GPU
    int *gdata;
    ce =  cudaMalloc( (void**) &gdata, memSize);
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating array (kcArrayToGPUint) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

    // copy to GPU
    ce = cudaMemcpy( gdata, idata, memSize, cudaMemcpyHostToDevice) ;
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating array (kcArrayToGPUint) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }



    // create MATLAB ref to GPU pointer
    /*mxArray* ptrVal=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);
    unsigned long  int * out = (unsigned long int *)mxGetPr(ptrVal);
    *out = (unsigned long int)gdata;
    mxSetField(plhs[0],0,KC_ARRAY_PTR, ptrVal);*/
    unsigned KC_PTR_SIZE int * ptr = (unsigned KC_PTR_SIZE int*)mxGetPr(mxGetField(plhs[0],0,KC_ARRAY_PTR));
    *ptr = (unsigned KC_PTR_SIZE int)gdata;

    unsigned int * type = (unsigned int*)mxGetPr(mxGetField(plhs[0],0,KC_ARRAY_TYPE));
    *type = KC_INT_ARRAY;

    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error finalizing new array allocation (kcArrayToGPUint) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }
    
}
