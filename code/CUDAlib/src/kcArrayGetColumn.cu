
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include <curand.h>

#include <helper_functions.h>
#include <helper_cuda.h>


#include "mex.h"

#include "kcDefs.h" //see for info on anything starting with KC_
#include "kcArrayFunctions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {



    mwSize *size  = (mwSize *)mxGetPr(mxGetField(prhs[0],0,KC_ARRAY_SIZE));

    KC_FP_TYPE *d_a;
    d_a = kcGetArrayData(prhs[0]);

    mwSize *size2 = (mwSize*)malloc(sizeof(mwSize)*2);
    size2[0] = size[0];
    size2[1] = 1; 


    int cNum = (int)mxGetScalar(prhs[1]);

    if(cNum < size[1]) {
        plhs[0] = kcSetupEmptyArray(2,size2);
        unsigned KC_PTR_SIZE int * ptr = (unsigned KC_PTR_SIZE int*)mxGetPr(mxGetField(plhs[0],0,KC_ARRAY_PTR));
        *ptr = (unsigned KC_PTR_SIZE int)(&(d_a[cNum*(size[0])]));
    }
    else {
        plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
        mexPrintf("Index out-of-bounds\n");
    }
}
