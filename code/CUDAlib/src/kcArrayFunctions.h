//functions for handling a MATLAB structure that holds CUDA array information.
//Warning: This h file inelegantly holds quite a few functions.
#include <cuda.h>
#include "mex.h"
#include "matrix.h"
 

#include <helper_cuda.h>

#include <kcDefs.h>

mxArray* kcSetupEmptyArray(unsigned int numDims, const mwSize * dims) {

    const char * field_names[] =  {KC_ARRAY_NUMEL,KC_ARRAY_NDIM,KC_ARRAY_SIZE,KC_ARRAY_PTR,KC_ARRAY_TYPE};

    mwSize structDims[1] = {1};
    mxArray*emptyArray = mxCreateStructArray(1,structDims, 5, field_names);
    
    int numelField = mxGetFieldNumber(emptyArray,KC_ARRAY_NUMEL);
    int dimField  = mxGetFieldNumber(emptyArray,KC_ARRAY_NDIM);
    int sizeField = mxGetFieldNumber(emptyArray,KC_ARRAY_SIZE);
    int ptrField  = mxGetFieldNumber(emptyArray,KC_ARRAY_PTR);
    int typeField = mxGetFieldNumber(emptyArray,KC_ARRAY_TYPE);
    

    mxArray* dimsArray = mxCreateNumericMatrix(1,numDims,mxINT64_CLASS,mxREAL);
    long long int * dimsPtr = (long long int*) mxGetPr(dimsArray);
    memcpy(dimsPtr,dims,sizeof(mwSize)*numDims);


    mxArray* numDimsArray = mxCreateNumericMatrix(1,1,mxUINT32_CLASS,mxREAL);
    unsigned int * numDimsPtr = (unsigned int*)mxGetPr(numDimsArray);
    numDimsPtr[0] = numDims;
    
    mxArray * typeVal = mxCreateNumericMatrix(1,1,mxUINT32_CLASS,mxREAL);
    unsigned int * typePtr = (unsigned int*) mxGetPr(typeVal);
    *typePtr = KC_DOUBLE_ARRAY;


    unsigned int numElements = 1;
    for(int i = 0; i <numDims;i++) {
        numElements = numElements*dims[i];
    }
    mxArray * numelVal = mxCreateNumericMatrix(1,1,mxUINT32_CLASS,mxREAL);
    unsigned int * numelPtr = (unsigned int*) mxGetPr(numelVal);
    *numelPtr = (unsigned int)numElements;

    mxArray* ptrVal=mxCreateNumericMatrix(1,1,KC_PTR_SIZE_MATLAB,mxREAL);
    unsigned KC_PTR_SIZE  int * out = (unsigned KC_PTR_SIZE int *)mxGetPr(ptrVal);
    *out = (unsigned KC_PTR_SIZE int)0;


    mxSetFieldByNumber(emptyArray,0,dimField,  numDimsArray);
    mxSetFieldByNumber(emptyArray,0,sizeField, dimsArray);
    mxSetFieldByNumber(emptyArray,0,ptrField,  ptrVal);
    mxSetFieldByNumber(emptyArray,0,typeField, typeVal);
    mxSetFieldByNumber(emptyArray,0,numelField, numelVal);

    return emptyArray;
}


KC_FP_TYPE * kcGetArrayData(const mxArray * arrayInfo) {
    if(mxIsStruct(arrayInfo)) {
        int arrayType = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_TYPE));
        if(arrayType == KC_DOUBLE_ARRAY) {
        
            KC_FP_TYPE * ptr = (KC_FP_TYPE *)(unsigned KC_PTR_SIZE int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_PTR));

            if(ptr == KC_NULL_ARRAY) {
                mexErrMsgTxt("Array value NULL.\n");
                return 0;
            }   
            else {
                return ptr;
            }
        }
        else {
            mexErrMsgTxt("Invalid GPU array type.\n");
        }
    }  
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

KC_FP_TYPE * kcGetArrayData(const mxArray * arrayInfo, unsigned int minSize) {

    KC_FP_TYPE * ptr = kcGetArrayData(arrayInfo);

    unsigned int size = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_NUMEL));
    if(size >= minSize) {
        return ptr;
    }
    else {
         mexErrMsgTxt("GPU array too small.\n");
        return 0;
    }
}



int * kcGetArrayDataInt(const mxArray * arrayInfo) {
    if(mxIsStruct(arrayInfo)) {
        int arrayType = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_TYPE));
        if(arrayType == KC_INT_ARRAY) {
        
            int * ptr = (int *)(unsigned KC_PTR_SIZE int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_PTR));
            

            if(ptr == KC_NULL_ARRAY) {
                mexErrMsgTxt("Array value NULL.\n");
            }   
            else {
                return ptr;
            }
        }
        else {
            mexErrMsgTxt("Invalid GPU array type.\n");
        }
    }  
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

int * kcGetArrayDataInt(const mxArray * arrayInfo, unsigned int minSize) {
    int * ptr = kcGetArrayDataInt(arrayInfo);
    
    unsigned int size = ((unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_NUMEL)));
    if(size >= minSize) {
        return ptr;
    }
    else {
        mexErrMsgTxt("GPU array too small.\n");
    }
    return 0;
}

void * kcGetArrayDataVoid(const mxArray * arrayInfo, bool nullError) {
    if(mxIsStruct(arrayInfo)) {
        
        void * ptr = (void *)(unsigned KC_PTR_SIZE int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_PTR));
        if(ptr == KC_NULL_ARRAY && nullError) {
            mexErrMsgTxt("Array value NULL.\n");
        }   
        else {
            return ptr;
        }
    }  
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");   
    }
    return 0;
}

unsigned int kcGetArrayNumEl(const mxArray * arrayInfo) {
    if(mxIsStruct(arrayInfo)) {
        int arrayType = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_TYPE));
        if(arrayType != 0) {
            int * ptr = (int *)(unsigned KC_PTR_SIZE int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_PTR));
            unsigned int size = ((unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_NUMEL)));
            
            if(ptr == KC_NULL_ARRAY) {
                mexErrMsgTxt("Array value NULL.\n");
            }
            else {
                return size;
            }
        }
        else {
            mexErrMsgTxt("Invalid GPU array type.\n");
        }
    }
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

long long int kcGetArraySize(const mxArray * arrayInfo, int dim) {
    if(mxIsStruct(arrayInfo)) {
        int arrayType = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_TYPE));
        if(arrayType != 0) {
            int * ptr = (int *)(unsigned KC_PTR_SIZE int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_PTR));
            long long int * size = ((long long int*)mxGetPr(mxGetField(arrayInfo,0,KC_ARRAY_SIZE)));
            
            if(ptr == KC_NULL_ARRAY) {
                mexErrMsgTxt("Array value NULL.\n");
            }
            else {
                return size[dim];
            }
        }
        else {
            mexErrMsgTxt("Invalid GPU array type.\n");
        }
    }
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}
