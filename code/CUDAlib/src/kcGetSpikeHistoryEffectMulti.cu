
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


// computes spike history effect
__global__ void kcSpikeHistoryEffect(KC_FP_TYPE * y, int * trIdx, KC_FP_TYPE * y_hist, KC_FP_TYPE * h_filt, int NH, int NT, KC_FP_TYPE * spe, int TT, int numNeur) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {

        for(int nn = 0; nn < numNeur; nn++) {

            for(int ii = trIdx[idx]; ii < trIdx[idx+1]; ii++)  {

                spe[ii+(nn*TT)] = 0;

                for(int jj = 0; jj < NH; jj++) {

                    if(jj < NH && ii<(trIdx[idx]+jj+1)) {
                        // offset to after the spike history train for each trial, so for first filter subtract 1, second filter subtract 2, etc. 
                        spe[ii+(nn*TT)] += y_hist[NH*(idx+1) + (ii-trIdx[idx]) - jj-1 + (nn*NH*NT)]*h_filt[jj+nn*NH];
                    }
                    else if(jj < NH) {
                        spe[ii+(nn*TT)] += y[ii-jj-1+(nn*TT)]*h_filt[jj+nn*NH];
                    }

                }

            }
    
        }

    }

}

//Computes the spike history effect
//args
//  0  = y (observations, on GPU), is TT 
//  1  = trIdx
//  2  = spike history (spikes before start of trials, NH*NTxnumNeuron x 1)
//  3  = spike history filters
//  4  = number of neurons

//outputs (left-hand side)
//  0  = spike history effect pointer

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    cudaError_t ce;

    //loads up trial information
    unsigned int TY = kcGetArrayNumEl(prhs[0]); // TY is total length of y
    int  numNeur    = mxGetScalar(prhs[4]);
    unsigned int TT = (int)TY/(int)numNeur; // TT is total number of bins for one neuron 
    unsigned int NT = kcGetArrayNumEl(prhs[1])-1;
    KC_FP_TYPE * y      = kcGetArrayData(prhs[0],TY);
    int * trIdx = kcGetArrayDataInt(prhs[1]);

    int NHN  = mxGetNumberOfElements(prhs[3]); // total number of filters, is NH times numNeur
    int NH   = int(NHN)/(int)numNeur; // number of filters for one neuron

    //loads filter values
    KC_FP_TYPE * h_filt;
    checkCudaErrors(cudaMalloc((void**)&h_filt,sizeof(KC_FP_TYPE)*NHN));
    checkCudaErrors(cudaMemcpy(h_filt,(KC_FP_TYPE*)mxGetPr(prhs[3]),sizeof(KC_FP_TYPE)*NHN,cudaMemcpyHostToDevice));

    //loads spike history before trials
    KC_FP_TYPE * y_hist = kcGetArrayData(prhs[2],NH*NT*numNeur);

    // sets up space for spike history effect
    KC_FP_TYPE * spe;
    checkCudaErrors(cudaMalloc((void**)&spe,sizeof(KC_FP_TYPE)*TY));

    //sets up CUDA variables
    int blockSize = 2;
    int numBlocks = (int)NT/(int)blockSize + ((NT%blockSize==0)?0:1);

    // computes spike history effect
    kcSpikeHistoryEffect<<< numBlocks,blockSize >>>(y,trIdx,y_hist,h_filt,NH,NT,spe,TT,numNeur);
    checkCudaErrors(cudaDeviceSynchronize());

    // push pointer to matlab
    mwSize dims[2] = {TY, 1};
    plhs[0] = kcSetupEmptyArray(2,dims);
    unsigned KC_PTR_SIZE int * ptr = (unsigned KC_PTR_SIZE int*)mxGetPr(mxGetField(plhs[0],0,KC_ARRAY_PTR));
    *ptr = (unsigned KC_PTR_SIZE int)spe;

    //clears up GPU variables
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(h_filt));

    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error at the end of kcLangevinStep.cu ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

}
