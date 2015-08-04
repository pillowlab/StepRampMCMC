
//#include "cuda/cuda.h"
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

#include "kcDefs.h"
#include "kcArrayFunctions.h"

//rough summing kernel (does not need to be efficient)
__global__ void kcSumLangevinVars(KC_FP_TYPE * der, KC_FP_TYPE * der_sum, KC_FP_TYPE * G, KC_FP_TYPE * G_sum, KC_FP_TYPE * ll, KC_FP_TYPE * ll_sum, int * mBlkIdx, int NT) {
    int nsum = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(nsum == 0) {
        der_sum[0] = 0;
        for(int idx = 0; idx < NT; idx++) {
            der_sum[0] += der[idx];
        }
    }
    else if(nsum == 1) {
        G_sum[0] *= -1;
                
        for(int idx = 0; idx < NT; idx++) {
            G_sum[0] -= G[idx];
        }

    }
    else if(nsum == 2) {
        ll_sum[0] = 0;
        for(int idx = 0; idx < NT; idx++) {
            ll_sum[0] += ll[idx];
        }
    }
    
    
}


//derivates of  firing rate function w.r.t. gamma (assuming fixed latent variables)
__device__ KC_FP_TYPE h(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt) {
    KC_FP_TYPE fr = (gamma*lambda>40)?(dt*gamma*lambda):(dt*KC_LOG(1+KC_EXP(gamma*lambda)));
    return fr;
}

__device__ KC_FP_TYPE dh(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt) {
    return dt*lambda/KC_MIN(KC_MAXN,(1+KC_EXP(-1*gamma*lambda)));
}


__device__ KC_FP_TYPE dh2_h(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt) {
    KC_FP_TYPE ex  = KC_EXP(gamma*lambda);
    KC_FP_TYPE nex = 1/ex;
    KC_FP_TYPE lex = (gamma*lambda>40)?(gamma*lambda):KC_MAX(KC_LOG(1+ex),KC_MINN);
    return (lambda*lambda)*dt*ex/KC_MAX(KC_MINN,((ex+2+nex)*lex));
}



// cimputes log p(single trial | gamma, fixed lambdas)
__global__ void kcBoundaryLikelihoodTrial(KC_FP_TYPE * y, KC_FP_TYPE * lambdas, int * crossingTimes, int * mBlkIdx, KC_FP_TYPE g, KC_FP_TYPE dt, int NT, KC_FP_TYPE * llSum, KC_FP_TYPE * trialSum, KC_FP_TYPE * trialSumRiemann) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {

        trialSum[idx]  = 0;
        trialSumRiemann[idx] = 0;
        llSum[idx] = 0;


        for(int ii = mBlkIdx[idx]; ii < mBlkIdx[idx+1]; ii++)  {

            KC_FP_TYPE trueLambda = fmin(1, ((ii-mBlkIdx[idx]) < crossingTimes[idx])?lambdas[ii]:1);
            
            KC_FP_TYPE ex    = KC_MIN(KC_EXP( g*trueLambda),KC_MAXN);
            KC_FP_TYPE nex   = KC_MAX(KC_EXP(-g*trueLambda),KC_MINN);
            KC_FP_TYPE logex = (g*trueLambda<80)?(KC_MAX(KC_LOG(1+ex),KC_MINN)):(g*trueLambda);
            llSum[idx] += y[ii]*KC_LOG(logex) - dt*logex;
            
            KC_FP_TYPE dr = dh(trueLambda,g,1);
            KC_FP_TYPE r  = KC_MAX(KC_MINN,h(trueLambda,g,1));

            trialSum[idx] += (y[ii]/r-dt)*dr;
            trialSumRiemann[idx] += -1*dh2_h(trueLambda,g,dt);
        }
    }
}
        

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    cudaError_t ce;



    unsigned int TT = kcGetArrayNumEl(prhs[0]);
    KC_FP_TYPE * lambda = kcGetArrayData(prhs[0]);
    int * crossingTimes = kcGetArrayDataInt(prhs[1]);
    KC_FP_TYPE * y      = kcGetArrayData(prhs[2],TT);

    int * trIdx = kcGetArrayDataInt(prhs[3]);
    unsigned int NT = kcGetArrayNumEl(prhs[3])-1;
    

    KC_FP_TYPE  g      = mxGetScalar(prhs[4]);
    KC_FP_TYPE  dt     = mxGetScalar(prhs[5]);

    if(mxGetClassID(prhs[6]) != KC_FP_TYPE_MATLAB) {
        mexErrMsgTxt("Prior matrix input wrong floating point type (kcLangevinStep)!");
    }
    KC_FP_TYPE * priorMat =  (KC_FP_TYPE *)mxGetPr(prhs[6]);

    KC_FP_TYPE * der_log_p_y;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y,sizeof(KC_FP_TYPE)*(NT)));    
    KC_FP_TYPE * der_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y_sum,sizeof(KC_FP_TYPE)*(1)));    

    KC_FP_TYPE * log_p_y;
    checkCudaErrors(cudaMalloc((void**)&log_p_y,sizeof(KC_FP_TYPE)*NT));    
    KC_FP_TYPE * log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&log_p_y_sum,sizeof(KC_FP_TYPE)*1));    

    
    KC_FP_TYPE * G_log_p_y1;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y1,sizeof(KC_FP_TYPE)*(NT)));    
    KC_FP_TYPE * G_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y_sum,sizeof(KC_FP_TYPE)*(1)*(1)));    
    checkCudaErrors(cudaMemcpy(G_log_p_y_sum,priorMat,sizeof(KC_FP_TYPE)*(1)*(1),cudaMemcpyHostToDevice));    

    int blockSize = 2;
    int numBlocks = (int)NT/(int)blockSize + ((NT%blockSize==0)?0:1);

    //gets each trials likelihood + derivates of gamma
    kcBoundaryLikelihoodTrial<<< numBlocks,blockSize >>>(y,lambda,crossingTimes,trIdx,g,dt, NT,log_p_y,der_log_p_y,G_log_p_y1);
    checkCudaErrors(cudaDeviceSynchronize());
    
    //sums up all the trials' likelihoods and derivatives with respect to gamma
    int nBlocksC = 3;
    int blockSizeC = 1;
    kcSumLangevinVars <<< nBlocksC,blockSizeC >>> (der_log_p_y, der_log_p_y_sum, G_log_p_y1, G_log_p_y_sum, log_p_y, log_p_y_sum,  trIdx, NT);
    checkCudaErrors(cudaDeviceSynchronize());


    if(nlhs > 0) {
        plhs[0] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[0]),log_p_y_sum,sizeof(KC_FP_TYPE)*1,cudaMemcpyDeviceToHost));
    }
    if(nlhs > 1) {
        plhs[1] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[1]),der_log_p_y_sum,sizeof(KC_FP_TYPE)*(1),cudaMemcpyDeviceToHost));
    }
    if(nlhs > 2) {
        plhs[2] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[2]),G_log_p_y_sum,sizeof(KC_FP_TYPE)*(1)*(1),cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(log_p_y));
    checkCudaErrors(cudaFree(log_p_y_sum));
    checkCudaErrors(cudaFree(der_log_p_y));
    checkCudaErrors(cudaFree(der_log_p_y_sum));
    checkCudaErrors(cudaFree(G_log_p_y1));
    checkCudaErrors(cudaFree(G_log_p_y_sum));

    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error at the end of kcLangevinStep.cu ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

}
