
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

//rough summing kernel (does not need to be efficient)
__global__ void kcSumLangevinVars(KC_FP_TYPE * der, KC_FP_TYPE * der_sum, KC_FP_TYPE * G, KC_FP_TYPE * G_sum, KC_FP_TYPE * ll, KC_FP_TYPE * ll_sum, int * mBlkIdx, int NT, KC_FP_TYPE gPrior, KC_FP_TYPE lPrior) {
    int nsum = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(nsum == 0) {
        der_sum[0] = lPrior;
        for(int idx = 0; idx < NT; idx++) {
            der_sum[0] += der[idx];
        }
    }
    else if(nsum == 1) {
        G_sum[0] = gPrior;
                
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
__device__ KC_FP_TYPE h(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt, KC_FP_TYPE modelInd) {
    if(modelInd > 0.001) {
        KC_FP_TYPE logex = KC_MAX(KC_MINN,(gamma*lambda>100)?(gamma*lambda):KC_MIN(log1p(exp(lambda*gamma)),KC_MAXN));
        return KC_MIN(KC_POW(logex*1.00000,modelInd)*dt,KC_MAXN);
    }
    else {
        return KC_MIN(exp(lambda*gamma)*dt,KC_MAXN);
    }
}

__device__ KC_FP_TYPE dh(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt, KC_FP_TYPE modelInd) {
    if( modelInd > 0.001) {
        KC_FP_TYPE logex = KC_MAX(KC_MINN,(gamma*lambda>100)?(gamma*lambda):KC_MIN(log1p(exp(lambda*gamma)),KC_MAXN));
        KC_FP_TYPE log_der = KC_MIN(lambda/(1+KC_MIN(KC_MAXN,KC_MAX(KC_MINN,exp(-lambda*gamma)))),KC_MAXN);
	KC_FP_TYPE der = modelInd*KC_POW(logex*1.00000,modelInd-1.00)*log_der;
        return der*dt;
    }
    else {
        return KC_MIN(dt*lambda*KC_EXP(gamma*lambda),KC_MAXN);
    }    
}


// computes log p(single trial | gamma, fixed lambdas)
__global__ void kcBoundaryLikelihoodTrial(KC_FP_TYPE * y, KC_FP_TYPE * lambdas, int * crossingTimes, int * mBlkIdx, KC_FP_TYPE g, KC_FP_TYPE dt, int NT, KC_FP_TYPE * llSum, KC_FP_TYPE * trialSum, KC_FP_TYPE * trialSumRiemann, KC_FP_TYPE modelInd) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {

        trialSum[idx]  = 0;
        trialSumRiemann[idx] = 0;
        llSum[idx] = 0;


        for(int ii = mBlkIdx[idx]; ii < mBlkIdx[idx+1]; ii++)  {

            KC_FP_TYPE trueLambda = fmin(1, ((ii-mBlkIdx[idx]) < crossingTimes[idx])?lambdas[ii]:1);
            
            KC_FP_TYPE fr    = KC_MAX(KC_MINN,h(trueLambda,g,1,modelInd));
            llSum[idx] += y[ii]*(KC_LOG(fr)+KC_LOG(dt)) - dt*fr -lgamma(y[ii]+1);
            
            KC_FP_TYPE dr = dh(trueLambda,g,1,modelInd);

            trialSum[idx] += (y[ii]/fr-dt)*dr;
            trialSumRiemann[idx] += -dt*dr*dr/fr;
        }
    }
}
        
//Computes the the log probability of a set of spike trains under the ramping model given a fixed set of latent variable
// as a function of \gamma (the bound height) along with first/second derivates w.r.t. \gamma
//args
//  0  = lambda (latent variables, on GPU. Same size as y)
//  1  = auxillary variable - threshold crossing time (latent variable boundary crossing time, on GPU. vector length number of trials: NT)
//  2  = y (observations, on GPU)
//  3  = trIdx (array that accesses the beta value used at each timepoint, y being indexed at 0. Includes final value that should be length of y)
//  4  = g (absorbing boundary effective height)
//  5  = dt (bin size in seconds)
//  6  = gPrior (contains Fisher information of gamma)
//  7  = lPrior (contains log prior probability of gamma)
//  8  = modelInd (power if use log1p transfer function, 0 if using exp)
//
//outputs (left-hand side)
//  0  = log p(y|lambdas,gamma)
//  1  = d/dg log p(y|lambdas,gamma)
//  2  = d^2/d^2g log p(y|lambdas,gamma)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    cudaError_t ce;

    //loads up trial information
    unsigned int TT = kcGetArrayNumEl(prhs[0]);
    int * crossingTimes = kcGetArrayDataInt(prhs[1]);
    KC_FP_TYPE * y      = kcGetArrayData(prhs[2],TT);

    int * trIdx = kcGetArrayDataInt(prhs[3]);
    unsigned int NT = kcGetArrayNumEl(prhs[3])-1;
    KC_FP_TYPE  dt     = mxGetScalar(prhs[5]);
    
    //loads gamma and latent variables
    KC_FP_TYPE  g      = mxGetScalar(prhs[4]);
    KC_FP_TYPE * lambda = kcGetArrayData(prhs[0]);

    //loads log prior probability of the gamma value
    if(mxGetClassID(prhs[6]) != KC_FP_TYPE_MATLAB) {
        mexErrMsgTxt("Prior matrix input wrong floating point type (kcLangevinStep)!");
    }

    KC_FP_TYPE gPrior = mxGetScalar(prhs[6]);
    KC_FP_TYPE lPrior = mxGetScalar(prhs[7]);
    KC_FP_TYPE modelInd = mxGetScalar(prhs[8]);

    //sets up space for computations on GPU
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

    //sets up CUDA variables
    int blockSize = 2;
    int numBlocks = (int)NT/(int)blockSize + ((NT%blockSize==0)?0:1);

    //gets each trials likelihood + derivates of gamma
    kcBoundaryLikelihoodTrial<<< numBlocks,blockSize >>>(y,lambda,crossingTimes,trIdx,g,dt, NT,log_p_y,der_log_p_y,G_log_p_y1,modelInd);
    checkCudaErrors(cudaDeviceSynchronize());
    
    //sums up all the trials' likelihoods and derivatives with respect to gamma
    int nBlocksC = 3;
    int blockSizeC = 1;
    kcSumLangevinVars <<< nBlocksC,blockSizeC >>> (der_log_p_y, der_log_p_y_sum, G_log_p_y1, G_log_p_y_sum, log_p_y, log_p_y_sum,  trIdx, NT, gPrior, lPrior);
    checkCudaErrors(cudaDeviceSynchronize());

    
    //pushes answers back to MATLAB
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
    
    //clears up GPU variables
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
