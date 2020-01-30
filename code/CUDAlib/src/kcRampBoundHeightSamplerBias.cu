
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
__global__ void kcSumLangevinVars(KC_FP_TYPE * der, KC_FP_TYPE * der_sum, KC_FP_TYPE * G, KC_FP_TYPE * G_sum, KC_FP_TYPE * ll, KC_FP_TYPE * ll_sum, KC_FP_TYPE * der_bias, KC_FP_TYPE * der_sum_bias, KC_FP_TYPE * G_bias, KC_FP_TYPE * G_sum_bias, int * mBlkIdx, int NT, KC_FP_TYPE gPrior, KC_FP_TYPE lPrior, KC_FP_TYPE bias_g, KC_FP_TYPE bias_l, KC_FP_TYPE * off_diag, KC_FP_TYPE * off_diag_sum) {
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
    else if(nsum == 3) {
        der_sum_bias[0] = bias_l;
        for(int idx = 0; idx < NT; idx++) {
            der_sum_bias[0] += der_bias[idx];
        }
    }
    else if(nsum == 4) {
        G_sum_bias[0] = bias_g;
                
        for(int idx = 0; idx < NT; idx++) {
            G_sum_bias[0] -= G_bias[idx];
        }
    }
    else if(nsum == 5) {
        off_diag_sum[0] = 0;
        for(int idx = 0; idx < NT; idx++) {
            off_diag_sum[0] -= off_diag[idx];
        }
    } 
    
}


//derivates of  firing rate function w.r.t. gamma (assuming fixed latent variables)
__device__ KC_FP_TYPE h(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt, KC_FP_TYPE bias) {
    KC_FP_TYPE fr = KC_MAX(KC_MIN(KC_MAXN,log1p(exp(lambda*gamma))+bias),KC_MINN);
    return fr*dt;
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



// computes log p(single trial | gamma, fixed lambdas)
__global__ void kcBoundaryLikelihoodTrial(KC_FP_TYPE * y, KC_FP_TYPE * lambdas, int * crossingTimes, int * mBlkIdx, KC_FP_TYPE g, KC_FP_TYPE dt, int NT, KC_FP_TYPE * llSum, KC_FP_TYPE * trialSum, KC_FP_TYPE * trialSumRiemann, KC_FP_TYPE bias, KC_FP_TYPE * trialSumB, KC_FP_TYPE * trialSumRiemannB, KC_FP_TYPE * trialRiemannOffDiag) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {

        trialSum[idx]  = 0;
        trialSumRiemann[idx] = 0;
        llSum[idx] = 0;
        trialSumB[idx] = 0;
        trialSumRiemannB[idx] = 0;
        trialRiemannOffDiag[idx] = 0;


        for(int ii = mBlkIdx[idx]; ii < mBlkIdx[idx+1]; ii++)  {

            KC_FP_TYPE trueLambda = fmin(1, ((ii-mBlkIdx[idx]) < crossingTimes[idx])?lambdas[ii]:1);
            
            KC_FP_TYPE r  = h(trueLambda,g,1,bias);
            llSum[idx] += y[ii]*(KC_LOG(r)+KC_LOG(dt)) - dt*r -lgamma(y[ii]+1);
            
            KC_FP_TYPE dr = dh(trueLambda,g,1);

            trialSum[idx] += (y[ii]/r-dt)*dr;
            //trialSumRiemann[idx] += -1*dh2_h(trueLambda,g,dt);
            trialSumRiemann[idx] += -1*dt*dr*dr/r;
            trialSumB[idx] += (y[ii]/r-dt);
            trialSumRiemannB[idx] += -1*dt/r;
            trialRiemannOffDiag[idx] += -1*dt*dr/r;
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
//  8  = bias
//  9 = bias gPrior
//  10 = bias lPrior
//
//outputs (left-hand side)
//  0  = log p(y|lambdas,gamma)
//  1  = d/dg log p(y|lambdas,gamma)
//  2  = d^2/d^2g log p(y|lambdas,gamma)
//  3
//  4
//  5
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
    KC_FP_TYPE bias   = mxGetScalar(prhs[8]);
    KC_FP_TYPE bias_g = mxGetScalar(prhs[9]);
    KC_FP_TYPE bias_l = mxGetScalar(prhs[10]);

    //sets up space for computations on GPU
    KC_FP_TYPE * der_log_p_y;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y,sizeof(KC_FP_TYPE)*(NT)));    
    KC_FP_TYPE * der_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y_sum,sizeof(KC_FP_TYPE)*(1)));    

    KC_FP_TYPE * der_log_p_y_bias;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y_bias,sizeof(KC_FP_TYPE)*(NT)));    
    KC_FP_TYPE * der_log_p_y_sum_bias;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y_sum_bias,sizeof(KC_FP_TYPE)*(1)));   

    KC_FP_TYPE * log_p_y;
    checkCudaErrors(cudaMalloc((void**)&log_p_y,sizeof(KC_FP_TYPE)*NT));    
    KC_FP_TYPE * log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&log_p_y_sum,sizeof(KC_FP_TYPE)*1));    

    KC_FP_TYPE * G_log_p_y1;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y1,sizeof(KC_FP_TYPE)*(NT)));    
    KC_FP_TYPE * G_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y_sum,sizeof(KC_FP_TYPE)*(1)*(1)));     

    KC_FP_TYPE * G_log_p_y1_bias;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y1_bias,sizeof(KC_FP_TYPE)*(NT)));    
    KC_FP_TYPE * G_log_p_y_sum_bias;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y_sum_bias,sizeof(KC_FP_TYPE)*(1)*(1)));           

    KC_FP_TYPE * G_off_diag;
    checkCudaErrors(cudaMalloc((void**)&G_off_diag,sizeof(KC_FP_TYPE)*(NT)));
    KC_FP_TYPE * G_off_diag_sum;
    checkCudaErrors(cudaMalloc((void**)&G_off_diag_sum,sizeof(KC_FP_TYPE)*(1)*(1)));

    //sets up CUDA variables
    int blockSize = 2;
    int numBlocks = (int)NT/(int)blockSize + ((NT%blockSize==0)?0:1);

    //gets each trials likelihood + derivates of gamma
    kcBoundaryLikelihoodTrial<<< numBlocks,blockSize >>>(y,lambda,crossingTimes,trIdx,g,dt, NT,log_p_y,der_log_p_y,G_log_p_y1,bias,der_log_p_y_bias,G_log_p_y1_bias,G_off_diag);
    checkCudaErrors(cudaDeviceSynchronize());
    
    //sums up all the trials' likelihoods and derivatives with respect to gamma
    int nBlocksC = 6;
    int blockSizeC = 1;
    kcSumLangevinVars <<< nBlocksC,blockSizeC >>> (der_log_p_y, der_log_p_y_sum, G_log_p_y1, G_log_p_y_sum, log_p_y, log_p_y_sum,der_log_p_y_bias, der_log_p_y_sum_bias, G_log_p_y1_bias, G_log_p_y_sum_bias, trIdx, NT, gPrior, lPrior, bias_g, bias_l,G_off_diag,G_off_diag_sum);
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
    if(nlhs > 3) {
        plhs[3] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[3]),der_log_p_y_sum_bias,sizeof(KC_FP_TYPE)*(1)*(1),cudaMemcpyDeviceToHost));
    }
    if(nlhs > 4) {
        plhs[4] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[4]),G_log_p_y_sum_bias,sizeof(KC_FP_TYPE)*(1)*(1),cudaMemcpyDeviceToHost));
    }
    if(nlhs > 5) {
        plhs[5] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[5]),G_off_diag_sum,sizeof(KC_FP_TYPE)*(1)*(1),cudaMemcpyDeviceToHost));
    }

    //clears up GPU variables
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(log_p_y));
    checkCudaErrors(cudaFree(log_p_y_sum));
    checkCudaErrors(cudaFree(der_log_p_y));
    checkCudaErrors(cudaFree(der_log_p_y_sum));
    checkCudaErrors(cudaFree(der_log_p_y_bias));
    checkCudaErrors(cudaFree(der_log_p_y_sum_bias));
    checkCudaErrors(cudaFree(G_log_p_y1));
    checkCudaErrors(cudaFree(G_log_p_y_sum));
    checkCudaErrors(cudaFree(G_log_p_y1_bias));
    checkCudaErrors(cudaFree(G_log_p_y_sum_bias));
    checkCudaErrors(cudaFree(G_off_diag));
    checkCudaErrors(cudaFree(G_off_diag_sum));

    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error at the end of kcLangevinStep.cu ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

}
