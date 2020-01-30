
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
__global__ void kcSumLangevinVars(KC_FP_TYPE * der, KC_FP_TYPE * der_sum, KC_FP_TYPE * G, KC_FP_TYPE * G_sum, KC_FP_TYPE * ll, KC_FP_TYPE * ll_sum, int * mBlkIdx, int NT, int NP, KC_FP_TYPE * gPrior, KC_FP_TYPE * lPrior) {
    int nsum = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(nsum == 0) {
        for(int idx = 0; idx < NP; idx++){
            der_sum[idx]=0;
            der_sum[idx]=lPrior[idx];
        }
        for(int jj = 0; jj < NP; jj++){
            for(int idx = 0; idx<NT; idx++){
                der_sum[jj] += der[jj*NT+idx];
            }
        }
    }
    else if(nsum == 1) {
        for(int idx = 0; idx < NP; idx++) {
            for(int idx2 = 0; idx2 < NP; idx2++) {
                    G_sum[idx+idx2*(NP)] = 0;
                    G_sum[idx+idx2*(NP)] = gPrior[idx*(NP)+idx2];
            }
        }
        for(int jj = 0; jj < NP; jj++) {
      	    for(int kk = 0; kk < NP; kk++) {
                for(int idx =0; idx < NT; idx++) {
                    G_sum[jj*(NP)+kk] -= G[idx+(jj*(NP)+kk)*NT];
                }
            }
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
__device__ KC_FP_TYPE h(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt, KC_FP_TYPE bias,KC_FP_TYPE steep,KC_FP_TYPE center) {
    KC_FP_TYPE fr = KC_MAX(KC_MIN( KC_MAXN, gamma / ( 1 + KC_EXP(-steep*(lambda-center)) ) + bias ),KC_MINN);
    return fr*dt;
}

__device__ KC_FP_TYPE dh_g(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE bias,KC_FP_TYPE steep,KC_FP_TYPE center) {
    return KC_MAX(KC_MIN( KC_MAXN, 1 / ( 1 + KC_EXP(-steep*(lambda-center)) ) ),KC_MINN);
}

__device__ KC_FP_TYPE dh_k(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE bias,KC_FP_TYPE steep,KC_FP_TYPE center) {
    KC_FP_TYPE num = gamma*(lambda-center)*KC_MAX(KC_MIN(KC_MAXN,KC_EXP(-steep*(lambda-center))),KC_MINN);
    KC_FP_TYPE denom = KC_MAX(KC_MIN(KC_MAXN,( 1 + KC_EXP(-steep*(lambda-center)) )*( 1 + KC_EXP(-steep*(lambda-center)) )),KC_MINN);
    return num/denom;
}

// computes log p(single trial | gamma, fixed lambdas, spike history)
__global__ void kcBoundaryLikelihoodTrialHist(KC_FP_TYPE * y, KC_FP_TYPE * lambdas, int * crossingTimes, int * mBlkIdx, KC_FP_TYPE g, KC_FP_TYPE dt, int NT, KC_FP_TYPE * llSum, KC_FP_TYPE * trialSum, KC_FP_TYPE * trialSumRiemann, int NP, KC_FP_TYPE bias, KC_FP_TYPE steep, KC_FP_TYPE center) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {

        for(int jj = 0; jj<NP; jj++){
            trialSum[idx+jj*NT]=0;
            for(int kk = 0; kk<NP; kk++){
                trialSumRiemann[idx+(jj*(NP)+kk)*NT]=0;
            }
        }
        llSum[idx] = 0;        

        for(int ii = mBlkIdx[idx]; ii < mBlkIdx[idx+1]; ii++)  {

            KC_FP_TYPE trueLambda = fmin(1, ((ii-mBlkIdx[idx]) < crossingTimes[idx])?lambdas[ii]:1);
            //KC_FP_TYPE trueLambda = fmin(1, lambdas[ii]);
            
            KC_FP_TYPE r  = h(trueLambda,g,1,bias,steep,center);
            llSum[idx] += y[ii]*(KC_LOG(r)+KC_LOG(dt)) - r*dt - KC_GAMMALN(y[ii]+1.0);

            for(int jj = 0; jj < NP ; jj++) {
            
            KC_FP_TYPE dr1 = 0;
            // 0 = gamma, 1 = bias, 2 = steep, 3 = center
            if(jj == 0) {
                dr1 = dh_g(trueLambda,g,bias,steep,center);
            }
            else if(jj == 1) {
                dr1 = 1;
            }
            else if(jj == 2) {
                dr1 = dh_k(trueLambda,g,bias,steep,center);
            }

            trialSum[idx+jj*NT] += (y[ii]/r-dt)*dr1;

                //for(int kk = 0; kk < NP; kk++) {
                for(int kk = jj; kk < NP; kk++) {

                    KC_FP_TYPE dr2 = 0;
                    // 0 = gamma, 1 = bias, 2 = steep, 3 = center
                    if(kk == 0) {
                        dr2 = dh_g(trueLambda,g,bias,steep,center);
                    }
                    else if(kk == 1) {
                        dr2 = 1;
                    }
                    else if(kk == 2) {
                        dr2 = dh_k(trueLambda,g,bias,steep,center);
                    }

            		trialSumRiemann[idx+(NP)*NT*jj+NT*kk] += -1*dt*dr1*dr2/r;

        		}
            }
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
//  6  = gPrior (Fisher information of sigmoid parameters)
//  7  = lPrior (derivative of log prior probability of sigmoid parameters)
//  8  = bias
//  9  = steep
//  10  = center
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

    unsigned int NP = 3;
    //loads Fisher information prior
    if(mxGetClassID(prhs[6]) != KC_FP_TYPE_MATLAB) {
        mexErrMsgTxt("Prior matrix input wrong floating point type (kcLangevinStep)!");
    }
    KC_FP_TYPE * gPrior;
    checkCudaErrors(cudaMalloc((void**)&gPrior,sizeof(KC_FP_TYPE)*(NP)*(NP)));
    checkCudaErrors(cudaMemcpy(gPrior,(KC_FP_TYPE*)mxGetPr(prhs[6]),sizeof(KC_FP_TYPE)*((NP)*(NP)),cudaMemcpyHostToDevice));


    //loads derivative of log prior probability of parameters
    if(mxGetClassID(prhs[7]) != KC_FP_TYPE_MATLAB) {
        mexErrMsgTxt("Prior matrix input wrong floating point type (kcLangevinStep)!");
    }
    KC_FP_TYPE * lPrior;
    checkCudaErrors(cudaMalloc((void**)&lPrior,sizeof(KC_FP_TYPE)*(NP)));
    checkCudaErrors(cudaMemcpy(lPrior,(KC_FP_TYPE*)mxGetPr(prhs[7]),sizeof(KC_FP_TYPE)*(NP),cudaMemcpyHostToDevice));

    KC_FP_TYPE bias   = mxGetScalar(prhs[8]);
    KC_FP_TYPE steep   = mxGetScalar(prhs[9]);
    KC_FP_TYPE center   = mxGetScalar(prhs[10]);
    
    //sets up space for computations on GPU
    KC_FP_TYPE * der_log_p_y;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y,sizeof(KC_FP_TYPE)*(NT)*(NP)));    
    KC_FP_TYPE * der_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y_sum,sizeof(KC_FP_TYPE)*(NP)*1));    

    KC_FP_TYPE * log_p_y;
    checkCudaErrors(cudaMalloc((void**)&log_p_y,sizeof(KC_FP_TYPE)*NT));    
    KC_FP_TYPE * log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&log_p_y_sum,sizeof(KC_FP_TYPE)*1));    

    KC_FP_TYPE * G_log_p_y1;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y1,sizeof(KC_FP_TYPE)*(NT)*(NP)*(NP)));    
    KC_FP_TYPE * G_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y_sum,sizeof(KC_FP_TYPE)*(NP)*(NP)));    

    //sets up CUDA variables
    int blockSize = 2;
    int numBlocks = (int)NT/(int)blockSize + ((NT%blockSize==0)?0:1);

    //gets each trials likelihood + derivatives of filter
    kcBoundaryLikelihoodTrialHist<<< numBlocks,blockSize >>>(y,lambda,crossingTimes,trIdx,g,dt, NT,log_p_y,der_log_p_y,G_log_p_y1,NP,bias,steep,center);
    checkCudaErrors(cudaDeviceSynchronize());
    
    //sums up all the trials' likelihoods and derivatives with respect to gamma
    int nBlocksC = 3;
    int blockSizeC = 1;
    kcSumLangevinVars <<< nBlocksC,blockSizeC >>> (der_log_p_y, der_log_p_y_sum, G_log_p_y1, G_log_p_y_sum, log_p_y, log_p_y_sum,  trIdx, NT, NP, gPrior, lPrior);
    checkCudaErrors(cudaDeviceSynchronize());

    
    //pushes answers back to MATLAB
    if(nlhs > 0) {
        plhs[0] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[0]),log_p_y_sum,sizeof(KC_FP_TYPE)*1,cudaMemcpyDeviceToHost));
    }
    if(nlhs > 1) {
        plhs[1] = mxCreateNumericMatrix(NP,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[1]),der_log_p_y_sum,sizeof(KC_FP_TYPE)*(NP)*(1),cudaMemcpyDeviceToHost));
    }
    if(nlhs > 2) {
        plhs[2] = mxCreateNumericMatrix(NP,NP,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[2]),G_log_p_y_sum,sizeof(KC_FP_TYPE)*(NP)*(NP),cudaMemcpyDeviceToHost));
    }
    
    //clears up GPU variables
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(log_p_y));
    checkCudaErrors(cudaFree(log_p_y_sum));
    checkCudaErrors(cudaFree(der_log_p_y));
    checkCudaErrors(cudaFree(der_log_p_y_sum));
    checkCudaErrors(cudaFree(G_log_p_y1));
    checkCudaErrors(cudaFree(G_log_p_y_sum));
    checkCudaErrors(cudaFree(lPrior));
    checkCudaErrors(cudaFree(gPrior));

    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error at the end of kcLangevinStep.cu ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

}
