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

//poison log likelihood for one observation
__device__ KC_FP_TYPE lh(KC_FP_TYPE y, KC_FP_TYPE x, KC_FP_TYPE g, KC_FP_TYPE dt, KC_FP_TYPE sh) {
    KC_FP_TYPE fr = KC_MAX(KC_MIN(KC_EXP( g*x + sh),KC_MAXN),KC_MINN);
    return y*(KC_LOG(fr)+KC_LOG(dt)) - dt*fr - KC_GAMMALN(y+1.0);
}

//sums up log likelihood of each trial given model parameters
__global__ void kcSumGBfinal(const KC_FP_TYPE * log_p_tr, KC_FP_TYPE * log_p,  const int NT) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < 1) {
        log_p[0] = 0;
        for(int ii = 0; ii < NT; ii++) {
            log_p[0] += log_p_tr[ii];
        }
    }
}

//averages log likelihood of each simulated path
// (one thread for each trial)
__global__ void kcSumGBlogpTr(const KC_FP_TYPE * log_p, KC_FP_TYPE * log_p_tr, const int NT, const int nSims) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {
        
        log_p_tr[idx] = 0;
        KC_FP_TYPE trSum = 0;
        KC_FP_TYPE log_x = 0;
        log_p_tr[idx] = KC_SQRT(-1.0);
        
        //computes log( 1/nSims * \sum exp( log p(y | sim paths)) )  for a single trial
        // does the sum in a slightly more numerical stable way than just blindly exponentiating all the log likleihoods
        
        for(int ii = 0; ii < nSims && isnan(log_p_tr[idx]);ii++) {

            trSum = 1 ;
            log_x = log_p[ii*NT+idx];
            for(int kk = 0; kk < ii; kk++) {
                trSum += KC_EXP(log_p[kk*NT+idx] - log_x);
            }
            for(int kk = ii+1; kk < nSims; kk++) {
                trSum += KC_EXP(log_p[kk*NT+idx] - log_x);
            }
            if(trSum > 1e-25 && !isnan(trSum) && !isinf(trSum)) {
                log_p_tr[idx] = log_x-KC_LOG((double)nSims)+KC_LOG(trSum);
                break;
            }
        }
        
    }
}

//simulates a ramping (diffusion-to-bound) path for each trial and computes likelihood
__global__ void kcSimGBPaths(const  KC_FP_TYPE * y, const int * trIdx, const int * betaIdx, KC_FP_TYPE * xx, const KC_FP_TYPE * b,const KC_FP_TYPE w2,const  KC_FP_TYPE l_0, const KC_FP_TYPE g, const KC_FP_TYPE dt, KC_FP_TYPE * log_p, const int NT, const int TT,  const int sim, KC_FP_TYPE * spe) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT ) {
        int trNum = idx;
        int T1  = trIdx[trNum];
        //xx contains zero mean Gaussian noise of variance \omega^2
        
        xx[T1] += l_0; //xx[T1] now contains initial point for simulated diffusion trajectory for this trial
        
        int currIdx = sim*(NT)+idx;
        log_p[currIdx] = lh(y[T1],xx[T1],g,dt,spe[T1]);
        for(int ii = T1+1; ii < trIdx[trNum+1];ii++) {
            //progates particle forward in time
            xx[ii] = (xx[ii-1] >= 1.0)?1.0:KC_MIN(xx[ii] + xx[ii-1]+b[betaIdx[ii]],1.0);
            //log likelihood of single observation (bin) y[ii] given diffusion path is at x[ii]
            log_p[currIdx] += lh(y[ii],xx[ii],g,dt,spe[ii]);
        }
    }
}

//Estimates the log probability of a set of spike trains under the ramping model given a set of fixed parameters
// This estimation is made by Monte Carlo simulations from the model to integrate out latent variable
//args
//  0  = y (observations)
//  1  = trIdx (array that accesses the beta value used at each timepoint, y being indexed at 0. Includes final value that should be length of y)
//  2  = betaIdxVector (array that gives coherence used at each bins of y. i.e., accesses the beta value used at each timepoint. values begin at 0 instead of 1 to be consistent with C, unlike MATLAB)
//  3  = spike history effect (same size as y)
//  4  = beta values
//  5  = w (variance of diffusion process)
//  6  = l_0 (starting lambda value)
//  7  = g (absorbing boundary effective height)
//  8  = dt (bin size in seconds)
//  9  = number of samples to use to estimate log probability of observations (I recommend using at least 1000)
//outputs (left-hand side)
//  0  = log p(y|\theta)
//  1  = log p(y|\theta) for each individual trial
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    cudaError_t ce;

    //load up trial data
    unsigned int TT = kcGetArrayNumEl(prhs[0]);
    KC_FP_TYPE * y      = kcGetArrayData(prhs[0]);
    int * trIdx = kcGetArrayDataInt(prhs[1]);
    unsigned int NT = kcGetArrayNumEl(prhs[1])-1;
    int * betaIdx = kcGetArrayDataInt(prhs[2],TT);

    // load spike history effect
    KC_FP_TYPE * spe    = kcGetArrayData(prhs[3]);

    //how many simulations to use to estimate log p(y|\theta)
    int trialsToSim    = (int)mxGetScalar(prhs[9]); 
    
    //load up parameters to simulate model
    if(mxGetClassID(prhs[4]) != KC_FP_TYPE_MATLAB) {
        mexErrMsgTxt("Beta input wrong floating point type (kcSimGaussianBound)!");
    }
    KC_FP_TYPE * b      = (KC_FP_TYPE *)mxGetPr(prhs[4]);
    int   numBetas = mxGetNumberOfElements(prhs[4]);
    KC_FP_TYPE * b_gpu;

    ce = cudaMalloc((void**)&b_gpu,sizeof(KC_FP_TYPE)*numBetas);
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating space for betas on device - first allocation in function (kcSimGaussianBound) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
    }    
    checkCudaErrors(cudaMemcpy(b_gpu,b,sizeof(KC_FP_TYPE)*numBetas,cudaMemcpyHostToDevice));

    KC_FP_TYPE  w      = mxGetScalar(prhs[5]);
    KC_FP_TYPE  l_0    = mxGetScalar(prhs[6]);
    KC_FP_TYPE  g      = mxGetScalar(prhs[7]);
    KC_FP_TYPE  dt     = mxGetScalar(prhs[8]);
    
    
    
    //setup CUDA variables + random number generator
    int randSize = TT + (((TT)%2==0)?0:1); 
    KC_FP_TYPE * xx;
    checkCudaErrors(cudaMalloc((void**)&xx,randSize*sizeof(KC_FP_TYPE)));
    curandGenerator_t curandGen = 0;
    curandStatus_t curandStatus;
    curandStatus = curandCreateGenerator(&curandGen,  CURAND_RNG_PSEUDO_DEFAULT);
    if(curandStatus != CURAND_STATUS_SUCCESS ) {
        mexPrintf("CURAND-1 error %d\n",(int)curandStatus);
        mexErrMsgTxt("CUDA errors");
    }
    
    struct timeval now;
    gettimeofday(&now,NULL);
    unsigned long long mySeed = (unsigned long long)now.tv_usec+(unsigned long long)(1e7*(unsigned long long)now.tv_sec);
    curandStatus = curandSetPseudoRandomGeneratorSeed(curandGen, mySeed);
    if(curandStatus != CURAND_STATUS_SUCCESS ) {
        mexPrintf("CURAND-2 error %d\n",(int)curandStatus);
        mexErrMsgTxt("CUDA errors");
    } 
    
    int blockSize = 2;
    int nBlocks   = NT/blockSize + ((NT%blockSize==0)?0:1);
    
    int blockSizeT = 2;
    int nBlocksT   = NT/blockSizeT + ((NT%blockSizeT==0)?0:1);
    
    //allocates sspace on GPU for simulating the likelihood
    KC_FP_TYPE * log_p;
    //KC_FP_TYPE * log_p_2;
    KC_FP_TYPE * log_p_tr;
    KC_FP_TYPE * sum_log_p;
    checkCudaErrors(cudaMalloc((void**)&log_p,sizeof(KC_FP_TYPE)*NT*trialsToSim));
    //checkCudaErrors(cudaMalloc((void**)&log_p_2,sizeof(KC_FP_TYPE)*NT*trialsToSim));
    checkCudaErrors(cudaMalloc((void**)&log_p_tr,sizeof(KC_FP_TYPE)*NT));
    checkCudaErrors(cudaMalloc((void**)&sum_log_p,sizeof(KC_FP_TYPE)*1));
    
    // generate AR1 noise
    for(int kk = 0; kk < trialsToSim; kk++) {
        //generates zero mean Gaussian noise with correct variance
        curandStatus = KC_RANDOM_NORMAL_FUNCTION(curandGen,xx,randSize,0,KC_SQRT(w));
        if(curandStatus != CURAND_STATUS_SUCCESS ) {
            mexPrintf("CURAND gen error %d\n",(int)curandStatus);
            mexErrMsgTxt("CUDA errors");
        }
        //checkCudaErrors(cudaDeviceSynchronize());

        //calculate path + logP
        kcSimGBPaths<<<nBlocks,blockSize>>>(y,trIdx,betaIdx,xx,b_gpu,w,l_0,g,dt,log_p,NT,TT,kk,spe);
        ce = cudaDeviceSynchronize();
        if(ce != cudaSuccess) {
            mexPrintf("Error in simulating of kcSimGaussianBound.cu  ");
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA errors");
        }
    }
   
    // log_p_2 = log_p;

    //average likelihood of each sampled path to get log p(y|\theta) for each trial
    kcSumGBlogpTr<<<nBlocksT,blockSizeT>>>(log_p,log_p_tr,NT,trialsToSim);
    checkCudaErrors(cudaDeviceSynchronize());
    
    //sums up log likelihood of each trial
    kcSumGBfinal<<<1,1>>>(log_p_tr,sum_log_p,NT);
    checkCudaErrors(cudaDeviceSynchronize());
    
    //copy back to host
    if(nlhs > 0) {
         plhs[0] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
         checkCudaErrors(cudaMemcpy((KC_FP_TYPE *)mxGetPr(plhs[0]),sum_log_p,1*sizeof(KC_FP_TYPE),cudaMemcpyDeviceToHost));
    }
    if(nlhs > 1) {
         plhs[1] = mxCreateNumericMatrix(NT,1,KC_FP_TYPE_MATLAB,mxREAL);
         checkCudaErrors(cudaMemcpy((KC_FP_TYPE *)mxGetPr(plhs[1]),log_p_tr,NT*sizeof(KC_FP_TYPE),cudaMemcpyDeviceToHost));
    }

    //free up CUDA variables
    checkCudaErrors(curandDestroyGenerator(curandGen));
    checkCudaErrors(cudaFree(xx));
    checkCudaErrors(cudaFree(b_gpu));
    checkCudaErrors(cudaFree(log_p));
    checkCudaErrors(cudaFree(log_p_tr));
    checkCudaErrors(cudaFree(sum_log_p));
}
