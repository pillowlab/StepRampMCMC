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


//simulates a ramping (diffusion-to-bound) path for each trial and computes likelihood
 //   kcComputeLL<<<nBlocks,blockSize>>>(y,trIdx,trCoh,T,pcnv,pcnv_idx,npadL,npadR,P,fr,dt,B,NT,NC,log_p_tr,dx,cv); 
__global__ void kcComputeLL(KC_FP_TYPE * y, const int * trIdx, const int * trCoh, const int * T, KC_FP_TYPE * pcnv, int * pcnv_idx, int * npadL, int * npadR, KC_FP_TYPE * P, KC_FP_TYPE * fr, const KC_FP_TYPE dt, const int B, const int NT, const int NC, KC_FP_TYPE * log_p_tr, KC_FP_TYPE dx, KC_FP_TYPE * cv) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT ) {

        log_p_tr[idx] = 0;

        int coh = trCoh[idx];

        // length of middle part of P (convolving part)
        int m = B-2;

        // length of pcnv for convolving
        int n = pcnv_idx[coh+1]-pcnv_idx[coh];

        for(int ii = trIdx[idx]; ii < trIdx[idx+1]; ii++)  {
            
            // do convolution for every index but first
            if(ii-trIdx[idx]>0){


                // convolve
                for(int kk = 0; kk < m+n-1; kk++) {

                    cv[idx*2*B+kk] = 0;

                    int jj_min = (kk >= n-1) ? kk - (n-1) : 0;
                    int jj_max = (kk < m-1)  ? kk : m-1;

                   
                    for(int jj = jj_min; jj <= jj_max; jj++){
                         
                        cv[idx*2*B+kk] += P[1+jj]*pcnv[pcnv_idx[coh]+kk-jj];
                        
                    }

                }
                
                // mass in first bin
                P[0] = P[0];
                for(int kk = 0; kk < npadL[coh]; kk++){
                    P[0] += cv[idx*2*B+kk];
                }

                // mass in middle
                for(int bb = 1; bb < B-1; bb++){
                    P[bb] = cv[idx*2*B+bb-1+npadL[coh]];
                }
/*                
                for(int kk = npadL[idx]; kk < m+n-1-npadR[coh]; kk++){
                    P[count] = cv[idx*2*B+kk];
                    count = int(count+1);
                } */

                // mass in last bin
                P[B-1] = P[B-1];
                for(int kk = B-1+npadL[coh]; kk < m+n-1; kk++){
                    P[B-1] += cv[idx*2*B+kk];
                }
                

            }
            
            // multiply spikes
            for(int jj = 0; jj < B; jj++) {
                KC_FP_TYPE like = KC_EXP(y[ii]*KC_LOG(fr[jj])-fr[jj]-KC_GAMMALN(y[ii]+1));
                P[jj] = P[jj]*like;
            }
            
        }


    // sum log likelihood for the trial
    for(int jj = 0; jj < B; jj++) {
        log_p_tr[idx] += P[jj];
    }

    log_p_tr[idx] = KC_LOG(log_p_tr[idx]*dx);

    }

}


//Estimates the log probability of a set of spike trains under the ramping model given a set of fixed parameters
// This estimation is made by Monte Carlo simulations from the model to integrate out latent variable
//args
//  0  = y (observations)
//  1  = trIdx (array that gives the start of each trial in y, y being indexed at 0. Includes final value that should be length of y)
//  2  = T (number of steps forward for each trial)
//  3  = pcnv (convolution kernels)
//  4  = pcnv_idx (index of starting point for each convolution kernel, with last index being length of pcnv)
//  5  = npadL (number of padded zeros on left for each kernel)
//  6  = npadR (number of padded zeros on right for each kernel)
//  7  = P (starting probability distribution)
//  8  = firing rate (vector of firing rate for each bin of P)
//  9 = dt (step size, should be the same as the bin size for spikes)
//  10 = trCoh (coherence for each trial)
//  11 = dx
//  12 = NC
//outputs (left-hand side)
//  0  = log p(y|\theta)
//  1  = log p(y|\theta) for each individual trial
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {

    //load up trial data
    unsigned int TT = kcGetArrayNumEl(prhs[0]);
    KC_FP_TYPE * y      = kcGetArrayData(prhs[0]);
    int * trIdx = kcGetArrayDataInt(prhs[1]);
    unsigned int NT = kcGetArrayNumEl(prhs[1])-1;


    //int * T = kcGetArrayDataInt(prhs[2]);
    int * T;
    checkCudaErrors(cudaMalloc((void**)&T,sizeof(int)*(NT)));
    checkCudaErrors(cudaMemcpy(T,(int*)mxGetPr(prhs[2]),sizeof(int)*(NT),cudaMemcpyHostToDevice));

    int NC = mxGetScalar(prhs[12]);

    //load up convolution indices
    //int * pcnv_idx = kcGetArrayDataInt(prhs[4]);
    //int * pcnv_idx;
    //checkCudaErrors(cudaMalloc((void**)&pcnv_idx,sizeof(int)*(NC+1)));
    //checkCudaErrors(cudaMemcpy(pcnv_idx,(int*)mxGetPr(prhs[4]),sizeof(int)*(NC+1),cudaMemcpyHostToDevice));
    int * pcnv_idx = kcGetArrayDataInt(prhs[4]);

    //load up convolution kernels
    //if(mxGetClassID(prhs[3]) != KC_FP_TYPE_MATLAB) {
    //    mexErrMsgTxt("Prior matrix input wrong floating point type (kcLangevinStep)!");
    //}
    //KC_FP_TYPE * pcnv;
    //checkCudaErrors(cudaMalloc((void**)&pcnv,sizeof(KC_FP_TYPE)*(pcnv_idx[NC])));
    //checkCudaErrors(cudaMemcpy(pcnv,(KC_FP_TYPE*)mxGetPr(prhs[3]),sizeof(KC_FP_TYPE)*(pcnv_idx[NC]),cudaMemcpyHostToDevice));
    KC_FP_TYPE * pcnv = kcGetArrayData(prhs[3]);

    //load up npadL
    //int * npadL = kcGetArrayDataInt(prhs[5]);
    int * npadL;
    checkCudaErrors(cudaMalloc((void**)&npadL,sizeof(int)*(NC)));
    checkCudaErrors(cudaMemcpy(npadL,(int*)mxGetPr(prhs[5]),sizeof(int)*(NC),cudaMemcpyHostToDevice));

    //load up npadR
    //int * npadR = kcGetArrayDataInt(prhs[6]);
    int * npadR;
    checkCudaErrors(cudaMalloc((void**)&npadR,sizeof(int)*(NC)));
    checkCudaErrors(cudaMemcpy(npadR,(int*)mxGetPr(prhs[6]),sizeof(int)*(NC),cudaMemcpyHostToDevice));

    unsigned int B = kcGetArrayNumEl(prhs[7]);

    //load up P
    KC_FP_TYPE * P = kcGetArrayData(prhs[7]);

    //load up fr
    KC_FP_TYPE * fr = kcGetArrayData(prhs[8]);

    KC_FP_TYPE dt = mxGetScalar(prhs[9]);

    int * trCoh = kcGetArrayDataInt(prhs[10]);

    KC_FP_TYPE dx = mxGetScalar(prhs[11]);

    int blockSize = 2;
    int nBlocks   = NT/blockSize + ((NT%blockSize==0)?0:1);

    //allocates sspace on GPU for simulating the likelihood
    KC_FP_TYPE * log_p_tr;
    KC_FP_TYPE * sum_log_p;
    KC_FP_TYPE * cv;
    checkCudaErrors(cudaMalloc((void**)&log_p_tr,sizeof(KC_FP_TYPE)*NT));
    checkCudaErrors(cudaMalloc((void**)&sum_log_p,sizeof(KC_FP_TYPE)*1));
    checkCudaErrors(cudaMalloc((void**)&cv,sizeof(KC_FP_TYPE)*NT*2*B));

    //propogate density forward to get log p(y|\theta) for each trial
    kcComputeLL<<<nBlocks,blockSize>>>(y,trIdx,trCoh,T,pcnv,pcnv_idx,npadL,npadR,P,fr,dt,B,NT,NC,log_p_tr,dx,cv); 
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
    checkCudaErrors(cudaFree(log_p_tr));
    checkCudaErrors(cudaFree(sum_log_p));
    checkCudaErrors(cudaFree(pcnv));
    checkCudaErrors(cudaFree(pcnv_idx));
    checkCudaErrors(cudaFree(npadL));
    checkCudaErrors(cudaFree(npadR));
    checkCudaErrors(cudaFree(P));
    checkCudaErrors(cudaFree(fr));
    checkCudaErrors(cudaFree(cv));

}
