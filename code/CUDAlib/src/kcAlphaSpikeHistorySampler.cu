
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
__global__ void kcSumLangevinVars(KC_FP_TYPE * der, KC_FP_TYPE * der_sum, KC_FP_TYPE * G, KC_FP_TYPE * G_sum, KC_FP_TYPE * ll, KC_FP_TYPE * ll_sum, int * mBlkIdx, int NT, int NH, KC_FP_TYPE * gPrior, KC_FP_TYPE * lPrior) {
    int nsum = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(nsum == 0) {
        for(int idx = 0; idx < NH+3; idx++){
            der_sum[idx]=lPrior[idx];
        }
        for(int jj = 0; jj < NH+3; jj++){
            for(int idx = 0; idx<NT; idx++){
                der_sum[jj] += der[jj*NT+idx];
            }
        }
    }
    else if(nsum == 1) {
        for(int idx = 0; idx < NH+3; idx++) {
            for(int idx2 = 0; idx2 < NH+3; idx2++) {
                    G_sum[idx+idx2*(NH+3)] = 0;
                    G_sum[idx+idx2*(NH+3)] = gPrior[idx*(NH+3)+idx2];
            }
        }
        for(int jj = 0; jj < NH+3; jj++) {
      	    for(int kk = 0; kk < NH+3; kk++) {
                for(int idx =0; idx < NT; idx++) {
                    G_sum[jj*(NH+3)+kk] -= G[idx+(jj*(NH+3)+kk)*NT];
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
__device__ KC_FP_TYPE h(KC_FP_TYPE alpha, KC_FP_TYPE dt, KC_FP_TYPE sh) {
    KC_FP_TYPE ex    = KC_MIN(KC_EXP(alpha + sh),KC_MAXN);
    KC_FP_TYPE r = KC_LOG(1+ex);
    return r;
}

__device__ KC_FP_TYPE dh(KC_FP_TYPE alpha, KC_FP_TYPE dt, KC_FP_TYPE sh, KC_FP_TYPE mult) {
    return dt*mult/(1+KC_MIN(KC_MAXN,KC_MAX(KC_MINN,KC_EXP(-1*alpha-sh))));
}

// computes log p(single trial | gamma, fixed lambdas, spike history)
__global__ void kcBoundaryLikelihoodTrialHist(KC_FP_TYPE * y, KC_FP_TYPE * spe, int * mBlkIdx, KC_FP_TYPE dt, int NT, KC_FP_TYPE * llSum, KC_FP_TYPE * trialSum, KC_FP_TYPE * trialSumRiemann, KC_FP_TYPE * alphas, KC_FP_TYPE * h_filt, KC_FP_TYPE * y_hist, int NH, KC_FP_TYPE * zs, KC_FP_TYPE * ss) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {

        for(int jj = 0; jj<NH+3; jj++){
            trialSum[idx+jj*NT]=0;
            for(int kk = 0; kk<NH+3; kk++){
                trialSumRiemann[idx+(jj*(NH+3)+kk)*NT]=0;
            }
        }
        llSum[idx] = 0;        

        int stepTime = zs[idx]; 
        int stepState = ss[idx]-1; 
        
        for(int ii = mBlkIdx[idx]; ii < mBlkIdx[idx+1]; ii++)  {
            
            KC_FP_TYPE alpha;
            int currentState = 0;

            // check this timing
            if( (ii-mBlkIdx[idx]) < stepTime ) {
                alpha = alphas[0];
            }
            else {
                alpha = alphas[stepState];
                currentState = stepState;
            }

            KC_FP_TYPE sh    = spe[ii];


            KC_FP_TYPE ex    = KC_EXP(alpha + sh);
            KC_FP_TYPE nex   = KC_MIN(KC_EXP(-alpha - sh),KC_MINN);
            KC_FP_TYPE r = KC_MIN(KC_MAX(KC_MINN,log1p(ex)),KC_MAXN);
            llSum[idx] += y[ii]*(KC_LOG(r)+KC_LOG(dt)) - dt*r - KC_GAMMALN(y[ii]+1.0);

	    for(int jj = 0; jj < NH+3 ; jj++) {
                        
            KC_FP_TYPE mult1 = 0;
            if(jj < 3 && jj==currentState){
                mult1 = 1;
            }
            else if(jj > 2 && jj < (NH+3) && ii<(mBlkIdx[idx]+(jj-3)+1)){
                mult1 = y_hist[NH*(idx+1) + (ii-mBlkIdx[idx]) - (jj-3) -1];
            }
            else if(jj > 2 && jj < (NH+3)){
                mult1 =  y[ii-(jj-3)-1];
            }

            KC_FP_TYPE dr = dh(alpha,1,sh,mult1);

            trialSum[idx+jj*NT] += (y[ii]/r-dt)*dr;
                
                for(int kk = jj; kk < NH+3; kk++) {
                //for(int kk = 0; kk < NH+3; kk++) {

                    KC_FP_TYPE mult2 = 0;
                    if(kk < 3 && kk==currentState){
                        mult2 = 1;
                    }
                    else if(kk > 2 && kk < (NH+3) && ii<(mBlkIdx[idx]+(kk-3)+1)){
                        mult2 = y_hist[NH*(idx+1) + (ii-mBlkIdx[idx]) - (kk-3) -1];
                    }
                    else if(kk > 2 && kk < (NH+3)){
                        mult2 =  y[ii-(kk-3)-1];
                    }

				    KC_FP_TYPE dr2 = dh(alpha,1,sh,mult2);
            		trialSumRiemann[idx+(NH+3)*NT*jj+NT*kk] += -1*dt*dr*dr2/r;

        		}
            }
        }
    }
}

// [log_alphah, der_log_alpha, der_log_h] = kcAlphaSpikeHistorySampler(gpu_y,gpu_trIndex,StepSamples.z(:,ss),StepSamples.s(:,ss),timeSeries.delta_t,gpu_spe,StepSamples.alpha(:,ss-1),StepSamples.hs(ss-1,:),der_log_prior_alpha,der_log_prior_h)
// Computes the the log probability of a set of spike trains under the ramping model given a fixed set of latent variable
// as a function of \gamma (the bound height) along with first/second derivates w.r.t. \gamma
// args
//  0  = y (observations, on GPU)
//  1  = trIdx
//  2  = StepSamples.z(:,ss) (switch times)
//  3  = StepSamples.s(:,ss) (switch states)
//  4  = dt (bin size in seconds)
//  5  = spe (spike history effect, TT x 1)
//  6  = alphas
//  7  = hs
//  8  = der log prior of alpha and hs
//  9  = fisher information of log prior of alpha and hs
//  10  = spike history (spikes before start of trials, NH*NT x 1)
//
//outputs (left-hand side)
//  0  = log p(y|alphas,hs)
//  1  = d/dg log p(y|alphas,hs)
//  2  = d^2/d^2g log p(y|alphas,hs)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    cudaError_t ce;

    //loads up trial information
    unsigned int TT = kcGetArrayNumEl(prhs[0]);
    KC_FP_TYPE * y      = kcGetArrayData(prhs[0],TT);

    int * trIdx = kcGetArrayDataInt(prhs[1]);
    unsigned int NT = kcGetArrayNumEl(prhs[1])-1;
    
    // load StepSamples.z
    KC_FP_TYPE * zs;
    checkCudaErrors(cudaMalloc((void**)&zs,sizeof(KC_FP_TYPE)*NT));   
    checkCudaErrors(cudaMemcpy(zs,(KC_FP_TYPE*)mxGetPr(prhs[2]),sizeof(KC_FP_TYPE)*NT,cudaMemcpyHostToDevice));

    // load StepSamples.s
    KC_FP_TYPE * ss;
    checkCudaErrors(cudaMalloc((void**)&ss,sizeof(KC_FP_TYPE)*NT));   
    checkCudaErrors(cudaMemcpy(ss,(KC_FP_TYPE*)mxGetPr(prhs[3]),sizeof(KC_FP_TYPE)*NT,cudaMemcpyHostToDevice));

    // load dt
    KC_FP_TYPE  dt     = mxGetScalar(prhs[4]);

    // load spike history effect
    KC_FP_TYPE * spe      = kcGetArrayData(prhs[5],TT);

    // load alphas
    KC_FP_TYPE * alphas;
    checkCudaErrors(cudaMalloc((void**)&alphas,sizeof(KC_FP_TYPE)*3));   
    checkCudaErrors(cudaMemcpy(alphas,(KC_FP_TYPE*)mxGetPr(prhs[6]),sizeof(KC_FP_TYPE)*3,cudaMemcpyHostToDevice));

    // load filter weights
    int NH  = mxGetNumberOfElements(prhs[7]);
    KC_FP_TYPE * h_filt;
    checkCudaErrors(cudaMalloc((void**)&h_filt,sizeof(KC_FP_TYPE)*NH));   
    checkCudaErrors(cudaMemcpy(h_filt,(KC_FP_TYPE*)mxGetPr(prhs[7]),sizeof(KC_FP_TYPE)*NH,cudaMemcpyHostToDevice));

    // load derivative of log prior
    KC_FP_TYPE * l_prior;
    checkCudaErrors(cudaMalloc((void**)&l_prior,sizeof(KC_FP_TYPE)*(NH+3)));   
    checkCudaErrors(cudaMemcpy(l_prior,(KC_FP_TYPE*)mxGetPr(prhs[8]),sizeof(KC_FP_TYPE)*(NH+3),cudaMemcpyHostToDevice));

    // load fisher information of log prior
    KC_FP_TYPE * g_prior;
    checkCudaErrors(cudaMalloc((void**)&g_prior,sizeof(KC_FP_TYPE)*(NH+3)*(NH+3)));   
    checkCudaErrors(cudaMemcpy(g_prior,(KC_FP_TYPE*)mxGetPr(prhs[9]),sizeof(KC_FP_TYPE)*(NH+3)*(NH+3),cudaMemcpyHostToDevice));

    //loads spike history before trials
    KC_FP_TYPE * y_hist = kcGetArrayData(prhs[10],NH*NT);
    
    //sets up space for computations on GPU
    KC_FP_TYPE * der_log_p_y;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y,sizeof(KC_FP_TYPE)*(NT)*(NH+3)));    
    KC_FP_TYPE * der_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y_sum,sizeof(KC_FP_TYPE)*(NH+3)*1));    

    KC_FP_TYPE * log_p_y;
    checkCudaErrors(cudaMalloc((void**)&log_p_y,sizeof(KC_FP_TYPE)*NT));    
    KC_FP_TYPE * log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&log_p_y_sum,sizeof(KC_FP_TYPE)*1));    

    KC_FP_TYPE * G_log_p_y1;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y1,sizeof(KC_FP_TYPE)*(NT)*(NH+3)*(NH+3)));    
    KC_FP_TYPE * G_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y_sum,sizeof(KC_FP_TYPE)*(NH+3)*(NH+3)));    

    //sets up CUDA variables
    int blockSize = 2;
    int numBlocks = (int)NT/(int)blockSize + ((NT%blockSize==0)?0:1);

    //gets each trials likelihood + derivatives of filter
    kcBoundaryLikelihoodTrialHist<<< numBlocks,blockSize >>>(y,spe,trIdx,dt,NT,log_p_y,der_log_p_y,G_log_p_y1,alphas,h_filt,y_hist,NH,zs,ss);
    checkCudaErrors(cudaDeviceSynchronize());
    
    //sums up all the trials' likelihoods and derivatives with respect to alpha and spike history filters
    int nBlocksC = 3;
    int blockSizeC = 1;
    kcSumLangevinVars <<< nBlocksC,blockSizeC >>> (der_log_p_y, der_log_p_y_sum, G_log_p_y1, G_log_p_y_sum, log_p_y, log_p_y_sum,  trIdx, NT, NH, g_prior, l_prior);
    checkCudaErrors(cudaDeviceSynchronize());

    
    //pushes answers back to MATLAB
    if(nlhs > 0) {
        plhs[0] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[0]),log_p_y_sum,sizeof(KC_FP_TYPE)*1,cudaMemcpyDeviceToHost));
    }
    if(nlhs > 1) {
        plhs[1] = mxCreateNumericMatrix(NH+3,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[1]),der_log_p_y_sum,sizeof(KC_FP_TYPE)*(NH+3)*(1),cudaMemcpyDeviceToHost));
    }
    if(nlhs > 2) {
        plhs[2] = mxCreateNumericMatrix(NH+3,NH+3,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[2]),G_log_p_y_sum,sizeof(KC_FP_TYPE)*(NH+3)*(NH+3),cudaMemcpyDeviceToHost));
    }
    
    //clears up GPU variables
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(log_p_y));
    checkCudaErrors(cudaFree(log_p_y_sum));
    checkCudaErrors(cudaFree(der_log_p_y));
    checkCudaErrors(cudaFree(der_log_p_y_sum));
    checkCudaErrors(cudaFree(G_log_p_y1));
    checkCudaErrors(cudaFree(G_log_p_y_sum));
    checkCudaErrors(cudaFree(h_filt));
    checkCudaErrors(cudaFree(g_prior));
    checkCudaErrors(cudaFree(l_prior));
    checkCudaErrors(cudaFree(zs));
    checkCudaErrors(cudaFree(ss));
    checkCudaErrors(cudaFree(alphas));

    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error at the end of kcLangevinStep.cu ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

}
