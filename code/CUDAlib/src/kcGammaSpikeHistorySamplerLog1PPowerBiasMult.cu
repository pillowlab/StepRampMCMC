
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
        for(int idx = 0; idx < NH+2; idx++){
            der_sum[idx]=lPrior[idx];
        }
        for(int jj = 0; jj < NH+2; jj++){
            for(int idx = 0; idx<NT; idx++){
                der_sum[jj] += der[jj*NT+idx];
            }
        }
    }
    else if(nsum == 1) {
        for(int idx = 0; idx < NH+2; idx++) {
            for(int idx2 = 0; idx2 < NH+2; idx2++) {
                    G_sum[idx+idx2*(NH+2)] = 0;
                    G_sum[idx+idx2*(NH+2)] = gPrior[idx*(NH+2)+idx2];
            }
        }
        for(int jj = 0; jj < NH+2; jj++) {
      	    for(int kk = 0; kk < NH+2; kk++) {
                for(int idx =0; idx < NT; idx++) {
                    G_sum[jj*(NH+2)+kk] -= G[idx+(jj*(NH+2)+kk)*NT];
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
__device__ KC_FP_TYPE h(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt, KC_FP_TYPE sh, KC_FP_TYPE bias, KC_FP_TYPE log_power) {
    KC_FP_TYPE logex = KC_MAX(KC_MINN,(gamma*lambda>100)?(gamma*lambda):KC_MIN(log1p(exp(lambda*gamma)),KC_MAXN));
    return KC_MIN((KC_POW(logex*1.00000,log_power)+bias)*KC_EXP(sh)*dt,KC_MAXN);
}

__device__ KC_FP_TYPE dhg(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt, KC_FP_TYPE sh, KC_FP_TYPE yh, KC_FP_TYPE log_power) {
    KC_FP_TYPE logex = KC_MAX(KC_MINN,(gamma*lambda>100)?(gamma*lambda):KC_MIN(log1p(exp(lambda*gamma)),KC_MAXN));
    KC_FP_TYPE log_der = lambda/(1+KC_MIN(KC_MAXN,KC_MAX(exp(-lambda*gamma),KC_MINN)));
    KC_FP_TYPE der = log_power*KC_POW(logex*1.00000,log_power-1.00)*log_der;
    return der*dt*KC_EXP(sh);
}

__device__ KC_FP_TYPE dhs(KC_FP_TYPE lambda, KC_FP_TYPE gamma, KC_FP_TYPE dt, KC_FP_TYPE sh, KC_FP_TYPE yh, KC_FP_TYPE bias, KC_FP_TYPE log_power) {
    KC_FP_TYPE logex = KC_MAX(KC_MINN,(gamma*lambda>100)?(gamma*lambda):KC_MIN(log1p(exp(lambda*gamma)),KC_MAXN));
    KC_FP_TYPE fr = KC_MIN((KC_POW(logex*1.00000,log_power)+bias),KC_MAXN);
    return dt*yh*KC_EXP(sh)*fr;
}


// computes log p(single trial | gamma, fixed lambdas, spike history)
__global__ void kcBoundaryLikelihoodTrialHist(KC_FP_TYPE * y, KC_FP_TYPE * spe, KC_FP_TYPE * lambdas, int * crossingTimes, int * mBlkIdx, KC_FP_TYPE g, KC_FP_TYPE dt, int NT, KC_FP_TYPE * llSum, KC_FP_TYPE * trialSum, KC_FP_TYPE * trialSumRiemann, KC_FP_TYPE * h_filt, KC_FP_TYPE * y_hist, int NH, KC_FP_TYPE bias, KC_FP_TYPE log_power) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {

        for(int jj = 0; jj<NH+2; jj++){
            trialSum[idx+jj*NT]=0;
            for(int kk = 0; kk<NH+2; kk++){
                trialSumRiemann[idx+(jj*(NH+2)+kk)*NT]=0;
            }
        }
        llSum[idx] = 0;        

        for(int ii = mBlkIdx[idx]; ii < mBlkIdx[idx+1]; ii++)  {

            KC_FP_TYPE trueLambda = fmin(1, ((ii-mBlkIdx[idx]) < crossingTimes[idx])?lambdas[ii]:1);
            //KC_FP_TYPE trueLambda = fmin(1, lambdas[ii]);
            
            KC_FP_TYPE sh    = spe[ii];

            KC_FP_TYPE r    = h(trueLambda,g,1,sh,bias,log_power);
            llSum[idx] += y[ii]*(KC_LOG(r)+KC_LOG(dt)) - dt*r - KC_GAMMALN(y[ii]+1.0);

	    for(int jj = 0; jj < NH+2 ; jj++) {
                        
            KC_FP_TYPE yh1 = 0;
            KC_FP_TYPE dr = 0;
            // if index is one of the first NH indices of y, the spike history depends on spikes in the time before the analyzed spike train y
            // in that case, we want the ii - jj spike of the y history
            if(jj < NH && ii<(mBlkIdx[idx]+jj+1)) {
                yh1 = y_hist[NH*(idx+1) + (ii-mBlkIdx[idx]) - jj-1];
                dr = dhs(trueLambda,g,1,sh,yh1,bias,log_power);
            }
            else if(jj < NH) {
                yh1 = y[ii-jj-1];
                dr = dhs(trueLambda,g,1,sh,yh1,bias,log_power);
            }
            else if(jj == NH) {
                yh1 = trueLambda;
                dr = dhg(trueLambda,g,1,sh,yh1,log_power);
            }
            else if(jj == NH+1) {
                dr = KC_EXP(sh);
            }

            trialSum[idx+jj*NT] += (y[ii]/r-dt)*dr;

                //for(int kk = jj+1; kk < NH+2; kk++) 
                for(int kk = 0; kk < NH+2; kk++) {

                    KC_FP_TYPE yh2 = 0;
                    KC_FP_TYPE dr2 = 0;
            		// if index is one of the first NH indices of y, the spike history depends on spikes in the time before the analyzed spike train y
            		// in that case, we want the ii - jj spike of the y history
               	    if(kk < NH && ii<(mBlkIdx[idx]+kk+1)) {
                        yh2 = y_hist[NH*(idx+1) + (ii-mBlkIdx[idx]) - kk -1];
                        dr2 = dhs(trueLambda,g,1,sh,yh2,bias,log_power);
            		}
                    else if(kk < NH) {
               	  		yh2 = y[ii-kk-1];
                        dr2 = dhs(trueLambda,g,1,sh,yh2,bias,log_power);
            		}
				    else if(kk == NH) {
	   	       		   	yh2 = trueLambda;
                        dr2 = dhg(trueLambda,g,1,sh,yh2,log_power);
		       		}
                    else if(kk == NH+1) {
                        dr2 = KC_EXP(sh);
                    }

            		trialSumRiemann[idx+(NH+2)*NT*jj+NT*kk] += -1*dt*dr*dr2/r;

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
//  5  = spe (spike history effect, TT x 1)
//  6  = dt (bin size in seconds)
//  7  = gPrior (Fisher information of log prior probability of filters and gamma)
//  8  = spike history filters
//  9  = spike history (spikes before start of trials, NH*NT x 1)
//  10 = lPrior (derivative of log prior probability of filters and gamma)
//  11 = bias
//  12 = power
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
    KC_FP_TYPE  dt     = mxGetScalar(prhs[6]);
    
    //loads gamma and latent variables
    KC_FP_TYPE  g      = mxGetScalar(prhs[4]);
    KC_FP_TYPE * lambda = kcGetArrayData(prhs[0]);
    
    //loads spike history effect
    KC_FP_TYPE * spe      = kcGetArrayData(prhs[5],TT);

    int NH  = mxGetNumberOfElements(prhs[8]);
    KC_FP_TYPE  bias      = mxGetScalar(prhs[11]);
    KC_FP_TYPE  log_power      = mxGetScalar(prhs[12]);

    //loads Fisher information prior
    if(mxGetClassID(prhs[7]) != KC_FP_TYPE_MATLAB) {
        mexErrMsgTxt("Prior matrix input wrong floating point type (kcLangevinStep)!");
    }
    KC_FP_TYPE * gPrior;
    checkCudaErrors(cudaMalloc((void**)&gPrior,sizeof(KC_FP_TYPE)*(NH+2)*(NH+2)));
    checkCudaErrors(cudaMemcpy(gPrior,(KC_FP_TYPE*)mxGetPr(prhs[7]),sizeof(KC_FP_TYPE)*((NH+2)*(NH+2)),cudaMemcpyHostToDevice));


    //loads derivative of log prior probability of parameters
    if(mxGetClassID(prhs[10]) != KC_FP_TYPE_MATLAB) {
        mexErrMsgTxt("Prior matrix input wrong floating point type (kcLangevinStep)!");
    }
    KC_FP_TYPE * lPrior;
    checkCudaErrors(cudaMalloc((void**)&lPrior,sizeof(KC_FP_TYPE)*(NH+2)));
    checkCudaErrors(cudaMemcpy(lPrior,(KC_FP_TYPE*)mxGetPr(prhs[10]),sizeof(KC_FP_TYPE)*(NH+2),cudaMemcpyHostToDevice));

    
    //loads filter values
    KC_FP_TYPE * h_filt;
    checkCudaErrors(cudaMalloc((void**)&h_filt,sizeof(KC_FP_TYPE)*NH));   
    checkCudaErrors(cudaMemcpy(h_filt,(KC_FP_TYPE*)mxGetPr(prhs[8]),sizeof(KC_FP_TYPE)*NH,cudaMemcpyHostToDevice));


    //loads spike history before trials
    KC_FP_TYPE * y_hist = kcGetArrayData(prhs[9],NH*NT);
    
    //sets up space for computations on GPU
    KC_FP_TYPE * der_log_p_y;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y,sizeof(KC_FP_TYPE)*(NT)*(NH+2)));    
    KC_FP_TYPE * der_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&der_log_p_y_sum,sizeof(KC_FP_TYPE)*(NH+2)*1));    

    KC_FP_TYPE * log_p_y;
    checkCudaErrors(cudaMalloc((void**)&log_p_y,sizeof(KC_FP_TYPE)*NT));    
    KC_FP_TYPE * log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&log_p_y_sum,sizeof(KC_FP_TYPE)*1));    

    KC_FP_TYPE * G_log_p_y1;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y1,sizeof(KC_FP_TYPE)*(NT)*(NH+2)*(NH+2)));    
    KC_FP_TYPE * G_log_p_y_sum;
    checkCudaErrors(cudaMalloc((void**)&G_log_p_y_sum,sizeof(KC_FP_TYPE)*(NH+2)*(NH+2)));    

    //sets up CUDA variables
    int blockSize = 2;
    int numBlocks = (int)NT/(int)blockSize + ((NT%blockSize==0)?0:1);

    //gets each trials likelihood + derivatives of filter
    kcBoundaryLikelihoodTrialHist<<< numBlocks,blockSize >>>(y,spe,lambda,crossingTimes,trIdx,g,dt, NT,log_p_y,der_log_p_y,G_log_p_y1,h_filt,y_hist,NH,bias,log_power);
    checkCudaErrors(cudaDeviceSynchronize());
    
    //sums up all the trials' likelihoods and derivatives with respect to gamma
    int nBlocksC = 3;
    int blockSizeC = 1;
    kcSumLangevinVars <<< nBlocksC,blockSizeC >>> (der_log_p_y, der_log_p_y_sum, G_log_p_y1, G_log_p_y_sum, log_p_y, log_p_y_sum,  trIdx, NT, NH, gPrior, lPrior);
    checkCudaErrors(cudaDeviceSynchronize());

    
    //pushes answers back to MATLAB
    if(nlhs > 0) {
        plhs[0] = mxCreateNumericMatrix(1,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[0]),log_p_y_sum,sizeof(KC_FP_TYPE)*1,cudaMemcpyDeviceToHost));
    }
    if(nlhs > 1) {
        plhs[1] = mxCreateNumericMatrix(NH+2,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[1]),der_log_p_y_sum,sizeof(KC_FP_TYPE)*(NH+2)*(1),cudaMemcpyDeviceToHost));
    }
    if(nlhs > 2) {
        plhs[2] = mxCreateNumericMatrix(NH+2,NH+2,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[2]),G_log_p_y_sum,sizeof(KC_FP_TYPE)*(NH+2)*(NH+2),cudaMemcpyDeviceToHost));
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
