
#include <math.h>

#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>


#include "mex.h"

#include "kcDefs.h" //see for info on anything starting with KC_
#include "kcArrayFunctions.h"


//this is just so only one small bit of data (6x6 matrix or something) needs to get pulled back to host
__global__ void kcVarStatsTrial(KC_FP_TYPE *w1, KC_FP_TYPE *w2, int * crossingTimes, int NT, int *mBlkIdx, KC_FP_TYPE * betas,int *betaIndVec,KC_FP_TYPE l_0, KC_FP_TYPE *lambdas) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {
        
        KC_FP_TYPE b = betas[betaIndVec[mBlkIdx[idx]]];
        int T = mBlkIdx[idx+1] - mBlkIdx[idx];
        int effLength = fmin(T*1.0,crossingTimes[idx]+1.0);//VALID?:   fmin(T*1.0,crossingTimes[idx]+1.0)
                                                          //or just   T
        w1[idx] = effLength/2.0; 
        w2[idx] = KC_POW(lambdas[mBlkIdx[idx]]-l_0,2.0);
        for(int ii = mBlkIdx[idx]+1;ii < mBlkIdx[idx]+effLength;ii++) {  
            //w2[idx] += KC_POW(lambdas[ii]-(lambdas[ii-1]+b),2.0);
            KC_FP_TYPE = lambdas[ii]-(lambdas[ii-1]+b);
            w2[idx] += dl*dl;
        }
        w2[idx] /= 2.0;
    }
}

//Function is used to calculate posterior parameters to Gibbs sample the diffusion 
//variance (\omega^2) in the ramping model given fixed latents (\lambda), \beta, and l_0.
//This function is mostly around to avoid pulling everything back to the host
//there are more efficient techniques to compute this, but this bit is so fast
//compared to the particle filter that optimization isn't necessary.
//
//args
//  0  = lambda (latent variables, on GPU. Same size as y)
//  1  = auxillary variable - threshold crossing time (latent variable boundary crossing time, on GPU. vector length number of trials: NT)
//  2  = trIdx (array that accesses the beta value used at each timepoint, y being indexed at 0. Includes final value that should be length of y)
//  3  = betaIdxVector (array that gives coherence used at each bins of y. i.e., accesses the beta value used at each timepoint. values begin at 0 instead of 1 to be consistent with C, unlike MATLAB)
//  4  = betas (the beta values)
//  5  = l_0 (initial diffusion value)
//
//outputs (left-hand side) Posterior parameter contribution to the inverse-gamma over \omega^2 for each trial
//  0  = posterior shape 
//  1  = posterior scale 
//
//   total posterior shape (usually called alpha in the PDF) is the sum of output 0 plus prior scale param
//   total posterior scale (usually called beta  in the PDF) is the sum of output 1 plus prior shape param

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {

    unsigned int TT = kcGetArrayNumEl(prhs[0]);
    KC_FP_TYPE * lambda = kcGetArrayData(prhs[0]);

    int * crossingTimes = kcGetArrayDataInt(prhs[1]);
    
    int * trIdx = kcGetArrayDataInt(prhs[2]);
    unsigned int NT = kcGetArrayNumEl(prhs[2])-1;

    int * betaIdxVector = kcGetArrayDataInt(prhs[3]);

    KC_FP_TYPE * b      = mxGetPr(prhs[4]);
    int   numBetas = mxGetNumberOfElements(prhs[4]);
    KC_FP_TYPE * b_gpu;
    checkCudaErrors(cudaMalloc((void**)&b_gpu,sizeof(KC_FP_TYPE)*numBetas));    
    checkCudaErrors(cudaMemcpy(b_gpu,b,sizeof(KC_FP_TYPE)*numBetas,cudaMemcpyHostToDevice));
    KC_FP_TYPE   l_0    = mxGetScalar(prhs[5]); 


    KC_FP_TYPE * w1;
    KC_FP_TYPE * w2;
    checkCudaErrors(cudaMalloc((void**)&w1,sizeof(KC_FP_TYPE)*NT));    
    checkCudaErrors(cudaMalloc((void**)&w2,sizeof(KC_FP_TYPE)*NT));    
    
    int blockSize = 2;
    int numBlocks = NT/blockSize + ((NT%blockSize==0)?0:1);
    kcVarStatsTrial<<< numBlocks,blockSize >>>(w1,w2,crossingTimes,NT,trIdx,b_gpu,betaIdxVector,l_0,lambda);

    if(nlhs > 1) {
        plhs[0] = mxCreateNumericMatrix(NT,1,KC_FP_TYPE_MATLAB,mxREAL);
        plhs[1] = mxCreateNumericMatrix(NT,1,KC_FP_TYPE_MATLAB,mxREAL);
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetData(plhs[0]),w1,sizeof(KC_FP_TYPE)*NT,cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetData(plhs[1]),w2,sizeof(KC_FP_TYPE)*NT,cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaFree(b_gpu));
    checkCudaErrors(cudaFree(w1));
    checkCudaErrors(cudaFree(w2));
}
