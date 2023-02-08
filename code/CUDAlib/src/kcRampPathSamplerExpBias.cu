
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
                    //KC_FP_TYPE can be assumed to mean "double", but originally
                    //this definition could also work with "float" for faster speed.
                    //float compatability is no longer supported in this function.
#include "kcArrayFunctions.h"

#define MAX_P 1e25
#define MIN_P 1e-25
__device__ KC_FP_TYPE positiveBound(KC_FP_TYPE a) {
    //return a;                                                                                                                                                                                                                                                                      
    if(isinf(a))
        return MAX_P;
    else
    return fmin(fmax(a,MIN_P),MAX_P);
}

__device__ KC_FP_TYPE h(KC_FP_TYPE z, KC_FP_TYPE gamma, KC_FP_TYPE dt, KC_FP_TYPE bias) {
    KC_FP_TYPE fr = KC_MAX(KC_MIN(KC_MAXN,exp(z*gamma)+bias),KC_MINN);
    return fr*dt;
}


//one thread per particle <<< nTrials,nParticles >>>
__global__ void kcMoveParticles(KC_FP_TYPE * y, KC_FP_TYPE * pos, KC_FP_TYPE * wt,  KC_FP_TYPE * b, int * betaIdxVector, KC_FP_TYPE l_0, KC_FP_TYPE g, KC_FP_TYPE w, KC_FP_TYPE dt, KC_FP_TYPE * randN, KC_FP_TYPE sigMult, KC_FP_TYPE * log_li, KC_FP_TYPE * lw, KC_FP_TYPE * lw2, KC_FP_TYPE * ncdf, KC_FP_TYPE * posc, int * trIdx, int NT, int TT, int numParticles, int t, KC_FP_TYPE bias) {
    int threadNum = blockIdx.x*blockDim.x + threadIdx.x;
    int tr_num = (int)threadNum / (int)numParticles;
    int p_num  = threadNum % numParticles;
    
    if(tr_num < NT) {
        int trLength = trIdx[tr_num+1] - trIdx[tr_num]; 
        
        if(t < trLength) {
            int row  = trIdx[tr_num] + t;
            int idx  = TT*p_num + row;
            int pidx = tr_num*numParticles+p_num;
            
            KC_FP_TYPE cb   = b[betaIdxVector[row]];
            KC_FP_TYPE sw   = sqrt(w);     
            
            KC_FP_TYPE mup = (t==0)?(l_0):(pos[idx-1]+cb);
            KC_FP_TYPE mu  =  mup; 
            
            KC_FP_TYPE sig2 = sigMult*w;
            KC_FP_TYPE sig  = sqrt(sig2);
            
            KC_FP_TYPE maxI = fmin(1.0-1e-20, fmax(  normcdf((1.0-mu)/sig),1e-20   ));
            pos[idx]    = fmin(1.0-1e-20, normcdfinv(maxI*randN[pidx])*sig + mu);
            posc[pidx]  = pos[idx];
            KC_FP_TYPE dpos = pos[idx]-mu;
            KC_FP_TYPE log_pi_k = -log(maxI)-0.5*log(2.0*M_PI*sig2) - 0.5/sig2*(dpos*dpos);
            
            //to be stored for each particle: ncdf, lw, lw2
            ncdf[idx]     = normcdf((1-mup)/sw); 
            
            KC_FP_TYPE dposp = pos[idx]-mup;
            KC_FP_TYPE log_p  = -0*log(maxI) -0.5*log(2*M_PI*w)- 0.5/w*(dposp*dposp);
            log_li[pidx]  = -h(pos[idx],g,dt,bias)+y[row]*(log(fmax(h(pos[idx],g,1.0,bias),1e-30))+log(dt))-lgamma(y[row]+1);
            
            KC_FP_TYPE pw = (t==0)?(log(1/(KC_FP_TYPE)numParticles) ):( log(fmax(wt[idx-1], 1e-30)) );
            lw[pidx]  = exp(pw+log_p+log_li[pidx]-log_pi_k);
            lw2[pidx] = exp(pw+log_p             -log_pi_k);

            //safety checks for numerical errors
            if(isnan(lw[pidx]) || isinf(lw[pidx]) || isnan(pos[idx]) || isinf(pos[idx]) || isnan(lw2[pidx]) || isinf(lw2[pidx])) {
                lw[pidx]  = 0;
                lw2[pidx] = 0;
                pos[idx]  = mup;
                posc[pidx] = mup;
            }
        } 
    }
}

//one thread per trial <<< nTrials,1 >>>
__global__ void kcNormalizeWeights(KC_FP_TYPE * y, KC_FP_TYPE * wt, KC_FP_TYPE * wt_p, KC_FP_TYPE * lw, KC_FP_TYPE * lw2, KC_FP_TYPE * nEff, KC_FP_TYPE * cumsum, int * trIdx, int NT, int TT, int numParticles, int t) {
    int tr_num = blockIdx.x*blockDim.x + threadIdx.x;

    if(tr_num < NT) {
        int trLength = trIdx[tr_num+1] - trIdx[tr_num]; 
        if(t < trLength) {
            int row = trIdx[tr_num] + t;
            
            //sum up and normalize weights
            KC_FP_TYPE weightSum  = 0; 
            KC_FP_TYPE weightSum2 = 0;
            for(int p_num = 0; p_num < numParticles; p_num++) {
                int pidx = tr_num*numParticles+p_num;
                weightSum  += lw[pidx];
                weightSum2 += lw2[pidx];
            }
            KC_FP_TYPE n_eff_den  = 0;
            weightSum  = fmax(weightSum,1e-20);
            weightSum2 = fmax(weightSum2,1e-20);
            for(int p_num = 0; p_num < numParticles; p_num++) {
                int idx      = TT*p_num + row;
                int pidx     = tr_num*numParticles+p_num;
                wt[idx]      = lw[pidx] /weightSum;
                wt_p[pidx]   = lw2[pidx]/weightSum2;
                n_eff_den   += wt[idx]*wt[idx];
                cumsum[pidx] = (p_num>0)?(cumsum[pidx-1]+wt[idx]):(wt[idx]);//for resampling
            }
            
            nEff[tr_num] = 1/n_eff_den;
        }
    }
}


//initial calculation - probability of each spike count coming from a rate at the bound
__global__ void kcSetupLG(KC_FP_TYPE * y,KC_FP_TYPE * lg,KC_FP_TYPE g, KC_FP_TYPE dt,int TT, KC_FP_TYPE bias) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < TT) {
        lg[idx] = exp( -h(1,g, dt,bias) + y[idx]*log(fmax(h(1,g,dt,bias),1e-30)) - lgamma(y[idx]+1));
    }
}

//one thread per particle <<< nTrials,nParticles >>>
// if particles look bad, resamples them from the distribution before the next step
__global__ void kcResampleParticles(KC_FP_TYPE * y, KC_FP_TYPE * pos, KC_FP_TYPE * posc, KC_FP_TYPE * wt, KC_FP_TYPE * log_li, KC_FP_TYPE * wt_p, int minEffParticles, KC_FP_TYPE * cumsum, KC_FP_TYPE * nEff, KC_FP_TYPE * randU, KC_FP_TYPE * p_cet_0, KC_FP_TYPE * p_cgt_0a, KC_FP_TYPE * p_cgt_0b, KC_FP_TYPE * ncdf, int * trIdx, int NT, int TT, int numParticles, int t) {
    int threadNum = blockIdx.x*blockDim.x + threadIdx.x;
    int tr_num = (int)threadNum / (int)numParticles;
    int p_num  = threadNum % numParticles;

    if(tr_num < NT) {

        int trLength = trIdx[tr_num+1] - trIdx[tr_num];
        if(t < trLength) {
            int pidx = tr_num*numParticles+p_num;
            int row = trIdx[tr_num] + t;
            int idx = TT*p_num + row;
            int pidx_new = pidx;
            if(nEff[tr_num] < minEffParticles) {
                int p_num_new;
                for(p_num_new = 0; p_num_new < numParticles-1 && randU[pidx] > cumsum[numParticles*tr_num+p_num_new]; p_num_new++) {
                    //everything taken care of in loop statement              
                }
                pidx_new = tr_num*numParticles+p_num_new;
                
                wt[idx]      = 1.0/(KC_FP_TYPE)numParticles; //weights are now uniform again
                
                pos[idx]     = posc[pidx_new];
                
            }
            KC_FP_TYPE wt_old     = (t==0)?(1.0/(KC_FP_TYPE)numParticles):(wt[idx-1]);

            p_cet_0[pidx]  = (1.0-ncdf[idx])*wt_old;
            p_cgt_0a[pidx] = exp(log_li[pidx])*wt_p[pidx]; //or pidx_new?
            p_cgt_0b[pidx] = ncdf[idx]*wt_old;
        }
    }
}

//one thread per trial <<< nTrials,1 >>>
//move bound crossing probabilities forward in time
__global__ void kcPropogateBoundaryDensity(KC_FP_TYPE * y, KC_FP_TYPE * p_clt, KC_FP_TYPE * p_cet, KC_FP_TYPE * p_cgt, KC_FP_TYPE * p_clte, KC_FP_TYPE * p_cpr, KC_FP_TYPE * p_cet_0, KC_FP_TYPE * p_cgt_0a, KC_FP_TYPE * p_cgt_0b, KC_FP_TYPE * lg, KC_FP_TYPE * nEff, int minEffParticles, KC_FP_TYPE * cumsum, int * trIdx, int NT, int TT, int numParticles, int t) {
    int tr_num = blockIdx.x*blockDim.x + threadIdx.x;   
    if(tr_num < NT) {
        int trLength = trIdx[tr_num+1] - trIdx[tr_num];
        if(t < trLength) {
            
            int row = trIdx[tr_num] + t;
            KC_FP_TYPE p_cet_s  = 0;
            KC_FP_TYPE p_cgt_sa = 0;
            KC_FP_TYPE p_cgt_sb = 0;
            for(int p_num = 0; p_num < numParticles; p_num++) {
                int pidx = tr_num*numParticles+p_num;
                //int idx = TT*p_num + row;
                p_cet_s  += p_cet_0[pidx];
                p_cgt_sa += p_cgt_0a[pidx];
                p_cgt_sb += p_cgt_0b[pidx];

                //finished a bit of the resampler that must run post-sampling for parallelization not to screw up, this will only be used again if this is last timestep in trial
                if(nEff[tr_num] < minEffParticles && t-1==trLength) {
                    cumsum[pidx] = 1/(KC_FP_TYPE)numParticles*(1+p_num);
                }
            }
            KC_FP_TYPE p_clte_old = ((t==0)?(0):(p_clte[row-1]));
            KC_FP_TYPE p_cgt_old = ((t==0)?(1):(p_cgt[row-1]));
            
            KC_FP_TYPE p_clt_1 = lg[row]*p_clte_old;
            KC_FP_TYPE p_cet_1 = lg[row]*(1.0-p_clte_old)*p_cet_s;
            KC_FP_TYPE p_cgt_1 = (1.0-p_clte_old)*p_cgt_sa*p_cgt_sb;

            p_cet[row]  = p_cet_1/(p_clt_1+p_cet_1+p_cgt_1);
            p_clte[row] = (p_cet_1+p_clt_1)/(p_clt_1+p_cet_1+p_cgt_1); //this is a little redudant, but I think it is convenient later?
            p_clt[row]  = p_clt_1/(p_clt_1+p_cet_1+p_cgt_1);
            p_cgt[row]  = p_cgt_1/(p_clt_1+p_cet_1+p_cgt_1);
            
            p_cpr[row] = p_cgt_old*p_cet_s; //compare this index in MATLAB code
        }
    }
}

//Finally do that backwards sampling, <<< NT, 1 >>>
__global__ void kcBackwardsSample(KC_FP_TYPE * sample, int * crossingTimes, KC_FP_TYPE * pos, KC_FP_TYPE * wt, KC_FP_TYPE * ncdf, KC_FP_TYPE * b, int * betaIdx, KC_FP_TYPE l_0, KC_FP_TYPE w, KC_FP_TYPE g, KC_FP_TYPE * p_cpr, KC_FP_TYPE * p_clte, KC_FP_TYPE * randUp, KC_FP_TYPE * randUb, KC_FP_TYPE * wt_p, KC_FP_TYPE * cumsum,  int * trIdx, int NT, int TT, int numParticles, int t) {
    int tr_num = blockIdx.x*blockDim.x + threadIdx.x;
    if(tr_num < NT) {
        int trLength = trIdx[tr_num+1] - trIdx[tr_num];
        int row = trIdx[tr_num] + t;
        if(t == trLength-1) {
            //if t=end of trial, start off the backwards sampling
            crossingTimes[tr_num] = trLength;

            //decide whether end trial has hit boundary
            if(randUb[tr_num] < p_clte[row]) {
                sample[row] = 1;
                crossingTimes[tr_num] = t;
            }
            //else select a particle to be end of trial (cumsum holds the CDF of the distribution over particles)
            else {
                int p_num;
                for(p_num = 0; p_num < numParticles-1 && randUp[tr_num] > cumsum[numParticles*tr_num+p_num]; p_num++) {
                }
                int idx = TT*p_num + row;
                sample[row] = pos[idx];

            }
        }
        else if(t < trLength-1 && t >= 0) {
            //else, propgate backwards
            
            //if previous sample had hit threshold
            if(sample[row+1] >= 1) {
                //if boundary already reached
                if(randUb[tr_num] < p_clte[row]/(p_cpr[row+1] + p_clte[row])) {
                    crossingTimes[tr_num] = t;
                    sample[row] = 1;
                }
                //gets pre-crossing particle
                else {
                    KC_FP_TYPE wtSum = 0;
                    int p_num;
                    for(p_num = 0; p_num < numParticles; p_num++) {
                        int idx = TT*p_num + row;
                        int pidx   = tr_num*numParticles+p_num;
                        wt_p[pidx] = wt[idx]*fmax(1.0-ncdf[idx+1],1e-25);
                        wtSum     += wt_p[pidx];
                    }
                    wtSum = fmax(wtSum,1e-30);
                    KC_FP_TYPE csum = wt_p[tr_num*numParticles+0]/wtSum;
                    for(p_num = 0; p_num < numParticles-1 && csum < randUp[tr_num]; p_num++) {
                        int pidx = tr_num*numParticles+p_num+1;
                        csum    += wt_p[pidx]/wtSum;
                    }
                    int idx = TT*p_num + row;
                    sample[row] = pos[idx];
                    
                }
                
            }
            //else, samples a particle
            else {
                KC_FP_TYPE wtSum = 0;
                int p_num;
                for(p_num = 0; p_num < numParticles; p_num++) {
                    int idx    = TT*p_num + row;
                    int pidx   = tr_num*numParticles+p_num;
                    wt_p[pidx] = wt[idx]*exp(-0.5/w*pow( sample[row+1] - (pos[idx] + b[betaIdx[row]]),2 ));
                    wtSum     += wt_p[pidx];
                }
                
                wtSum = fmax(wtSum,1e-30);
                KC_FP_TYPE csum = wt_p[tr_num*numParticles+0]/wtSum;
                for(p_num = 0; p_num < numParticles-1 && csum < randUp[tr_num]; p_num++) {
                    int pidx = tr_num*numParticles+p_num+1;
                    csum    += wt_p[pidx]/wtSum;
                }
                
                int idx = TT*p_num + row;
                sample[row] = pos[idx];
            }
        }
    }
}

/*
 Performs a forward sweep of the path after backwards sampling
 Draws from prior for steps post-threshold crossing (for conjugate sampling of parameters)
 Calculates som statistics for later sampling

trial number given by CUDA thread
 */
__global__ void kcForwardFinalPass( KC_FP_TYPE* lambda, const int * crossingTimes, const KC_FP_TYPE * randUni, const KC_FP_TYPE* b, const int * betaIndVec,const  KC_FP_TYPE l_0, const KC_FP_TYPE w, const int* trIdx,const  int NT, KC_FP_TYPE * beta_sum) {
    int tr_num = blockIdx.x*blockDim.x+threadIdx.x;
    if(tr_num < NT) {
        int t_0 = trIdx[tr_num];
        beta_sum[tr_num] = 0;
        int trLength = trIdx[tr_num+1] - trIdx[tr_num];
        
        KC_FP_TYPE cb   = b[betaIndVec[t_0]];
        
        for(int t = 0; t < trLength; t++) {
            if(t == crossingTimes[tr_num]) {
                //samples the first value of lambda to cross the bound (truncated normal, > 1)
                KC_FP_TYPE mu = (t > 0)?(lambda[t_0 + t-1]+cb):l_0;
                KC_FP_TYPE minS = normcdf((1-mu)/sqrt(w));
                if(minS >= 1.0-1e-5) {
                    lambda[t_0 + t] = 1;
                }
                else {
                    lambda[t_0 + t] = mu+sqrt(w)*normcdfinv( minS + (1-minS)*randUni[t_0+t]);
                }
            }
            else if(t > crossingTimes[tr_num]) {
                lambda[t_0 + t] = lambda[t_0 + t - 1] + cb + KC_SQRT(w)*normcdfinv( randUni[t_0+t]);
            }
            beta_sum[tr_num] += (t>0 && t <= crossingTimes[tr_num])?(lambda[t_0 + t] - lambda[t_0 + t-1]):0; //only include lambdas up until first threshold crossing to look at drift rates
        }
    }
}

//single thread kernel to assemble stats of the ramps across trials for sampling beta,l_0
__global__ void kcAssembleSamplingStatistics(KC_FP_TYPE * sigMat, KC_FP_TYPE * muVec, const KC_FP_TYPE* lambda, const int * crossingTimes, const KC_FP_TYPE * beta_sum,const int*betaIndVec,const  KC_FP_TYPE l_0, const KC_FP_TYPE w, const int* trIdx,  const  int NT, const int numBetas) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx == 0) {
        for(int trNum = 0; trNum < NT; trNum++) {
            int t_0 = trIdx[trNum];
            int cb  = betaIndVec[t_0];
            
            int trLength = trIdx[trNum+1] - trIdx[trNum];
            sigMat[(cb)*(numBetas+1) + cb] += fmin(1.0*crossingTimes[trNum],trLength-1.0)/w; 
            sigMat[(numBetas)*(numBetas+1) + numBetas] += 1.0/w;
            
            muVec[cb] += beta_sum[trNum]/w;
            muVec[numBetas] += lambda[t_0]/w;
        }
    }
}


//Samples a single set of latent paths from the ramping model for a set of trials given fixed parameters 
//args
//  0  = new lambda (output, should be pre-allocated on GPU, same size as y)
//  1  = new auxiliary variable for threshold crossing (output, should be pre-allocated on GPU, vector of length number of trials)
//  2  = y (observations)
//  3  = trIdx (array that accesses the beta value used at each timepoint, y being indexed at 0. Includes final value that should be length of y)
//  4  = betaIdxVector (array that gives coherence used at each bins of y. i.e., accesses the beta value used at each timepoint. values begin at 0 instead of 1 to be consistent with C, unlike MATLAB)
//  5  = betas (the beta values)
//  6  = w (variance of diffusion process)
//  7  = l_0 (starting lambda value)
//  8  = g (absorbing boundary effective height)
//  9  = dt (bin/timestep size)
//  10 = numParticles
//  11 = minEffParticles (how many effective particles per trial to keep around)
//  12 = sigMult (used for particle proposals, proposal variance is sigMult*w)
//  13 = maxTrialLength
//  14 = beta/l_0 sampling vec param c (uses this as output for sampling betas, l_0)
//  15 = beta/l_0 sampling vec param p uses this as output for sampling betas, l_0)
//  16 = bias

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    
    cudaError_t ce;


    curandStatus_t cre;
    /*ce = cudaSetDevice(KC_GPU_DEVICE);
    if(ce != cudaSuccess) {
        mexPrintf("Error initializing device (kcParticleFilterProp.cu) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
        }*/



    //init data                                                                                                                                                                                                                                                                      
    unsigned int TT = kcGetArrayNumEl(prhs[0]);

    KC_FP_TYPE * lambdaTarget = kcGetArrayData(prhs[0]);
    int * auxiliaryTarget = kcGetArrayDataInt(prhs[1]);

    KC_FP_TYPE * y      = kcGetArrayData(prhs[2],TT);


    int * trIdx = kcGetArrayDataInt(prhs[3]);

    unsigned int NT = kcGetArrayNumEl(prhs[3])-1;
    int * betaIdxVector = kcGetArrayDataInt(prhs[4]);

    

    KC_FP_TYPE * b      = mxGetPr(prhs[5]);
    int   numBetas = mxGetNumberOfElements(prhs[5]);

    KC_FP_TYPE * b_gpu;

    ce = cudaMalloc((void**)&b_gpu,sizeof(KC_FP_TYPE)*numBetas);
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating space for betas on GPU - first allocation in function (particle filter) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }
    ce = cudaMemcpy(b_gpu,b,sizeof(KC_FP_TYPE)*numBetas,cudaMemcpyHostToDevice);
    if(ce != cudaSuccess) {
        mexPrintf("Error moving betas to GPU (particle filter) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }

    KC_FP_TYPE  w      = mxGetScalar(prhs[6]);
    KC_FP_TYPE  l_0    = mxGetScalar(prhs[7]);
    KC_FP_TYPE  g      = mxGetScalar(prhs[8]);
    KC_FP_TYPE  dt     = mxGetScalar(prhs[9]);

    int  numParticles    = mxGetScalar(prhs[10]);
    int  minEffParticles = mxGetScalar(prhs[11]);
    int  sigMult         = mxGetScalar(prhs[12]);
    int  maxTrialLength  = mxGetScalar(prhs[13]);

    KC_FP_TYPE  bias     = mxGetScalar(prhs[16]);

    //particle weights/probabilities of hitting the bound
    KC_FP_TYPE * p_clte;
    KC_FP_TYPE * p_cet;
    KC_FP_TYPE * p_cgt;
    KC_FP_TYPE * p_clt;
    KC_FP_TYPE * p_cpr;
    checkCudaErrors(cudaMalloc((void**)&p_clte, TT*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&p_cet,  TT*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&p_cgt,  TT*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&p_clt,  TT*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&p_cpr,  TT*sizeof(KC_FP_TYPE)));



    KC_FP_TYPE * wt;
    KC_FP_TYPE * wt_p;
    KC_FP_TYPE * pos;//particle positions
    checkCudaErrors(cudaMalloc((void**)&wt,   (TT)*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&wt_p, (NT)*numParticles*sizeof(KC_FP_TYPE)));
    ce = cudaMalloc((void**)&pos,  (TT)*numParticles*sizeof(KC_FP_TYPE));
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating pos ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
   }



    KC_FP_TYPE * log_li;  
    KC_FP_TYPE * posc;    //for resampling
    KC_FP_TYPE * lw;  //unnormalized weights
    KC_FP_TYPE * lw2;
    KC_FP_TYPE * ncdf;
    KC_FP_TYPE * p_cet_0;
    KC_FP_TYPE * p_cgt_0a;
    KC_FP_TYPE * p_cgt_0b;
    KC_FP_TYPE * lg; //log p(y|at boundary)
    KC_FP_TYPE * cumsum;
    KC_FP_TYPE * beta_sum;
    checkCudaErrors(cudaMalloc((void**)&log_li,   NT*numParticles*sizeof(KC_FP_TYPE)));
    //checkCudaErrors(cudaMalloc((void**)&log_lic,  NT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&posc,     NT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&lw,       NT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&lw2,      NT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&ncdf,     TT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&p_cet_0,  NT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&p_cgt_0a, NT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&p_cgt_0b, NT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&cumsum,   NT*numParticles*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&beta_sum, NT*sizeof(KC_FP_TYPE)));
    checkCudaErrors(cudaMalloc((void**)&lg,       TT*sizeof(KC_FP_TYPE)));

    KC_FP_TYPE * nEff;
    checkCudaErrors(cudaMalloc((void**)&nEff,   NT*sizeof(KC_FP_TYPE)));

    int randSize  = (NT*numParticles) + ((NT*numParticles)%2==0?0:1);
    int randSizeS = (NT) + (NT%2==0?0:1);
    int randSizeT = (TT) + (TT%2==0?0:1);
    
    KC_FP_TYPE * randN;
    KC_FP_TYPE * randNs;
    KC_FP_TYPE * randTs;
    ce = cudaMalloc((void**)&randN,  randSize *sizeof(KC_FP_TYPE));
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating randN ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
    }
    ce = cudaMalloc((void**)&randNs, randSizeS*sizeof(KC_FP_TYPE));
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating randNs ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
    }
    ce = cudaMalloc((void**)&randTs, randSizeT*sizeof(KC_FP_TYPE));
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating randTs ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
    }
    
    //setup the random number generator
    curandGenerator_t curandGen = 0;
    curandStatus_t curandStatus;
    curandStatus = curandCreateGenerator(&curandGen,  CURAND_RNG_PSEUDO_DEFAULT);
    if(curandStatus != CURAND_STATUS_SUCCESS) {
        char buffer [50];
        sprintf(buffer, "Error initializing random number generator (%d).\n",(int)curandStatus);
        mexErrMsgTxt(buffer);
    }

    struct timeval now;
    gettimeofday(&now,NULL);
    unsigned long long mySeed = (unsigned long long)now.tv_usec+(unsigned long long)(1e7*(unsigned long long)now.tv_sec);
    curandStatus = curandSetPseudoRandomGeneratorSeed(curandGen, mySeed);
    //curandStatus = curandSetPseudoRandomGeneratorSeed(curandGen, (unsigned int)time(NULL));
    if(curandStatus != CURAND_STATUS_SUCCESS) {
        char buffer [50];
        sprintf(buffer, "Error random number seed (%d).\n",(int)curandStatus);
        mexErrMsgTxt(buffer);
    }
    curandStatus = curandGenerateSeeds(curandGen);
    if(curandStatus != CURAND_STATUS_SUCCESS) {
        char buffer [50];
        sprintf(buffer, "Error random number generating seed (%d).\n",(int)curandStatus);
        mexErrMsgTxt(buffer);
    }
    //cudaThreadSetLimit(cudaLimitStackSize, 1024);
    
    //setup initial particle positions
    int blockSize , nBlocks;
    int blockSizeT, nBlocksT;
    int blockSizeN, nBlocksN;

    blockSizeT = 4;
    nBlocksT   = TT/blockSizeT + ((TT%blockSizeT==0)?0:1);

    blockSizeN = 1;
    nBlocksN   = NT/blockSizeN + ((NT%blockSizeN==0)?0:1);

    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error before kcSetupLG ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }


    //__global__ void kcSetupLG(KC_FP_TYPE * y,KC_FP_TYPE * lg,KC_FP_TYPE g, KC_FP_TYPE dt,int TT,KC_FP_TYPE bias) {
    kcSetupLG <<< nBlocksT, blockSizeT >>> (y,lg,g,dt,TT,bias);
    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error after kcSetupLG<<<%d,%d>>> ",nBlocksT,blockSizeT);
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }

    blockSize = 8;
    int totalThreads = numParticles*NT;
    nBlocks   = totalThreads/blockSize + ((totalThreads%blockSize==0)?0:1);

    //mexPrintf("Max trial length = %d, blockSizes = %d,%d, nBlocks = %d,%d\n", maxTrialLength,blockSize,blockSizeN,nBlocks,nBlocksN);
    //forward pass loop
    for (int ii = 0; ii < maxTrialLength;ii++) {
        //move all particles foward
        cre = KC_RANDOM_UNIFORM_FUNCTION(curandGen,randN,randSize); //random sample steps for all particles
        ce = cudaDeviceSynchronize();
        if(ce != cudaSuccess) {
            int currDev;
            cudaGetDevice(&currDev);
            mexPrintf("Error synchronizing post-rand draw 1 Size=%d ii=%d,  current device=%d   ",randSize,ii,currDev);
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }
        if(cre != CURAND_STATUS_SUCCESS) {
            mexPrintf("Error after rand generation in particle propogation. Size=%d ii=%d ",randSize,ii);
            mexPrintf(" (%d)\n", (int)cre);
            mexErrMsgTxt("CUDA Errors");
        }
        

        kcMoveParticles <<< nBlocks, blockSize >>> (y,pos,wt, b_gpu,betaIdxVector,l_0,g,w,dt,randN, sigMult,log_li,lw,lw2,ncdf, posc, trIdx, NT, TT, numParticles, ii, bias);
        ce = cudaDeviceSynchronize();
        if(ce != cudaSuccess) {
            int currDev;
            cudaGetDevice(&currDev);
            mexPrintf("Error after kcMoveParticles<<<%d,%d>>> ii=%d/%d, dev=%d ",nBlocks,blockSize,ii,maxTrialLength,currDev);
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }

        //normalize weights
        kcNormalizeWeights <<< nBlocksN,blockSizeN >>> (y,wt,wt_p, lw, lw2, nEff, cumsum, trIdx, NT, TT, numParticles, ii);
        ce = cudaDeviceSynchronize();
        if(ce != cudaSuccess) {
            mexPrintf("Error after kcNormalizeWeights<<<%d,%d>>> ii=%d/%d ",nBlocksN,blockSizeN,ii,maxTrialLength);
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }

        //check effective num particles, resample when necessary
        cre = KC_RANDOM_UNIFORM_FUNCTION(curandGen,randN, randSize);
        if(cre != CURAND_STATUS_SUCCESS) {
            mexPrintf("Error after rand generation in resampler. ii=%d/%d ",ii,maxTrialLength);
            mexPrintf(" (%d)\n", (int)cre);
            mexErrMsgTxt("CUDA Errors");
        }

        kcResampleParticles <<< nBlocks, blockSize >>> (y,pos,posc,wt,log_li,wt_p, minEffParticles,cumsum,nEff,randN,p_cet_0,p_cgt_0a,p_cgt_0b,ncdf,trIdx, NT, TT, numParticles, ii); 
        ce = cudaDeviceSynchronize();
        if(ce != cudaSuccess) {
            mexPrintf("Error after kcResampleParticles<<<%d,%d>>> ii=%d/%d ",nBlocks,blockSize,ii,maxTrialLength);
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }

        //move passage density foward
        //__global__ void kcPropogateBoundaryDensity(KC_FP_TYPE * y, KC_FP_TYPE * p_clt, KC_FP_TYPE * p_cet, KC_FP_TYPE * p_cgt, KC_FP_TYPE * p_clte, KC_FP_TYPE * p_cpr, KC_FP_TYPE * p_cet_0, KC_FP_TYPE * p_cgt_0a, KC_FP_TYPE * p_cgt_0b, KC_FP_TYPE * lg, int * trIdx, KC_FP_TYPE * nEff, int minEffParticles, KC_FP_TYPE * cumsum, int t, int NT, int TT, int numParticles) {
        kcPropogateBoundaryDensity <<< nBlocksN,blockSizeN >>> (y,p_clt,p_cet,p_cgt,p_clte,p_cpr,p_cet_0,p_cgt_0a, p_cgt_0b, lg, nEff, minEffParticles, cumsum,trIdx, NT, TT, numParticles, ii);
        ce = cudaDeviceSynchronize();
        if(ce != cudaSuccess) {
            mexPrintf("Error after kcPropogateBoundaryDensity<<<%d,%d>>> ii=%d/%d ",nBlocksN,blockSizeN,ii,maxTrialLength);
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }

    }


    //backwards sample the particles
    for (int jj = maxTrialLength-1; jj >= 0; jj--) {

        cre = KC_RANDOM_UNIFORM_FUNCTION(curandGen,randN, randSizeS);
        if(cre != CURAND_STATUS_SUCCESS) {
            mexPrintf("Error after rand generation in backwards sampler (1). jj=%d/%d ",jj,maxTrialLength);
            mexPrintf(" (%d)\n", (int)cre);
            mexErrMsgTxt("CUDA Errors");
        }

        cre = KC_RANDOM_UNIFORM_FUNCTION(curandGen,randNs,randSizeS);
        //ce = cudaDeviceSynchronize();
        if(cre != CURAND_STATUS_SUCCESS) {
            mexPrintf("Error after rand generation in backwards sampler (2). jj=%d/%d ",jj,maxTrialLength);
            mexPrintf(" (%d)\n", (int)cre);
            mexErrMsgTxt("CUDA Errors");
        }
        ce = cudaDeviceSynchronize();
        if(ce != cudaSuccess) {
            mexPrintf("Error synchronizing before kcBackwardsSample (post random generation) jj=%d/%d ",jj,maxTrialLength);
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }

        kcBackwardsSample <<< nBlocksN,blockSizeN >>> (lambdaTarget, auxiliaryTarget, pos, wt, ncdf, b_gpu, betaIdxVector, l_0,  w, g, p_cpr, p_clte, randN, randNs, wt_p, cumsum, trIdx, NT, TT, numParticles, jj); 
        ce = cudaDeviceSynchronize();
        if(ce != cudaSuccess) {
            mexPrintf("Error after kcBackwardsSample<<<%d,%d>>> jj=%d/%d ",nBlocksN,blockSizeN,jj,maxTrialLength);
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }
    }
    
    cre = KC_RANDOM_UNIFORM_FUNCTION(curandGen,randTs, randSizeT);
    //ce = cudaDeviceSynchronize();
    if(cre != CURAND_STATUS_SUCCESS) {
        mexPrintf("Error after rand generation in final sampler (2).  ");
        mexPrintf(" (%d)\n", (int)cre);
        mexErrMsgTxt("CUDA Errors");
    }
    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error synchronizing before kcForwardFinalPass (post random generation) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }
    
    //samples all latent variables beyond bound hit time
    kcForwardFinalPass <<< nBlocksN,blockSizeN >>> (lambdaTarget, auxiliaryTarget,  randTs, b_gpu, betaIdxVector, l_0,  w, trIdx, NT, beta_sum); 
    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error after kcForwardFinalPass ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }
    
    //gets some statistics about the latent variables put together to be able to sample the drift rates
    KC_FP_TYPE * sampling_c;
    KC_FP_TYPE * sampling_p;
    checkCudaErrors(cudaMalloc((void**)&sampling_c, sizeof(KC_FP_TYPE)*(numBetas+1)));
    checkCudaErrors(cudaMalloc((void**)&sampling_p, sizeof(KC_FP_TYPE)*(numBetas+1)*(numBetas+1)));

    checkCudaErrors(cudaMemcpy(sampling_c,(KC_FP_TYPE*)mxGetPr(prhs[14]), sizeof(KC_FP_TYPE)*(numBetas+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(sampling_p,(KC_FP_TYPE*)mxGetPr(prhs[15]), sizeof(KC_FP_TYPE)*(numBetas+1)*(numBetas+1),cudaMemcpyHostToDevice));

    kcAssembleSamplingStatistics<<<1,1>>>(sampling_p, sampling_c, lambdaTarget, auxiliaryTarget, beta_sum,betaIdxVector,l_0,  w, trIdx,  NT, numBetas);

    
    checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(prhs[14]),sampling_c, sizeof(KC_FP_TYPE)*(numBetas+1),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(prhs[15]),sampling_p, sizeof(KC_FP_TYPE)*(numBetas+1)*(numBetas+1),cudaMemcpyDeviceToHost));


    //free up memory
    cre = curandDestroyGenerator(curandGen);
    if(cre != CURAND_STATUS_SUCCESS) {
        mexPrintf("Error destroying rand generator (%d)\n", (int)cre);
        mexErrMsgTxt("CUDA Errors");
    }
    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error synchronizing post-rand generator destruction (particleFilter) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }


    checkCudaErrors(cudaFree(b_gpu));
    checkCudaErrors(cudaFree(p_clte));
    checkCudaErrors(cudaFree(p_cet));
    checkCudaErrors(cudaFree(p_cgt));
    checkCudaErrors(cudaFree(p_clt));
    checkCudaErrors(cudaFree(p_cpr));
    checkCudaErrors(cudaFree(pos));

    checkCudaErrors(cudaFree(wt));
    ce = cudaFree(wt_p);
    if(ce != cudaSuccess) {
        mexPrintf("Error freeing memory in particle filter (wt_p) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }


    checkCudaErrors(cudaFree(log_li));
    checkCudaErrors(cudaFree(posc));
    checkCudaErrors(cudaFree(lw));
    checkCudaErrors(cudaFree(lw2));    
    checkCudaErrors(cudaFree(ncdf));
    checkCudaErrors(cudaFree(p_cet_0));
    checkCudaErrors(cudaFree(p_cgt_0a));
    checkCudaErrors(cudaFree(p_cgt_0b));
    checkCudaErrors(cudaFree(lg));
    checkCudaErrors(cudaFree(cumsum));
    checkCudaErrors(cudaFree(beta_sum));
    checkCudaErrors(cudaFree(sampling_c));
    checkCudaErrors(cudaFree(sampling_p));

    checkCudaErrors(cudaFree(nEff));

    checkCudaErrors(cudaFree(randN));
    checkCudaErrors(cudaFree(randNs));
    checkCudaErrors(cudaFree(randTs));
    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error at the end ofthe particle filter ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }
}
