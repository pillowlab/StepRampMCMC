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

/*
 * log_p_y
 *   log likelihood for a poisson
 *    y = observed count
 *    dt = length of observation
 */
//__device__ KC_FP_TYPE log_p_y( KC_FP_TYPE y,  KC_FP_TYPE rate,  KC_FP_TYPE dt, KC_FP_TYPE sh) {
//    return y*(KC_LOG(rate)+KC_LOG(dt)) - dt*rate;// - KC_GAMMALN(y+1)
//}
__device__ KC_FP_TYPE log_p_y(KC_FP_TYPE y, KC_FP_TYPE rate, KC_FP_TYPE dt, KC_FP_TYPE sh) {
    KC_FP_TYPE ex = rate*KC_MAX(KC_MIN(KC_EXP(sh),KC_MAXN),KC_MINN);
    return y*(KC_LOG(ex)+KC_LOG(dt)) - dt*ex - KC_GAMMALN(y+1.0);
}

/* kcSampleSMStates
 *  kernel runs on each trial (not timebin)
 *  outputs:
 *    z = jump times per each trial
 *    s = which state jumped to
 *    sampleStats = (3,2,NT) array, spike counts observed in each hidden state (divided up by trial)
 *  inputs
 *    y = spike counts
 *    trialIndex = index for y ( first spike count for trial i is y[trialIndex[i]] and the last spike count is y[trialIndex[i+1]-1]
 *                 y is indexed at 0. This array includes final value that should be length of y)
 *    trialCoh   = coherence level for each trial (coherence controls prior jump time distribution and jump to state probability)
 *                 coherence labels/indices begin at 0 instead of 1 to be consistent with C, unlike MATLAB
 *    NT         = number of trials
 *    alpha      = (3,1) array, spike rates
 *    phi        = (numCoherences,1) jump probabilities (p(s=3) = phi, p(s=2) = 1-phi), trial coherence dependent
 *    delta_t    = length of each timebin
 *    maxJump    = the longest to calculate out possible jump time values for
 *    randU      = (NT,1) array a set of uniform random numbers on [0,1]  
 *    
 *    nbPDF = (maxJump,numberOfCoherences) array, negative binomial pdf values (up to some limit) for each of the parameters of coherences 
 *
 *    jumpToProbs = (maxJump*NT,2) preallocated space to do calculations over
 */
__global__ void kcSampleSMStates(KC_FP_TYPE * z, KC_FP_TYPE * s, KC_FP_TYPE * sampleStats, KC_FP_TYPE * y, int * trialIndex, int * trialCoh, int NT, KC_FP_TYPE * alphas, KC_FP_TYPE * phi, KC_FP_TYPE delta_t, int maxJump, KC_FP_TYPE * randU, KC_FP_TYPE * nbPDF, KC_FP_TYPE * jumpToProbs, KC_FP_TYPE * spe, int TT, int numNeur) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < NT) {
        int T1 = trialIndex[idx];
        int T  = trialIndex[idx+1]-T1;
        
        //index in jumpToProbs for jumping to state 2
        int jumpT1_2 = idx*(maxJump*2);
        
        //index in jumpToProbs for jumping to state 3
        int jumpT1_3 = idx*(maxJump*2) + maxJump;
        
        int cohIndex = trialCoh[idx]*maxJump;
        
        KC_FP_TYPE p2 = (phi[trialCoh[idx]] < 1)?KC_LOG(1-phi[trialCoh[idx]]):0;
        KC_FP_TYPE p3 = KC_LOG(phi[trialCoh[idx]]);
        
        //calculate jump time probabilities for jump time happening within observed window (else model says jump happens after trial observations end)
        for(int ii = T-1; ii >= 0; ii--) {
            //taking a cumulative sum over p(y_{ii:end}|z=ii,s=2 or 3)

            //loop over neurons
            KC_FP_TYPE neuronSum1 = 0;
            KC_FP_TYPE neuronSum2 = 0;
            for(int nn = 0; nn < numNeur; nn++) {
                neuronSum1 += log_p_y(y[T1+ii+(TT*nn)],alphas[1+nn*3],delta_t,spe[T1+ii+(TT*nn)]);
                neuronSum2 += log_p_y(y[T1+ii+(TT*nn)],alphas[2+nn*3],delta_t,spe[T1+ii+(TT*nn)]);
            }
            jumpToProbs[jumpT1_2+ii] = ((ii < T-1)?(jumpToProbs[jumpT1_2+ii+1]):(0)) + neuronSum1;
            jumpToProbs[jumpT1_3+ii] = ((ii < T-1)?(jumpToProbs[jumpT1_3+ii+1]):(0)) + neuronSum2;
        }
        
        KC_FP_TYPE initStateCumsum = 0;
        
        KC_FP_TYPE maxLog = 0;
        for(int ii = 0; ii < maxJump; ii++) {
            // p (y_{1:t}|z==ii<=T), my comments are starting indexes at 1 while the code starts at 0
            if(ii < T) {

                KC_FP_TYPE p_y_init = 0;
                for(int nn = 0; nn < numNeur; nn++) {
                    p_y_init += log_p_y(y[T1+ii+(TT*nn)],alphas[0+nn*3],delta_t,spe[T1+ii+(TT*nn)]);
                }
                initStateCumsum += p_y_init;


                if(ii < T-1) {
                    jumpToProbs[jumpT1_2+ii+1] += initStateCumsum;
                    jumpToProbs[jumpT1_3+ii+1] += initStateCumsum;
                }
            }
            else {
                jumpToProbs[jumpT1_2+ii] = initStateCumsum;
                jumpToProbs[jumpT1_3+ii] = initStateCumsum;
            }
            
            
            jumpToProbs[jumpT1_2+ii] = jumpToProbs[jumpT1_2+ii] + nbPDF[cohIndex+ii] + p2;
            jumpToProbs[jumpT1_3+ii] = jumpToProbs[jumpT1_3+ii] + nbPDF[cohIndex+ii] + p3;
            
            
            maxLog = KC_MAX(KC_MAX(maxLog,jumpToProbs[jumpT1_2+ii]),jumpToProbs[jumpT1_3+ii]);
            //maxLog = jumpToProbs[jumpT1_2+ii]+jumpToProbs[jumpT1_3+ii];
        }
        //maxLog /= (maxJump*2.0);
        KC_FP_TYPE maxNumToExp = 8;
        KC_FP_TYPE minNumToExp = 2;
        KC_FP_TYPE extraConst = 0; //this helps numerical stability when going from log p to p (quick and dirty method)
        if(maxLog > maxNumToExp) {
            extraConst = maxLog-maxNumToExp;
        }
        else if(maxLog < minNumToExp) {
            extraConst = minNumToExp-maxLog;
        }
        
        KC_FP_TYPE totalProbCumsum = 0;
        for(int ii = 0; ii  < maxJump; ii++) {
            jumpToProbs[jumpT1_3+ii] = KC_EXP(jumpToProbs[jumpT1_3+ii] + extraConst);
            if(phi[trialCoh[idx]] < 1.0) {
                jumpToProbs[jumpT1_2+ii] = KC_EXP(jumpToProbs[jumpT1_2+ii] + extraConst);
                totalProbCumsum += jumpToProbs[jumpT1_3+ii] + jumpToProbs[jumpT1_2+ii];
            }
            else {
                totalProbCumsum += jumpToProbs[jumpT1_3+ii];
                jumpToProbs[jumpT1_2+ii] = 0.0;
            }
        }
        
        
        //goes back through and finds a sampling time + sample to state

        KC_FP_TYPE post_cdf = 0;
        int switchFound = -1;
        int switchTime  = 0;
        KC_FP_TYPE randn = randU[idx] * totalProbCumsum;
        for(int ii = 0; ii < maxJump && switchFound < 1; ii++) {
            post_cdf += jumpToProbs[jumpT1_2+ii];
            if(post_cdf > randn && phi[trialCoh[idx]] < 1) {
                switchFound = 2;
                switchTime  = ii;
            }
            else {
                post_cdf += jumpToProbs[jumpT1_3+ii];
                if(post_cdf > randn) {
                    switchFound = 3;
                    switchTime  = ii;
                }
            }
        }
        
        if(switchFound <= 0) {
            //just to make sure it doesn't crash
            switchFound = (KC_LOG(randU[idx])>p3)?2:3;
            switchTime  = 101;
        }
        
        s[idx] = switchFound;
        z[idx] = switchTime;
        
        //sum up observed spike count info
        sampleStats[idx*6]   = KC_MIN((KC_FP_TYPE)switchTime,(KC_FP_TYPE)T);
        sampleStats[idx*6+3] = 0;
        sampleStats[idx*6+4] = 0;
        sampleStats[idx*6+5] = 0;
        
        if(switchFound == 2) {
            sampleStats[idx*6+1] = ((KC_FP_TYPE)T)-sampleStats[idx*6] ;
            sampleStats[idx*6+2] = 0.0;
            for(int ii = 0; ii < T;ii++) {
                if(ii<switchTime) {
                    sampleStats[idx*6+3] += y[T1+ii];
                }
                else {
                    sampleStats[idx*6+4] += y[T1+ii];
                }
            }
        }
        else {
            sampleStats[idx*6+2] = ((KC_FP_TYPE)T)-sampleStats[idx*6] ;
            sampleStats[idx*6+1] = 0.0;
            for(int ii = 0; ii < T;ii++) {
                if(ii<switchTime) {
                    sampleStats[idx*6+3] += y[T1+ii];
                }
                else {
                    sampleStats[idx*6+5] += y[T1+ii];
                }
            }
        }
        
    }
}

/*
 * [SMSamples.z(:,ss) SMSamples.s(:,ss) SMSamples.spikeStats(:,:,ss)] = kcStepTimeSampler(gpu_y,gpu_trIndex,gpu_trCoh,SMSamples.alpha(:,ss-1),SMSamples.phi(:,ss-1),nbPDF,nbCDF,gpu_spe);
    
 *  Inputs:
 *    0 = y (spikes) - one long vector of all the spike times for all trials (GPU array)
 *    1 = trial index - 0:end-1 are the trial start times (GPU array)
 *    2 = trial coherence - on GPU, coherence levels per each trial (GPU array)
 *    3 = alpha, firing rates per each state (MATLAB array)
 *    4 = phi, probability of switiching to state 3 for each coherence (MATLAB array)
 *    5 = nbPDF, negative binomial pdf values (up to some limit) for each of the parameters of coherences  nbPDF(k,c) = P(z=k| p_c,r) (MATLAB array)
 *    6 = delta_t, length of each timebins
 *    7 = spe, spike history effect (GPU array)
 *    8 = numNeurons
 *
 *  Outputs (all in MATLAB array form)
 *    0 = z, switching times per each trial, size (NT,1)
 *    1 = s, which state was switched to per each trial (either 2 or 3), size (NT,1)
 *    2 = spikeStats, summary statistics on how many spikes were fired per each state of the semi-markov model and how many observations per state, size (3,2)
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    
    //load up the GPU array inputs
    unsigned int TY = kcGetArrayNumEl(prhs[0]); // TY is total length of y
    int  numNeur    = mxGetScalar(prhs[8]);
    unsigned int TT = (int)TY/(int)numNeur; // TT is total number of bins for one neuron 
    KC_FP_TYPE * y  = kcGetArrayData(prhs[0]);
    unsigned int NT = kcGetArrayNumEl(prhs[1])-1;
    int * trIndex   = kcGetArrayDataInt(prhs[1]);
    int * cohIndex  = kcGetArrayDataInt(prhs[2],NT);
    KC_FP_TYPE * spe = kcGetArrayData(prhs[7]);
    
    //put the precalculated negative binomial PDF, CDF values onto the GPU
    const mwSize * precalcSize = mxGetDimensions(prhs[5]);
    int maxJump = precalcSize[0];
    int NC = precalcSize[1];
    //mexPrintf("Sampling SM states. Max jump = %d, NC = %d, TT = %d, NT = %d\n",maxJump,NC,TT,NT);
    
    KC_FP_TYPE * nbPDF;
    checkCudaErrors(cudaMalloc((void**)&nbPDF,sizeof(KC_FP_TYPE)*NC*maxJump));   
    checkCudaErrors(cudaMemcpy(nbPDF,(KC_FP_TYPE*)mxGetPr(prhs[5]),sizeof(KC_FP_TYPE)*NC*maxJump,cudaMemcpyHostToDevice));
    
    KC_FP_TYPE  dt     = mxGetScalar(prhs[6]);
    
    //put model parameters onto the GPU
    
    KC_FP_TYPE * alphas;
    checkCudaErrors(cudaMalloc((void**)&alphas,sizeof(KC_FP_TYPE)*3*numNeur)); // number of alphas is 3 times numNeurons   
    checkCudaErrors(cudaMemcpy(alphas,(KC_FP_TYPE*)mxGetPr(prhs[3]),sizeof(KC_FP_TYPE)*3*numNeur,cudaMemcpyHostToDevice));
    
    KC_FP_TYPE * phi;
    checkCudaErrors(cudaMalloc((void**)&phi,sizeof(KC_FP_TYPE)*NC));   
    checkCudaErrors(cudaMemcpy(phi,(KC_FP_TYPE*)mxGetPr(prhs[4]),sizeof(KC_FP_TYPE)*NC,cudaMemcpyHostToDevice));
    
    
    
    //setup space on GPU for sampling
    //  z,s,sampleStats
    //   log_post2 - size(TT,1)
    //   log_post3 - size(TT,1)
    KC_FP_TYPE * log_post2;
    KC_FP_TYPE * log_post3;
    checkCudaErrors(cudaMalloc((void**)&log_post2,sizeof(KC_FP_TYPE)*TT));   
    checkCudaErrors(cudaMalloc((void**)&log_post3,sizeof(KC_FP_TYPE)*TT)); 
    
    KC_FP_TYPE * z;
    checkCudaErrors(cudaMalloc((void**)&z,sizeof(KC_FP_TYPE)*NT)); 
    KC_FP_TYPE * s;
    checkCudaErrors(cudaMalloc((void**)&s,sizeof(KC_FP_TYPE)*NT)); 
    KC_FP_TYPE * sampleStats;
    checkCudaErrors(cudaMalloc((void**)&sampleStats,sizeof(KC_FP_TYPE)*6*NT)); 
    
    
    KC_FP_TYPE * calculationSpace;
    checkCudaErrors(cudaMalloc((void**)&calculationSpace,sizeof(KC_FP_TYPE)*maxJump*NT*2)); 
    
    
    
    //setup random number generator
    curandGenerator_t curandGen = 0;
    curandStatus_t curandStatus;
    curandStatus = curandCreateGenerator(&curandGen,  CURAND_RNG_PSEUDO_DEFAULT);
    if(curandStatus != CURAND_STATUS_SUCCESS ) {
        mexPrintf("CURAND-1 error %d\n",(int)curandStatus);
        mexErrMsgTxt("CUDA errors sampling semi markov ");
    }
    struct timeval now;
    gettimeofday(&now,NULL);
    unsigned long long mySeed = (unsigned long long)now.tv_usec+(unsigned long long)(1e7*(unsigned long long)now.tv_sec);
    curandStatus = curandSetPseudoRandomGeneratorSeed(curandGen, mySeed);
    if(curandStatus != CURAND_STATUS_SUCCESS ) {
        mexPrintf("CURAND-2 error %d\n",(int)curandStatus);
        mexErrMsgTxt("CUDA errors sampling semi markov");
    }
    
    //generate a uniform random number set (size NT*2)
    KC_FP_TYPE * randU;
    int randSize = NT+((NT%2==0)?0:1);
    checkCudaErrors(cudaMalloc((void**)&randU,sizeof(KC_FP_TYPE)*randSize));   
    
    curandStatus = KC_RANDOM_UNIFORM_FUNCTION(curandGen,randU,randSize);
    cudaDeviceSynchronize();
    
    
    //sample the states
    kcSampleSMStates<<<NT,1>>>(z, s, sampleStats, y, trIndex, cohIndex, NT, alphas,  phi, dt, maxJump, randU, nbPDF, calculationSpace, spe, TT, numNeur);
 
    cudaDeviceSynchronize();
    
    //combine the sample stats
    KC_FP_TYPE * sampleStats_local;
    sampleStats_local = (KC_FP_TYPE*)malloc(sizeof(KC_FP_TYPE)*6*NT);
    checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)sampleStats_local,sampleStats,sizeof(KC_FP_TYPE)*6*NT,cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    plhs[2] = mxCreateNumericMatrix(3,2,KC_FP_TYPE_MATLAB,mxREAL);
    KC_FP_TYPE * sampleStats_sum = (KC_FP_TYPE*)mxGetPr(plhs[2]);
    for(int jj = 0; jj < 6; jj++) {
        sampleStats_sum[jj]  = 0;
        for(int ii = 0; ii < NT; ii++) {
            sampleStats_sum[jj] += sampleStats_local[ii*6 + jj];
        }
    }
    
    //move sampled values to MATLAB
    plhs[0] = mxCreateNumericMatrix(NT,1,KC_FP_TYPE_MATLAB,mxREAL);
    plhs[1] = mxCreateNumericMatrix(NT,1,KC_FP_TYPE_MATLAB,mxREAL);
    checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[0]),z,sizeof(KC_FP_TYPE)*NT,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((KC_FP_TYPE*)mxGetPr(plhs[1]),s,sizeof(KC_FP_TYPE)*NT,cudaMemcpyDeviceToHost));
    
    
    //clear out random number generator
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(curandDestroyGenerator(curandGen));
    
    
    //clear GPU values
    //  negative binomial distribution items
    checkCudaErrors(cudaFree(nbPDF));
    //  model params
    checkCudaErrors(cudaFree(alphas));
    checkCudaErrors(cudaFree(phi));
    //  sampler stuff
    checkCudaErrors(cudaFree(log_post2));
    checkCudaErrors(cudaFree(log_post3));
    checkCudaErrors(cudaFree(z));
    checkCudaErrors(cudaFree(s));
    checkCudaErrors(cudaFree(sampleStats));
    free(sampleStats_local);
    checkCudaErrors(cudaFree(calculationSpace));
    //  random nums
    checkCudaErrors(cudaFree(randU));
    
}
