#ifndef   _KCUDA_CONSTS
#define   _KCUDA_CONSTS

#define KC_GPU_DEVICE  0

/*option for chosing to compile the CUDA functions as single or double precision.
* The code was initially intended to have an easy switch between double/single
* precision for speed, but this didn't percolate through and some functions are
* only double precision compatible.
*/
#define KC_DOUBLE_PRECISION 1


#if KC_DOUBLE_PRECISION >0
    #define KC_FP_TYPE double
    #define KC_FP_TYPE_MATLAB mxDOUBLE_CLASS

    #define KC_LOG log
    #define KC_EXP exp
    #define KC_GAMMALN lgamma
    #define KC_SQRT sqrt
    #define KC_MIN fmin
    #define KC_MAX fmax
    #define KC_NORMCDF normcdf
    #define KC_MINN 1e-30
    #define KC_MAXN 1e30
    #define KC_POW pow

    #define KC_RANDOM_UNIFORM_FUNCTION curandGenerateUniformDouble
    #define KC_RANDOM_NORMAL_FUNCTION curandGenerateNormalDouble
#else
    #define KC_FP_TYPE float
    #define KC_FP_TYPE_MATLAB mxSINGLE_CLASS

    #define KC_LOG logf
    #define KC_EXP expf
    #define KC_GAMMALN lgammaf
    #define KC_SQRT sqrtf
    #define KC_MIN fminf
    #define KC_MAX fmaxf
    #define KC_NORMCDF normcdff
    #define KC_MINN 1e-15
    #define KC_MAXN 1e15
    #define KC_POW powf

    #define KC_RANDOM_UNIFORM_FUNCTION curandGenerateUniform
    #define KC_RANDOM_NORMAL_FUNCTION curandGenerateNormal   
#endif

//For 32 or 64 bit compilation

#define KC_PTR_SIZE long long
#define KC_PTR_SIZE_MATLAB mxUINT64_CLASS



#include <cuda.h>

//Array types
#define KC_DOUBLE_ARRAY 1
#define KC_INT_ARRAY    2
#define KC_FLOAT_ARRAY  3

//Struct field names
#define KC_ARRAY_NUMEL "numel"
#define KC_ARRAY_NDIM  "ndim"
#define KC_ARRAY_SIZE  "size"
#define KC_ARRAY_PTR   "ptr"
#define KC_ARRAY_TYPE  "type"

#define KC_HANDLE_SPARSE_PTR   "sparseHandlePtr"
#define KC_HANDLE_RAND_PTR   "randHandlePtr"

#define KC_NULL_ARRAY   0

#include <sys/time.h>


#endif


#include <math.h>
#ifndef  M_PI
#define M_PI 3.14159265358979323846
#define M_SQRT2 1.41421356237309504880
#define M_SQRT_2 0.707106781186547524401
#define M_1_SQRTPI 0.564189583547756286948
#endif 




