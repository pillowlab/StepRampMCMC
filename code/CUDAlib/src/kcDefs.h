#ifndef   _KCUDA_CONSTS
#define   _KCUDA_CONSTS

#define KC_GPU_DEVICE  0

//for single or double precision

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
// #include < time.h >
// #include <windows.h> //I've ommited this line.
// #if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
//   #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
// #else
//   #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
// #endif
//  
// struct timezone 
// {
//   int  tz_minuteswest; /* minutes W of Greenwich */
//   int  tz_dsttime;     /* type of dst correction */
// };
//  
// int gettimeofday(struct timeval *tv, struct timezone *tz)
// {
//   FILETIME ft;
//   unsigned __int64 tmpres = 0;
//   static int tzflag;
//  
//   if (NULL != tv)
//   {
//     GetSystemTimeAsFileTime(&ft);
//  
//     tmpres |= ft.dwHighDateTime;
//     tmpres <<= 32;
//     tmpres |= ft.dwLowDateTime;
//  
//     /*converting file time to unix epoch*/
//     tmpres -= DELTA_EPOCH_IN_MICROSECS; 
//     tmpres /= 10;  /*convert into microseconds*/
//     tv->tv_sec = (long)(tmpres / 1000000UL);
//     tv->tv_usec = (long)(tmpres % 1000000UL);
//   }
//  
//   if (NULL != tz)
//   {
//     if (!tzflag)
//     {
//       _tzset();
//       tzflag++;
//     }
//     tz->tz_minuteswest = _timezone / 60;
//     tz->tz_dsttime = _daylight;
//   }
//  
//   return 0;
// }




#endif


#include <math.h>
#ifndef  M_PI
#define M_PI 3.14159265358979323846
#define M_SQRT2 1.41421356237309504880
#define M_SQRT_2 0.707106781186547524401
#define M_1_SQRTPI 0.564189583547756286948
#endif 




