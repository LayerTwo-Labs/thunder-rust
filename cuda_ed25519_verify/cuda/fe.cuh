#ifndef FE_H
#define FE_H

#include "fixedint.cuh"


/*
    fe means field element.
    Here the field is \Z/(2^255-19).
    An element t, entries t[0]...t[9], represents the integer
    t[0]+2^26 t[1]+2^51 t[2]+2^77 t[3]+2^102 t[4]+...+2^230 t[9].
    Bounds on each t[i] vary depending on context.
*/


typedef int32_t fe[10];


__device__ void fe_0(fe h);
__device__ void fe_1(fe h);

__device__ void fe_frombytes(fe h, const unsigned char *s);
__device__ void fe_tobytes(unsigned char *s, const fe h);

__device__ void fe_copy(fe h, const fe f);
__device__ int fe_isnegative(const fe f);
__device__ int fe_isnonzero(const fe f);
__device__ int fe_iszero(const fe f);
__device__ void fe_cmov(fe f, const fe g, unsigned int b);
__device__ void fe_cswap(fe f, fe g, unsigned int b);

__device__ void fe_neg(fe h, const fe f);
__device__ void fe_add(fe h, const fe f, const fe g);
__device__ void fe_invert(fe out, const fe z);
__device__ void fe_sq(fe h, const fe f);
__device__ void fe_sq2(fe h, const fe f);
__device__ void fe_mul(fe h, const fe f, const fe g);
__device__ void fe_mul121666(fe h, fe f);
__device__ void fe_pow22523(fe out, const fe z);
__device__ void fe_sub(fe h, const fe f, const fe g);

#endif
