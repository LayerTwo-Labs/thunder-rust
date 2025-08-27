#ifndef GE_H
#define GE_H

#include "fe.cuh"


/*
ge means group element.

Here the group is the set of pairs (x,y) of field elements (see fe.h)
satisfying -x^2 + y^2 = 1 + d x^2y^2
where d = -121665/121666.

Representations:
  ge_p2 (projective): (X:Y:Z) satisfying x=X/Z, y=Y/Z
  ge_p3 (extended): (X:Y:Z:T) satisfying x=X/Z, y=Y/Z, XY=ZT
  ge_p1p1 (completed): ((X:Z),(Y:T)) satisfying x=X/Z, y=Y/T
  ge_precomp (Duif): (y+x,y-x,2dxy)
*/

typedef struct {
  fe X;
  fe Y;
  fe Z;
} ge_p2;

typedef struct {
  fe X;
  fe Y;
  fe Z;
  fe T;
} ge_p3;

typedef struct {
  fe X;
  fe Y;
  fe Z;
  fe T;
} ge_p1p1;

typedef struct {
  fe yplusx;
  fe yminusx;
  fe xy2d;
} ge_precomp;

typedef struct {
  fe YplusX;
  fe YminusX;
  fe Z;
  fe T2d;
} ge_cached;

__device__ void ge_p3_tobytes(unsigned char *s, const ge_p3 *h);
__device__ void ge_tobytes(unsigned char *s, const ge_p2 *h);
__device__ int ge_frombytes_negate_vartime(ge_p3 *h, const unsigned char *s);

__device__ void ge_add(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q);
__device__ void ge_sub(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q);
__device__ void ge_double_scalarmult_vartime(ge_p2 *r, const unsigned char *a, const ge_p3 *A, const unsigned char *b);
__device__ void ge_madd(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q);
__device__ void ge_msub(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q);
__device__ void ge_scalarmult_base(ge_p3 *h, const unsigned char *a);

__device__ void ge_p1p1_to_p2(ge_p2 *r, const ge_p1p1 *p);
__device__ void ge_p1p1_to_p3(ge_p3 *r, const ge_p1p1 *p);
__device__ void ge_p2_0(ge_p2 *h);
__device__ void ge_p2_dbl(ge_p1p1 *r, const ge_p2 *p);
__device__ void ge_p3_0(ge_p3 *h);
__device__ void ge_p3_dbl(ge_p1p1 *r, const ge_p3 *p);
__device__ void ge_p3_to_cached(ge_cached *r, const ge_p3 *p);
__device__ void ge_p3_to_p2(ge_p2 *r, const ge_p3 *p);
__device__ void ge_neg(ge_p3 *r, const ge_p3 *p);
__device__ int ge_is_identity(const ge_p3 *p);
__device__ void ge_identity(ge_p3 *r);

// Bucket accumulation functions for Pippenger MSM
__device__ void ge_add_p3_p3(ge_p3 *r, const ge_p3 *p, const ge_p3 *q);
__device__ void ge_bucket_add(ge_p3 *bucket, const ge_p3 *point);
__device__ void ge_bucket_add_signed(ge_p3 *bucket, const ge_p3 *point, int negate);

// Debug helper functions for systematic MSM testing
__host__ void dump_p3(const char* tag, const ge_p3* p);
__host__ int ge_equal(const ge_p3* a, const ge_p3* b);
__host__ bool ge_p3_equal_canonical(const ge_p3* A, const ge_p3* B);
__host__ void ge_copy(ge_p3* dest, const ge_p3* src);
__host__ void ge_identity_host(ge_p3 *r);
__host__ void ge_neg_host(ge_p3 *r, const ge_p3 *p);
__host__ void ge_p3_add_inplace(ge_p3* acc, const ge_p3* p);
__host__ void scalar_from_u64(unsigned char s[32], uint64_t k);
__host__ void ge_scalar_mul_ref(ge_p3* out, const ge_p3* P, const unsigned char scalar[32]);

// Host function for encoding points to bytes using device kernels
extern "C" __host__ void ge_p3_tobytes_host(unsigned char *s, const ge_p3 *h);

// Host function for scalar multiplication
extern "C" __host__ void host_scalar_multiply_256(ge_p3* result, const ge_p3* base_point, const unsigned char scalar[32]);

#endif
