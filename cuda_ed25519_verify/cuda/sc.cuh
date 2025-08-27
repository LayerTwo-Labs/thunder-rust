#ifndef SC_H
#define SC_H

/*
The set of scalars is \Z/l
where l = 2^252 + 27742317777372353535851937790883648493.
*/

__device__ void sc_reduce(unsigned char *s);
__device__ void sc_muladd(unsigned char *s, const unsigned char *a, const unsigned char *b, const unsigned char *c);
__device__ void sc_add(unsigned char *s, const unsigned char *a, const unsigned char *b);
__device__ void sc_sub(unsigned char *s, const unsigned char *a, const unsigned char *b);

// Batch verification scalar combination functions
__device__ void sc_from_128bit(unsigned char *s, const unsigned char *coeff);
__device__ void sc_mul_128bit_scalar(unsigned char *result, const unsigned char *coeff, const unsigned char *scalar);
__device__ void sc_accumulate_basepoint_scalar(unsigned char *accumulator, const unsigned char *coeff, const unsigned char *signature_scalar);
__device__ void sc_compute_pubkey_scalar(unsigned char *result, const unsigned char *coeff_reduced, const unsigned char *hash_scalar);
__device__ void sc_compute_r_scalar(unsigned char *result, const unsigned char *coeff_reduced);
__device__ __host__ void sc_neg(unsigned char *out, const unsigned char *in);

#endif
