#include "ge.cuh"
#include "precomp_data.cuh"
#include <cstdio>


/*
r = p + q
*/

__device__ void ge_add(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q) {
    fe t0;
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->YplusX);
    fe_mul(r->Y, r->Y, q->YminusX);
    fe_mul(r->T, q->T2d, p->T);
    fe_mul(r->X, p->Z, q->Z);
    fe_add(t0, r->X, r->X);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_add(r->Z, t0, r->T);
    fe_sub(r->T, t0, r->T);
}


__device__ void slide(signed char *r, const unsigned char *a) {
    int i;
    int b;
    int k;

    for (i = 0; i < 256; ++i) {
        r[i] = 1 & (a[i >> 3] >> (i & 7));
    }

    for (i = 0; i < 256; ++i)
        if (r[i]) {
            for (b = 1; b <= 6 && i + b < 256; ++b) {
                if (r[i + b]) {
                    if (r[i] + (r[i + b] << b) <= 15) {
                        r[i] += r[i + b] << b;
                        r[i + b] = 0;
                    } else if (r[i] - (r[i + b] << b) >= -15) {
                        r[i] -= r[i + b] << b;

                        for (k = i + b; k < 256; ++k) {
                            if (!r[k]) {
                                r[k] = 1;
                                break;
                            }

                            r[k] = 0;
                        }
                    } else {
                        break;
                    }
                }
            }
        }
}

/*
r = a * A + b * B
where a = a[0]+256*a[1]+...+256^31 a[31].
and b = b[0]+256*b[1]+...+256^31 b[31].
B is the Ed25519 base point (x,4/5) with x positive.
*/

__device__ void ge_double_scalarmult_vartime(ge_p2 *r, const unsigned char *a, const ge_p3 *A, const unsigned char *b) {
    signed char aslide[256];
    signed char bslide[256];
    ge_cached Ai[8]; /* A,3A,5A,7A,9A,11A,13A,15A */
    ge_p1p1 t;
    ge_p3 u;
    ge_p3 A2;
    int i;
    slide(aslide, a);
    slide(bslide, b);
    ge_p3_to_cached(&Ai[0], A);
    ge_p3_dbl(&t, A);
    ge_p1p1_to_p3(&A2, &t);
    ge_add(&t, &A2, &Ai[0]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[1], &u);
    ge_add(&t, &A2, &Ai[1]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[2], &u);
    ge_add(&t, &A2, &Ai[2]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[3], &u);
    ge_add(&t, &A2, &Ai[3]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[4], &u);
    ge_add(&t, &A2, &Ai[4]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[5], &u);
    ge_add(&t, &A2, &Ai[5]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[6], &u);
    ge_add(&t, &A2, &Ai[6]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[7], &u);
    ge_p2_0(r);

    for (i = 255; i >= 0; --i) {
        if (aslide[i] || bslide[i]) {
            break;
        }
    }

    for (; i >= 0; --i) {
        ge_p2_dbl(&t, r);

        if (aslide[i] > 0) {
            ge_p1p1_to_p3(&u, &t);
            ge_add(&t, &u, &Ai[aslide[i] / 2]);
        } else if (aslide[i] < 0) {
            ge_p1p1_to_p3(&u, &t);
            ge_sub(&t, &u, &Ai[(-aslide[i]) / 2]);
        }

        if (bslide[i] > 0) {
            ge_p1p1_to_p3(&u, &t);
            ge_madd(&t, &u, &Bi[bslide[i] / 2]);
        } else if (bslide[i] < 0) {
            ge_p1p1_to_p3(&u, &t);
            ge_msub(&t, &u, &Bi[(-bslide[i]) / 2]);
        }

        ge_p1p1_to_p2(r, &t);
    }
}


__constant__ fe d = {
    -10913610, 13857413, -15372611, 6949391, 114729, -8787816, -6275908, -3247719, -18696448, -12055116
};

__constant__ fe sqrtm1 = {
    -32595792, -7943725, 9377950, 3500415, 12389472, -272473, -25146209, -2005654, 326686, 11406482
};

__device__ int ge_frombytes_negate_vartime(ge_p3 *h, const unsigned char *s) {
    fe u;
    fe v;
    fe v3;
    fe vxx;
    fe check;
    fe_frombytes(h->Y, s);
    fe_1(h->Z);
    fe_sq(u, h->Y);
    fe_mul(v, u, d);
    fe_sub(u, u, h->Z);     /* u = y^2-1 */
    fe_add(v, v, h->Z);     /* v = dy^2+1 */
    fe_sq(v3, v);
    fe_mul(v3, v3, v);      /* v3 = v^3 */
    fe_sq(h->X, v3);
    fe_mul(h->X, h->X, v);
    fe_mul(h->X, h->X, u);  /* x = uv^7 */
    fe_pow22523(h->X, h->X); /* x = (uv^7)^((q-5)/8) */
    fe_mul(h->X, h->X, v3);
    fe_mul(h->X, h->X, u);  /* x = uv^3(uv^7)^((q-5)/8) */
    fe_sq(vxx, h->X);
    fe_mul(vxx, vxx, v);
    fe_sub(check, vxx, u);  /* vx^2-u */

    if (fe_isnonzero(check)) {
        fe_add(check, vxx, u); /* vx^2+u */

        if (fe_isnonzero(check)) {
            return -1;
        }

        fe_mul(h->X, h->X, sqrtm1);
    }

    if (fe_isnegative(h->X) == (s[31] >> 7)) {
        fe_neg(h->X, h->X);
    }

    fe_mul(h->T, h->X, h->Y);
    return 0;
}


/*
r = p + q
*/

__device__ void ge_madd(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q) {
    fe t0;
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->yplusx);
    fe_mul(r->Y, r->Y, q->yminusx);
    fe_mul(r->T, q->xy2d, p->T);
    fe_add(t0, p->Z, p->Z);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_add(r->Z, t0, r->T);
    fe_sub(r->T, t0, r->T);
}


/*
r = p - q
*/

__device__ void ge_msub(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q) {
    fe t0;

    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->yminusx);
    fe_mul(r->Y, r->Y, q->yplusx);
    fe_mul(r->T, q->xy2d, p->T);
    fe_add(t0, p->Z, p->Z);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_sub(r->Z, t0, r->T);
    fe_add(r->T, t0, r->T);
}


/*
r = p
*/

__device__ void ge_p1p1_to_p2(ge_p2 *r, const ge_p1p1 *p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
}



/*
r = p
*/

__device__ void ge_p1p1_to_p3(ge_p3 *r, const ge_p1p1 *p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
    fe_mul(r->T, p->X, p->Y);
}


__device__ void ge_p2_0(ge_p2 *h) {
    fe_0(h->X);
    fe_1(h->Y);
    fe_1(h->Z);
}



/*
r = 2 * p
*/

__device__ void ge_p2_dbl(ge_p1p1 *r, const ge_p2 *p) {
    fe t0;

    fe_sq(r->X, p->X);
    fe_sq(r->Z, p->Y);
    fe_sq2(r->T, p->Z);
    fe_add(r->Y, p->X, p->Y);
    fe_sq(t0, r->Y);
    fe_add(r->Y, r->Z, r->X);
    fe_sub(r->Z, r->Z, r->X);
    fe_sub(r->X, t0, r->Y);
    fe_sub(r->T, r->T, r->Z);
}


__device__ void ge_p3_0(ge_p3 *h) {
    fe_0(h->X);
    fe_1(h->Y);
    fe_1(h->Z);
    fe_0(h->T);
}


/*
r = 2 * p
*/

__device__ void ge_p3_dbl(ge_p1p1 *r, const ge_p3 *p) {
    ge_p2 q;
    ge_p3_to_p2(&q, p);
    ge_p2_dbl(r, &q);
}



/*
r = p
*/

__constant__ fe d2 = {
    -21827239, -5839606, -30745221, 13898782, 229458, 15978800, -12551817, -6495438, 29715968, 9444199
};

__device__ void ge_p3_to_cached(ge_cached *r, const ge_p3 *p) {
    fe_add(r->YplusX, p->Y, p->X);
    fe_sub(r->YminusX, p->Y, p->X);
    fe_copy(r->Z, p->Z);
    fe_mul(r->T2d, p->T, d2);
}


/*
r = p
*/

__device__ void ge_p3_to_p2(ge_p2 *r, const ge_p3 *p) {
    fe_copy(r->X, p->X);
    fe_copy(r->Y, p->Y);
    fe_copy(r->Z, p->Z);
}


__device__ void ge_p3_tobytes(unsigned char *s, const ge_p3 *h) {
    fe recip;
    fe x;
    fe y;
    fe_invert(recip, h->Z);
    fe_mul(x, h->X, recip);
    fe_mul(y, h->Y, recip);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}


__device__ unsigned char equal(signed char b, signed char c) {
    unsigned char ub = b;
    unsigned char uc = c;
    unsigned char x = ub ^ uc; /* 0: yes; 1..255: no */
    uint64_t y = x; /* 0: yes; 1..255: no */
    y -= 1; /* large: yes; 0..254: no */
    y >>= 63; /* 1: yes; 0: no */
    return (unsigned char) y;
}

__device__ unsigned char negative(signed char b) {
    uint64_t x = b; /* 18446744073709551361..18446744073709551615: yes; 0..255: no */
    x >>= 63; /* 1: yes; 0: no */
    return (unsigned char) x;
}

__device__ void cmov(ge_precomp *t, const ge_precomp *u, unsigned char b) {
    fe_cmov(t->yplusx, u->yplusx, b);
    fe_cmov(t->yminusx, u->yminusx, b);
    fe_cmov(t->xy2d, u->xy2d, b);
}


__device__ void select(ge_precomp *t, int pos, signed char b) {
    ge_precomp minust;
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);
    fe_1(t->yplusx);
    fe_1(t->yminusx);
    fe_0(t->xy2d);
    cmov(t, &base[pos][0], equal(babs, 1));
    cmov(t, &base[pos][1], equal(babs, 2));
    cmov(t, &base[pos][2], equal(babs, 3));
    cmov(t, &base[pos][3], equal(babs, 4));
    cmov(t, &base[pos][4], equal(babs, 5));
    cmov(t, &base[pos][5], equal(babs, 6));
    cmov(t, &base[pos][6], equal(babs, 7));
    cmov(t, &base[pos][7], equal(babs, 8));
    fe_copy(minust.yplusx, t->yminusx);
    fe_copy(minust.yminusx, t->yplusx);
    fe_neg(minust.xy2d, t->xy2d);
    cmov(t, &minust, bnegative);
}

/*
h = a * B
where a = a[0]+256*a[1]+...+256^31 a[31]
B is the Ed25519 base point (x,4/5) with x positive.

Preconditions:
  a[31] <= 127
*/

__device__ void ge_scalarmult_base(ge_p3 *h, const unsigned char *a) {
    signed char e[64];
    signed char carry;
    ge_p1p1 r;
    ge_p2 s;
    ge_precomp t;
    int i;

    for (i = 0; i < 32; ++i) {
        e[2 * i + 0] = (a[i] >> 0) & 15;
        e[2 * i + 1] = (a[i] >> 4) & 15;
    }

    /* each e[i] is between 0 and 15 */
    /* e[63] is between 0 and 7 */
    carry = 0;

    for (i = 0; i < 63; ++i) {
        e[i] += carry;
        carry = e[i] + 8;
        carry >>= 4;
        e[i] -= carry << 4;
    }

    e[63] += carry;
    /* each e[i] is between -8 and 8 */
    ge_p3_0(h);

    for (i = 1; i < 64; i += 2) {
        select(&t, i / 2, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }

    ge_p3_dbl(&r, h);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p3(h, &r);

    for (i = 0; i < 64; i += 2) {
        select(&t, i / 2, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}


/*
r = p - q
*/

__device__ void ge_sub(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q) {
    fe t0;
    
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->YminusX);
    fe_mul(r->Y, r->Y, q->YplusX);
    fe_mul(r->T, q->T2d, p->T);
    fe_mul(r->X, p->Z, q->Z);
    fe_add(t0, r->X, r->X);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_sub(r->Z, t0, r->T);
    fe_add(r->T, t0, r->T);
}


__device__ void ge_tobytes(unsigned char *s, const ge_p2 *h) {
    fe recip;
    fe x;
    fe y;
    fe_invert(recip, h->Z);
    fe_mul(x, h->X, recip);
    fe_mul(y, h->Y, recip);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}

/*
Negate a point in extended coordinates
For point (X:Y:Z:T), the negation is (-X:Y:Z:-T)

Input:
  p = point in extended coordinates

Output:
  r = -p in extended coordinates
*/

__device__ void ge_neg(ge_p3 *r, const ge_p3 *p) {
    fe_neg(r->X, p->X);  // X = -X
    fe_copy(r->Y, p->Y); // Y = Y  
    fe_copy(r->Z, p->Z); // Z = Z
    fe_neg(r->T, p->T);  // T = -T
}

/*
Check if a point in extended coordinates is the identity element
The identity in extended coordinates is (0:1:1:0)

Input:
  p = point in extended coordinates

Output:
  returns 1 if p is the identity, 0 otherwise
*/

__device__ int ge_is_identity(const ge_p3 *p) {
    // Check for empty bucket (cleared with Z=0): (0,1,0,0)  
    int z_is_zero = fe_iszero(p->Z);
    if (z_is_zero) {
        int x_is_zero = fe_iszero(p->X);
        int t_is_zero = fe_iszero(p->T);
        return x_is_zero && t_is_zero;  // Don't check Y for empty buckets
    }
    
    // Check canonical identity (0:1:1:0): X = 0, Y = Z, and T = 0
    int x_is_zero = fe_iszero(p->X);
    int t_is_zero = fe_iszero(p->T);
    
    // Check if Y = Z by computing Y - Z and checking if it's zero
    fe y_minus_z;
    fe_sub(y_minus_z, p->Y, p->Z);
    int y_equals_z = fe_iszero(y_minus_z);
    
    return x_is_zero && y_equals_z && t_is_zero;
}

/*
Set a point to the identity element in extended coordinates
The identity in extended coordinates is (0:1:1:0)

Output:
  r = identity point
*/

__device__ void ge_identity(ge_p3 *r) {
    fe_0(r->X);  // X = 0
    fe_1(r->Y);  // Y = 1
    fe_1(r->Z);  // Z = 1
    fe_0(r->T);  // T = 0
}

/*
Add two points in extended coordinates (p3 + p3 -> p3)
This is used for bucket accumulation in Pippenger MSM

Input:
  p, q = points in extended coordinates

Output:
  r = p + q in extended coordinates
*/

__device__ void ge_add_p3_p3(ge_p3 *r, const ge_p3 *p, const ge_p3 *q) {
    // Convert q to cached form for efficient addition
    ge_cached q_cached;
    ge_p3_to_cached(&q_cached, q);
    
    // Add using existing ge_add (p3 + cached -> p1p1)
    ge_p1p1 t;
    ge_add(&t, p, &q_cached);
    
    // Convert result back to p3
    ge_p1p1_to_p3(r, &t);
}

/*
Accumulate a point into a bucket (bucket += point)
Optimized for the common case where bucket might be identity

Input:
  bucket = current bucket sum (input/output, modified in place)
  point = point to add to bucket

Output:
  bucket is updated with bucket + point
*/

__device__ void ge_bucket_add(ge_p3 *bucket, const ge_p3 *point) {
    // Check if bucket is identity (common case for first addition to bucket)
    if (ge_is_identity(bucket)) {
        // If bucket is identity, just copy the point
        fe_copy(bucket->X, point->X);
        fe_copy(bucket->Y, point->Y);
        fe_copy(bucket->Z, point->Z);
        fe_copy(bucket->T, point->T);
    } else {
        // Otherwise do full addition
        ge_add_p3_p3(bucket, bucket, point);
    }
}

/*
Add a point to bucket with negation support for signed digits
Used in Pippenger window processing when digit can be negative

Input:
  bucket = current bucket sum (input/output, modified in place)
  point = point to add (or subtract if negate=1)
  negate = 1 to subtract point, 0 to add point

Output:
  bucket is updated with bucket ± point
*/

__device__ void ge_bucket_add_signed(ge_p3 *bucket, const ge_p3 *point, int negate) {
    if (negate) {
        // Create negated point and add it
        ge_p3 neg_point;
        ge_neg(&neg_point, point);
        ge_bucket_add(bucket, &neg_point);
    } else {
        // Regular addition
        ge_bucket_add(bucket, point);
    }
}

/*
Debug helper functions for systematic MSM testing
*/

/*
Print a point in extended coordinates for debugging
*/
__host__ void dump_p3(const char* tag, const ge_p3* p) {
    std::printf("%s: ge_p3{\n", tag);
    std::printf("  X: [");
    for (int i = 0; i < 10; i++) {
        std::printf("%08x", p->X[i]);
        if (i < 9) std::printf(" ");
    }
    std::printf("]\n");
    std::printf("  Y: [");
    for (int i = 0; i < 10; i++) {
        std::printf("%08x", p->Y[i]);
        if (i < 9) std::printf(" ");
    }
    std::printf("]\n");
    std::printf("  Z: [");
    for (int i = 0; i < 10; i++) {
        std::printf("%08x", p->Z[i]);
        if (i < 9) std::printf(" ");
    }
    std::printf("]\n");
    std::printf("  T: [");
    for (int i = 0; i < 10; i++) {
        std::printf("%08x", p->T[i]);
        if (i < 9) std::printf(" ");
    }
    std::printf("]\n}\n");
}

/*
Compare two points for equality (not constant-time, for testing only)
Returns 1 if equal, 0 if different
*/
__host__ int ge_equal(const ge_p3* a, const ge_p3* b) {
    // Compare all field elements component-wise
    for (int i = 0; i < 10; i++) {
        if (a->X[i] != b->X[i]) return 0;
        if (a->Y[i] != b->Y[i]) return 0;
        if (a->Z[i] != b->Z[i]) return 0;
        if (a->T[i] != b->T[i]) return 0;
    }
    return 1;
}

/*
Copy a point (host version)
*/
__host__ void ge_copy(ge_p3* dest, const ge_p3* src) {
    for (int i = 0; i < 10; i++) {
        dest->X[i] = src->X[i];
        dest->Y[i] = src->Y[i];
        dest->Z[i] = src->Z[i];
        dest->T[i] = src->T[i];
    }
}

/*
Set a point to the identity element in extended coordinates (host version)
The identity in extended coordinates is (0:1:1:0)
*/
__host__ void ge_identity_host(ge_p3 *r) {
    // X = 0 (all limbs zero)
    for (int i = 0; i < 10; i++) {
        r->X[i] = 0;
    }
    
    // Y = 1 (first limb 1, rest zero)
    r->Y[0] = 1;
    for (int i = 1; i < 10; i++) {
        r->Y[i] = 0;
    }
    
    // Z = 1 (first limb 1, rest zero)  
    r->Z[0] = 1;
    for (int i = 1; i < 10; i++) {
        r->Z[i] = 0;
    }
    
    // T = 0 (all limbs zero)
    for (int i = 0; i < 10; i++) {
        r->T[i] = 0;
    }
}

/*
Host field element operations for point negation
*/
__host__ void fe_copy_host(fe h, const fe f) {
    for (int i = 0; i < 10; i++) {
        h[i] = f[i];
    }
}

__host__ void fe_neg_host(fe h, const fe f) {
    for (int i = 0; i < 10; i++) {
        h[i] = -f[i];
    }
}

/*
Negate a point in extended coordinates (host version)
For point (X:Y:Z:T), the negation is (-X:Y:Z:-T)
*/
__host__ void ge_neg_host(ge_p3 *r, const ge_p3 *p) {
    fe_neg_host(r->X, p->X);  // X = -X
    fe_copy_host(r->Y, p->Y); // Y = Y  
    fe_copy_host(r->Z, p->Z); // Z = Z
    fe_neg_host(r->T, p->T);  // T = -T
}

// Device kernel wrapper for point encoding
__global__ void ge_p3_tobytes_kernel(unsigned char *s, const ge_p3 *h) {
    ge_p3_tobytes(s, h);
}

/*
Check if point is canonical identity using byte encoding (reliable for projective coordinates)
*/
__host__ bool ge_p3_equal_canonical_identity(const ge_p3* p) {
    unsigned char encoded[32];
    
    // Copy point to device and encode
    ge_p3 *d_p;
    unsigned char *d_encoded;
    
    cudaMalloc(&d_p, sizeof(ge_p3));
    cudaMalloc(&d_encoded, 32);
    
    cudaMemcpy(d_p, p, sizeof(ge_p3), cudaMemcpyHostToDevice);
    
    // Encode point on device
    ge_p3_tobytes_kernel<<<1, 1>>>(d_encoded, d_p);
    cudaDeviceSynchronize();
    
    // Copy encoding back
    cudaMemcpy(encoded, d_encoded, 32, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_p);
    cudaFree(d_encoded);
    
    // Check if encoded form is canonical identity: 0x01 followed by 31 zeros
    if (encoded[0] != 0x01) return false;
    
    for (int i = 1; i < 32; i++) {
        if (encoded[i] != 0) return false;
    }
    
    return true;
}

/*
Compare two points for equality using canonical encoding (host version)
This handles projective representation differences correctly
*/
__host__ bool ge_p3_equal_canonical(const ge_p3* A, const ge_p3* B) {
    unsigned char a_enc[32], b_enc[32];
    
    // Copy points to device
    ge_p3 *d_A, *d_B;
    unsigned char *d_a_enc, *d_b_enc;
    
    cudaMalloc(&d_A, sizeof(ge_p3));
    cudaMalloc(&d_B, sizeof(ge_p3));
    cudaMalloc(&d_a_enc, 32);
    cudaMalloc(&d_b_enc, 32);
    
    cudaMemcpy(d_A, A, sizeof(ge_p3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(ge_p3), cudaMemcpyHostToDevice);
    
    // Encode points on device
    ge_p3_tobytes_kernel<<<1, 1>>>(d_a_enc, d_A);
    ge_p3_tobytes_kernel<<<1, 1>>>(d_b_enc, d_B);
    cudaDeviceSynchronize();
    
    // Copy encodings back
    cudaMemcpy(a_enc, d_a_enc, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_enc, d_b_enc, 32, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_a_enc);
    cudaFree(d_b_enc);
    
    // Compare encodings
    return memcmp(a_enc, b_enc, 32) == 0;
}

// Device kernel wrapper for point addition
__global__ void ge_add_kernel(ge_p3* result, const ge_p3* a, const ge_p3* b) {
    ge_add_p3_p3(result, a, b);
}

/*
Add point to accumulator in place (host version) with identity fast-paths
acc = acc + p
*/
__host__ void ge_p3_add_inplace(ge_p3* acc, const ge_p3* p) {
    // Fast path: if acc is identity, just copy p
    if (ge_p3_equal_canonical_identity(acc)) {
        ge_copy(acc, p);
        return;
    }
    
    // Fast path: if p is identity, no change needed
    if (ge_p3_equal_canonical_identity(p)) {
        return;
    }
    
    // Use device function on host - copy to device, compute, copy back
    ge_p3 *d_acc, *d_p, *d_result;
    
    cudaMalloc(&d_acc, sizeof(ge_p3));
    cudaMalloc(&d_p, sizeof(ge_p3));  
    cudaMalloc(&d_result, sizeof(ge_p3));
    
    cudaMemcpy(d_acc, acc, sizeof(ge_p3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, sizeof(ge_p3), cudaMemcpyHostToDevice);
    
    // Launch single-thread kernel for addition
    ge_add_kernel<<<1, 1>>>(d_result, d_acc, d_p);
    cudaDeviceSynchronize();
    
    cudaMemcpy(acc, d_result, sizeof(ge_p3), cudaMemcpyDeviceToHost);
    
    cudaFree(d_acc);
    cudaFree(d_p);
    cudaFree(d_result);
}

/*
Convert small integer to 32-byte little-endian scalar
*/
__host__ void scalar_from_u64(unsigned char s[32], uint64_t k) {
    memset(s, 0, 32);
    for (int i = 0; i < 8; i++) {
        s[i] = (k >> (8 * i)) & 0xFF;
    }
}

// Device kernel wrappers
__global__ void ge_scalarmult_base_kernel(ge_p3* result, const unsigned char* scalar) {
    ge_scalarmult_base(result, scalar);
}

__global__ void ge_identity_kernel(ge_p3* result) {
    ge_identity(result);
}

/*
Reference scalar multiplication using existing working implementation  
Computes out = scalar * P using the individual verification path as reference
*/
__host__ void ge_scalar_mul_ref(ge_p3* out, const ge_p3* P, const unsigned char scalar[32]) {
    // For now, implement simple basepoint case only
    // Use device basepoint scalar multiplication
    ge_p3 *d_result;
    unsigned char *d_scalar;
    
    cudaMalloc(&d_result, sizeof(ge_p3));
    cudaMalloc(&d_scalar, 32);
    
    cudaMemcpy(d_scalar, scalar, 32, cudaMemcpyHostToDevice);
    
    // Launch single-thread kernel for basepoint scalar multiplication
    ge_scalarmult_base_kernel<<<1, 1>>>(d_result, d_scalar);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out, d_result, sizeof(ge_p3), cudaMemcpyDeviceToHost);
    
    cudaFree(d_result);
    cudaFree(d_scalar);
}

// Device kernel wrapper for ge_p3_tobytes
__global__ void ge_p3_tobytes_wrapper_kernel(unsigned char *s, const ge_p3 *h) {
    ge_p3_tobytes(s, h);
}

/*
Host function for encoding ge_p3 points to bytes using device kernels
Uses static reusable device buffers to eliminate allocation overhead
*/
extern "C" __host__ void ge_p3_tobytes_host(unsigned char *s, const ge_p3 *h) {
    // Use static reusable device buffers to eliminate micro-alloc/free thrash
    static unsigned char *d_s = nullptr;
    static ge_p3 *d_h = nullptr;
    
    // Allocate device buffers only once (reused for entire program)
    if (!d_s) {
        cudaError_t err = cudaMalloc(&d_s, 32);
        if (err != cudaSuccess) {
            fprintf(stderr, "FATAL: ge_p3_tobytes_host cudaMalloc failed: %s\n", cudaGetErrorString(err));
            abort();
        }
        err = cudaMalloc(&d_h, sizeof(ge_p3));
        if (err != cudaSuccess) {
            fprintf(stderr, "FATAL: ge_p3_tobytes_host cudaMalloc failed: %s\n", cudaGetErrorString(err));
            abort();
        }
    }
    
    cudaError_t err = cudaMemcpy(d_h, h, sizeof(ge_p3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL: ge_p3_tobytes_host cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    
    ge_p3_tobytes_wrapper_kernel<<<1, 1>>>(d_s, d_h);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL: ge_p3_tobytes_host kernel execution failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    
    err = cudaMemcpy(s, d_s, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL: ge_p3_tobytes_host cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    
    // No cudaFree - buffers reused throughout program (CUDA cleans up at exit)
}

/*
Host function for scalar multiplication: result = scalar * base_point
*/
extern "C" __host__ void host_scalar_multiply_256(ge_p3* result, const ge_p3* base_point, const unsigned char scalar[32]) {
    ge_identity_host(result);
    
    // Early exit for zero scalar
    bool is_zero = true;
    for (int i = 0; i < 32; i++) {
        if (scalar[i] != 0) {
            is_zero = false;
            break;
        }
    }
    if (is_zero) return;
    
    // Copy base point for doubling
    ge_p3 addend;
    ge_copy(&addend, base_point);
    
    // Process scalar bits from least significant to most significant
    for (int byte_idx = 0; byte_idx < 32; byte_idx++) {
        unsigned char byte = scalar[byte_idx];
        
        for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
            bool bit_is_set = byte & (1 << bit_idx);
            
            // If current bit is set, add the current addend to result
            if (bit_is_set) {
                ge_p3_add_inplace(result, &addend);
            }
            
            // Double the addend for next bit (2^(i+1))
            ge_p3_add_inplace(&addend, &addend);
        }
    }
}
