#include "fixedint.cuh"
#include "sc.cuh"

__device__ uint64_t sc_load3(const unsigned char *in) {
    uint64_t result;

    result = (uint64_t) in[0];
    result |= ((uint64_t) in[1]) << 8;
    result |= ((uint64_t) in[2]) << 16;

    return result;
}

__device__ uint64_t sc_load4(const unsigned char *in) {
    uint64_t result;

    result = (uint64_t) in[0];
    result |= ((uint64_t) in[1]) << 8;
    result |= ((uint64_t) in[2]) << 16;
    result |= ((uint64_t) in[3]) << 24;
    
    return result;
}

/*
Input:
  s[0]+256*s[1]+...+256^63*s[63] = s

Output:
  s[0]+256*s[1]+...+256^31*s[31] = s mod l
  where l = 2^252 + 27742317777372353535851937790883648493.
  Overwrites s in place.
*/

__device__ void sc_reduce(unsigned char *s) {
    int64_t s0 = 2097151 & sc_load3(s);
    int64_t s1 = 2097151 & (sc_load4(s + 2) >> 5);
    int64_t s2 = 2097151 & (sc_load3(s + 5) >> 2);
    int64_t s3 = 2097151 & (sc_load4(s + 7) >> 7);
    int64_t s4 = 2097151 & (sc_load4(s + 10) >> 4);
    int64_t s5 = 2097151 & (sc_load3(s + 13) >> 1);
    int64_t s6 = 2097151 & (sc_load4(s + 15) >> 6);
    int64_t s7 = 2097151 & (sc_load3(s + 18) >> 3);
    int64_t s8 = 2097151 & sc_load3(s + 21);
    int64_t s9 = 2097151 & (sc_load4(s + 23) >> 5);
    int64_t s10 = 2097151 & (sc_load3(s + 26) >> 2);
    int64_t s11 = 2097151 & (sc_load4(s + 28) >> 7);
    int64_t s12 = 2097151 & (sc_load4(s + 31) >> 4);
    int64_t s13 = 2097151 & (sc_load3(s + 34) >> 1);
    int64_t s14 = 2097151 & (sc_load4(s + 36) >> 6);
    int64_t s15 = 2097151 & (sc_load3(s + 39) >> 3);
    int64_t s16 = 2097151 & sc_load3(s + 42);
    int64_t s17 = 2097151 & (sc_load4(s + 44) >> 5);
    int64_t s18 = 2097151 & (sc_load3(s + 47) >> 2);
    int64_t s19 = 2097151 & (sc_load4(s + 49) >> 7);
    int64_t s20 = 2097151 & (sc_load4(s + 52) >> 4);
    int64_t s21 = 2097151 & (sc_load3(s + 55) >> 1);
    int64_t s22 = 2097151 & (sc_load4(s + 57) >> 6);
    int64_t s23 = (sc_load4(s + 60) >> 3);
    int64_t carry0;
    int64_t carry1;
    int64_t carry2;
    int64_t carry3;
    int64_t carry4;
    int64_t carry5;
    int64_t carry6;
    int64_t carry7;
    int64_t carry8;
    int64_t carry9;
    int64_t carry10;
    int64_t carry11;
    int64_t carry12;
    int64_t carry13;
    int64_t carry14;
    int64_t carry15;
    int64_t carry16;

    s11 += s23 * 666643;
    s12 += s23 * 470296;
    s13 += s23 * 654183;
    s14 -= s23 * 997805;
    s15 += s23 * 136657;
    s16 -= s23 * 683901;
    s23 = 0;
    s10 += s22 * 666643;
    s11 += s22 * 470296;
    s12 += s22 * 654183;
    s13 -= s22 * 997805;
    s14 += s22 * 136657;
    s15 -= s22 * 683901;
    s22 = 0;
    s9 += s21 * 666643;
    s10 += s21 * 470296;
    s11 += s21 * 654183;
    s12 -= s21 * 997805;
    s13 += s21 * 136657;
    s14 -= s21 * 683901;
    s21 = 0;
    s8 += s20 * 666643;
    s9 += s20 * 470296;
    s10 += s20 * 654183;
    s11 -= s20 * 997805;
    s12 += s20 * 136657;
    s13 -= s20 * 683901;
    s20 = 0;
    s7 += s19 * 666643;
    s8 += s19 * 470296;
    s9 += s19 * 654183;
    s10 -= s19 * 997805;
    s11 += s19 * 136657;
    s12 -= s19 * 683901;
    s19 = 0;
    s6 += s18 * 666643;
    s7 += s18 * 470296;
    s8 += s18 * 654183;
    s9 -= s18 * 997805;
    s10 += s18 * 136657;
    s11 -= s18 * 683901;
    s18 = 0;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry12 = (s12 + (1 << 20)) >> 21;
    s13 += carry12;
    s12 -= carry12 << 21;
    carry14 = (s14 + (1 << 20)) >> 21;
    s15 += carry14;
    s14 -= carry14 << 21;
    carry16 = (s16 + (1 << 20)) >> 21;
    s17 += carry16;
    s16 -= carry16 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    carry13 = (s13 + (1 << 20)) >> 21;
    s14 += carry13;
    s13 -= carry13 << 21;
    carry15 = (s15 + (1 << 20)) >> 21;
    s16 += carry15;
    s15 -= carry15 << 21;
    s5 += s17 * 666643;
    s6 += s17 * 470296;
    s7 += s17 * 654183;
    s8 -= s17 * 997805;
    s9 += s17 * 136657;
    s10 -= s17 * 683901;
    s17 = 0;
    s4 += s16 * 666643;
    s5 += s16 * 470296;
    s6 += s16 * 654183;
    s7 -= s16 * 997805;
    s8 += s16 * 136657;
    s9 -= s16 * 683901;
    s16 = 0;
    s3 += s15 * 666643;
    s4 += s15 * 470296;
    s5 += s15 * 654183;
    s6 -= s15 * 997805;
    s7 += s15 * 136657;
    s8 -= s15 * 683901;
    s15 = 0;
    s2 += s14 * 666643;
    s3 += s14 * 470296;
    s4 += s14 * 654183;
    s5 -= s14 * 997805;
    s6 += s14 * 136657;
    s7 -= s14 * 683901;
    s14 = 0;
    s1 += s13 * 666643;
    s2 += s13 * 470296;
    s3 += s13 * 654183;
    s4 -= s13 * 997805;
    s5 += s13 * 136657;
    s6 -= s13 * 683901;
    s13 = 0;
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;
    carry0 = (s0 + (1 << 20)) >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry2 = (s2 + (1 << 20)) >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry4 = (s4 + (1 << 20)) >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry1 = (s1 + (1 << 20)) >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry3 = (s3 + (1 << 20)) >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry5 = (s5 + (1 << 20)) >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;
    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry11 = s11 >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;
    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;

    s[0] = (unsigned char) (s0 >> 0);
    s[1] = (unsigned char) (s0 >> 8);
    s[2] = (unsigned char) ((s0 >> 16) | (s1 << 5));
    s[3] = (unsigned char) (s1 >> 3);
    s[4] = (unsigned char) (s1 >> 11);
    s[5] = (unsigned char) ((s1 >> 19) | (s2 << 2));
    s[6] = (unsigned char) (s2 >> 6);
    s[7] = (unsigned char) ((s2 >> 14) | (s3 << 7));
    s[8] = (unsigned char) (s3 >> 1);
    s[9] = (unsigned char) (s3 >> 9);
    s[10] = (unsigned char) ((s3 >> 17) | (s4 << 4));
    s[11] = (unsigned char) (s4 >> 4);
    s[12] = (unsigned char) (s4 >> 12);
    s[13] = (unsigned char) ((s4 >> 20) | (s5 << 1));
    s[14] = (unsigned char) (s5 >> 7);
    s[15] = (unsigned char) ((s5 >> 15) | (s6 << 6));
    s[16] = (unsigned char) (s6 >> 2);
    s[17] = (unsigned char) (s6 >> 10);
    s[18] = (unsigned char) ((s6 >> 18) | (s7 << 3));
    s[19] = (unsigned char) (s7 >> 5);
    s[20] = (unsigned char) (s7 >> 13);
    s[21] = (unsigned char) (s8 >> 0);
    s[22] = (unsigned char) (s8 >> 8);
    s[23] = (unsigned char) ((s8 >> 16) | (s9 << 5));
    s[24] = (unsigned char) (s9 >> 3);
    s[25] = (unsigned char) (s9 >> 11);
    s[26] = (unsigned char) ((s9 >> 19) | (s10 << 2));
    s[27] = (unsigned char) (s10 >> 6);
    s[28] = (unsigned char) ((s10 >> 14) | (s11 << 7));
    s[29] = (unsigned char) (s11 >> 1);
    s[30] = (unsigned char) (s11 >> 9);
    s[31] = (unsigned char) (s11 >> 17);
}

/*
Input:
  a[0]+256*a[1]+...+256^31*a[31] = a mod l
  b[0]+256*b[1]+...+256^31*b[31] = b mod l

Output:
  s[0]+256*s[1]+...+256^31*s[31] = (a+b) mod l
  where l = 2^252 + 27742317777372353535851937790883648493.
*/

__device__ void sc_add(unsigned char *s, const unsigned char *a, const unsigned char *b) {
    int64_t a0 = 2097151 & sc_load3(a);
    int64_t a1 = 2097151 & (sc_load4(a + 2) >> 5);
    int64_t a2 = 2097151 & (sc_load3(a + 5) >> 2);
    int64_t a3 = 2097151 & (sc_load4(a + 7) >> 7);
    int64_t a4 = 2097151 & (sc_load4(a + 10) >> 4);
    int64_t a5 = 2097151 & (sc_load3(a + 13) >> 1);
    int64_t a6 = 2097151 & (sc_load4(a + 15) >> 6);
    int64_t a7 = 2097151 & (sc_load3(a + 18) >> 3);
    int64_t a8 = 2097151 & sc_load3(a + 21);
    int64_t a9 = 2097151 & (sc_load4(a + 23) >> 5);
    int64_t a10 = 2097151 & (sc_load3(a + 26) >> 2);
    int64_t a11 = (sc_load4(a + 28) >> 7);
    
    int64_t b0 = 2097151 & sc_load3(b);
    int64_t b1 = 2097151 & (sc_load4(b + 2) >> 5);
    int64_t b2 = 2097151 & (sc_load3(b + 5) >> 2);
    int64_t b3 = 2097151 & (sc_load4(b + 7) >> 7);
    int64_t b4 = 2097151 & (sc_load4(b + 10) >> 4);
    int64_t b5 = 2097151 & (sc_load3(b + 13) >> 1);
    int64_t b6 = 2097151 & (sc_load4(b + 15) >> 6);
    int64_t b7 = 2097151 & (sc_load3(b + 18) >> 3);
    int64_t b8 = 2097151 & sc_load3(b + 21);
    int64_t b9 = 2097151 & (sc_load4(b + 23) >> 5);
    int64_t b10 = 2097151 & (sc_load3(b + 26) >> 2);
    int64_t b11 = (sc_load4(b + 28) >> 7);

    int64_t s0 = a0 + b0;
    int64_t s1 = a1 + b1;
    int64_t s2 = a2 + b2;
    int64_t s3 = a3 + b3;
    int64_t s4 = a4 + b4;
    int64_t s5 = a5 + b5;
    int64_t s6 = a6 + b6;
    int64_t s7 = a7 + b7;
    int64_t s8 = a8 + b8;
    int64_t s9 = a9 + b9;
    int64_t s10 = a10 + b10;
    int64_t s11 = a11 + b11;
    int64_t s12 = 0;
    
    int64_t carry0, carry1, carry2, carry3, carry4, carry5, carry6, carry7, carry8, carry9, carry10, carry11;

    carry0 = (s0 + (1 << 20)) >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry2 = (s2 + (1 << 20)) >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry4 = (s4 + (1 << 20)) >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry1 = (s1 + (1 << 20)) >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry3 = (s3 + (1 << 20)) >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry5 = (s5 + (1 << 20)) >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    
    // Apply reduction with Barrett-style approach
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;
    
    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry11 = s11 >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;
    
    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;

    s[0] = (unsigned char) (s0 >> 0);
    s[1] = (unsigned char) (s0 >> 8);
    s[2] = (unsigned char) ((s0 >> 16) | (s1 << 5));
    s[3] = (unsigned char) (s1 >> 3);
    s[4] = (unsigned char) (s1 >> 11);
    s[5] = (unsigned char) ((s1 >> 19) | (s2 << 2));
    s[6] = (unsigned char) (s2 >> 6);
    s[7] = (unsigned char) ((s2 >> 14) | (s3 << 7));
    s[8] = (unsigned char) (s3 >> 1);
    s[9] = (unsigned char) (s3 >> 9);
    s[10] = (unsigned char) ((s3 >> 17) | (s4 << 4));
    s[11] = (unsigned char) (s4 >> 4);
    s[12] = (unsigned char) (s4 >> 12);
    s[13] = (unsigned char) ((s4 >> 20) | (s5 << 1));
    s[14] = (unsigned char) (s5 >> 7);
    s[15] = (unsigned char) ((s5 >> 15) | (s6 << 6));
    s[16] = (unsigned char) (s6 >> 2);
    s[17] = (unsigned char) (s6 >> 10);
    s[18] = (unsigned char) ((s6 >> 18) | (s7 << 3));
    s[19] = (unsigned char) (s7 >> 5);
    s[20] = (unsigned char) (s7 >> 13);
    s[21] = (unsigned char) (s8 >> 0);
    s[22] = (unsigned char) (s8 >> 8);
    s[23] = (unsigned char) ((s8 >> 16) | (s9 << 5));
    s[24] = (unsigned char) (s9 >> 3);
    s[25] = (unsigned char) (s9 >> 11);
    s[26] = (unsigned char) ((s9 >> 19) | (s10 << 2));
    s[27] = (unsigned char) (s10 >> 6);
    s[28] = (unsigned char) ((s10 >> 14) | (s11 << 7));
    s[29] = (unsigned char) (s11 >> 1);
    s[30] = (unsigned char) (s11 >> 9);
    s[31] = (unsigned char) (s11 >> 17);
}

/*
Input:
  a[0]+256*a[1]+...+256^31*a[31] = a mod l
  b[0]+256*b[1]+...+256^31*b[31] = b mod l

Output:
  s[0]+256*s[1]+...+256^31*s[31] = (a-b) mod l
  where l = 2^252 + 27742317777372353535851937790883648493.
*/

__device__ void sc_sub(unsigned char *s, const unsigned char *a, const unsigned char *b) {
    // Compute a - b mod l by computing a + (l - b) mod l
    // First compute l - b
    unsigned char l_minus_b[32];
    
    // Load order l = 2^252 + 27742317777372353535851937790883648493
    // In little endian: edd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010
    static const unsigned char l[32] = {
        0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
        0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10
    };
    
    // Compute l - b with borrow propagation
    int borrow = 0;
    for (int i = 0; i < 32; i++) {
        int diff = l[i] - b[i] - borrow;
        if (diff < 0) {
            diff += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        l_minus_b[i] = (unsigned char)diff;
    }
    
    // Now compute a + (l - b) mod l
    sc_add(s, a, l_minus_b);
}



/*
Input:
  a[0]+256*a[1]+...+256^31*a[31] = a
  b[0]+256*b[1]+...+256^31*b[31] = b
  c[0]+256*c[1]+...+256^31*c[31] = c

Output:
  s[0]+256*s[1]+...+256^31*s[31] = (ab+c) mod l
  where l = 2^252 + 27742317777372353535851937790883648493.
*/

__device__ void sc_muladd(unsigned char *s, const unsigned char *a, const unsigned char *b, const unsigned char *c) {
    int64_t a0 = 2097151 & sc_load3(a);
    int64_t a1 = 2097151 & (sc_load4(a + 2) >> 5);
    int64_t a2 = 2097151 & (sc_load3(a + 5) >> 2);
    int64_t a3 = 2097151 & (sc_load4(a + 7) >> 7);
    int64_t a4 = 2097151 & (sc_load4(a + 10) >> 4);
    int64_t a5 = 2097151 & (sc_load3(a + 13) >> 1);
    int64_t a6 = 2097151 & (sc_load4(a + 15) >> 6);
    int64_t a7 = 2097151 & (sc_load3(a + 18) >> 3);
    int64_t a8 = 2097151 & sc_load3(a + 21);
    int64_t a9 = 2097151 & (sc_load4(a + 23) >> 5);
    int64_t a10 = 2097151 & (sc_load3(a + 26) >> 2);
    int64_t a11 = (sc_load4(a + 28) >> 7);
    int64_t b0 = 2097151 & sc_load3(b);
    int64_t b1 = 2097151 & (sc_load4(b + 2) >> 5);
    int64_t b2 = 2097151 & (sc_load3(b + 5) >> 2);
    int64_t b3 = 2097151 & (sc_load4(b + 7) >> 7);
    int64_t b4 = 2097151 & (sc_load4(b + 10) >> 4);
    int64_t b5 = 2097151 & (sc_load3(b + 13) >> 1);
    int64_t b6 = 2097151 & (sc_load4(b + 15) >> 6);
    int64_t b7 = 2097151 & (sc_load3(b + 18) >> 3);
    int64_t b8 = 2097151 & sc_load3(b + 21);
    int64_t b9 = 2097151 & (sc_load4(b + 23) >> 5);
    int64_t b10 = 2097151 & (sc_load3(b + 26) >> 2);
    int64_t b11 = (sc_load4(b + 28) >> 7);
    int64_t c0 = 2097151 & sc_load3(c);
    int64_t c1 = 2097151 & (sc_load4(c + 2) >> 5);
    int64_t c2 = 2097151 & (sc_load3(c + 5) >> 2);
    int64_t c3 = 2097151 & (sc_load4(c + 7) >> 7);
    int64_t c4 = 2097151 & (sc_load4(c + 10) >> 4);
    int64_t c5 = 2097151 & (sc_load3(c + 13) >> 1);
    int64_t c6 = 2097151 & (sc_load4(c + 15) >> 6);
    int64_t c7 = 2097151 & (sc_load3(c + 18) >> 3);
    int64_t c8 = 2097151 & sc_load3(c + 21);
    int64_t c9 = 2097151 & (sc_load4(c + 23) >> 5);
    int64_t c10 = 2097151 & (sc_load3(c + 26) >> 2);
    int64_t c11 = (sc_load4(c + 28) >> 7);
    int64_t s0;
    int64_t s1;
    int64_t s2;
    int64_t s3;
    int64_t s4;
    int64_t s5;
    int64_t s6;
    int64_t s7;
    int64_t s8;
    int64_t s9;
    int64_t s10;
    int64_t s11;
    int64_t s12;
    int64_t s13;
    int64_t s14;
    int64_t s15;
    int64_t s16;
    int64_t s17;
    int64_t s18;
    int64_t s19;
    int64_t s20;
    int64_t s21;
    int64_t s22;
    int64_t s23;
    int64_t carry0;
    int64_t carry1;
    int64_t carry2;
    int64_t carry3;
    int64_t carry4;
    int64_t carry5;
    int64_t carry6;
    int64_t carry7;
    int64_t carry8;
    int64_t carry9;
    int64_t carry10;
    int64_t carry11;
    int64_t carry12;
    int64_t carry13;
    int64_t carry14;
    int64_t carry15;
    int64_t carry16;
    int64_t carry17;
    int64_t carry18;
    int64_t carry19;
    int64_t carry20;
    int64_t carry21;
    int64_t carry22;

    s0 = c0 + a0 * b0;
    s1 = c1 + a0 * b1 + a1 * b0;
    s2 = c2 + a0 * b2 + a1 * b1 + a2 * b0;
    s3 = c3 + a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
    s4 = c4 + a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;
    s5 = c5 + a0 * b5 + a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1 + a5 * b0;
    s6 = c6 + a0 * b6 + a1 * b5 + a2 * b4 + a3 * b3 + a4 * b2 + a5 * b1 + a6 * b0;
    s7 = c7 + a0 * b7 + a1 * b6 + a2 * b5 + a3 * b4 + a4 * b3 + a5 * b2 + a6 * b1 + a7 * b0;
    s8 = c8 + a0 * b8 + a1 * b7 + a2 * b6 + a3 * b5 + a4 * b4 + a5 * b3 + a6 * b2 + a7 * b1 + a8 * b0;
    s9 = c9 + a0 * b9 + a1 * b8 + a2 * b7 + a3 * b6 + a4 * b5 + a5 * b4 + a6 * b3 + a7 * b2 + a8 * b1 + a9 * b0;
    s10 = c10 + a0 * b10 + a1 * b9 + a2 * b8 + a3 * b7 + a4 * b6 + a5 * b5 + a6 * b4 + a7 * b3 + a8 * b2 + a9 * b1 + a10 * b0;
    s11 = c11 + a0 * b11 + a1 * b10 + a2 * b9 + a3 * b8 + a4 * b7 + a5 * b6 + a6 * b5 + a7 * b4 + a8 * b3 + a9 * b2 + a10 * b1 + a11 * b0;
    s12 = a1 * b11 + a2 * b10 + a3 * b9 + a4 * b8 + a5 * b7 + a6 * b6 + a7 * b5 + a8 * b4 + a9 * b3 + a10 * b2 + a11 * b1;
    s13 = a2 * b11 + a3 * b10 + a4 * b9 + a5 * b8 + a6 * b7 + a7 * b6 + a8 * b5 + a9 * b4 + a10 * b3 + a11 * b2;
    s14 = a3 * b11 + a4 * b10 + a5 * b9 + a6 * b8 + a7 * b7 + a8 * b6 + a9 * b5 + a10 * b4 + a11 * b3;
    s15 = a4 * b11 + a5 * b10 + a6 * b9 + a7 * b8 + a8 * b7 + a9 * b6 + a10 * b5 + a11 * b4;
    s16 = a5 * b11 + a6 * b10 + a7 * b9 + a8 * b8 + a9 * b7 + a10 * b6 + a11 * b5;
    s17 = a6 * b11 + a7 * b10 + a8 * b9 + a9 * b8 + a10 * b7 + a11 * b6;
    s18 = a7 * b11 + a8 * b10 + a9 * b9 + a10 * b8 + a11 * b7;
    s19 = a8 * b11 + a9 * b10 + a10 * b9 + a11 * b8;
    s20 = a9 * b11 + a10 * b10 + a11 * b9;
    s21 = a10 * b11 + a11 * b10;
    s22 = a11 * b11;
    s23 = 0;
    carry0 = (s0 + (1 << 20)) >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry2 = (s2 + (1 << 20)) >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry4 = (s4 + (1 << 20)) >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry12 = (s12 + (1 << 20)) >> 21;
    s13 += carry12;
    s12 -= carry12 << 21;
    carry14 = (s14 + (1 << 20)) >> 21;
    s15 += carry14;
    s14 -= carry14 << 21;
    carry16 = (s16 + (1 << 20)) >> 21;
    s17 += carry16;
    s16 -= carry16 << 21;
    carry18 = (s18 + (1 << 20)) >> 21;
    s19 += carry18;
    s18 -= carry18 << 21;
    carry20 = (s20 + (1 << 20)) >> 21;
    s21 += carry20;
    s20 -= carry20 << 21;
    carry22 = (s22 + (1 << 20)) >> 21;
    s23 += carry22;
    s22 -= carry22 << 21;
    carry1 = (s1 + (1 << 20)) >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry3 = (s3 + (1 << 20)) >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry5 = (s5 + (1 << 20)) >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    carry13 = (s13 + (1 << 20)) >> 21;
    s14 += carry13;
    s13 -= carry13 << 21;
    carry15 = (s15 + (1 << 20)) >> 21;
    s16 += carry15;
    s15 -= carry15 << 21;
    carry17 = (s17 + (1 << 20)) >> 21;
    s18 += carry17;
    s17 -= carry17 << 21;
    carry19 = (s19 + (1 << 20)) >> 21;
    s20 += carry19;
    s19 -= carry19 << 21;
    carry21 = (s21 + (1 << 20)) >> 21;
    s22 += carry21;
    s21 -= carry21 << 21;
    s11 += s23 * 666643;
    s12 += s23 * 470296;
    s13 += s23 * 654183;
    s14 -= s23 * 997805;
    s15 += s23 * 136657;
    s16 -= s23 * 683901;
    s23 = 0;
    s10 += s22 * 666643;
    s11 += s22 * 470296;
    s12 += s22 * 654183;
    s13 -= s22 * 997805;
    s14 += s22 * 136657;
    s15 -= s22 * 683901;
    s22 = 0;
    s9 += s21 * 666643;
    s10 += s21 * 470296;
    s11 += s21 * 654183;
    s12 -= s21 * 997805;
    s13 += s21 * 136657;
    s14 -= s21 * 683901;
    s21 = 0;
    s8 += s20 * 666643;
    s9 += s20 * 470296;
    s10 += s20 * 654183;
    s11 -= s20 * 997805;
    s12 += s20 * 136657;
    s13 -= s20 * 683901;
    s20 = 0;
    s7 += s19 * 666643;
    s8 += s19 * 470296;
    s9 += s19 * 654183;
    s10 -= s19 * 997805;
    s11 += s19 * 136657;
    s12 -= s19 * 683901;
    s19 = 0;
    s6 += s18 * 666643;
    s7 += s18 * 470296;
    s8 += s18 * 654183;
    s9 -= s18 * 997805;
    s10 += s18 * 136657;
    s11 -= s18 * 683901;
    s18 = 0;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry12 = (s12 + (1 << 20)) >> 21;
    s13 += carry12;
    s12 -= carry12 << 21;
    carry14 = (s14 + (1 << 20)) >> 21;
    s15 += carry14;
    s14 -= carry14 << 21;
    carry16 = (s16 + (1 << 20)) >> 21;
    s17 += carry16;
    s16 -= carry16 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    carry13 = (s13 + (1 << 20)) >> 21;
    s14 += carry13;
    s13 -= carry13 << 21;
    carry15 = (s15 + (1 << 20)) >> 21;
    s16 += carry15;
    s15 -= carry15 << 21;
    s5 += s17 * 666643;
    s6 += s17 * 470296;
    s7 += s17 * 654183;
    s8 -= s17 * 997805;
    s9 += s17 * 136657;
    s10 -= s17 * 683901;
    s17 = 0;
    s4 += s16 * 666643;
    s5 += s16 * 470296;
    s6 += s16 * 654183;
    s7 -= s16 * 997805;
    s8 += s16 * 136657;
    s9 -= s16 * 683901;
    s16 = 0;
    s3 += s15 * 666643;
    s4 += s15 * 470296;
    s5 += s15 * 654183;
    s6 -= s15 * 997805;
    s7 += s15 * 136657;
    s8 -= s15 * 683901;
    s15 = 0;
    s2 += s14 * 666643;
    s3 += s14 * 470296;
    s4 += s14 * 654183;
    s5 -= s14 * 997805;
    s6 += s14 * 136657;
    s7 -= s14 * 683901;
    s14 = 0;
    s1 += s13 * 666643;
    s2 += s13 * 470296;
    s3 += s13 * 654183;
    s4 -= s13 * 997805;
    s5 += s13 * 136657;
    s6 -= s13 * 683901;
    s13 = 0;
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;
    carry0 = (s0 + (1 << 20)) >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry2 = (s2 + (1 << 20)) >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry4 = (s4 + (1 << 20)) >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry1 = (s1 + (1 << 20)) >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry3 = (s3 + (1 << 20)) >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry5 = (s5 + (1 << 20)) >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;
    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry11 = s11 >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;
    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    
    s[0] = (unsigned char) (s0 >> 0);
    s[1] = (unsigned char) (s0 >> 8);
    s[2] = (unsigned char) ((s0 >> 16) | (s1 << 5));
    s[3] = (unsigned char) (s1 >> 3);
    s[4] = (unsigned char) (s1 >> 11);
    s[5] = (unsigned char) ((s1 >> 19) | (s2 << 2));
    s[6] = (unsigned char) (s2 >> 6);
    s[7] = (unsigned char) ((s2 >> 14) | (s3 << 7));
    s[8] = (unsigned char) (s3 >> 1);
    s[9] = (unsigned char) (s3 >> 9);
    s[10] = (unsigned char) ((s3 >> 17) | (s4 << 4));
    s[11] = (unsigned char) (s4 >> 4);
    s[12] = (unsigned char) (s4 >> 12);
    s[13] = (unsigned char) ((s4 >> 20) | (s5 << 1));
    s[14] = (unsigned char) (s5 >> 7);
    s[15] = (unsigned char) ((s5 >> 15) | (s6 << 6));
    s[16] = (unsigned char) (s6 >> 2);
    s[17] = (unsigned char) (s6 >> 10);
    s[18] = (unsigned char) ((s6 >> 18) | (s7 << 3));
    s[19] = (unsigned char) (s7 >> 5);
    s[20] = (unsigned char) (s7 >> 13);
    s[21] = (unsigned char) (s8 >> 0);
    s[22] = (unsigned char) (s8 >> 8);
    s[23] = (unsigned char) ((s8 >> 16) | (s9 << 5));
    s[24] = (unsigned char) (s9 >> 3);
    s[25] = (unsigned char) (s9 >> 11);
    s[26] = (unsigned char) ((s9 >> 19) | (s10 << 2));
    s[27] = (unsigned char) (s10 >> 6);
    s[28] = (unsigned char) ((s10 >> 14) | (s11 << 7));
    s[29] = (unsigned char) (s11 >> 1);
    s[30] = (unsigned char) (s11 >> 9);
    s[31] = (unsigned char) (s11 >> 17);
}
__device__ void sc_from_128bit(unsigned char *s, const unsigned char *coeff) {
    // Zero-extend the 128-bit coefficient to 64 bytes and reduce
    unsigned char expanded[64];
    
    // Copy the 128-bit coefficient to lower 16 bytes
    for (int i = 0; i < 16; i++) {
        expanded[i] = coeff[i];
    }
    
    // Zero the upper 48 bytes
    for (int i = 16; i < 64; i++) {
        expanded[i] = 0;
    }
    
    // Use sc_reduce to get the final 32-byte result
    sc_reduce(expanded);
    
    // Copy the reduced result
    for (int i = 0; i < 32; i++) {
        s[i] = expanded[i];
    }
}

/*
Multiply a 128-bit coefficient by a 32-byte scalar mod l
Input:
  coeff[0]+256*coeff[1]+...+256^15*coeff[15] = 128-bit coefficient
  scalar[0]+256*scalar[1]+...+256^31*scalar[31] = 32-byte scalar mod l

Output:
  result[0]+256*result[1]+...+256^31*result[31] = (coeff * scalar) mod l
*/

__device__ void sc_mul_128bit_scalar(unsigned char *result, const unsigned char *coeff, const unsigned char *scalar) {
    // First reduce the 128-bit coefficient to a 32-byte scalar
    unsigned char coeff_reduced[32];
    sc_from_128bit(coeff_reduced, coeff);
    
    // Now multiply: result = coeff_reduced * scalar + 0
    unsigned char zero[32] = {0};
    sc_muladd(result, coeff_reduced, scalar, zero);
}

/*
Compute the combined basepoint scalar S0 = Σ z_i * s_i mod l
Accumulates multiple terms: S0 += z_i * s_i for each signature

Input:
  accumulator[32] = current sum (input/output)
  coeff[16] = 128-bit random coefficient z_i
  signature_scalar[32] = signature scalar s_i

Output:
  accumulator is updated with += z_i * s_i mod l
*/

__device__ void sc_accumulate_basepoint_scalar(unsigned char *accumulator, const unsigned char *coeff, const unsigned char *signature_scalar) {
    unsigned char term[32];
    sc_mul_128bit_scalar(term, coeff, signature_scalar);
    
    unsigned char new_accumulator[32];
    sc_add(new_accumulator, accumulator, term);
    
    // Copy result back to accumulator
    for (int i = 0; i < 32; i++) {
        accumulator[i] = new_accumulator[i];
    }
}

/*
Compute public key scalar: S_i = z_i * k_i mod l (positive for negated point convention)
Input:
  coeff_reduced[32] = already reduced 32-byte scalar z_i
  hash_scalar[32] = hash scalar k_i = H(R_i || A_i || M_i) reduced mod l

Output:
  result[32] = z_i * k_i mod l (positive to match negated A_i points)
*/

__device__ void sc_compute_pubkey_scalar(unsigned char *result, const unsigned char *coeff_reduced, const unsigned char *hash_scalar) {
    // Compute z_i * k_i for negated point convention
    // With negated points, MSM computes -S_i * A_true, and we want -z_i * k_i * A_true
    // So we need S_i = z_i * k_i (positive)
    unsigned char tmp[32], zero[32] = {0};
    sc_muladd(tmp, coeff_reduced, hash_scalar, zero); // result = z_i * k_i
    sc_neg(result, tmp);
}

/*
Compute R point scalar: T_i = z_i mod l (positive for negated point convention)
Input:
  coeff_reduced[32] = already reduced 32-byte scalar z_i

Output:
  result[32] = z_i mod l (positive to match negated R_i points)
*/

__device__ void sc_compute_r_scalar(unsigned char *result, const unsigned char *coeff_reduced) {
    //memcpy(result, coeff_reduced, 32);          // copy, do NOT sc_neg()
    // sc_reduce32_local(result);     // already canonical
    sc_neg(result, coeff_reduced); // negate to match negated R_i points
}

/*
Modular negation: out = (L - in) mod L
Input:
  in[32] = 32-byte scalar mod L

Output:
  out[32] = (L - in) mod L, with special case for 0
*/

__device__ __host__ void sc_neg(unsigned char *out, const unsigned char *in) {
    // Ed25519 order L = 2^252 + 27742317777372353535851937790883648493
    static const unsigned char L[32] = {
        0xed,0xd3,0xf5,0x5c,0x1a,0x63,0x12,0x58,
        0xd6,0x9c,0xf7,0xa2,0xde,0xf9,0xde,0x14,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x10
    };
    
    // Special case: if in == 0, then out = 0 (since -0 = 0)
    int is_zero = 1;
    for (int i = 0; i < 32; i++) {
        if (in[i] != 0) {
            is_zero = 0;
            break;
        }
    }
    if (is_zero) {
        for (int i = 0; i < 32; i++) {
            out[i] = 0;
        }
        return;
    }
    
    // Compute out = L - in with borrow propagation
    int borrow = 0;
    for (int i = 0; i < 32; i++) {
        int diff = (int)L[i] - (int)in[i] - borrow;
        if (diff < 0) {
            diff += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out[i] = (unsigned char)diff;
    }
}
