#include <bl_config.h>
#include "bl_dgemm_kernel.h"
#include <riscv_vector.h>

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i,j) c[ (j)*ldc + (i) ]

void AddDot8x4(double*, double*, double*, int);

void bl_dgemm_rvv_8x4( int    k,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        AddDot8x4(&a(0, l), &b(l, 0), &c(0, 0), ldc);
    }
}

void AddDot8x4(double* a, double* b, double* c, int ldc)
{
    vfloat64m1_t c00, c10, c20, c30,
                vecA0;

    size_t vl = vsetvl_e64m1(8);
    
    double b0, b1, b2, b3;
    b0 = *b; b1 = *(b+1); b2 = *(b+2); b3 = *(b+3);

    for (int i = 0; i < 8; i += vl) {
        vecA0 = vle64_v_f64m1(a+i, vl);
        c00 = vle64_v_f64m1(&c(i, 0), vl);
        c10 = vle64_v_f64m1(&c(i, 1), vl);
        c20 = vle64_v_f64m1(&c(i, 2), vl);
        c30 = vle64_v_f64m1(&c(i, 3), vl);

        c00 = vfmacc_vf_f64m1(c00, b0, vecA0, vl);
        c10 = vfmacc_vf_f64m1(c10, b1, vecA0, vl);
        c20 = vfmacc_vf_f64m1(c20, b2, vecA0, vl);
        c30 = vfmacc_vf_f64m1(c30, b3, vecA0, vl);

        vse64_v_f64m1(&c(i, 0), c00, vl);
        vse64_v_f64m1(&c(i, 1), c10, vl);
        vse64_v_f64m1(&c(i, 2), c20, vl);
        vse64_v_f64m1(&c(i, 3), c30, vl);
    }
}