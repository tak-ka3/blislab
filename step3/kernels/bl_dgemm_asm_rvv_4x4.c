#include <bl_config.h>
#include "bl_dgemm_kernel.h"
#include "riscv_vector.h"

#define MYKERNEL4x4 \
    "vsetvli t4, t0, e64, m2, ta, ma    \n\t"\
    "vle64.v    v0,     (%[PA])     \n\t"\
    "addi       %[PA],  %[PA],   4*8     \n\t"\
    "addi       t1,     %[PB],  1*8 \n\t"\
    "addi       t2,     %[PB],  2*8 \n\t"\
    "addi       t3,     %[PB],  3*8 \n\t"\
    "fld        ft0,    (%[PB])   \n\t"\
    "fld        ft1,    (t1)    \n\t"\
    "fld        ft2,    (t2)    \n\t"\
    "fld        ft3,    (t3)    \n\t"\
    "vfmv.v.f   v8,     ft0     \n\t"\
    "vfmv.v.f   v10,     ft1     \n\t"\
    "vfmv.v.f   v12,    ft2     \n\t"\
    "vfmv.v.f   v14,    ft3     \n\t"\
    "vfmacc.vv  v16,    v8,     v0  \n\t"\
    "vfmacc.vv  v18,    v10,     v0  \n\t"\
    "vfmacc.vv  v20,    v12,    v0  \n\t"\
    "vfmacc.vv  v22,    v14,    v0  \n\t"\
    "sub        t0,     t0,     t4  \n\t"\

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i,j) c[ (j)*ldc + (i) ]

void AddDot4x4(double*, double*, double*, int);

void bl_dgemm_asm_rvv_4x4( int    k,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l;

    for ( l = 0; l < k; ++l )
    {                 
        AddDot4x4(&a(0, l), &b(l, 0), &c(0, 0), ldc);
    }

}

void AddDot4x4(double* a, double* b, double* c, int ldc)
{
    int bk = 4;
    // printf("a = %f, b - %f, c = %f\n", *a, *b, *c);
    double *c0 = c;
    double *c1 = c0 + ldc;
    double *c2 = c1 + ldc;
    double *c3 = c2 + ldc;

    asm volatile(
        "vsetvli     zero, zero, e64, m2, ta, ma \n\t"
        "fmv.w.x    ft11,   zero    \n\t"
        "vfmv.v.f   v16,    ft11    \n\t"
        "vfmv.v.f   v18,    ft11    \n\t"
        "vfmv.v.f   v20,    ft11    \n\t"
        "vfmv.v.f   v22,    ft11    \n\t"
        "mv         t0,     %[BK]   \n\t"

        "LOOP%=:                      \n\t"
        MYKERNEL4x4
        "bnez       t0,     LOOP%=   \n\t"
        "vsetvli zero, zero, e64, m2, ta, ma    \n\t"
        "vse64.v    v16,    (%[C0])     \n\t"
        "vse64.v    v18,    (%[C1])     \n\t"
        "vse64.v    v20,    (%[C2])     \n\t"
        "vse64.v    v22,    (%[C3])     \n\t"

        "M4x4_END%=:                     \n\t"

        :[C0]"+r"(c0),[C1]"+r"(c1),[C2]"+r"(c2),[C3]"+r"(c3),[PB]"+r"(b),[PA]"+r"(a)
        :[BK]"r"(bk)
        :"cc", "t0", "t4","t5","t6","t3","t1","t2",
        "ft11", "ft0", "ft1", "ft2","ft3","ft4", "ft5", "ft6","ft7",
        "v0", "v1", "v2", "v3","v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11","v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );
    return;
}