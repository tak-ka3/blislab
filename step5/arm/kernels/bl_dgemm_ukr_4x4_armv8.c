#include <bl_config.h>
#include "bl_dgemm_kernel.h"

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i,j) c[ (j)*ldc + (i) ]


#include<arm_neon.h>

/*
void bl_dgemm_ukr( int    k,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < DGEMM_NR; ++j )
        { 
            for ( i = 0; i < DGEMM_MR; ++i )
            { 
                c( i, j ) += a( i, l ) * b( l, j );
            }
        }
    }

}

*/

void bl_dgemm_ukr( int    k,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
  //4x4
  float64x2_t C00, C20, C01, C21, C02, C22, C03, C23;
  float64x2_t A0, A2;
  float64x2_t B0, B2;

  float64x2_t c_tmp;
  
  int l;

  C00=vmovq_n_f64( 0.0 );
  C20=vmovq_n_f64( 0.0 );
  
  C01=vmovq_n_f64( 0.0 );
  C21=vmovq_n_f64( 0.0 );
  
  C02=vmovq_n_f64( 0.0 );
  C22=vmovq_n_f64( 0.0 );

  C03=vmovq_n_f64( 0.0 );
  C23=vmovq_n_f64( 0.0 );
  
  for ( l = 0; l < k; ++l )
  {
    A0=vld1q_f64(a);
    A2=vld1q_f64(a+2);

    B0=vld1q_f64(b);
    B2=vld1q_f64(b+2);

    C00=vfmaq_laneq_f64(C00, A0, B0, 0);
    C20=vfmaq_laneq_f64(C20, A2, B0, 0);

    C01=vfmaq_laneq_f64(C01, A0, B0, 1);
    C21=vfmaq_laneq_f64(C21, A2, B0, 1);

    C02=vfmaq_laneq_f64(C02, A0, B2, 0);
    C22=vfmaq_laneq_f64(C22, A2, B2, 0);

    C03=vfmaq_laneq_f64(C03, A0, B2, 1);
    C23=vfmaq_laneq_f64(C23, A2, B2, 1);
    
    a+=4;
    b+=4;
  }

  //write back  
  c_tmp=vld1q_f64(&c(0,0));
  C00 = vaddq_f64(c_tmp, C00);
  vst1q_f64(&c(0,0), C00);

  c_tmp=vld1q_f64(&c(2,0));
  C20 = vaddq_f64(c_tmp, C20);
  vst1q_f64(&c(2,0), C20);

  c_tmp=vld1q_f64(&c(0,1));
  C01 = vaddq_f64(c_tmp, C01);
  vst1q_f64(&c(0,1), C01);

  c_tmp=vld1q_f64(&c(2,1));
  C21 = vaddq_f64(c_tmp, C21);
  vst1q_f64(&c(2,1), C21);

  c_tmp=vld1q_f64(&c(0,2));
  C02 = vaddq_f64(c_tmp, C02);
  vst1q_f64(&c(0,2), C02);

  c_tmp=vld1q_f64(&c(2,2));
  C22 = vaddq_f64(c_tmp, C22);
  vst1q_f64(&c(2,2), C22);

  c_tmp=vld1q_f64(&c(0,3));
  C03 = vaddq_f64(c_tmp, C03);
  vst1q_f64(&c(0,3), C03);

  c_tmp=vld1q_f64(&c(2,3));
  C23 = vaddq_f64(c_tmp, C23);
  vst1q_f64(&c(2,3), C23);
  
}
