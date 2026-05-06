#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xj_v = _mm512_loadu_ps(x);
  __m512 yj_v = _mm512_loadu_ps(y);
  __m512 mj_v = _mm512_loadu_ps(m);

  __m512i idx_v = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

  for(int i=0;i<N;i++)
  {
    __m512 xi_v = _mm512_set1_ps(x[i]);
    __m512 yi_v = _mm512_set1_ps(y[i]);

    __m512i i_v = _mm512_set1_epi32(i);
    __mmask16 mask = _mm512_cmp_epi32_mask(idx_v, i_v, _MM_CMPINT_NE);

    __m512 rx_v = _mm512_sub_ps(xi_v, xj_v);
    __m512 ry_v = _mm512_sub_ps(yi_v, yj_v);

    __m512 rx2_v = _mm512_mul_ps(rx_v, rx_v);
    __m512 ry2_v = _mm512_mul_ps(ry_v, ry_v);
    __m512 r2_v = _mm512_add_ps(rx2_v, ry2_v);
    
    __m512 inv_r_v = _mm512_rsqrt14_ps(r2_v);
    __m512 inv_r3_v = _mm512_mul_ps(_mm512_mul_ps(inv_r_v, inv_r_v), inv_r_v);

    __m512 del_fx = _mm512_mul_ps(_mm512_mul_ps(rx_v, mj_v), inv_r3_v);
    __m512 del_fy = _mm512_mul_ps(_mm512_mul_ps(ry_v, mj_v), inv_r3_v);

    fx[i] -= _mm512_mask_reduce_add_ps(del_fx, mask);
    fy[i] -= _mm512_mask_reduce_add_ps(del_fy, mask);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
