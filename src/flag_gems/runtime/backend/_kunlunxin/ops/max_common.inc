#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_simd.h"

// max_dim / max raw payload v4 (P800 xpu3, cluster path).
// One Triton program == one cluster (64 cores). Cores take rows with a
// core-stride; each core drives its own GM2LM/LM2GM DMA.
//
// Techniques (all measured on [4096,4096] fp32 vs alternatives):
//  * GM2LM staging into double-buffered 3KB LM chunks, with the next chunk
//    issued (GM2LM_ASYNC) before processing the current one and the fence
//    placed before the issue. The pipeline is FLAT across rows: chunk (r,c+1)
//    is prefetched while chunk (r,c) is reduced, and row r+1's first chunk
//    while row r's last one is reduced. Direct global vector loads are 14x
//    slower; SM staging reads 2.4x slower than LM.
//  * Row results are staged in SHARED memory (one slot per core, one bank
//    per round) and flushed cooperatively: a handful of contiguous 256B-1KB
//    SM2GM transfers per cluster instead of two tiny per-row global stores
//    (two inline scalar stores cost 16us/row; batched LM2GM 4.5us; SM
//    cooperation ~1.5us).
//  * The winning 16/32-element vector is copied to a small LM array when a
//    new maximum appears, so the first-index scan touches 16 lanes once per
//    row.
//  * bf16 goes through the exact "top 16 bits of f32" identity on packed
//    int32x16 vectors (32 bf16 per vector).

#define CHUNK_BYTES 3072
#define NBUF 2
#define FLUSH_R 8  // rounds staged in shared memory between flushes

// type codes
#define T_F32 0
#define T_F16 1
#define T_BF16 2
#define T_I32 3
#define T_I64 4
#define T_I16 5
#define T_I8 6
#define T_U8 7
#define T_F64 8

// ---------------- chunk processors ------------------------------------------

static __device__ __attribute__((always_inline)) void proc_f32(
    __local__ char *buf, int len, long long base, float *best, int *bidx,
    int *first, __local__ float *svl, int *sv_base) {
  __local__ float *f = (__local__ float *)buf;
  int nv = len >> 4;
  int v = 0;
  for (; v + 3 < nv; v += 4) {
    float m0 = vrmax_float32x16(*(__local__ float32x16_t *)(f + ((v + 0) << 4)));
    float m1 = vrmax_float32x16(*(__local__ float32x16_t *)(f + ((v + 1) << 4)));
    float m2 = vrmax_float32x16(*(__local__ float32x16_t *)(f + ((v + 2) << 4)));
    float m3 = vrmax_float32x16(*(__local__ float32x16_t *)(f + ((v + 3) << 4)));
    float ma = m0 > m1 ? m0 : m1;
    float mb = m2 > m3 ? m2 : m3;
    float m = ma > mb ? ma : mb;
    if (*first || m > *best) {
      *best = m;
      int q = (m == m0) ? 0 : (m == m1) ? 1 : (m == m2) ? 2 : 3;
      const __local__ float *w = f + ((v + q) << 4);
      for (int l = 0; l < 16; ++l) svl[l] = w[l];
      *sv_base = (int)base + ((v + q) << 4);
      *first = 0;
    }
  }
  for (; v < nv; ++v) {
    float32x16_t x = *(__local__ float32x16_t *)(f + (v << 4));
    float m = vrmax_float32x16(x);
    if (*first || m > *best) {
      *best = m;
      const __local__ float *w = f + (v << 4);
      for (int l = 0; l < 16; ++l) svl[l] = w[l];
      *sv_base = (int)base + (v << 4);
      *first = 0;
    }
  }
  for (int i = nv << 4; i < len; ++i) {
    float x = f[i];
    if (*first || x > *best) {
      *best = x;
      *bidx = (int)base + i;
      *sv_base = -1;
      *first = 0;
    }
  }
}

static __device__ __attribute__((always_inline)) void proc_f16(
    __local__ char *buf, int len, long long base, _Float16 *best, int *bidx,
    int *first, __local__ _Float16 *svl, int *sv_base) {
  __local__ _Float16 *f = (__local__ _Float16 *)buf;
  int nv = len >> 5;
  int v = 0;
  for (; v + 1 < nv; v += 2) {
    _Float16 m0 = vrmax_float16x32(*(__local__ float16x32_t *)(f + ((v + 0) << 5)));
    _Float16 m1 = vrmax_float16x32(*(__local__ float16x32_t *)(f + ((v + 1) << 5)));
    _Float16 m = m0 > m1 ? m0 : m1;
    if (*first || m > *best) {
      *best = m;
      int q = (m == m0) ? 0 : 1;
      const __local__ _Float16 *w = f + ((v + q) << 5);
      for (int l = 0; l < 32; ++l) svl[l] = w[l];
      *sv_base = (int)base + ((v + q) << 5);
      *first = 0;
    }
  }
  for (; v < nv; ++v) {
    float16x32_t x = *(__local__ float16x32_t *)(f + (v << 5));
    _Float16 m = vrmax_float16x32(x);
    if (*first || m > *best) {
      *best = m;
      const __local__ _Float16 *w = f + (v << 5);
      for (int l = 0; l < 32; ++l) svl[l] = w[l];
      *sv_base = (int)base + (v << 5);
      *first = 0;
    }
  }
  for (int i = nv << 5; i < len; ++i) {
    _Float16 x = f[i];
    if (*first || x > *best) {
      *best = x;
      *bidx = (int)base + i;
      *sv_base = -1;
      *first = 0;
    }
  }
}

static __device__ __attribute__((always_inline)) void proc_bf16(
    __local__ char *buf, int len, long long base, float *best, int *bidx,
    int *first, __local__ unsigned short *svl, int *sv_base) {
  __local__ unsigned short *w = (__local__ unsigned short *)buf;
  int nv = len >> 5;  // int32x16 vectors (32 bf16 each)
  int v = 0;
  for (; v + 1 < nv; v += 2) {
    float mg[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      int32x16_t pr = *(__local__ int32x16_t *)(w + ((v + k) << 5));
      int32x16_t hi = pr << 16;
      int32x16_t lo = pr & (int32x16_t)(-65536);
      float mh = vrmax_float32x16(*(float32x16_t *)&hi);
      float ml = vrmax_float32x16(*(float32x16_t *)&lo);
      mg[k] = mh > ml ? mh : ml;
    }
    float m = mg[0] > mg[1] ? mg[0] : mg[1];
    if (*first || m > *best) {
      *best = m;
      int q = (m == mg[0]) ? 0 : 1;
      const __local__ unsigned short *p = w + ((v + q) << 5);
      for (int l = 0; l < 32; ++l) svl[l] = p[l];
      *sv_base = (int)base + ((v + q) << 5);
      *first = 0;
    }
  }
  for (; v < nv; ++v) {
    int32x16_t pr = *(__local__ int32x16_t *)(w + (v << 5));
    int32x16_t hi = pr << 16;
    int32x16_t lo = pr & (int32x16_t)(-65536);
    float mh = vrmax_float32x16(*(float32x16_t *)&hi);
    float ml = vrmax_float32x16(*(float32x16_t *)&lo);
    float m = mh > ml ? mh : ml;
    if (*first || m > *best) {
      *best = m;
      const __local__ unsigned short *p = w + (v << 5);
      for (int l = 0; l < 32; ++l) svl[l] = p[l];
      *sv_base = (int)base + (v << 5);
      *first = 0;
    }
  }
  for (int i = nv << 5; i < len; ++i) {
    unsigned bits = ((unsigned)w[i]) << 16;
    float fv = *(float *)&bits;
    if (*first || fv > *best) {
      *best = fv;
      *bidx = (int)base + i;
      *sv_base = -1;
      *first = 0;
    }
  }
}

static __device__ __attribute__((always_inline)) void proc_i32(
    __local__ char *buf, int len, long long base, int *best, int *bidx,
    int *first, __local__ int *svl, int *sv_base) {
  __local__ int *f = (__local__ int *)buf;
  int nv = len >> 4;
  int v = 0;
  for (; v + 3 < nv; v += 4) {
    int m0 = vrmax_int32x16(*(__local__ int32x16_t *)(f + ((v + 0) << 4)));
    int m1 = vrmax_int32x16(*(__local__ int32x16_t *)(f + ((v + 1) << 4)));
    int m2 = vrmax_int32x16(*(__local__ int32x16_t *)(f + ((v + 2) << 4)));
    int m3 = vrmax_int32x16(*(__local__ int32x16_t *)(f + ((v + 3) << 4)));
    int ma = m0 > m1 ? m0 : m1;
    int mb = m2 > m3 ? m2 : m3;
    int m = ma > mb ? ma : mb;
    if (*first || m > *best) {
      *best = m;
      int q = (m == m0) ? 0 : (m == m1) ? 1 : (m == m2) ? 2 : 3;
      const __local__ int *w = f + ((v + q) << 4);
      for (int l = 0; l < 16; ++l) svl[l] = w[l];
      *sv_base = (int)base + ((v + q) << 4);
      *first = 0;
    }
  }
  for (; v < nv; ++v) {
    int32x16_t x = *(__local__ int32x16_t *)(f + (v << 4));
    int m = vrmax_int32x16(x);
    if (*first || m > *best) {
      *best = m;
      const __local__ int *w = f + (v << 4);
      for (int l = 0; l < 16; ++l) svl[l] = w[l];
      *sv_base = (int)base + (v << 4);
      *first = 0;
    }
  }
  for (int i = nv << 4; i < len; ++i) {
    int x = f[i];
    if (*first || x > *best) {
      *best = x;
      *bidx = (int)base + i;
      *sv_base = -1;
      *first = 0;
    }
  }
}

#define PROC_SCALAR(FNAME, T)                                               \
  static __device__ __attribute__((always_inline)) void FNAME(              \
      __local__ char *buf, int len, long long base, T *best, int *bidx,     \
      int *first) {                                                         \
    __local__ T *f = (__local__ T *)buf;                                    \
    for (int i = 0; i < len; ++i) {                                         \
      T x = f[i];                                                           \
      if (*first || x > *best) {                                            \
        *best = x;                                                          \
        *bidx = (int)base + i;                                              \
        *first = 0;                                                         \
      }                                                                     \
    }                                                                       \
  }

PROC_SCALAR(proc_i64, long long)
PROC_SCALAR(proc_i16, short)
PROC_SCALAR(proc_i8, signed char)
PROC_SCALAR(proc_u8, unsigned char)
PROC_SCALAR(proc_f64, double)

// ---------------- resolve saved lanes -> exact first index -----------------

static __device__ __attribute__((always_inline)) int resolve_f32(
    __local__ float *svl, float best, int sv_base) {
  for (int l = 0; l < 16; ++l)
    if (svl[l] == best) return sv_base + l;
  return sv_base;
}

static __device__ __attribute__((always_inline)) int resolve_f16(
    __local__ _Float16 *svl, _Float16 best, int sv_base) {
  for (int l = 0; l < 32; ++l)
    if (svl[l] == best) return sv_base + l;
  return sv_base;
}

static __device__ __attribute__((always_inline)) int resolve_bf16(
    __local__ unsigned short *svl, float best, int sv_base) {
  const unsigned b = ((unsigned *)&best)[0];
  for (int l = 0; l < 32; ++l)
    if (((unsigned)svl[l]) << 16 == b) return sv_base + l;
  return sv_base;
}

static __device__ __attribute__((always_inline)) int resolve_i32(
    __local__ int *svl, int best, int sv_base) {
  for (int l = 0; l < 16; ++l)
    if (svl[l] == best) return sv_base + l;
  return sv_base;
}

// stage a value's bytes into the shared value bank for round rr, core cid
static __device__ __attribute__((always_inline)) void stage_val(
    char *sval, int idx, int esz, float bestf,
    _Float16 besth, int besti, long long bestl, short bestw,
    signed char bestb, unsigned char bestub, double bestd, int type_code) {
  char *p = sval + idx * esz;
  switch (type_code) {
    case T_F32: {
      unsigned b = ((unsigned *)&bestf)[0];
      *(unsigned *)(p) = b;
      break;
    }
    case T_F16: {
      unsigned short b = ((unsigned short *)&besth)[0];
      *(unsigned short *)(p) = b;
      break;
    }
    case T_BF16: {
      unsigned b = ((unsigned *)&bestf)[0];
      *(unsigned short *)(p) = (unsigned short)(b >> 16);
      break;
    }
    case T_I32: {
      *(int *)(p) = besti;
      break;
    }
    case T_I64: {
      *(long long *)(p) = bestl;
      break;
    }
    case T_I16: {
      *(short *)(p) = bestw;
      break;
    }
    case T_I8: {
      *(signed char *)(p) = bestb;
      break;
    }
    case T_U8: {
      *(unsigned char *)(p) = bestub;
      break;
    }
    default: {
      *(double *)(p) = bestd;
      break;
    }
  }
}
