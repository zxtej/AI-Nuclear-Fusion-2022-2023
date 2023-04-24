//Preprocessor things for compilation of tnp
#ifndef XLOW
  #define XLOW 0.f
#endif
#ifndef YLOW
  #define YLOW 0.f
#endif
#ifndef ZLOW
  #define ZLOW 0.f
#endif
#ifndef XHIGH
  #define XHIGH 0.f
#endif
#ifndef YHIGH
  #define YHIGH 0.f
#endif
#ifndef ZHIGH
  #define ZHIGH 0.f
#endif
#ifndef DX
  #define DX 0
#endif
#ifndef DY
  #define DY 0
#endif
#ifndef DZ
  #define DZ 0
#endif
#ifndef NX
  #define NX 0
#endif
#ifndef NY
  #define NY 0
#endif
#ifndef NZ
  #define NZ 0
#endif
#ifndef R_VISC
  #define R_VISC 0
#endif
#define A_VISC 1 - R_VISC
#define A_VISC_1 1.f / A_VISC
void kernel vector_cross_mul(global float *A0, global const float *B0,
                             global const float *C0, global float *A1,
                             global const float *B1, global const float *C1,
                             global float *A2, global const float *B2,
                             global const float *C2) {
  int i = get_global_id(0); // Get index of the current element to be processed
  A0[i] = B1[i] * C2[i] - B2[i] * C1[i]; // Do the operation
  A1[i] = B2[i] * C0[i] - B0[i] * C2[i];
  A2[i] = B0[i] * C1[i] - B1[i] * C0[i];
}

void kernel vector_mul(global float *A, global const float *B,
                       global const float *C) {
  int i = get_global_id(0); // Get index of the current element to be processed
  A[i] = B[i] * C[i];       // Do the operation
}

void kernel vector_muls_addv(global float *A, global const float *B,
                             global const float *C) {
  float Bb = B[0];
  int i = get_global_id(0); // Get index of current element processed
  A[i] = Bb * A[i] + C[i];  // Do the operation
}

void kernel vector_muls(global float *A, const float B) {
  int i = get_global_id(0); // Get index of current element processed
  A[i] *= B;         // Do the operation
}

void kernel vector_mul_complex(global float2 *A, global float2 *B, global float2 *C) {
  int i = get_global_id(0); // Get index of the current element to be processed
  float2 b = B[i], c = C[i];
  A[i] = (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
}

void kernel
tnp_k(global const float8 *a1, global const float8 *a2,                   // E, B coeff
      global float *x0, global float *y0, global float *z0, // initial pos
      global float *x1, global float *y1, global float *z1,
      const float Bcoeff, const float Ecoeff, // Bcoeff, Ecoeff
      const unsigned int n, const unsigned int ncalc // n, ncalc
      ) {
  uint id = get_global_id(0);
  uint prev_idx = UINT_MAX;
  float xprev = x0[id], yprev = y0[id], zprev = z0[id],
        x = x1[id], y = y1[id], z = z1[id];
  float8 temp, pos;
  float8 store0, store1, store2, store3, store4, store5;
  for (int t = 0; t < ncalc; t++) {
    if(x <= XLOW || x >= XHIGH || y <= YLOW || y >= YHIGH || z <= ZLOW || z >= ZHIGH) break;
    float xy = x * y, xz = x * z, yz = y * z, xyz = x * yz;
    uint idx = ((uint)((z - ZLOW) / DZ) * NZ + (uint)((y - YLOW) / DY)) * NY + (uint)((x - XLOW) / DX);
    // round down the cells - this is intentional
    idx *= 3;
    pos = (float8)(1.f, x, y, z, xy, xz, yz, xyz);

    // Is there no better way to do this? Why does float8 not have dot()?
    if (prev_idx != idx) {
      store0 = a1[idx]; store1 = a1[idx+1]; store2 = a1[idx+2]; store3 = a2[idx]; store4 = a2[idx+1]; store5 = a2[idx+2];
      prev_idx = idx;
    }
    temp = store0 * pos;
    float xE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7;
    temp = store1 * pos;
    float yE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7;
    temp = store2 * pos;
    float zE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7;
    temp = store3 * pos;
    float xP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7;
    temp = store4 * pos;
    float yP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7;
    temp = store5 * pos;
    float zP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7;

    xP *= Bcoeff;
    yP *= Bcoeff;
    zP *= Bcoeff;
    xE *= Ecoeff;
    yE *= Ecoeff;
    zE *= Ecoeff;

    float xyP = xP * yP, yzP = yP * zP, xzP = xP * zP;
    float xxP = xP * xP, yyP = yP * yP, zzP = zP * zP;
    float b_det = 1.f / (A_VISC * A_VISC + xxP + yyP + zzP);

    float vxdt = x - xprev;
    float vydt = y - yprev;
    float vzdt = z - zprev;

    xprev = x;
    yprev = y;
    zprev = z;

    float xprep = fma(A_VISC, zP - yP, xyP + xzP); //xyP + xzP + A_VISC * (zP - yP);
    float yprep = fma(A_VISC, xP - zP, xyP + yzP); //xyP + yzP + A_VISC * (xP - zP);
    float zprep = fma(A_VISC, yP - xP, xzP + yzP); //xzP + yzP + A_VISC * (yP - xP);
    //x += vxdt * (1.f - (yyP + zzP - xprep) * b_det) + xE * b_det * (A_VISC_1 * (xxP + xprep) + A_VISC);
    //y += vydt * (1.f - (xxP + zzP - yprep) * b_det) + yE * b_det * (A_VISC_1 * (yyP + yprep) + A_VISC);
    //z += vzdt * (1.f - (xxP + yyP - zprep) * b_det) + zE * b_det * (A_VISC_1 * (zzP + zprep) + A_VISC);
    // fma time
    x += fma(vxdt, fma(b_det, xprep - yyP - zzP, 1.f), xE * b_det * fma(A_VISC_1, xxP + xprep, A_VISC));
    y += fma(vydt, fma(b_det, yprep - xxP - zzP, 1.f), yE * b_det * fma(A_VISC_1, yyP + yprep, A_VISC));
    z += fma(vzdt, fma(b_det, zprep - xxP - yyP, 1.f), zE * b_det * fma(A_VISC_1, zzP + zprep, A_VISC));
  }
  x0[id] = xprev;
  y0[id] = yprev;
  z0[id] = zprev;
  x1[id] = x;
  y1[id] = y;
  z1[id] = z;
}