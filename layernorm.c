#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const float EPS = 1e-6;

void layernorm_forward(int B, int T, int C, float *x, float *w, float *bias,
                       float *mean, float *rstd, float *out, float eps) {
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < C; ++i) {
        mean[b * T + t] += x[b * T * C + t * C + i]; // # B,T,1
      }
      mean[b * T + t] /= C;
      float mu = mean[b * T + t];

      float var = 0;
      float xshift[C];
      for (int i = 0; i < C; ++i) {
        xshift[i] = x[b * T * C + t * C + i] - mu;
        var += pow(xshift[i], 2);
      }
      float rstd_v = pow(var / C + eps, -0.5);
      rstd[b * T + t] = rstd_v;
      for (int i = 0; i < C; ++i) {
        out[b * T * C + t * C + i] = xshift[i] * rstd_v * w[i] + bias[i];
      }
    }
  }
}

int main() {

  int B = 2; // batch
  int T = 3;
  int C = 4;

  float *x = (float *)malloc(B * T * C * sizeof(float));
  float *w = (float *)malloc(C * sizeof(float));
  float *b = (float *)malloc(C * sizeof(float));
  float *out = (float *)malloc(B * T * C * sizeof(float));
  float *out_ref = (float *)malloc(B * T * C * sizeof(float));
  float *mean = (float *)malloc(B * T * sizeof(float));
  float *rstd = (float *)malloc(B * T * sizeof(float));
  float *dout = (float *)malloc(B * T * C * sizeof(float));

  // read data from reference information
  FILE *fp = fopen("ln.bin", "rb");
  if (fp == NULL) {
    printf("File ln.bin not found!");
    return 1;
  }

  fread(x, sizeof(float), B * T * C, fp);
  fread(w, sizeof(float), C, fp);
  fread(b, sizeof(float), C, fp);
  fread(out_ref, sizeof(float), B * T * C, fp);

  layernorm_forward(B, T, C, x, w, b, mean, rstd, out, EPS);
  printf("%f\n", out[0]);
  printf("%f\n", out_ref[0]);
}
