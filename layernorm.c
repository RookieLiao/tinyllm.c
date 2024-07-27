#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const float EPS = 1e-5f;

void layernorm_forward(float *inp, float *w, float *bias, float *mean,
                       float *rstd, float *out, int B, int T, int C) {
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      float m = 0.0f;

      // seek to the input position inp[b,t,:]
      float *x = inp + b * T * C + t * C;
      for (int i = 0; i < C; ++i) {
        m += x[i]; // # B,T,1
      }
      m /= C;

      float var = 0.0f;
      for (int i = 0; i < C; ++i) {
        float xshift = x[i] - m;
        var += pow(xshift, 2);
      }
      var /= C;
      float rstd_v = pow(var + EPS, -0.5);

      // seek to the output position out[b,t:]
      float *out_p = out + b * T * C + t * C;
      for (int i = 0; i < C; ++i) {
        out_p[i] = (x[i] - m) * rstd_v * w[i] + bias[i];
      }

      mean[b * T + t] = m;
      rstd[b * T + t] = rstd_v;
    }
  }
}

void layernorm_backward(float *dout, float *dw, float *db, float *dx,
                        float *inp, float *w, float *bias, float *mean,
                        float *rstd, int B, int T, int C) {

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      // seek to the position [b,t,:]
      float *x = inp + b * T * C + t * C;
      float *dout_p = dout + b * T * C + t * C;
      float *dx_p = dx + b * T * C + t * C;

      float m = mean[b * T + t];
      float v = rstd[b * T + t];

      // first: two reduce operations along with C
      float dnorm_mean = 0.0f;
      float dnorm_norm_mean = 0.0f;
      for (int i = 0; i < C; ++i) {
        // recompute norm (saving memory at the cost of compute)
        float norm_v = (x[i] - m) * v;
        float dnorm_v = dout_p[i] * w[i];
        dnorm_mean += dnorm_v;
        dnorm_norm_mean += dnorm_v * norm_v;
      }
      dnorm_mean /= C;
      dnorm_norm_mean /= C;

      for (int i = 0; i < C; ++i) {
        float norm_v = (x[i] - m) * v;
        float dnorm_v = dout_p[i] * w[i];
        dw[i] += dout_p[i] * norm_v;
        db[i] += dout_p[i];
        dx_p[i] += (dnorm_v - dnorm_mean - norm_v * dnorm_norm_mean) * v;
      }
    }
  }
}

int check_correct(float *ref, float *pred, int size) {
  for (int i = 0; i < size; ++i) {
    if (fabs(ref[i] - pred[i]) > 1e-4) {
      printf("pred: %f and ref: %f\n", pred[i], ref[i]);
      printf("failed!\n");
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

int main() {

  int B = 2; // batch
  int T = 3;
  int C = 4;

  float *x = (float *)malloc(B * T * C * sizeof(float));
  float *w = (float *)malloc(C * sizeof(float));
  float *b = (float *)malloc(C * sizeof(float));
  float *out = (float *)malloc(B * T * C * sizeof(float));
  float *dout = (float *)malloc(B * T * C * sizeof(float));
  float *mean = (float *)malloc(B * T * sizeof(float));
  float *rstd = (float *)malloc(B * T * sizeof(float));
  float *dw = (float *)malloc(C * sizeof(float));
  float *db = (float *)malloc(C * sizeof(float));
  float *dx = (float *)malloc(B * T * C * sizeof(float));

  // reference
  float *out_ref = (float *)malloc(B * T * C * sizeof(float));
  float *dx_ref = (float *)malloc(B * T * C * sizeof(float));
  float *mean_ref = (float *)malloc(B * T * sizeof(float));
  float *rstd_ref = (float *)malloc(B * T * sizeof(float));
  float *dw_ref = (float *)malloc(C * sizeof(float));
  float *db_ref = (float *)malloc(C * sizeof(float));

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
  fread(mean_ref, sizeof(float), B * T, fp);
  fread(rstd_ref, sizeof(float), B * T, fp);
  fread(dout, sizeof(float), B * T * C, fp);
  fread(dx_ref, sizeof(float), B * T * C, fp);
  fread(dw_ref, sizeof(float), C, fp);
  fread(db_ref, sizeof(float), C, fp);

  layernorm_forward(x, w, b, mean, rstd, out, B, T, C);

  if (check_correct(out_ref, out, B * T * C) == EXIT_FAILURE) {
    printf("check out fail!\n");
    return EXIT_FAILURE;
  }

  if (check_correct(mean_ref, mean, C) == EXIT_FAILURE) {
    printf("check mean failed!\n");
    return EXIT_FAILURE;
  }

  if (check_correct(rstd_ref, rstd, C) == EXIT_FAILURE) {
    printf("check rstd failed!\n");
    return EXIT_FAILURE;
  }

  printf("forward success!\n");

  // initialize dw, db and dx to zeros
  layernorm_backward(dout, dw, db, dx, x, w, b, mean, rstd, B, T, C);

  if (check_correct(dw_ref, dw, C) == EXIT_FAILURE) {
    printf("check dw fail!\n");
    return EXIT_FAILURE;
  }

  if (check_correct(db_ref, db, C) == EXIT_FAILURE) {
    printf("check db fail!\n");
    return EXIT_FAILURE;
  }

  if (check_correct(dx_ref, dx, B * T * C) == EXIT_FAILURE) {
    printf("check dx fail!\n");
    return EXIT_FAILURE;
  }

  printf("backward success!\n");
  return EXIT_SUCCESS;
}
