#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const float EPS = 1e-5f;

// make it from -1 to 1
static float randf(void) { return (float)rand() / RAND_MAX * 2.f - 1.f; }

void rmsnorm_forward(float* out, float* x, const float* weight, const int B, const int T, const int C) {
  for (int b = 0; b < B; ++b) {
    // move ptr of x and out
    x += b * T * C;
    out += b * T * C;
    for (int t = 0; t < T; ++t) {
      float* inp_t = x + t * C;
      float* out_t = out + t * C;
      float ss = 0.f; // sum of squares
      for (int i = 0; i < C; ++i) { ss += inp_t[i] * inp_t[i]; }
      ss /= C;
      ss = 1.f / sqrtf(ss + EPS); // (rsqrt)
      for (int i = 0; i < C; ++i) { out_t[i] = inp_t[i] * ss * weight[i]; }
    }
  }
}

int check_tensor(float* ref, float* pred, const int size, const char* label) {
  printf("start to check %s\n", label);
  for (int i = 0; i < size; ++i) {
    if (fabs(ref[i] - pred[i]) > 1e-4) {
      printf("pred: %f and ref: %f\n", pred[i], ref[i]);
      printf("failed!\n");
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

int main(void) {
  const int B = 2;
  const int T = 4;
  const int C = 8;

  float* x = (float*)calloc(B * T * C, sizeof(float));
  float* w = (float*)calloc(C, sizeof(float));
  float* out = (float*)calloc(T * T * C, sizeof(float));
  float* ref = (float*)calloc(T * T * C, sizeof(float));

  // load
  FILE* fp = fopen("rmsnorm.bin", "rb");
  if (fp == NULL) {
    printf("Failed to open rmsnorm.bin!");
    return EXIT_FAILURE;
  }

  fread(x, sizeof(float), B * T * C, fp);
  fread(w, sizeof(float), C, fp);
  fread(ref, sizeof(float), B * T * C, fp);
  fclose(fp);

  rmsnorm_forward(out, x, w, B, T, C);

  if (check_tensor(ref, out, B * T * C, "out") == EXIT_FAILURE) { return EXIT_FAILURE; }
  printf("rmsnorm forward successfully!\n");

  free(x);
  free(w);
  free(out);
  free(ref);
  return EXIT_SUCCESS;
}
