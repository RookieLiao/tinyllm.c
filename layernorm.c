#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

  int B = 2; // batch
  int T = 3;
  int C = 4;

  float *x = (float *)malloc(B * T * C * sizeof(float));
  float *w = (float *)malloc(C * sizeof(float));
  float *b = (float *)malloc(C * sizeof(float));
  float *out = (float *)malloc(B * T * C * sizeof(float));
  float *mean = (float *)malloc(B * T * sizeof(float));
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
  fread(out, sizeof(float), B * T * C, fp);
}
