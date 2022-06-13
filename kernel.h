#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

float* Kernel(float *data, int envmap_w, int envmap_h, int num_runs = 1);