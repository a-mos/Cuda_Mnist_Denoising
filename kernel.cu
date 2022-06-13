#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define BLOCKSIZE_2D 16
#define BLOCKSIZE_1D 1024

__global__ void Conv2D(float* image, float* kernel, float* bias, float* out, int envmap_w, int envmap_h, int k_size, int channels_in, int channels_out, int activation_mode = 0) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < envmap_w && y < envmap_h) {
		for (int c_out = 0; c_out < channels_out; ++c_out) {
			float sum = bias[c_out];
			for (int k_x = -k_size / 2; k_x <= k_size / 2; ++k_x) {
				for (int k_y = -k_size / 2; k_y <= k_size / 2; ++k_y) {
					for (int c_in = 0; c_in < channels_in; ++c_in) {
						float img_val = (x + k_x >= 0 && x + k_x < envmap_w && y + k_y >= 0 && y + k_y < envmap_h) ? image[((x + k_x) * envmap_w + (y + k_y)) * channels_in + c_in] : 0;
						float kernel_val = kernel[(((k_size / 2 + k_x) * k_size + (k_size / 2 + k_y)) * channels_in + c_in) * channels_out + c_out];
						sum += img_val * kernel_val;
					}
				}
			}
			if (activation_mode == 0) {
				// No activation
				out[(x * envmap_w + y) * channels_out + c_out] = sum;
			} else if (activation_mode == 1) {
				// ReLU
				out[(x * envmap_w + y) * channels_out + c_out] = max(sum, 0.0f);
			} else {
				// Sigmoid
				out[(x * envmap_w + y) * channels_out + c_out] = 1. / (1 + exp(-sum));
			}
		}
	}
}

__global__ void ConvShared2D(float* image, float* kernel, float* bias, float* out, int envmap_w, int envmap_h, int k_size, int channels_in, int channels_out, int activation_mode = 0) {
	int x = threadIdx.x + blockIdx.x * (BLOCKSIZE_2D - k_size + 1);
	int y = threadIdx.y + blockIdx.y * (BLOCKSIZE_2D - k_size + 1);

	extern __shared__ float shm[];

	int xx = x - k_size / 2;
	int yy = y - k_size / 2;

	for (int c_in = 0; c_in < channels_in; ++c_in) {
		if (xx < envmap_w && xx >= 0 && yy < envmap_h && yy >= 0) {
			shm[(threadIdx.x * BLOCKSIZE_2D + threadIdx.y) * channels_in + c_in] = image[(xx * envmap_w + yy) * channels_in + c_in];
		}
		else {
			shm[(threadIdx.x * BLOCKSIZE_2D + threadIdx.y) * channels_in + c_in] = 0;
		}
	}

	__syncthreads();

	if (x < envmap_w && y < envmap_h && threadIdx.x < BLOCKSIZE_2D - k_size + 1 && threadIdx.y < BLOCKSIZE_2D - k_size + 1) {
		for (int c_out = 0; c_out < channels_out; ++c_out) {
			float sum = bias[c_out];
			for (int k_x = -k_size / 2; k_x <= k_size / 2; ++k_x) {
				for (int k_y = -k_size / 2; k_y <= k_size / 2; ++k_y) {
					for (int c_in = 0; c_in < channels_in; ++c_in) {
						int xxx = 1 + threadIdx.x + k_x;
						int yyy = 1 + threadIdx.y + k_y;
						float img_val = (xxx >= 0 && xxx < BLOCKSIZE_2D && yyy >= 0 && yyy < BLOCKSIZE_2D) ? shm[(xxx * BLOCKSIZE_2D + yyy) * channels_in + c_in] : 0;
						float kernel_val = kernel[(((k_size / 2 + k_x) * k_size + (k_size / 2 + k_y)) * channels_in + c_in) * channels_out + c_out];
						sum += img_val * kernel_val;
					}
				}
			}
			if (activation_mode == 0) {
				// No activation
				out[(x * envmap_w + y) * channels_out + c_out] = sum;
			}
			else if (activation_mode == 1) {
				// ReLU
				out[(x * envmap_w + y) * channels_out + c_out] = max(sum, 0.0f);
			}
			else {
				// Sigmoid
				out[(x * envmap_w + y) * channels_out + c_out] = 1. / (1 + exp(-sum));
			}
		}
	}
}


__global__ void ReLU(float* image, int size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < size && image[x] < 0) {
		image[x] = 0;
	}
}

__global__ void Sigmoid(float* image, int size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < size) {
		image[x] = 1. / (1 + exp(-image[x]));
	}
}

__global__ void MaxPool2D(float* image, float* out, int stride, int envmap_w, int envmap_h, int channels) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < envmap_w / stride && y < envmap_h / stride) {
		int xx = stride * x;
		int yy = stride * y;
		for (int c_out = 0; c_out < channels; ++c_out) {
			float m = 0;
			for (int k_x = 0; k_x < stride; ++k_x) {
				for (int k_y = 0; k_y < stride; ++k_y) {
					m = max(m, image[((xx + k_x) * envmap_w + (yy + k_y)) * channels + c_out]);
				}
			}
			out[(x * (envmap_w / stride) + y) * channels + c_out] = m;
		}
	}
}

__global__ void ConvTransposed2D(float* image, float* kernel, float* bias, float* out, int envmap_w, int envmap_h, int k_size, int stride, int channels_in, int channels_out, int activation_mode = 0) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < envmap_w && y < envmap_h) {
		for (int c_out = 0; c_out < channels_out; ++c_out) {
			float sum = bias[c_out];
			for (int k_x = -k_size / 2; k_x <= k_size / 2; ++k_x) {
				for (int k_y = -k_size / 2; k_y <= k_size / 2; ++k_y) {
					for (int c_in = 0; c_in < channels_in; ++c_in) {
						int xs = x + k_x - k_size / 2;
						int ys = y + k_y - k_size / 2;
						float img_val = (xs >= 0 && xs < envmap_w && ys >= 0 && ys < envmap_h && !(xs % stride) && !(ys % stride)) ? image[((xs / stride) * (envmap_w / stride) + (ys / stride)) * channels_in + c_in] : 0;
						float kernel_val = kernel[(((k_size / 2 + k_x) * k_size + (k_size / 2 + k_y)) * channels_out + c_out) * channels_in + c_in];
						sum += img_val * kernel_val;
					}
				}
			}
			if (activation_mode == 0) {
				// No activation
				out[(x * envmap_w + y) * channels_out + c_out] = sum;
			} else if (activation_mode == 1) {
				// ReLU
				out[(x * envmap_w + y) * channels_out + c_out] = max(sum, 0.0f);
			} else {
				// Sigmoid
				out[(x * envmap_w + y) * channels_out + c_out] = 1. / (1 + exp(-sum));
			}
		}
	}
}


__global__ void ConvSharedTransposed2D(float* image, float* kernel, float* bias, float* out, int envmap_w, int envmap_h, int k_size, int stride, int channels_in, int channels_out, int activation_mode = 0) {
	int x = threadIdx.x + blockIdx.x * (BLOCKSIZE_2D - k_size + 1);
	int y = threadIdx.y + blockIdx.y * (BLOCKSIZE_2D - k_size + 1);

	extern __shared__ float shm[];

	int xx = x - k_size / 2 - 1;
	int yy = y - k_size / 2 - 1;

	for (int c_in = 0; c_in < channels_in; ++c_in) {
		if (xx < envmap_w && xx >= 0 && yy < envmap_h && yy >= 0 && !(xx % stride) && !(yy % stride)) {
			shm[(threadIdx.x * BLOCKSIZE_2D + threadIdx.y) * channels_in + c_in] = image[((xx / stride) * (envmap_w / stride) + (yy / stride)) * channels_in + c_in];
		}
		else {
			shm[(threadIdx.x * BLOCKSIZE_2D + threadIdx.y) * channels_in + c_in] = 0;
		}
	}

	__syncthreads();

	if (x < envmap_w && y < envmap_h && threadIdx.x < BLOCKSIZE_2D - k_size + 1 && threadIdx.y < BLOCKSIZE_2D - k_size + 1) {
		for (int c_out = 0; c_out < channels_out; ++c_out) {
			float sum = bias[c_out];
			for (int k_x = -k_size / 2; k_x <= k_size / 2; ++k_x) {
				for (int k_y = -k_size / 2; k_y <= k_size / 2; ++k_y) {
					for (int c_in = 0; c_in < channels_in; ++c_in) {
						int xxx = 1 + threadIdx.x + k_x;
						int yyy = 1 + threadIdx.y + k_y;
						float img_val = (xxx >= 0 && xxx < BLOCKSIZE_2D && yyy >= 0 && yyy < BLOCKSIZE_2D) ? shm[(xxx * BLOCKSIZE_2D + yyy) * channels_in + c_in] : 0;
						float kernel_val = kernel[(((k_size / 2 + k_x) * k_size + (k_size / 2 + k_y)) * channels_out + c_out) * channels_in + c_in];
						sum += img_val * kernel_val;
					}
				}
			}
			if (activation_mode == 0) {
				// No activation
				out[(x * envmap_w + y) * channels_out + c_out] = sum;
			}
			else if (activation_mode == 1) {
				// ReLU
				out[(x * envmap_w + y) * channels_out + c_out] = max(sum, 0.0f);
			}
			else {
				// Sigmoid
				out[(x * envmap_w + y) * channels_out + c_out] = 1. / (1 + exp(-sum));
			}
		}
	}
}


float* Kernel(float* data, int envmap_w, int envmap_h, int num_runs = 1) {

	// Padding in shared convs is 1 (ksize/2)
	const dim3 dimGrid(ceil((float)(envmap_w) / (BLOCKSIZE_2D - 2)), ceil((float)(envmap_h) / (BLOCKSIZE_2D - 2)));
	const dim3 dimBlock(BLOCKSIZE_2D, BLOCKSIZE_2D);
	int channels_hidden = 32;

	FILE* f = fopen("model_weights.bin", "rb");
	float *conv1_gpu, *conv2_gpu, *conv3_gpu, *bias1_gpu, *bias2_gpu, *bias3_gpu;
	float *conv_trans1_gpu, *conv_trans2_gpu, *bias_trans1_gpu, *bias_trans2_gpu;
	
	float* result_cpu = (float*)malloc(sizeof(float) * envmap_w * envmap_h);
	float* conv1 = (float*)malloc(3 * 3 * 1 * channels_hidden * sizeof(float));
	float* bias1 = (float*)malloc(channels_hidden * sizeof(float));
	float* conv2 = (float*)malloc(3 * 3 * channels_hidden * channels_hidden * sizeof(float));
	float* bias2 = (float*)malloc(channels_hidden * sizeof(float));
	float* conv3 = (float*)malloc(3 * 3 * channels_hidden * 1 * sizeof(float));
	float* bias3 = (float*)malloc(1 * sizeof(float));
	float* conv_trans1 = (float*)malloc(3 * 3 * channels_hidden * channels_hidden * sizeof(float));
	float* bias_trans1 = (float*)malloc(channels_hidden * sizeof(float));
	float* conv_trans2 = (float*)malloc(3 * 3 * channels_hidden * channels_hidden * sizeof(float));
	float* bias_trans2 = (float*)malloc(channels_hidden * sizeof(float));

	fread(conv1, sizeof(float), 3 * 3 * 1 * channels_hidden, f);
	fread(bias1, sizeof(float), channels_hidden, f);
	fread(conv2, sizeof(float), 3 * 3 * channels_hidden * channels_hidden, f);
	fread(bias2, sizeof(float), channels_hidden, f);
	fread(conv_trans1, sizeof(float), 3 * 3 * channels_hidden * channels_hidden, f);
	fread(bias_trans1, sizeof(float), channels_hidden, f);
	fread(conv_trans2, sizeof(float), 3 * 3 * channels_hidden * channels_hidden, f);
	fread(bias_trans2, sizeof(float), channels_hidden, f);
	fread(conv3, sizeof(float), 3 * 3 * channels_hidden * 1, f);
	fread(bias3, sizeof(float), 1, f);

	cudaMalloc(&conv1_gpu, 3 * 3 * 1 * channels_hidden * sizeof(float));
	cudaMalloc(&conv2_gpu, 3 * 3 * channels_hidden * channels_hidden * sizeof(float));
	cudaMalloc(&conv3_gpu, 3 * 3 * channels_hidden * 1 * sizeof(float));
	cudaMalloc(&bias1_gpu, channels_hidden * sizeof(float));
	cudaMalloc(&bias2_gpu, channels_hidden * sizeof(float));
	cudaMalloc(&bias3_gpu, 1 * sizeof(float));
	cudaMalloc(&conv_trans1_gpu, 3 * 3 * channels_hidden * channels_hidden * sizeof(float));
	cudaMalloc(&conv_trans2_gpu, 3 * 3 * channels_hidden * channels_hidden * sizeof(float));
	cudaMalloc(&bias_trans1_gpu, channels_hidden * sizeof(float));
	cudaMalloc(&bias_trans2_gpu, channels_hidden * sizeof(float));

	float *conv_result1_gpu, *conv_result2_gpu, *conv_result3_gpu, *conv_trans_result1_gpu, *conv_trans_result2_gpu;
	float* maxpool_result1_gpu, * maxpool_result2_gpu, * maxpool_result3_gpu, *image_gpu;

	cudaMalloc(&image_gpu, sizeof(float) * envmap_w * envmap_h);
	cudaMalloc(&conv_result1_gpu, sizeof(float) * channels_hidden * envmap_w * envmap_h);
	cudaMalloc(&maxpool_result1_gpu, sizeof(float) * channels_hidden * (envmap_w / 2) * (envmap_h / 2));
	cudaMalloc(&conv_result2_gpu, sizeof(float) * channels_hidden * (envmap_w / 2) * (envmap_h / 2));
	cudaMalloc(&maxpool_result2_gpu, sizeof(float) * channels_hidden * (envmap_w / 4) * (envmap_h / 4));
	cudaMalloc(&conv_trans_result1_gpu, sizeof(float) * channels_hidden * (envmap_w / 2) * (envmap_h / 2));
	cudaMalloc(&conv_trans_result2_gpu, sizeof(float) * channels_hidden * envmap_w * envmap_h);
	cudaMalloc(&conv_result3_gpu, sizeof(float) * envmap_w * envmap_h);

	const auto start_with_copy = std::chrono::high_resolution_clock::now();
	cudaMemcpy(image_gpu, data, sizeof(float) * envmap_w * envmap_h, cudaMemcpyHostToDevice);
	cudaMemcpy(conv1_gpu, conv1, 3 * 3 * 1 * channels_hidden * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(conv2_gpu, conv2, 3 * 3 * channels_hidden * channels_hidden * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(conv3_gpu, conv3, 3 * 3 * channels_hidden * 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias1_gpu, bias1, channels_hidden * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias2_gpu, bias2, channels_hidden * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias3_gpu, bias3, 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(conv_trans1_gpu, conv_trans1, 3 * 3 * channels_hidden * channels_hidden * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(conv_trans2_gpu, conv_trans2, 3 * 3 * channels_hidden * channels_hidden * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias_trans1_gpu, bias_trans1, channels_hidden * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias_trans2_gpu, bias_trans2, channels_hidden * sizeof(float), cudaMemcpyHostToDevice);
	const auto start_forward = std::chrono::high_resolution_clock::now();
	cudaDeviceSynchronize();

	for (int retry = 0; retry < num_runs; ++retry) {
		//No layer merging with activations
		//Conv2D <<<dimGrid, dimBlock>>> (image_gpu, conv1_gpu, bias1_gpu, conv_result1_gpu, envmap_w, envmap_h, 3, 1, channels_hidden);
		//ReLU <<<(int)ceil(envmap_w * envmap_h * channels_hidden / (float)BLOCKSIZE_1D), BLOCKSIZE_1D >>> (conv_result1_gpu, envmap_w * envmap_h * channels_hidden);
		//MaxPool2D << <dimGrid, dimBlock >> > (conv_result1_gpu, maxpool_result1_gpu, 2, envmap_w, envmap_h, channels_hidden);
		//Conv2D << <dimGrid, dimBlock >> > (maxpool_result1_gpu, conv2_gpu, bias2_gpu, conv_result2_gpu, envmap_w / 2, envmap_h / 2, 3, channels_hidden, channels_hidden);
		//ReLU << <(int)ceil(envmap_w * envmap_h * (channels_hidden / 4) / (float)BLOCKSIZE_1D), BLOCKSIZE_1D >> > (conv_result2_gpu, envmap_w * envmap_h * (channels_hidden / 4));
		//MaxPool2D << <dimGrid, dimBlock >> > (conv_result2_gpu, maxpool_result2_gpu, 2, envmap_w / 2, envmap_h / 2, channels_hidden);
		//ConvTransposed2D << <dimGrid, dimBlock >> > (maxpool_result2_gpu, conv_trans1_gpu, bias_trans1_gpu, conv_trans_result1_gpu, envmap_w / 2, envmap_h / 2, 3, 2, channels_hidden, channels_hidden);
		//ReLU << <(int)ceil(envmap_w * envmap_h * (channels_hidden / 4) / (float)BLOCKSIZE_1D), BLOCKSIZE_1D >> > (conv_trans_result1_gpu, envmap_w * envmap_h * (channels_hidden / 4));
		//ConvTransposed2D << <dimGrid, dimBlock >> > (conv_trans_result1_gpu, conv_trans2_gpu, bias_trans2_gpu, conv_trans_result2_gpu, envmap_w, envmap_h, 3, 2, channels_hidden, channels_hidden);
		//ReLU << <(int)ceil(envmap_w * envmap_h * channels_hidden / (float)BLOCKSIZE_1D), BLOCKSIZE_1D >> > (conv_trans_result2_gpu, envmap_w * envmap_h * channels_hidden);
		//Conv2D << <dimGrid, dimBlock >> > (conv_trans_result2_gpu, conv3_gpu, bias3_gpu, conv_result3_gpu, envmap_w, envmap_h, 3, channels_hidden, 1);
		//Sigmoid << <(int)ceil(envmap_w * envmap_h / (float)BLOCKSIZE_1D), BLOCKSIZE_1D >> > (conv_result3_gpu, envmap_w * envmap_h * 1);

		//With layer merging
		//Conv2D << <dimGrid, dimBlock>> > (image_gpu, conv1_gpu, bias1_gpu, conv_result1_gpu, envmap_w, envmap_h, 3, 1, channels_hidden, 1);
		//MaxPool2D << <dimGrid, dimBlock >> > (conv_result1_gpu, maxpool_result1_gpu, 2, envmap_w, envmap_h, channels_hidden);
		//Conv2D << <dimGrid, dimBlock>> > (maxpool_result1_gpu, conv2_gpu, bias2_gpu, conv_result2_gpu, envmap_w / 2, envmap_h / 2, 3, channels_hidden, channels_hidden, 1);
		//MaxPool2D << <dimGrid, dimBlock >> > (conv_result2_gpu, maxpool_result2_gpu, 2, envmap_w / 2, envmap_h / 2, channels_hidden);
		//ConvTransposed2D << <dimGrid, dimBlock>> > (maxpool_result2_gpu, conv_trans1_gpu, bias_trans1_gpu, conv_trans_result1_gpu, envmap_w / 2, envmap_h / 2, 3, 2, channels_hidden, channels_hidden, 1);
		//ConvTransposed2D << <dimGrid, dimBlock>> > (conv_trans_result1_gpu, conv_trans2_gpu, bias_trans2_gpu, conv_trans_result2_gpu, envmap_w, envmap_h, 3, 2, channels_hidden, channels_hidden, 1);
		//Conv2D << <dimGrid, dimBlock>> > (conv_trans_result2_gpu, conv3_gpu, bias3_gpu, conv_result3_gpu, envmap_w, envmap_h, 3, channels_hidden, 1, 2);

		//With layer merging and shared memory
		ConvShared2D << <dimGrid, dimBlock, BLOCKSIZE_2D* BLOCKSIZE_2D * 1 * sizeof(float) >> > (image_gpu, conv1_gpu, bias1_gpu, conv_result1_gpu, envmap_w, envmap_h, 3, 1, channels_hidden, 1);
		MaxPool2D << <dimGrid, dimBlock >> > (conv_result1_gpu, maxpool_result1_gpu, 2, envmap_w, envmap_h, channels_hidden);
		ConvShared2D << <dimGrid, dimBlock, BLOCKSIZE_2D* BLOCKSIZE_2D * 32 * sizeof(float) >> > (maxpool_result1_gpu, conv2_gpu, bias2_gpu, conv_result2_gpu, envmap_w / 2, envmap_h / 2, 3, channels_hidden, channels_hidden, 1);
		MaxPool2D << <dimGrid, dimBlock >> > (conv_result2_gpu, maxpool_result2_gpu, 2, envmap_w / 2, envmap_h / 2, channels_hidden);
		ConvSharedTransposed2D << <dimGrid, dimBlock, BLOCKSIZE_2D * BLOCKSIZE_2D * 32 * sizeof(float) >> > (maxpool_result2_gpu, conv_trans1_gpu, bias_trans1_gpu, conv_trans_result1_gpu, envmap_w / 2, envmap_h / 2, 3, 2, channels_hidden, channels_hidden, 1);
		ConvSharedTransposed2D << <dimGrid, dimBlock, BLOCKSIZE_2D* BLOCKSIZE_2D * 32 * sizeof(float) >> > (conv_trans_result1_gpu, conv_trans2_gpu, bias_trans2_gpu, conv_trans_result2_gpu, envmap_w, envmap_h, 3, 2, channels_hidden, channels_hidden, 1);
		ConvShared2D << <dimGrid, dimBlock, BLOCKSIZE_2D * BLOCKSIZE_2D * 32 * sizeof(float) >> > (conv_trans_result2_gpu, conv3_gpu, bias3_gpu, conv_result3_gpu, envmap_w, envmap_h, 3, channels_hidden, 1, 2);
	}

	cudaDeviceSynchronize();

	const auto stop_forward = std::chrono::high_resolution_clock::now();
	cudaMemcpy(result_cpu, conv_result3_gpu, sizeof(float) * 1 * envmap_w * envmap_h, cudaMemcpyDeviceToHost);
	const auto end_with_copy = std::chrono::high_resolution_clock::now();

	double elapsed_time_wc = std::chrono::duration<double, std::milli>(stop_forward - start_forward).count();
	double elapsed_time_total = std::chrono::duration<double, std::milli>(end_with_copy - start_with_copy).count();

	if (num_runs > 1) {
		std::cout << "Num runs: " << num_runs << "\n";
		std::cout << "Total GPU time: " << elapsed_time_wc << " ms\n";
		std::cout << "AVG one forward pass GPU time: " << elapsed_time_wc / num_runs << " ms\n";
	} else {
		std::cout << "Forward pass GPU time (without copy): " << elapsed_time_wc << " ms\n";
		std::cout << "Forward pass GPU time (with copy): " << elapsed_time_total << " ms\n";
		std::cout << "GPU Copy time: " << elapsed_time_total - elapsed_time_wc << " ms\n";
	}

	free(conv1);
	free(conv2);
	free(conv3);
	free(bias1);
	free(bias2);
	free(bias3);
	free(conv_trans1);
	free(conv_trans2);
	free(bias_trans1);
	free(bias_trans2);

	cudaFree(conv1_gpu);
	cudaFree(conv2_gpu);
	cudaFree(conv3_gpu);
	cudaFree(bias1_gpu);
	cudaFree(bias2_gpu);
	cudaFree(bias3_gpu);
	cudaFree(conv_trans1_gpu);
	cudaFree(conv_trans2_gpu);
	cudaFree(bias_trans1_gpu);
	cudaFree(bias_trans2_gpu);

	cudaFree(image_gpu);
	cudaFree(conv_result1_gpu);
	cudaFree(conv_result2_gpu);
	cudaFree(conv_result3_gpu);
	cudaFree(maxpool_result1_gpu);
	cudaFree(maxpool_result2_gpu);
	cudaFree(conv_trans_result1_gpu);
	cudaFree(conv_trans_result2_gpu);

	return result_cpu;
}