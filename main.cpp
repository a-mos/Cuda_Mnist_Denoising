#include <iostream>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "kernel.h"

float* img_to_float(unsigned char* pixmap, int w, int h, int c) {
	float* out = (float *)malloc(w * h * c * sizeof(float));
	for (int i = 0; i < w * h * c; ++i) {
		out[i] = pixmap[i] / 255.;
	}
	return out;
}

unsigned char* img_to_uchar(float* pixmap, int w, int h, int c) {
	unsigned char* out = (unsigned char*)malloc(w * h * c * sizeof(unsigned char));
	for (int i = 0; i < w * h * c; ++i) {
		out[i] = pixmap[i] * 255;
	}
	return out;
}

int main(int argc, const char** argv) {

	if (argc < 2) {
		std::cout << "Usage: ./denoiser {input_img_path} [optional -benchmark N]";
		return 1;
	}

	int num_runs_benchmark = 1;
	if (argc > 3) {
		if (strcmp(argv[2], "-benchmark") == 0) {
			num_runs_benchmark = std::stoi(argv[3]);
		}
	}

	int envmap_w, envmap_h, channels;
	auto image_path = argv[1];

	unsigned char* char_buffer = stbi_load(image_path, &envmap_w, &envmap_h, &channels, 1);
	std::cout << "Img shape: " << envmap_h << "x" << envmap_w << "x" << channels << "\n";

	float* img_float = img_to_float(char_buffer, envmap_w, envmap_h, channels);
	float* net_out = Kernel(img_float, envmap_w, envmap_h, num_runs_benchmark);
	unsigned char* result = img_to_uchar(net_out, envmap_w, envmap_h, channels);

	std::string output_name = std::string(image_path);
	std::string suffix = "_denoised";
	output_name.insert(output_name.end() - 4, suffix.begin(), suffix.end());
	stbi_write_png(output_name.c_str(), envmap_w, envmap_h, 1, result, 0);
	
	stbi_image_free(char_buffer);
	free(result);
	free(img_float);
	free(net_out);
}