#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include "forwardKernel.h"
#include "backwardKernel.h"
#include "cosweightKernel.h"
#include "dtvtools.h"
#include <ATen/ATen.h>
#include "infer.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cmath>

#include <tuple>
#include <vector>
#include <torch/script.h>
#include <chrono>
#include <Windows.h>
#include <filesystem>
#include <format>
#define PI 3.14159265359
#define imageshape { 1, 1, 100,1600,1600 }
#define sinoshape { 1, 1, 32, 2940, 2304 }
#define sinoshape1 { 1, 1, 32, 2939, 2303 }


std::chrono::high_resolution_clock::time_point print_time_elapsed_and_return_current_time(
    std::chrono::high_resolution_clock::time_point t1, const std::string& event_name) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - t1);
    std::cout << event_name << " - Time taken: " << time_span.count() << " milliseconds" << std::endl;
    return current_time;
}

torch::Tensor load_raw_file_to_tensor(std::string filename, int batch_size, int volume_depth, int volume_height, int volume_width, int deviceId) {
    // open file
    std::ifstream raw_file(filename, std::ios::binary);

    // get file size
    raw_file.seekg(0, std::ios::end);
    std::streampos file_size = raw_file.tellg();
    raw_file.seekg(0, std::ios::beg);

    // read file data into vector
    std::vector<char> data(file_size);
    raw_file.read(data.data(), file_size);

    // create tensor from vector data
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(data.data(), { batch_size, volume_depth, volume_height, volume_width }, options);
    /*std::cout << input[0][0][19][10];*/
    // move tensor to device
    input = input.to(torch::kCUDA, deviceId);
    return input;
}

torch::Tensor load_raw_file_to_cpu(std::string filename, int batch_size, int volume_depth, int volume_height, int volume_width, int deviceId) {
    // open file
    std::ifstream raw_file(filename, std::ios::binary);

    // get file size
    raw_file.seekg(0, std::ios::end);
    std::streampos file_size = raw_file.tellg();
    raw_file.seekg(0, std::ios::beg);

    // read file data into vector
    std::vector<char> data(file_size);
    raw_file.read(data.data(), file_size);

    // create tensor from vector data
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(data.data(), { batch_size, volume_depth, volume_height, volume_width }, options);
    /*std::cout << input[0][0][19][10];*/
    // move tensor to device
    return input;
}

void write_tensor_to_binary_file(torch::Tensor& tensor, const std::string& filename) {
    // get tensor data pointer
    float* data_ptr = tensor.data<float>();
    // write tensor data to binary file
    std::ofstream outfile(filename, std::ios::binary);
    outfile.write((char*)data_ptr, tensor.numel() * sizeof(float));
    outfile.close();
    //data_ptr = NULL;
}


static void print_cuda_use()
{
    size_t free_byte;
    size_t total_byte;

    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db_1 = (total_db - free_db) / 1024.0 / 1024.0;
    std::cout << "Now used GPU memory " << used_db_1 << "  MB\n";
}


torch::Tensor ramp_filter(int projWidth) {
    torch::Tensor filter = torch::ones({ 1, 1, 1, projWidth });
    int mid = std::floor(projWidth / 2);
    for (int i = 0; i < projWidth; ++i) {

        if ((i - mid) % 2 == 0) {
            filter[0][0][0][i] = 0;
        }
        else {
            filter[0][0][0][i] = -0.5 / (M_PI * M_PI * (i - mid) * (i - mid));
        }
        if (i == mid) {
            filter[0][0][0][i] = 1.0 / 8.0;
        }
    }
    filter = filter;
    return filter;
}


torch::Tensor bp(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectMatrix,float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum, int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    volume.zero_();
    int projWidth = 2304;
    int projHeight = 2940;
    torch::Tensor residual = torch::zeros(sinoshape).to(torch::kCUDA, device);
    torch::Tensor result = torch::zeros(imageshape).to(torch::kCUDA, device);
    auto time = print_time_elapsed_and_return_current_time(start_time, "start");
    for (int i = 0; i < 1000; i++)
    {
        forward_F(residual, volume, volumeSize, detectorSize, projectMatrix, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "forward");
        residual.sub_(real);
        result.zero_();
        backward_F(result, residual, volumeSize, detectorSize, projectMatrix,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "backward");
        volume .sub_(result);
    }
    volume = volume.cpu().contiguous();
    std::string filename = "../data/test/highF/8/sirt.raw";
    write_tensor_to_binary_file(volume, filename);
    volume = volume.to(torch::kCUDA, device);
    time = print_time_elapsed_and_return_current_time(time, "save");
    return volume;
}

void fdk(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectVector, float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum, int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID) 
{
    int projWidth = 2303;
    int projHeight = 2939;
    torch::Tensor ramp = ramp_filter(projWidth).to(torch::kCUDA, device);
    torch::nn::functional::Conv2dFuncOptions conv_options;
    conv_options.stride({ 1, 1 }).padding({ 0, static_cast<int>(projWidth / 2) });
    torch::Tensor sino = torch::zeros(sinoshape1).to(torch::kCUDA, device);
    sino.copy_(real);
    real = real.cpu().contiguous();
    sino = sino.view({ 1, 1, AngleNum, projHeight, projWidth });
    sino = cosweight(sino, detectorSize, projectVector, 0);
    sino = sino.view({ 1, 1, AngleNum * projHeight, projWidth });
    sino = torch::nn::functional::conv2d((sino), ramp, conv_options);
    sino = sino.view({ 1, 1, AngleNum, projHeight, projWidth });
    volume.copy_(backward(sino, volumeSize, detectorSize, projectVector,
        volbiasz, dSampleInterval, dSliceInterval,
        sourceRadius, sourceZpos, fBiaz, SID, device));
    /*volume = volume.cpu().contiguous();
    std::string filename = "../data/test/shenzhenhigh/results/1/fdkResults.raw";
    write_tensor_to_binary_file(volume, filename);
    volume = volume.to(torch::kCUDA, device);*/
    real = real.to(torch::kCUDA, device);
}

int main() {
    float volbiasz = -925.0f ;
    float dSampleInterval = 0.125f;
    float dSliceInterval = 0.25f;
    float sourceRadius = 302.8957 ;
    float sourceZpos = -582.5868 ;
    float fBiaz = 4299.0880 ;
    float SID = 1806.7501  ;
    long device = 0;
    const int AngleNum = 32;
    int projWidth = 2304; // Replace with the appropriate value from shenzhenDetectorSize[0]
    int projHeight = 2940;
    const int SystemNum = 114;
    //
    auto start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor volumeSize = torch::tensor({ 1600, 1600, 100 }, torch::kInt).to(torch::kCUDA, device);
    torch::Tensor detectorSize = torch::tensor({ projWidth, projHeight }, torch::kInt).to(torch::kCUDA, device);
    torch::Tensor detectorSize_r = torch::tensor({ 2303, 2939 }, torch::kInt).to(torch::kCUDA, device);
    torch::Tensor projectVector = load_raw_file_to_tensor(
        "../data/test/highF/proj_vec.raw",
        1, 1, AngleNum * SystemNum, 12, device);
    projectVector = projectVector.view({ AngleNum * SystemNum, 12 });
    torch::Tensor projectVector_r = load_raw_file_to_tensor(
        "../data/test/shenzhenhigh/flyTostation/4/proj_vec_r/0001.raw",
        1, 1, AngleNum * SystemNum, 12, device);
    projectVector_r = projectVector_r.view({ AngleNum * SystemNum, 12 });
    //
    torch::Tensor real = load_raw_file_to_tensor("../data/test/highF/8/gM.raw", 1, AngleNum, projHeight, projWidth, device);
    torch::Tensor volume = torch::zeros(imageshape).to(torch::kCUDA, device);
    real = real.view({ 1, 1, AngleNum , projHeight, projWidth });
    
    torch::Tensor real_r = torch::zeros(sinoshape1).to(torch::kCUDA, device);
    rotation(real_r,real,  detectorSize, detectorSize_r, projectVector,projectVector_r, device);
    fdk(real_r, volume, volumeSize, detectorSize_r, projectVector_r, volbiasz, dSampleInterval, dSliceInterval, AngleNum, SystemNum, device, sourceRadius, sourceZpos, fBiaz, SID);
    bp(real, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, AngleNum, SystemNum, device, sourceRadius, sourceZpos, fBiaz, SID);   
  
}
