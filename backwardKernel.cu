#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "backwardKernel.h"

#define BLOCK_X 32
#define BLOCK_Y 32
#define BLOCK_A 1
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

texture<float, cudaTextureType3D, cudaReadModeElementType> sinoTexture;

__global__ void backwardKernel(float* volume, const uint3 volumeSize, const uint2 detectorSize, const float* projectVector, const uint index,const int anglesNum,const float3 volumeCenter, const float2 detectorCenter,
                               const float volbiasz, const float dSampleInterval, const float dSliceInterval, const float sourceRadius, const float sourceZpos, const float fBiaz, const float  SID){
    uint3 volumeIdx = make_uint3(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, blockIdx.z*blockDim.z+threadIdx.z);
    if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y){
        return;
    }

    for(int k=0;k<volumeSize.z;k++){
        float value = 0.0f;
        for(int angleIdx = index;angleIdx < index + BLOCK_A;angleIdx++){
            float3 sourcePosition = make_float3(projectVector[angleIdx*12], projectVector[angleIdx*12+1], projectVector[angleIdx*12+2]);
            float3 detectorPosition = make_float3(projectVector[angleIdx*12+3], projectVector[angleIdx*12+4], projectVector[angleIdx*12+5]);
            float3 u = make_float3(projectVector[angleIdx*12+6], projectVector[angleIdx*12+7], projectVector[angleIdx*12+8]);
            float3 v = make_float3(projectVector[angleIdx*12+9], projectVector[angleIdx*12+10], projectVector[angleIdx*12+11]);
            float3 coordinates = make_float3((volumeCenter.x + volumeIdx.x) * dSampleInterval, (volumeCenter.y + volumeIdx.y) * dSampleInterval,(volumeCenter.z + k) * dSliceInterval + volbiasz);
            float fScale = __fdividef(1.0f, det3(u, v, sourcePosition-coordinates));
            fScale = det3(u, v, sourcePosition-coordinates) == 0 ? 0 : fScale;
            float detectorX = fScale * det3(coordinates-sourcePosition,v,sourcePosition-detectorPosition)-detectorCenter.x;
            float detectorY = fScale * det3(u, coordinates-sourcePosition,sourcePosition-detectorPosition)-detectorCenter.y;
            value +=  tex3D(sinoTexture, detectorX+0.5f, detectorY+0.5f, angleIdx + 0.5f);
        }
        int idx = k * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
        float beamlength = volumeSize.z*dSliceInterval;
        atomicAdd(&volume[idx], value / anglesNum/beamlength );
    }
}

torch::Tensor backward(torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
                        const float volbiasz, const float dSampleInterval, const float dSliceInterval,
                        const float sourceRadius, const float sourceZpos, const float fBiaz, const float  SID,
                        const long device){
    CHECK_INPUT(sino);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");

    int angles = projectVector.size(0);
    auto out = torch::zeros({sino.size(0), 1, _volumeSize[2].item<int>(), _volumeSize[1].item<int>(), _volumeSize[0].item<int>()}).to(sino.device());
    float* outPtr = out.data<float>();
    float* sinoPtr = sino.data<float>();

    cudaSetDevice(device);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    sinoTexture.addressMode[0] = cudaAddressModeBorder;
    sinoTexture.addressMode[1] = cudaAddressModeBorder;
    sinoTexture.addressMode[2] = cudaAddressModeBorder;
    sinoTexture.filterMode = cudaFilterModeLinear;
    sinoTexture.normalized = false;

    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    float2 detectorCenter = make_float2(detectorSize) / -2.0;
    for(int batch = 0;batch < sino.size(0); batch++){
        float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
        float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;

        cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles);
        cudaArray *sinoArray;
        cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, detectorSize.x*sizeof(float), detectorSize.x, detectorSize.y);
        copyParams.dstArray = sinoArray;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.extent = m_extent;
        cudaMemcpy3D(&copyParams);
        cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
        const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, 1);
        for (int angle = 0; angle < angles; angle+=BLOCK_A){
           backwardKernel<<<gridSize, blockSize>>>(outPtrPitch, volumeSize, detectorSize, (float*)projectVector.data<float>(), angle,angles,volumeCenter,detectorCenter,
                                                   volbiasz, dSampleInterval, dSliceInterval, sourceRadius, sourceZpos, fBiaz, SID);
        }
      cudaUnbindTexture(sinoTexture);
      cudaFreeArray(sinoArray);
    }
    return out;
}

__global__ void backwardKernel_F(float* volume, const uint3 volumeSize, const uint2 detectorSize, const float* projectVector, const uint index, const int anglesNum, const float3 volumeCenter, const float2 detectorCenter,
	const float volbiasz, const float dSampleInterval, const float dSliceInterval, const float sourceRadius, const float sourceZpos, const float fBiaz, const float  SID, const uint systemNum) {
	uint3 volumeIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y || volumeIdx.z >= systemNum) {
		return;
	}
	unsigned projectVectorIdx = volumeIdx.z * 12;
	float3 sourcePosition = make_float3(projectVector[projectVectorIdx], projectVector[projectVectorIdx + 1], projectVector[projectVectorIdx + 2]);
	float3 detectorPosition = make_float3(projectVector[projectVectorIdx + 3], projectVector[projectVectorIdx + 4], projectVector[projectVectorIdx + 5]);
	float3 u = make_float3(projectVector[projectVectorIdx + 6], projectVector[projectVectorIdx + 7], projectVector[projectVectorIdx + 8]);
	float3 v = make_float3(projectVector[projectVectorIdx + 9], projectVector[projectVectorIdx + 10], projectVector[projectVectorIdx + 11]);

	for (int k = 0; k < volumeSize.z; k++) {
		float3 coordinates = make_float3((volumeCenter.x + volumeIdx.x) * dSampleInterval, (volumeCenter.y + volumeIdx.y) * dSampleInterval, (volumeCenter.z + k) * dSliceInterval + volbiasz);
		float fScale = __fdividef(1.0f, det3(u, v, sourcePosition - coordinates));
		fScale = det3(u, v, sourcePosition - coordinates) == 0 ? 0 : fScale;
		float detectorX = fScale * det3(coordinates - sourcePosition, v, sourcePosition - detectorPosition) - detectorCenter.x;
		float detectorY = fScale * det3(u, coordinates - sourcePosition, sourcePosition - detectorPosition) - detectorCenter.y;
		float value = tex3D(sinoTexture, detectorX + 0.5f, detectorY + 0.5f, index + 0.5f);
		int idx = k * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
		float beamlength = volumeSize.z*dSliceInterval;
		atomicAdd(&volume[idx], value / anglesNum / systemNum/beamlength);
	}
}

void backward_F(torch::Tensor out, torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
	const float volbiasz, const float dSampleInterval, const float dSliceInterval,
	const float sourceRadius, const float sourceZpos, const float fBiaz, const float SID,
	const int systemNum, const long device) {
	CHECK_INPUT(sino);
	CHECK_INPUT(_volumeSize);
	AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
	CHECK_INPUT(_detectorSize);
	AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
	CHECK_INPUT(projectVector);
	AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");
	int angles = projectVector.size(0) / systemNum;
	float* outPtr = out.data<float>();
	float* sinoPtr = sino.data<float>();

	cudaSetDevice(device);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	sinoTexture.addressMode[0] = cudaAddressModeBorder;
	sinoTexture.addressMode[1] = cudaAddressModeBorder;
	sinoTexture.addressMode[2] = cudaAddressModeBorder;
	sinoTexture.filterMode = cudaFilterModeLinear;
	sinoTexture.normalized = false;

	uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
	uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
	float3 volumeCenter = make_float3(volumeSize) / -2.0;
	float2 detectorCenter = make_float2(detectorSize) / -2.0;

	for (int batch = 0; batch < sino.size(0); batch++) {
		float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
		float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;

		cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles);
		cudaArray* sinoArray;
		cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, detectorSize.x * sizeof(float), detectorSize.x, detectorSize.y);
		copyParams.dstArray = sinoArray;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		copyParams.extent = m_extent;
		cudaMemcpy3D(&copyParams);
		cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

		const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, BLOCK_A);
		const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, systemNum / BLOCK_A );
		auto projVec = projectVector.reshape({ angles, systemNum * 12 });
		for (int angle = 0; angle < angles; angle++) {
			backwardKernel_F << <gridSize, blockSize >> > (outPtrPitch, volumeSize, detectorSize, (float*)projVec[angle].data<float>(), angle, angles, volumeCenter, detectorCenter,
				volbiasz, dSampleInterval, dSliceInterval, sourceRadius, sourceZpos, fBiaz, SID, systemNum);
			cudaDeviceSynchronize();
		}
		cudaUnbindTexture(sinoTexture);
		cudaFreeArray(sinoArray);
	}
}

__global__ void rotationKernal(float* out, const uint2 detectorSize, const uint2 targetdetectorSize, const float* projectVector,
	const float* targetprojectVector, const uint index, const int anglesNum, const float2 detectorCenter, const float2 targetdetectorCenter)
{ 
	uint3 targetdetectorIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (targetdetectorIdx.x >= targetdetectorSize.x || targetdetectorIdx.y >= targetdetectorSize.y ) {
		return;
	}
	float3 sourcePosition = make_float3(projectVector[0], projectVector[1], projectVector[2]);
	float3 detectorPosition = make_float3(projectVector[3], projectVector[4], projectVector[5]);
	float3 u = make_float3(projectVector[6], projectVector[7], projectVector[8]);
	float3 v = make_float3(projectVector[9], projectVector[10], projectVector[11]);
	float3 targetdetectorPosition = make_float3(targetprojectVector[3], targetprojectVector[4], targetprojectVector[5]);
	float3 targetu = make_float3(targetprojectVector[6], targetprojectVector[7], targetprojectVector[8]);
	float3 targetv = make_float3(targetprojectVector[9], targetprojectVector[10], targetprojectVector[11]);
	float targetdetectorX = targetdetectorIdx.x + targetdetectorCenter.x;
	float targetdetectorY = targetdetectorIdx.y + targetdetectorCenter.y;
	float3 coordinates = targetdetectorPosition+ targetdetectorX* targetu+ targetdetectorY* targetv;
	float fScale = 0;
	if ((det3(u, v, sourcePosition - coordinates)) != 0) { fScale = __fdividef(1.0f, det3(u, v, sourcePosition - coordinates)); }
	float detectorX = fScale * det3(coordinates - sourcePosition, v, sourcePosition - detectorPosition) - detectorCenter.x;
	float detectorY = fScale * det3(u, coordinates - sourcePosition, sourcePosition - detectorPosition) - detectorCenter.y;
	float value = tex3D(sinoTexture, detectorX + 0.5f, detectorY + 0.5f, index + 0.5f);
	unsigned sinogramIdx = index * targetdetectorSize.x * targetdetectorSize.y + targetdetectorIdx.y * targetdetectorSize.x + targetdetectorIdx.x;
	atomicAdd(&out[sinogramIdx], value);
}


void rotation(torch::Tensor out, torch::Tensor sino, torch::Tensor _detectorSize, torch::Tensor _detectorSize1, torch::Tensor projectVector,
	torch::Tensor projectVector1, const long device)
{
	CHECK_INPUT(sino);
	CHECK_INPUT(_detectorSize);
	AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 3");
	CHECK_INPUT(_detectorSize1);
	AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
	CHECK_INPUT(projectVector);
	AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");
	int angles = projectVector.size(0);
	out.zero_();
	float* outPtr = out.data<float>();
	float* sinoPtr = sino.data<float>();

	cudaSetDevice(device);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	sinoTexture.addressMode[0] = cudaAddressModeBorder;
	sinoTexture.addressMode[1] = cudaAddressModeBorder;
	sinoTexture.addressMode[2] = cudaAddressModeBorder;
	sinoTexture.filterMode = cudaFilterModeLinear;
	sinoTexture.normalized = false;

	uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
	float2 detectorCenter = make_float2(detectorSize) / -2.0;
	uint2 detectorSize1 = make_uint2(_detectorSize1[0].item<int>(), _detectorSize1[1].item<int>());
	float2 detectorCenter1 = make_float2(detectorSize1) / -2.0;

	for (int batch = 0; batch < sino.size(0); batch++) {
		float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
		float* outPtrPitch = outPtr + detectorSize1.x * detectorSize1.y * angles * batch;

		cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles);
		cudaArray* sinoArray;
		cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, detectorSize.x * sizeof(float), detectorSize.x, detectorSize.y);
		copyParams.dstArray = sinoArray;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		copyParams.extent = m_extent;
		cudaMemcpy3D(&copyParams);
		cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

		const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, BLOCK_A);
		const dim3 gridSize = dim3(detectorSize1.x / BLOCK_X + 1, detectorSize1.y / BLOCK_Y + 1,  BLOCK_A);
		auto projVec = projectVector.reshape({ angles, 12 });
		auto targetprojVec = projectVector1.reshape({ angles, 12 });
		for (int angle = 0; angle < angles; angle++) {
			rotationKernal << <gridSize, blockSize >> > (outPtrPitch, detectorSize, detectorSize1, (float*)projVec[angle].data<float>(), (float*)targetprojVec[angle].data<float>(),
				angle, angles, detectorCenter, detectorCenter1);
			cudaDeviceSynchronize();
		}
		cudaUnbindTexture(sinoTexture);
		cudaFreeArray(sinoArray);
	}
}

