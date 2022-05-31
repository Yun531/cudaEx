﻿#pragma warning(disable:4996)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Windows.h>

#define NUM_CPU_THREADS (4)

#define ROW_SIZE (1024)
#define K_SIZE (512)
#define COL_SIZE (1024)
#define DATA_SIZE (8)  // 4, 2의 경우 active block은 많지만 block당 thread의 수가 적어 성능이 하락한다

#define WORK_LOAD (10)

#define MAT_SIZE_A (ROW_SIZE * K_SIZE)
#define MAT_SIZE_B (K_SIZE * COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE * COL_SIZE)

DS_timer* timer;

#define TIMER_HOST 0
#define TIMER_KERNEL 1
#define TIMER_KERNEL_SH 2
#define TIMER_HtoD 3
#define TIMER_DtoH 4
#define NUM_TIMER (TIMER_DtoH + 1)

void setTimer(void);


//#define dMemAlloc(_P, _type, _size) cudaMalloc(&_P, sizeof(_type)*_size);


/*__global__ void matMul_kernel(float* _A, float* _B, float* _C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * blockDim.x * gridDim.x + col;

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++) {
		for (int i = 0; i < WORK_LOAD; i++) {
			_C[index] += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
		}
	}
}*/


/*__global__ void matMul_kernel_shared(float* _A, float* _B, float* _C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * blockDim.x * gridDim.x + col;

	__shared__ float sA[DATA_SIZE][K_SIZE];
	__shared__ float sB[K_SIZE][DATA_SIZE];
	__shared__ float sC[DATA_SIZE][DATA_SIZE];

	for (int k = 0; k < K_SIZE; k++) {
		sA[row % DATA_SIZE][k] = _A[row * K_SIZE + k];
		sB[k][col % DATA_SIZE] = _B[col + k * COL_SIZE];
	}

	__syncthreads();

	sC[row % DATA_SIZE][col % DATA_SIZE] = 0;
	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++) {
		for (int i = 0; i < WORK_LOAD; i++) {
			sC[row % DATA_SIZE][col % DATA_SIZE] += sA[row % DATA_SIZE][k] * sB[k][col % DATA_SIZE];
		}
	}
	_C[index] = sC[row % DATA_SIZE][col % DATA_SIZE];

}*/

/*void main(void) {
	timer = NULL; setTimer();

	float* dA, * dB, * dC;
	dA = dB = dC = NULL;


	dMemAlloc(dA, float, MAT_SIZE_A);
	dMemAlloc(dB, float, MAT_SIZE_B);
	dMemAlloc(dC, float, MAT_SIZE_C);


	timer->onTimer(TIMER_HOST);
	for (int r = 0; r < ROW_SIZE; r++) {
		for (int c = 0; c < COL_SIZE; c++) {
			for (int k = 0; k < K_SIZE; k++) {
				for (int i = 0; i < WORK_LOAD; i++) {
					hostC[r][c] += A[r][k] * B[k][c];
				}
			}
		}
	}
	timer->offTimer(TIMER_HOST);

	timer->onTimer(TIMER_HtoD);
	cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
	timer->offTimer(TIMER_HtoD);


	dim3 gridDim1(COL_SIZE / 32, ROW_SIZE / 32);
	dim3 blockDim1(32, 32);
	dim3 gridDim2(COL_SIZE / DATA_SIZE, ROW_SIZE / DATA_SIZE);
	dim3 blockDim2(DATA_SIZE, DATA_SIZE);

	timer->onTimer(TIMER_KERNEL);
	matMul_kernel << <gridDim1, blockDim1 >> > (dA, dB, dC);
	cudaThreadSynchronize();
	timer->offTimer(TIMER_KERNEL);

	timer->onTimer(TIMER_KERNEL_SH);
	matMul_kernel_shared << <gridDim2, blockDim2 >> > (dA, dB, dC);
	cudaThreadSynchronize();
	timer->offTimer(TIMER_KERNEL_SH);

	timer->onTimer(TIMER_DtoH);
	cudaMemcpy(deviceC, dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
	timer->offTimer(TIMER_DtoH);

	bool isCorrect = true;

	float* pHostC = &hostC[0][0];
	float* pDeviceC = &deviceC[0][0];

	for (int i = 0; i < MAT_SIZE_C; i++) {
		if (pHostC[i] != pDeviceC[i]) {
			printf("[%d] %.2f, %.2f\n", i, pHostC[i], pDeviceC[i]);
			isCorrect = false;
			break;
		}
	}

	if (isCorrect) {
		printf("Result is correct!\n");
	}
	else {
		printf("Result is not correct!!!!!!\n");
	}

	timer->printTimer();
	if (timer != NULL) {
		delete timer;
	}



}*/

#define T_SIZE 3*600*400

void main() {
	FILE* infile = fopen("catSample.bmp", "rb");
	FILE* outfile = fopen("Lenna_small.raw", "wb");

	BITMAPFILEHEADER hf;
	fread(&hf, sizeof(BITMAPFILEHEADER), 1, infile);

	BITMAPINFOHEADER hInfo;
	fread(&hInfo, sizeof(BITMAPINFOHEADER), 1, infile);

	BYTE* lpImg = (BYTE*)malloc(hInfo.biSizeImage * sizeof(unsigned char));
	BYTE* lpOutImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));

	fread(lpImg, sizeof(unsigned char), hInfo.biSizeImage, infile);

	BGRtoRGB(lpImg, lpOutImg, T_SIZE);

	change(lpOutImg);

	fwrite(lpOutImg, sizeof(unsigned char), T_SIZE, outfile);

	fclose(infile);
	fclose(outfile);

}

void BGRtoRGB(BYTE* BGR, BYTE* RGB, unsigned int Size) // RGB로 변환 함수
{
	for (unsigned int i = 0; i < Size; i = i + 3)
	{
		RGB[i] = BGR[i + 2];
		RGB[i + 1] = BGR[i + 1];
		RGB[i + 2] = BGR[i];
	}
}

void change(BYTE* Image)    // 상하 반전 함수
{
	unsigned int i, j, ch;
	for (i = 0; i < 256 / 2; i++)

		for (j = 0; j < 768; j++)
		{
			ch = Image[i * 768 + j];
			Image[i * 768 + j] = Image[(256 - i - 1) * 768 + j];
			Image[(256 - i - 1) * 768 + j] = ch;

		}

}



void setTimer(void) {
	timer = new DS_timer(NUM_TIMER);

	timer->initTimers();
	timer->setTimerName(TIMER_HOST, "CPU code");
	timer->setTimerName(TIMER_KERNEL, "Kernel launch");
	timer->setTimerName(TIMER_KERNEL_SH, "Kernel launch (shared ver)");
	timer->setTimerName(TIMER_HtoD, "[Data transter] host->device");
	timer->setTimerName(TIMER_DtoH, "[Data transfer] device->host");
}