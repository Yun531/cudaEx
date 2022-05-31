#pragma warning(disable:4996)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Windows.h>

#define SIZE 600*400
#define T_SIZE 3*600*400

#define NUM_CPU_THREADS (4)

DS_timer* timer;

#define TIMER_HOST 0
#define TIMER_KERNEL 1
#define TIMER_KERNEL_SH 2
#define TIMER_HtoD 3
#define TIMER_DtoH 4
#define NUM_TIMER (TIMER_DtoH + 1)

void setTimer(void);

void brightFilter(BYTE* sourceBGR, BYTE* outBGR);
void grayFilter(BYTE* sourceBGR, BYTE* outBGR);
void invertFilter(BYTE* sourceBGR, BYTE* outBGR);
void sepiaFilter(BYTE* sourceBGR, BYTE* outBGR);


void main() {
	FILE* infile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\catSample.bmp", "rb");
	FILE* brightfile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\brightResult.bmp", "wb");
	FILE* graytfile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\grayResult.bmp", "wb");
	FILE* invertfile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\invertResult.bmp", "wb");
	FILE* sepiatfile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\sepiaResult.bmp", "wb");

	BITMAPFILEHEADER hf;
	fread(&hf, sizeof(BITMAPFILEHEADER), 1, infile);

	BITMAPINFOHEADER hInfo;
	fread(&hInfo, sizeof(BITMAPINFOHEADER), 1, infile);

	BYTE* Img = (BYTE*)malloc(hInfo.biSizeImage * sizeof(unsigned char));
	BYTE* brightImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));      //결과값 저장 배열
	BYTE* grayImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));
	BYTE* invertImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));
	BYTE* sepiaImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));



	fread(Img, sizeof(unsigned char), hInfo.biSizeImage, infile);

	
	for(int i = 0; i < SIZE; i++) {
		brightFilter(&Img[i * 3], &brightImg[i * 3]);
		grayFilter(&Img[i * 3], &grayImg[i * 3]);
		invertFilter(&Img[i * 3], &invertImg[i * 3]);
		sepiaFilter(&Img[i * 3], &sepiaImg[i * 3]);
	}


	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), brightfile);         //bright파일 저장
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), brightfile);
	fseek(brightfile, hf.bfOffBits, SEEK_SET);
	fwrite(brightImg, sizeof(unsigned char), hInfo.biSizeImage, brightfile);

	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), graytfile);         //gray파일 저장
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), graytfile);
	fseek(graytfile, hf.bfOffBits, SEEK_SET);
	fwrite(grayImg, sizeof(unsigned char), hInfo.biSizeImage, graytfile);

	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), invertfile);         //invert파일 저장
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), invertfile);
	fseek(invertfile, hf.bfOffBits, SEEK_SET);
	fwrite(invertImg, sizeof(unsigned char), hInfo.biSizeImage, invertfile);

	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), sepiatfile);         //gray파일 저장
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), sepiatfile);
	fseek(sepiatfile, hf.bfOffBits, SEEK_SET);
	fwrite(sepiaImg, sizeof(unsigned char), hInfo.biSizeImage, sepiatfile);

	free(Img); free(brightImg); free(grayImg); free(invertImg); free(sepiaImg);
	
	fclose(infile);
	fclose(brightfile);
	fclose(graytfile);
	fclose(invertfile);
	fclose(sepiatfile);

}



void brightFilter(BYTE* sourceBGR, BYTE* outBGR) {
	outBGR[0] = (sourceBGR[0] + sourceBGR[0] * .2f) > 255 ? 255 : (sourceBGR[0] + sourceBGR[0] * .2f);
	outBGR[1] = (sourceBGR[1] + sourceBGR[1] * .2f) > 255 ? 255 : (sourceBGR[1] + sourceBGR[1] * .2f);
	outBGR[2] = (sourceBGR[2] + sourceBGR[2] * .2f) > 255 ? 255 : (sourceBGR[2] + sourceBGR[2] * .2f);
}

void grayFilter(BYTE* sourceBGR, BYTE* outBGR) {
	BYTE gray = sourceBGR[0] * .114f + sourceBGR[1] * .587f + sourceBGR[2] * .299f;

	//outBGR[0] = gray;
	//outBGR[1] = gray;
	//outBGR[2] = gray;

	outBGR[0] = sourceBGR[0];   //노
	outBGR[1] = sourceBGR[1];   //분홍
	outBGR[2] = sourceBGR[2];   //하늘
}

void invertFilter(BYTE* sourceBGR, BYTE* outBGR) {
	outBGR[0] = 255 - sourceBGR[0];
	outBGR[1] = 255 - sourceBGR[1];
	outBGR[2] = 255 - sourceBGR[2];
}

void sepiaFilter(BYTE* sourceBGR, BYTE* outBGR) {
	BYTE B_temp = sourceBGR[0] * .131f + sourceBGR[1] * .534f + sourceBGR[2] * .272f;
	BYTE G_temp = sourceBGR[0] * .168f + sourceBGR[1] * .686f + sourceBGR[2] * .349f;
	BYTE R_temp = sourceBGR[0] * .189f + sourceBGR[1] * .769f + sourceBGR[2] * .393f;

	outBGR[0] = (B_temp > 255) ? 255 : B_temp;
	outBGR[1] = (G_temp > 255) ? 255 : G_temp;
	outBGR[2] = (R_temp > 255) ? 255 : R_temp;
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