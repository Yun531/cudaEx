#pragma warning(disable:4996)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Windows.h>

//#define SIZE 600*400*4*4
//#define T_SIZE 3*SIZE

#define NUM_CPU_THREADS (8)

DS_timer* timer;

#define TIMER_HOST 0
#define TIMER_HOST_OPENMP 1
#define TIMER_KERNEL 2
#define TIMER_KERNEL_SH 3
#define TIMER_HtoD 4
#define TIMER_DtoH 5
#define NUM_TIMER 6

#define dMemAlloc(_P, _type, _size) cudaMalloc(&_P, sizeof(_type)*_size);


void setTimer(void);

void brightFilter(BYTE* sourceBGR, BYTE* outBGR);
void darkFilter(BYTE* sourceBGR, BYTE* outBGR);
void grayFilter(BYTE* sourceBGR, BYTE* outBGR);
void invertFilter(BYTE* sourceBGR, BYTE* outBGR);


__global__ void imageFilter(unsigned char* dImg, unsigned char* dBrightImg, unsigned char* dDarkImg, unsigned char* dGrayImg, unsigned char* dInvertImg, int SIZE) {
	int index = blockIdx.y*(gridDim.x*blockDim.x*blockDim.y*blockDim.z) + blockIdx.x*(blockDim.x*blockDim.y*blockDim.z) + threadIdx.x;
	if (index > SIZE) {
		return;
	}
	index = index * 3;

	dBrightImg[index] = (dImg[index] + dImg[index] * .3f) > 255 ? 255 : (dImg[index] + dImg[index] * .3f);
	dBrightImg[index + 1] = (dImg[index + 1] + dImg[index + 1] * .3f) > 255 ? 255 : (dImg[index + 1] + dImg[index + 1] * .3f);
	dBrightImg[index + 2] = (dImg[index + 2] + dImg[index + 2] * .3f) > 255 ? 255 : (dImg[index + 2] + dImg[index + 2] * .3f);

	dDarkImg[index] = dImg[index] * .7f;
	dDarkImg[index + 1] = dImg[index + 1] * .7f;
	dDarkImg[index + 2] = dImg[index + 2] * .7f;

	BYTE gray = dImg[index] * .114f + dImg[index + 1] * .587f + dImg[index + 2] * .299f;
	dGrayImg[index] = gray;
	dGrayImg[index + 1] = gray;
	dGrayImg[index + 2] = gray;

	dInvertImg[index] = 255 - dImg[index];
	dInvertImg[index + 1] = 255 - dImg[index + 1];
	dInvertImg[index + 2] = 255 - dImg[index + 2];
}

__global__ void imageFilterShared(unsigned char* dImg, unsigned char* dBrightImg, unsigned char* dDarkImg, unsigned char* dGrayImg, unsigned char* dInvertImg, int SIZE) {
	int temp;
	int index = blockIdx.y * (gridDim.x * blockDim.x * blockDim.y * blockDim.z) + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.x;
	
	index = index * 3;
	temp = index % (1024*3);

	__shared__ unsigned char sImg[1024 * 3];    //SIZE를 벗어나는 index를 가진 쓰레드들이 sIng[]에 이상한 값을 저장할 수 있지면 다시 접근 안해서 상관 X
	sImg[temp] = dImg[index];
	sImg[temp + 1] = dImg[index + 1];
	sImg[temp + 2] = dImg[index + 2];

	__syncthreads();
	if (index > SIZE) {
		return;
	}

	dBrightImg[index] = (sImg[temp] + sImg[temp] * .3f) > 255 ? 255 : (sImg[temp] + sImg[temp] * .3f);
	dBrightImg[index + 1] = (sImg[temp + 1] + sImg[temp + 1] * .3f) > 255 ? 255 : (sImg[temp + 1] + sImg[temp + 1] * .3f);
	dBrightImg[index + 2] = (sImg[temp + 2] + sImg[temp + 2] * .3f) > 255 ? 255 : (sImg[temp + 2] + sImg[temp + 2] * .3f);

	dDarkImg[index] = sImg[temp] * .7f;
	dDarkImg[index + 1] = sImg[temp + 1] * .7f;
	dDarkImg[index + 2] = sImg[temp + 2] * .7f;

	BYTE gray = sImg[temp] * .114f + sImg[temp + 1] * .587f + sImg[temp + 2] * .299f;
	dGrayImg[index] = gray;
	dGrayImg[index + 1] = gray;
	dGrayImg[index + 2] = gray;

	dInvertImg[index] = 255 - sImg[temp];
	dInvertImg[index + 1] = 255 - sImg[temp + 1];
	dInvertImg[index + 2] = 255 - sImg[temp + 2];
}

void main() {
	timer = NULL; setTimer();
	
	int mode;
	FILE* infile = NULL;
	printf("필터를 적용할 사진을 선택하십시오. ('1' or '2')\n> ");
	scanf(" %d", &mode);
	if (mode == 1) {
		infile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\catSample.bmp", "rb");			//작업을 진행할 bmp파일(파일명*경로 재지정)
	}
	else if(mode == 2) {
		infile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\scenery.bmp", "rb");		//작업을 진행할 bmp파일(파일명*경로 재지정)
	}
	else {
		printf("\n잘못된 입력입니다.\n");
		printf("실행을 종료합니다.\n");
		return;
	}
	
	FILE* brightfile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\brightResult.bmp", "wb");
	FILE* darkfile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\darkResult.bmp", "wb");
	FILE* grayfile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\grayResult.bmp", "wb");
	FILE* invertfile = fopen("C:\\Users\\User\\source\\repos\\Yun531\\cudaEx\\invertResult.bmp", "wb");
	

	BITMAPFILEHEADER hf;														//bmp파일 읽기
	fread(&hf, sizeof(BITMAPFILEHEADER), 1, infile);
	BITMAPINFOHEADER hInfo;
	fread(&hInfo, sizeof(BITMAPINFOHEADER), 1, infile);
	BYTE* Img = (BYTE*)malloc(hInfo.biSizeImage * sizeof(unsigned char));
	fseek(infile, hf.bfOffBits, SEEK_SET);
	fread(Img, sizeof(unsigned char), hInfo.biSizeImage, infile);

	int SIZE = hInfo.biWidth * hInfo.biHeight;
	int T_SIZE = 3 * SIZE;

	BYTE* brightImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));			//CPU결과값 저장 공간
	BYTE* darkImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));
	BYTE* grayImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));
	BYTE* invertImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));

	BYTE* openmpbrightImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));		//openmp결과값 저장 공간
	BYTE* openmpdarkImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));
	BYTE* openmpgrayImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));
	BYTE* openmpinvertImg = (BYTE*)malloc(T_SIZE * sizeof(unsigned char));


	unsigned char * dImg, * dBrightImg, * dDarkImg, * dGrayImg, * dInvertImg;			//device결과값 저장 공간
	dImg = dBrightImg = dDarkImg = dGrayImg = dInvertImg = NULL;


	dMemAlloc(dImg, unsigned char, T_SIZE);
	dMemAlloc(dBrightImg, unsigned char, T_SIZE);
	dMemAlloc(dDarkImg, unsigned char, T_SIZE);
	dMemAlloc(dGrayImg, unsigned char, T_SIZE);
	dMemAlloc(dInvertImg, unsigned char, T_SIZE);
	

	timer->onTimer(TIMER_HOST);								 //CPU버전(시작)
	for(int i = 0; i < SIZE; i++) {
		brightFilter(&Img[i * 3], &brightImg[i * 3]);
		darkFilter(&Img[i * 3], &darkImg[i * 3]);
		grayFilter(&Img[i * 3], &grayImg[i * 3]);
		invertFilter(&Img[i * 3], &invertImg[i * 3]);
		
	}
	timer->offTimer(TIMER_HOST);							////CPU버전(종료)


	timer->onTimer(TIMER_HOST_OPENMP);						//openMP버전(시작)
	int m = SIZE / NUM_CPU_THREADS;
	int* start = new int[NUM_CPU_THREADS];
	int* end = new int[NUM_CPU_THREADS];

	for (int i = 0; i < NUM_CPU_THREADS; i++) {
		start[i] = m * i;
		end[i] = m * (i + 1);
		if (i == (NUM_CPU_THREADS - 1)) {
			end[i] = SIZE;
		}
	}

#pragma omp parallel num_threads(NUM_CPU_THREADS)
{	int tid = omp_get_thread_num();

	for (int i = start[tid]; i < end[tid]; i++) {
		brightFilter(&Img[i * 3], &openmpbrightImg[i * 3]);
		darkFilter(&Img[i * 3], &openmpdarkImg[i * 3]);
		grayFilter(&Img[i * 3], &openmpgrayImg[i * 3]);
		invertFilter(&Img[i * 3], &openmpinvertImg[i * 3]);
	}
}
	timer->offTimer(TIMER_HOST_OPENMP);						//openMP버전(종료)


	dim3 gridDim1((SIZE + 1023) / 1024);
	dim3 blockDim1(1024);

	timer->onTimer(TIMER_HtoD);															//HtoD(시작)
	cudaMemcpy(dImg, Img, sizeof(unsigned char) * T_SIZE, cudaMemcpyHostToDevice);
	timer->offTimer(TIMER_HtoD);														//HtoD(종료)
	
	timer->onTimer(TIMER_KERNEL);						//device버전(시작)
	imageFilter << <gridDim1, blockDim1 >> > (dImg, dBrightImg, dDarkImg, dGrayImg, dInvertImg, SIZE);
	cudaThreadSynchronize();
	timer->offTimer(TIMER_KERNEL);						//device버전(종료)

	timer->onTimer(TIMER_KERNEL_SH);						//deviceShared버전(시작)
	imageFilterShared << <gridDim1, blockDim1 >> > (dImg, dBrightImg, dDarkImg, dGrayImg, dInvertImg, SIZE);
	cudaThreadSynchronize();
	timer->offTimer(TIMER_KERNEL_SH);						//deviceShared버전(종료)

	timer->onTimer(TIMER_DtoH);															//DtoH(시작)
	cudaMemcpy(openmpbrightImg, dBrightImg, sizeof(unsigned char) * T_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(openmpdarkImg, dDarkImg, sizeof(unsigned char) * T_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(openmpgrayImg, dGrayImg, sizeof(unsigned char) * T_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(openmpinvertImg, dInvertImg, sizeof(unsigned char) * T_SIZE, cudaMemcpyDeviceToHost);
	timer->offTimer(TIMER_DtoH);														//DtoH(종료)

	



	int temp;															//결과 확인 코드
	for (temp = 0; temp < SIZE; temp++) {
		int k = temp * 3;
		if ((openmpbrightImg[k] != brightImg[k]) || (openmpdarkImg[k] != darkImg[k]) || (openmpgrayImg[k] != grayImg[k]) || (openmpinvertImg[k] != invertImg[k])) {
			break;
		}
		if ((openmpbrightImg[k + 1] != brightImg[k + 1]) || (openmpdarkImg[k + 1] != darkImg[k + 1]) || (openmpgrayImg[k + 1] != grayImg[k + 1]) || (openmpinvertImg[k + 1] != invertImg[k + 1])) {
			break;
		}
		if ((openmpbrightImg[k + 2] != brightImg[k + 2]) || (openmpdarkImg[k + 2] != darkImg[k + 2]) || (openmpgrayImg[k + 2] != grayImg[k + 2]) || (openmpinvertImg[k + 2] != invertImg[k + 2])) {
			break;
		}
	}
	if (temp == SIZE)
		printf("Result is correct!\n");
	else
		printf("[%d] Result is not correct!!!!!!\n", temp);


	//파일 저장

	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), brightfile);         //bright파일 저장
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), brightfile);
	fseek(brightfile, hf.bfOffBits, SEEK_SET);
	fwrite(brightImg, sizeof(unsigned char), hInfo.biSizeImage, brightfile);

	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), darkfile);         //dark파일 저장
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), darkfile);
	fseek(darkfile, hf.bfOffBits, SEEK_SET);
	fwrite(darkImg, sizeof(unsigned char), hInfo.biSizeImage, darkfile);

	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), grayfile);         //gray파일 저장
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), grayfile);
	fseek(grayfile, hf.bfOffBits, SEEK_SET);
	fwrite(grayImg, sizeof(unsigned char), hInfo.biSizeImage, grayfile);

	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), invertfile);         //invert파일 저장
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), invertfile);
	fseek(invertfile, hf.bfOffBits, SEEK_SET);
	fwrite(invertImg, sizeof(unsigned char), hInfo.biSizeImage, invertfile);


	timer->printTimer();     //성능 출력
	if (timer != NULL) {
		delete timer;
	}

	free(Img); free(brightImg); free(grayImg); free(invertImg); free(darkImg);
	free(openmpbrightImg); free(openmpgrayImg); free(openmpinvertImg); free(openmpdarkImg);
	cudaFree(dImg); cudaFree(dBrightImg); cudaFree(dGrayImg); cudaFree(dInvertImg); cudaFree(dDarkImg);
	
	fclose(infile);
	fclose(brightfile);
	fclose(darkfile);
	fclose(grayfile);
	fclose(invertfile);
	

}



void brightFilter(BYTE* sourceBGR, BYTE* outBGR) {
	outBGR[0] = (sourceBGR[0] + sourceBGR[0] * .3f) > 255 ? 255 : (sourceBGR[0] + sourceBGR[0] * .3f);
	outBGR[1] = (sourceBGR[1] + sourceBGR[1] * .3f) > 255 ? 255 : (sourceBGR[1] + sourceBGR[1] * .3f);
	outBGR[2] = (sourceBGR[2] + sourceBGR[2] * .3f) > 255 ? 255 : (sourceBGR[2] + sourceBGR[2] * .3f);
}

void darkFilter(BYTE* sourceBGR, BYTE* outBGR) {
	outBGR[0] = sourceBGR[0] * .7f;
	outBGR[1] = sourceBGR[1] * .7f;
	outBGR[2] = sourceBGR[2] * .7f;
}

void grayFilter(BYTE* sourceBGR, BYTE* outBGR) {
	BYTE gray = sourceBGR[0] * .114f + sourceBGR[1] * .587f + sourceBGR[2] * .299f;

	outBGR[0] = gray;
	outBGR[1] = gray;
	outBGR[2] = gray;
}

void invertFilter(BYTE* sourceBGR, BYTE* outBGR) {
	outBGR[0] = 255 - sourceBGR[0];
	outBGR[1] = 255 - sourceBGR[1];
	outBGR[2] = 255 - sourceBGR[2];
}


void setTimer(void) {
	timer = new DS_timer(NUM_TIMER);

	timer->initTimers();
	timer->setTimerName(TIMER_HOST, "CPU code");
	timer->setTimerName(TIMER_HOST_OPENMP, "CPU code(openmp)");
	timer->setTimerName(TIMER_KERNEL, "Kernel launch");
	timer->setTimerName(TIMER_KERNEL_SH, "Kernel launch (shared ver)");
	timer->setTimerName(TIMER_HtoD, "[Data transter] host->device");
	timer->setTimerName(TIMER_DtoH, "[Data transfer] device->host");
}
