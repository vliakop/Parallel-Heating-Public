#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"



#define NXPROB      20                 /* x dimension of problem grid */
#define NYPROB      20                 /* y dimension of problem grid */
#define STEPS       100                /* number of time steps */

struct Parms { 
  float cx;
  float cy;
} parms = {0.1, 0.1};


/**************************************************************************
 *  subroutine initdat
 ****************************************************************************/
__global__ void inidat(int *d_dimensions, float *d_u){

	int pos = (blockIdx.x * blockDim.x) + threadIdx.x;
	d_u[pos] = (float)(blockIdx.x*((*d_dimensions) - blockIdx.x - 1)*threadIdx.x*((*d_dimensions) - threadIdx.x - 1));
	__syncthreads();

}

/**************************************************************************
 *  subroutine prdat
 ****************************************************************************/
	
	void prdat(int X, int Y, float u[NXPROB][NYPROB], char *filename){
		
		FILE *fp;
		fp = fopen(filename, "w+");
		if(fp == NULL){
			printf("Couldn't open %s\n", filename);
			return;
		}
		int x, y;
		for (x = 0; x < X; x++){
			for(y = 0; y < Y; y++){
				fprintf(fp, "%6.1lf", u[x][y]);
				if(y == X - 1){
					fprintf(fp, "\n");
				}
				else{
					fprintf(fp, " ");
				}
			}
		}
		return;
	}

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
__global__ void update(int *d_dimensions, float *d_cx, float *d_cy, float *d_u1, float *d_u2) {
	
	int index;
	index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(blockIdx.x == 0 || threadIdx.x == 0 || blockIdx.x == (*d_dimensions - 1) || threadIdx.x == (*d_dimensions - 1)){ /* ! on last thread-clause */
		d_u2[index] = 0.0;
		return;
	}
	d_u2[index] = d_u1[index] + (*d_cx) * (d_u1[threadIdx.x + (blockIdx.x + 1)* blockDim.x] + 
					d_u1[threadIdx.x + (blockIdx.x - 1)* blockDim.x] - 2.0 * d_u1[index]) + 
					(*d_cy) * (d_u1[(threadIdx.x + 1) + blockIdx.x * blockDim.x] +
					d_u1[(threadIdx.x - 1) + blockIdx.x * blockDim.x] - 2.0 * d_u1[index]);
	__syncthreads();
	return;
}



int main (int argc, char *argv[]){

	/* host declerations */
	int	iz, it, dimensions, arraySize;
	float  u[2][NXPROB][NYPROB];     /* array for grid */
	
	dimensions = NXPROB;
	arraySize = NXPROB*NYPROB*sizeof(float);

	
	/* device declerations */
	int *d_dimensions;
	float *d_cx, *d_cy;
	float *d_u[2];
	
	/* Device Memory Allocation */
	cudaMalloc((void **)&d_dimensions, sizeof(int)); /* X, Y dimensions */
	cudaMalloc((void **)&d_cx, sizeof(float));		 /* Parameter 1 */
	cudaMalloc((void **)&d_cy, sizeof(float));		 /* Parameter 2*/
	cudaMalloc((void **)&d_u[0], arraySize); 		 /* array at time t */
	cudaMalloc((void **)&d_u[1], arraySize);		 /* array at time t+1 */

	/* Device memcpy */
	cudaMemcpy(d_dimensions, &dimensions, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cx, &parms.cx, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cy, &parms.cy, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u[0], u[0], arraySize, cudaMemcpyHostToDevice);
	
	/* Initialise the table */
	printf("Grid size: X= %d  Y= %d  Time steps= %d\n",NXPROB,NYPROB,STEPS);
	printf("Initializing grid and writing initial.dat file...\n");
	inidat << <NXPROB, NYPROB >> >(d_dimensions, d_u[0]);
	cudaMemcpy(u[0], d_u[0], arraySize, cudaMemcpyDeviceToHost);
	prdat(NXPROB, NYPROB, u[0], "init.dat");
	
	/* Calculate d_u[1] */
	iz = 0;
	for (it = 0; it < STEPS; it++){
		update << <NXPROB, NYPROB >> >(d_dimensions, d_cx, d_cy, d_u[iz], d_u[1 - iz]);
		iz = 1 - iz;
	}
	
	cudaMemcpy(u[1], d_u[1], arraySize, cudaMemcpyDeviceToHost);
	
	/* Write final output, call X graph and finalize MPI */
	printf("Writing final.dat file, which was calculated using CUDA C, and generating graph...\n");
	prdat(NXPROB, NYPROB, u[1], "final.dat");
	
	cudaFree(d_dimensions);
	cudaFree(d_cx);
	cudaFree(d_cy);
	cudaFree(d_u[0]);
	cudaFree(d_u[1]);
	return 0;
 }  
