
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define NXPROB      128                 /* x dimension of problem grid */
#define NYPROB      128             /* y dimension of problem grid */
#define STEPS       100                /* number of time steps */
#define UPTAG       0                  /* message tag */
#define DTAG        1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define MASTER      0                  /* taskid of first process */

struct Parms {
  float cx;
  float cy;
} parms = {0.1, 0.1};

void inidat(int width,int height,int blockSize,int nprocs,int rank,float* u1,float* u2);

int main (int argc, char *argv[]){
  clock_t begin, end;
  double time_run;
  begin = clock();
  void  prtdat(), update();
  int taskid,
      my_rank,
      comm_size,
      nproc,
      ndims,
      offset,
      dims[2],
      periods[2],
      reorder,coords[2],
      *rcounts,
      *displs,
      i,j,a,m,
      blockSize;
  MPI_Comm cart;

  
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  periods[0] = 0;
  periods[1] = 0;
  reorder = 1;
  dims[0]=dims[1]= (int) sqrt(nproc);
  blockSize = NXPROB / (int) sqrt(nproc);

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);

  float  u[2][blockSize+2][blockSize+2];
  int up,down,left,right;
  int iz = 0;

 
  
  
  MPI_Request recv[4];
  MPI_Request send[4];
  MPI_Status st[4];

  MPI_Datatype row,outer_row,inner_row,col,outer_col,inner_col;
  MPI_Type_vector(blockSize,1,1,MPI_FLOAT,&row);
  MPI_Type_vector(blockSize,1,blockSize+2,MPI_FLOAT,&col);
  MPI_Type_commit(&row);
  MPI_Type_commit(&col);

  MPI_Datatype type, subarray;
  int sizes[2]    = {blockSize+2,blockSize+2};
  int subsizes[2] = {blockSize,blockSize};
  int starts[2]   = {1,1};
  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &type);
  MPI_Type_create_resized(type, 0, blockSize*sizeof(float), &subarray);
  MPI_Type_commit(&subarray);

  inidat(NXPROB,NYPROB,blockSize,nproc,my_rank,&u[0][0][0],&u[1][0][0]);



  /*find neighbors*/
  MPI_Cart_shift(cart, 1, 1, &up, &down);
  MPI_Cart_shift(cart, 0, 1, &left,&right );

  for (i = 1; i <= STEPS; i++){
    MPI_Isend(&u[iz][1][1],1,row,up,UPTAG,cart,&send[0]);
    MPI_Isend(&u[iz][blockSize][1],1,row,down,DTAG,cart,&send[1]);
    MPI_Isend(&u[iz][1][1],1,col,left,LTAG,cart,&send[2]);
    MPI_Isend(&u[iz][1][blockSize],1,col,right,RTAG,cart,&send[3]);

    MPI_Irecv(&u[iz][0][1],1,row,up,DTAG,cart,&recv[0]);
    MPI_Irecv(&u[iz][blockSize+2-1][1],1,row,down,UPTAG,cart,&recv[1]);
    MPI_Irecv(&u[iz][1][0],1,col,left,RTAG,cart,&recv[2]);
    MPI_Irecv(&u[iz][1][blockSize+2-1],1,col,right,LTAG,cart,&recv[3]);

    update(0,blockSize-2,0,blockSize-2,blockSize+2,&u[iz][2][2],&u[1-iz][2][2]);

    MPI_Waitall(4,recv,MPI_STATUS_IGNORE);

    update(0,1,0,blockSize,blockSize+2,&u[iz][1][1],&u[1-iz][1][1]);
    update(0,1,0,blockSize,blockSize+2,&u[iz][blockSize][1],&u[1-iz][blockSize][1]);
    update(0,blockSize-2,0,1,blockSize+2,&u[iz][2][1],&u[1-iz][2][1]);
    update(0,blockSize-2,0,1,blockSize+2,&u[iz][2][blockSize],&u[1-iz][2][blockSize]);

    MPI_Waitall(4,send,st);
    iz = 1-iz;
  }

  if (my_rank == MASTER) {
    float final_heat[NXPROB*NYPROB];
    MPI_Gather(&u[0][0][0],1,subarray,&final_heat,blockSize*blockSize,MPI_FLOAT,MASTER,cart);
    //prtdat(NXPROB, NYPROB, final_heat, "final.dat");
    end = clock();
    time_run = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%f\n", time_run);
     for(a=0;a<blockSize;a++){
       for(m=0;m<sqrt(nproc);m++){
         for(i=0;i<NXPROB;i+=blockSize){
           for(j=0;j<blockSize;j++)
            printf("%6.1f\t",(float)final_heat[a*NXPROB+m*blockSize+i*NXPROB+j]);
         }
         printf("\n");
       }
     }
  }else{
   	MPI_Gather(&u[0][0][0],1,subarray,NULL,blockSize*blockSize,MPI_FLOAT,MASTER,cart);
   }
    MPI_Finalize();
    return 0;

}



/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int startX, int endX, int startY,int endY,int offset, float *u1, float *u2)
{
   int ix, iy;
   for (ix = startX; ix < endX; ix++){
      for (iy = startY; iy < endY; iy++){
          if(*(u1+ix*offset+iy) == 0.0)
            *(u2+ix*offset+iy) = 0.0;
          else
            *(u2+ix*offset+iy) = *(u1+ix*offset+iy)  +
                          parms.cx * (*(u1+(ix+1)*offset+iy) +
                          *(u1+(ix-1)*offset+iy) -
                          2.0 * *(u1+ix*offset+iy)) +
                          parms.cy * (*(u1+ix*offset+iy+1) +
                         *(u1+ix*offset+iy-1) -
                          2.0 * *(u1+ix*offset+iy));
      }
    }
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int width,int height,int blockSize,int nprocs,int rank,float* u1,float* u2) {

  int x,y;
  int bx = (rank % (int)sqrt(nprocs))*blockSize;
  int by = (rank / (int)sqrt(nprocs))*blockSize;

  bx = bx-1;
  by = by-1;
  for(x=0; x < blockSize+2; x++){
    for(y=0; y < blockSize+2; y++){
      if(bx+x < 0 || by+y < 0 || bx+x >= height || by+y >= width)
        u1[x*(blockSize+2)+y] = 0.0;
      else
        u1[x*(blockSize+2)+y] = (float)((bx+x)*(width-(bx+x)-1.0)*(by+y)*(height-(by+y)-1.0));
      u2[x*(blockSize+2)+y] = 0.0;
    }
    if(x == 0 || x == blockSize + 1){
      u1[x*(blockSize+2)+0] = 0.0;
      u1[x*(blockSize+2)+(blockSize+1)] = 0.0;
    }
  }
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *u1, char *fnam) {
	int ix, iy;
	FILE *fp;

	fp = fopen(fnam, "w");
	for (iy = ny-1; iy >= 0; iy--) {
	  for (ix = 0; ix <= nx-1; ix++) {
		fprintf(fp, "%6.1f", *(u1+ix*ny+iy));
		if (ix != nx-1)
		  fprintf(fp, " ");
		else
		  fprintf(fp, "\n");
		}
	  }
	fclose(fp);
}
