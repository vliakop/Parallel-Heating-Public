
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NXPROB      20                 /* x dimension of problem grid */
#define NYPROB      20               /* y dimension of problem grid */
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

  void  prtdat(), update();
  int my_rank,
      nproc,
      dims[2],
      periods[2],
      reorder,
      i,j,a,m,
      blockSize;
  MPI_Comm cart;
  
  int up,down,left,right;
  
  MPI_Request recv[4];
  MPI_Request send[4];
  MPI_Datatype inner_up_row,inner_down_row,
  			   outer_up_row,outer_down_row,
  			   inner_left_col,inner_right_col,
  			   outer_left_col,outer_right_col,
  				type,subarray;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  periods[0] = 0;
  periods[1] = 0;
  reorder = 1;
  dims[0]=dims[1]= (int) sqrt(nproc);
  blockSize = NXPROB / (int) sqrt(nproc);


  int arraySize[]  = {blockSize+2,blockSize+2};
  int subsizes[] = {blockSize,blockSize};
  int starts[]   = {1,1};
  MPI_Type_create_subarray(2, arraySize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &type);
  MPI_Type_create_resized(type, 0, blockSize*sizeof(float), &subarray);
  MPI_Type_commit(&subarray);

  
  int rowSubsize[2] = {1,blockSize};
  int inner_row_up_start[2]   = {1,1};
  int inner_row_down_start[2]   = {blockSize,1};
  int outer_row_up_start[2] = {0,1};
  int outer_row_down_start[2] = {blockSize+1,1};

  int colSubSize[2] = {blockSize,1};
  int inner_col_left_start[2] = {1,1};
  int inner_col_right_start[2] = {1,blockSize};
  int outer_col_left_start[2] = {1,0};
  int outer_col_right_start[2] = {1,blockSize+1};

  MPI_Type_create_subarray(2, arraySize, rowSubsize, inner_row_up_start, MPI_ORDER_C, MPI_FLOAT, &inner_up_row);
  MPI_Type_create_subarray(2, arraySize, rowSubsize, inner_row_down_start, MPI_ORDER_C, MPI_FLOAT, &inner_down_row);
  MPI_Type_create_subarray(2, arraySize, colSubSize, inner_col_left_start, MPI_ORDER_C, MPI_FLOAT, &inner_left_col);
  MPI_Type_create_subarray(2, arraySize, colSubSize, inner_col_right_start, MPI_ORDER_C, MPI_FLOAT, &inner_right_col);
  
  MPI_Type_create_subarray(2, arraySize, rowSubsize, outer_row_up_start, MPI_ORDER_C, MPI_FLOAT, &outer_up_row);
  MPI_Type_create_subarray(2, arraySize, rowSubsize, outer_row_down_start, MPI_ORDER_C, MPI_FLOAT, &outer_down_row);
  MPI_Type_create_subarray(2, arraySize, colSubSize, outer_col_left_start, MPI_ORDER_C, MPI_FLOAT, &outer_left_col);
  MPI_Type_create_subarray(2, arraySize, colSubSize, outer_col_right_start, MPI_ORDER_C, MPI_FLOAT, &outer_right_col);



  MPI_Type_commit(&inner_up_row);
  MPI_Type_commit(&inner_down_row);
  MPI_Type_commit(&inner_left_col);
  MPI_Type_commit(&inner_right_col);
  MPI_Type_commit(&outer_up_row);
  MPI_Type_commit(&outer_down_row);
  MPI_Type_commit(&outer_left_col);
  MPI_Type_commit(&outer_right_col);


  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);
  
  float* u1 = (float*)(malloc((blockSize+2)*(blockSize+2)*sizeof(float)));
  float* u2 = (float*)(malloc((blockSize+2)*(blockSize+2)*sizeof(float)));
  inidat(NXPROB,NYPROB,blockSize,nproc,my_rank,u1,u2);

 //  if (my_rank == MASTER) {
 //  	float init_heat[NXPROB*NYPROB];    
	// MPI_Gather(u1,1,subarray,&init_heat,blockSize*blockSize,MPI_FLOAT,MASTER,cart);
	// for(a=0;a<blockSize;a++){
	//   for(m=0;m<sqrt(nproc);m++){
	//     for(i=0;i<NXPROB;i+=blockSize){
	//       for(j=0;j<blockSize;j++)
	//         printf("%6.1f\t",(float)init_heat[a*NXPROB+m*blockSize+i*NXPROB+j]);
	//     }
	//     printf("\n");
	//   }
	// }
	// printf("\n");
 //  }else
 //    MPI_Gather(u1,1,subarray,NULL,blockSize*blockSize,MPI_FLOAT,MASTER,cart);

  
  
  /*find neighbors*/
  MPI_Cart_shift(cart, 1, 1, &up, &down);
  MPI_Cart_shift(cart, 0, 1, &left,&right);
  float* temp = NULL;
  for (i = 1; i <= STEPS; i++){
  	
	MPI_Isend(u1,1,inner_up_row,up,UPTAG,MPI_COMM_WORLD,&send[0]);
	MPI_Irecv(u1,1,outer_down_row,down,UPTAG,MPI_COMM_WORLD,&recv[0]);

	MPI_Isend(u1,1,inner_down_row,down,DTAG,MPI_COMM_WORLD,&send[1]);
	MPI_Irecv(u1,1,outer_up_row,up,DTAG,MPI_COMM_WORLD,&recv[1]);

	MPI_Isend(u1,1,inner_left_col,left,LTAG,MPI_COMM_WORLD,&send[2]);
	MPI_Irecv(u1,1,outer_right_col,right,LTAG,MPI_COMM_WORLD,&recv[2]);

	MPI_Isend(u1,1,inner_right_col,right,RTAG,MPI_COMM_WORLD,&send[3]);
	MPI_Irecv(u1,1,outer_left_col,left,RTAG,MPI_COMM_WORLD,&recv[3]);
	
	/*inner array*/
	update(2,blockSize-1,2,blockSize-1,blockSize+2,u1,u2);
	
	MPI_Waitall(4,recv,MPI_STATUSES_IGNORE);

	/*left col*/
	update(1,blockSize,1,1,blockSize+2,u1,u2);
	/*right col*/
	update(1,blockSize,blockSize,blockSize,blockSize+2,u1,u2);
	/*up row*/
	update(1,1,2,blockSize-1,blockSize+2,u1,u2);
	/*down row*/
	update(blockSize,blockSize,2,blockSize-1,blockSize+2,u1,u2);
	
	MPI_Waitall(4,send,MPI_STATUSES_IGNORE);
		
	temp = u1;
	u1 = u2;
	u2 = temp;
	temp = NULL;
  }
 
  // if (my_rank == MASTER) {
	 //    float final_heat[NXPROB*NYPROB];
	 //    MPI_Gather(u1,1,subarray,&final_heat,blockSize*blockSize,MPI_FLOAT,MASTER,cart);
	 //    for(a=0;a<blockSize;a++){
	 //      for(m=0;m<sqrt(nproc);m++){
	 //        for(i=0;i<NXPROB;i+=blockSize){
	 //          for(j=0;j<blockSize;j++)
	 //            printf("%6.1f\t",(float)final_heat[a*NXPROB+m*blockSize+i*NXPROB+j]);
	 //        }
	 //        printf("\n");
	 //      }
	 //    }
  //  }else
  //  		MPI_Gather(u1,1,subarray,NULL,blockSize*blockSize,MPI_FLOAT,MASTER,cart);

  MPI_Finalize();
  return 0;
}



/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int startX, int endX, int startY,int endY,int offset, float *u1, float *u2)
{
   int ix, iy;
   float up = 0.0,left = 0.0,right = 0.0 ,down = 0.0;
   for (ix = startX; ix <= endX; ix++){
      for (iy = startY; iy <= endY; iy++){
          if(u1[ix*offset+iy] == 0.0)
            u2[ix*offset+iy] = 0.0;
          else{
      		up = u1[(ix-1)*offset+iy];
      		down = u1[(ix+1)*offset+iy];
      		left = u1[ix*offset+iy-1];
      		right = u1[ix*offset+iy+1];
          
            u2[ix*offset+iy]= u1[ix*offset+iy] +
                          parms.cx * ( down + up -
                          2.0 * u1[ix*offset+iy]) +
                          parms.cy * (right + left -
                          2.0 * u1[ix*offset+iy]);
            }
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
