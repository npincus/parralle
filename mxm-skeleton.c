/* Parallel matrix-vector multiplication with 2D block decomposition
 * The last argument is assumed to be the communicator for a 2D Cartesian
 * topology and we assume the matrix a and vector x are already distribtued
 * with the vector x along the rightmost column of processors.
 */
#include "mpi.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

MatrixMatrixMultiply2D(int n, double *a, double *x, double *y, MPI_Comm comm_2d)
{ 
  int ROW=0, COL=1; /* Improve readability */ 
  int i, j, nlocal; 
  double *py; /* Will store partial dot products */ 
  int npes, dims[2], periods[2], keep_dims[2]; 
  int myrank, mycoords[2]; 
  int source_rank, dest_rank, coords[2]; 
  MPI_Status status; 
  MPI_Comm comm_row, comm_col; 

  /* Get information about the communicator */ 
  MPI_Comm_size(comm_2d, &npes); 
  MPI_Comm_rank(comm_2d, &myrank); 

  /* Compute the size of the square grid */ 
  dims[ROW] = dims[COL] = sqrt(npes); 

  nlocal = n/dims[ROW]; 
  int  Nlocal = nlocal*nlocal;
  MPI_Cart_coords(comm_2d, myrank, 2, mycoords); /* Get my coordinates */

  /* Create the row-based sub-topology */
  keep_dims[ROW] = 0;
  keep_dims[COL] = 1;
  MPI_Cart_sub(comm_2d, keep_dims, &comm_row);
  /* Create the column-based sub-topology */
  /****************************************/
  keep_dims[ROW] = 1;
  keep_dims[COL] = 0;
  MPI_Cart_sub(comm_2d, keep_dims, &comm_col);
  //define left right up down
  int right,left;
  int up,down;
  MPI_Cart_shift(comm_col,COL,1,&up,&down);
  MPI_Cart_shift(comm_row,ROW,1,&right,&left);

    /****************************************/
    for(int j=0;j<dims[ROW];j++) {
        /* Perform local matrix-matrix multiply */
        for (int t = 0; t < nlocal; t++) {
            for (int U = 0; U < nlocal; U++) {
                float sum = 0.0;
                for (int k = 0; k < nlocal; k++)
                    sum += a[t * nlocal + k] * x[k * nlocal + U];
                y[t * nlocal + U] += sum;
            }
            /* perform shift to right*/
        MPI_Sendrecv(
                a,Nlocal,MPI_DOUBLE,right,0,
                a,Nlocal,MPI_DOUBLE,left,0,
                comm_row,MPI_STATUS_IGNORE);
        /* perform shift to up*/
        MPI_Sendrecv(
                x,Nlocal,MPI_DOUBLE,up,1,
                x,Nlocal,MPI_DOUBLE,down,1,
                comm_col,MPI_STATUS_IGNORE);
    }
 
  /* free local communicators */
  MPI_Comm_free(&comm_row); /* Free up communicator */ 
  MPI_Comm_free(&comm_col); /* Free up communicator */
} 
