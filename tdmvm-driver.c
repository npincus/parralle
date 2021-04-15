/* test driver program for 2D block matrix vector multiply */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void MatrixVectorMultiply2D(int n, double *a, double *x, double *y, MPI_Comm comm2d);

int main (int argc, char *argv[])
{
  int ROW = 0, COL = 1;  /* for readability */  
  int   numtasks, taskid;
  int i, j, N, n, nlocal, Nlocal; 
  double *a, *x, *y, *ycheck, *alocal, *xlocal, *ylocal;
  MPI_Datatype blocktype;
  int *disps, dims[2], periods[2];;
  int proc;
  MPI_Comm comm2d;
  int my2drank;
  int nrowblocks, ncolblocks, rowsize;
  int mycoords[2], coords[2];
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  if (argc != 2) {
    if (taskid == 0) {
      fprintf(stderr, "Usage: %s <n>\n", argv[0]);
      fprintf(stderr, "where n is a multiple of the square root of the number of tasks\n");
    }   
    MPI_Finalize();
    exit(0);
  }

  /* Read row/column dimension from command line */
  n = atoi(argv[1]);

  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  dims[ROW] = dims[COL] = sqrt(numtasks);

  if (n%dims[ROW] != 0) {
    if (taskid == 0) {
      fprintf(stderr, "Usage: %s <n>\n", argv[0]);
      fprintf(stderr, "where n is a multiple of the square root of the number of tasks\n");
    }   
    MPI_Finalize();
    exit(0);
  }

  periods[ROW] = periods[COL] = 0;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2d);
  MPI_Comm_rank(comm2d, &my2drank);
  MPI_Cart_coords(comm2d, my2drank, 2, mycoords);

  N=n*n; //size of matrix n x n
  nrowblocks = ncolblocks = sqrt(numtasks);
  nlocal = n/nrowblocks;
  Nlocal = nlocal*nlocal;
  rowsize = nrowblocks*Nlocal;

  if (my2drank == 0) {
    a = (double *) malloc(N*sizeof(double));
    x = (double *) malloc(n*sizeof(double));
    y = (double *) malloc(n*sizeof(double));
    ycheck = (double *) malloc(n*sizeof(double));

    disps = (int *) malloc(numtasks*sizeof(int));

    /* Initialize A and x */
    for (i = 0; i < n; i++){
      for (j = 0; j < n; j++)
        a[i*n+j] = i*2+j;
      x[i] = i;
    }

    /* Compute displacements for block distribution datatype */
    for (i = 0; i < nrowblocks; i++)
      for (j = 0; j < ncolblocks; j++) {
        coords[ROW] = i;
        coords[COL] = j;
        MPI_Cart_rank(comm2d, coords, &proc);
        disps[proc] = i*rowsize + j*nlocal;
      }

  }

  /* Allocate space for local matrix and vectors */
  alocal = (double *) malloc(Nlocal*sizeof(double));
  xlocal = (double *) malloc(nlocal*sizeof(double));
  ylocal = (double *) malloc(nlocal*sizeof(double));

  MPI_Type_vector(nlocal, nlocal, n, MPI_DOUBLE, &blocktype);
  MPI_Type_commit(&blocktype);

  /* Distribute a and x in 2D block distribution */
  if (my2drank == 0) {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < nlocal; j++)
        alocal[i*nlocal+j] = a[i*n+j];
    for (i = 1; i < numtasks; i++) 
      MPI_Send(a+disps[i], 1, blocktype, i, 1, comm2d);
    for (i = 0; i < nrowblocks; i++) { 
      coords[ROW] = i;
      coords[COL] = dims[COL] - 1;
      MPI_Cart_rank(comm2d, coords, &proc);
      MPI_Send(x + i*nlocal, nlocal, MPI_DOUBLE, proc, 2, comm2d);
    }
  }

  if (my2drank !=0) { 
    MPI_Recv(alocal, Nlocal, MPI_DOUBLE, 0, 1, comm2d, &status);
    if (mycoords[COL] == (dims[COL]-1))
      MPI_Recv(xlocal, nlocal, MPI_DOUBLE, 0, 2, comm2d, &status);
  }

  /* Each process calls MatrixVectorMultiply2D */
  
  MatrixVectorMultiply2D(n, alocal, xlocal, ylocal, comm2d); 

  /* Gather results back to root process */
  if (mycoords[COL] == dims[COL] - 1)
    MPI_Send(ylocal, nlocal, MPI_DOUBLE, 0, 3, comm2d);
  if (my2drank == 0)
    for (i = 0; i < ncolblocks; i++) {
      coords[ROW] = i;
      coords[COL] = dims[COL] - 1;
      MPI_Cart_rank(comm2d, coords, &proc);
      MPI_Recv(y + i*nlocal, nlocal, MPI_DOUBLE, proc, 3, comm2d, &status);
    }

  /* Check that results are correct */
  if (my2drank == 0) {
    for (i = 0; i < n; i++) {
      ycheck[i] = 0;
      for (j = 0; j < n; j++)
        ycheck[i] += a[i*n+j]*x[j];
    }
    for (i = 0; i < n; i++)
      if (y[i] != ycheck[i])
        fprintf(stderr, "Discrepancy: y[%d] = %f, ycheck[%d] = %f\n", i, y[i], i, ycheck[i]);
    printf("Done with tdmvm, y[%d] = %f\n", n-1, y[n-1]);
  }

  MPI_Finalize();

}
