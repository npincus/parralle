INCLUDE =
CC = mpicc
CFLAGS = -O2 -lm

%.o:	%.c
	$(CC) -c $^ $(INCLUDE) $(CFLAGS)


tdmvm:	tdmvm-skeleton.o tdmvm-driver.o
	$(CC) -o $@ $^ $(CFLAGS)


mxm:	mxm-skeleton.o mxm-driver.o
	$(CC) -o $@ $^ $(CFLAGS)