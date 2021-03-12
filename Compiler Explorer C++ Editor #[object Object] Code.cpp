#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <vector>
#include <functional>
#include <algorithm>
#include  <execution>
#include <chrono>
#include <iostream>
#include <array>
#include <utility>
namespace {
    constexpr int  Y(){
        return 512;
    }
    constexpr int  X(){
        return 512;
    }
    constexpr double North(){
        return 0.0;
    }
    constexpr double South(){
        return 100.0;
    }
    constexpr double East(){
        return 100.0;
    }
    constexpr double West(){
        return 100.0;
    }
    constexpr double Epsilon(){
        return 0.1;
    }
    std::pair<double,int>  work(std::vector<double> &&now,
     const std::vector<int> mask) noexcept{
        constexpr auto N=X();
        constexpr auto M=Y();
        double diff=Epsilon();
        int iterations=0;
        int iterations_print=1;
        auto left=std::vector<double>(N*M);
        auto Right=std::vector<double>(N*M);
        auto up=std::vector<double>(N*M);
        auto down=std::vector<double>(N*M);
        auto old= std::vector<double>(N*M);
        std::vector<double> hor=std::vector<double>(N*M);
        std::vector<double> vert=std::vector<double>(N*M);
        std::vector<double> diffvec=std::vector<double>(N*M);
        for ( ; Epsilon()<=diff;iterations++ ){
            //makes 4 copys
            left=std::vector<double>(now);
            Right=std::vector<double>(now);
            up=std::vector<double>(now);
            down=std::vector<double>(now);
            old= std::vector<double>(now);


            //makes 4 copy off now rotated 1left,1right,Nleft,NRight
            std::rotate(std::execution::par,left.begin(), left.begin() + 1, left.end());
            std::rotate(std::execution::par,Right.rbegin(), Right.rbegin() + 1, Right.rend());     
            std::rotate(std::execution::par,up.begin(), up.begin() + N, up.end());
            std::rotate(std::execution::par,down.rbegin(), down.rbegin() + N, down.rend());

            /* this block avrages*/
            //sums right and left into hor
            std::transform(std::execution::par,left.begin(), left.end(), Right.begin(), hor.begin(), std::plus<>{});
            //sums up and down into vert
            std::transform(std::execution::par,up.begin(), up.end(), down.begin(), vert.begin(), std::plus<>{});
            //sums and devides by 4 up and left and puts into now
            std::transform(std::execution::par,vert.begin(), vert.end(), hor.begin(), now.begin(),[](double lhs,double rhs) -> double { return (lhs+rhs)/4.0; });
            // todo fill in borders again
            std::transform(std::execution::par,mask.begin(), mask.end(), now.begin(),
                now.begin(),[](double mask,double curr) -> double{
                        if(mask==-1)
                            return curr;
                        else return mask;
                   } );
            //exists for clarity on how to sum diffrances between 2 containers
    
            //steals right as its not useful right now to fill diffrances
            std::transform(std::execution::par,old.begin(), old.end(), now.begin(),
                diffvec.begin(),[](double old,double now) -> double{
                   return std::abs(now-old);
                   } );
            diff = *std::max_element(diffvec.begin(), diffvec.end());
            if ( iterations == iterations_print )
            {
            printf ( "  %8d  %f\n", iterations, diff );
            iterations_print *= 2 ;
            }
        } 
        return std::pair<double,int>{diff,iterations};
    }

}
/******************************************************************************/

int main ( int argc, char *argv[] ) noexcept

/******************************************************************************/
/*
  Purpose:
    MAIN is the main program for HEATED2D.
  Discussion:
    This code solves the steady state heat equation on a rectangular region.
    The unparuential version of this program needs approximately
    18/epsilon iterations to complete. 
    The physical region, and the boundary conditions, are suggested
    by this diagram;
                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100
    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:
                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1
    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:
      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )
    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.
   
    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:
      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )
    If this process is repeated often enough, the difference between successive 
    estimates of the solution will go to zero.
    This program carries out such an iteration, using a tolerance specified by
    the user, and writes the final estimate of the solution to a file that can
    be used for graphic processing.
  Licensing:
    This code is distributed under the GNU LGPL license. 
  Modified:
    24 February 2021
  Author:
    Original C version by Michael Quinn.
    Modified C version by John Burkardt and Shirley Moore.
  Reference:
    Michael Quinn,
    Parallel Programming in C with MPI and OpenMP,
    McGraw-Hill, 2004,
    ISBN13: 978-0071232654,
    LC: QA76.73.C15.Q55.
  Parameters:
    double DIFF, the norm of the change in the solution from one iteration
    to the next.
    double MEAN, the average of the boundary values, used to initialize
    the values of the solution in the interior.
    double U[M][N], the solution at the previous iteration.
    double W[M][N], the solution computed at the latest iteration.
*/
{
    constexpr int M=Y();
    constexpr int N=X();
    constexpr double north=North();
    constexpr double south=South();
    constexpr double east=East();
    constexpr double west=West();
    constexpr double meanNorth = (N-2)*north;
    constexpr double meanSouth = (N-2)*south;
    constexpr double meanEast = (M-2)*east;
    constexpr double meanWest = (M-2)*south;
    constexpr double sum=meanNorth+meanSouth+meanEast+meanWest;
    constexpr double mean=sum / static_cast < double >  ( 2 * M + 2 * N - 4 );
    constexpr double epsilon = Epsilon();
    std::vector<double> now=std::vector<double> (N*M, mean);
    std::vector<int> mask=std::vector<int>(N*M,-1);
    std::transform(mask.begin(), mask.end(), mask.begin(),
                   mask.begin(), [idx=0](double curr,double curr2) mutable  -> double { 
                        if(idx<N){
                            idx++;
                            return north;
                        }
                        if(idx>(N-1)*M){
                            idx++;
                            return south;
                        }
                        if(idx%N==0){
                            idx++;
                            return east;
                        }
                        if((idx+1)%N==0){
                            idx++;
                            return west;
                        }
                        idx++;
                        return curr;

                    });
     std::transform(mask.begin(), mask.end(), now.begin(),
                   now.begin(),[](double mask,double curr) -> double{
                        if(mask==-1)
                            return curr;
                        else return mask;
                   } );

    printf ( "\n" );
    printf ( "HEAT2D\n" );
    printf ( "  unparuential C++ version\n" );
    printf ( "  A program to solve for the steady state temperature distribution\n" );
    printf ( "  over a rectangular plate.\n" );
    printf ( "\n" );
    printf ( "  Spatial grid of %d by %d points.\n", M, N );
    printf ( "  MEAN = %f\n", mean ); 
    printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon );     
    printf ( "\n" );  
/*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/

  printf ( "\n" );
  printf ( " Iteration  Change\n" );
  printf ( "\n" );
  auto t1 = std::chrono::high_resolution_clock::now();
  std::pair<double,int> shell=work(std::move(now),mask);
  int iterations =shell.second;
  double diff =shell.first;
  auto t2 = std::chrono::high_resolution_clock::now();
  auto wtime=std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
  printf ( "\n" );
  printf ( "  %8d  %f\n", iterations, diff );
  printf ( "\n" );
  printf ( "  Error tolerance achieved.\n" );
  std::cout<< "Wallclock time =" << wtime.count() << " milliseconds\n";
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "HEAT2D:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;

}