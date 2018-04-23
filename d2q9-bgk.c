/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdbool.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"


#define MASTER 0
#define NTYPES 9

/*-------- Global Variables ------------ */
int rank;                  // Variable to store the rank of the process
int size;                  // Variable to store the size of the mpi processess.
int local_rows;
int local_cols;
int bigX;
int bigY;
int haloTop;
int haloBottom;

int myStartInd;
int myEndInd;

int numOfObstacles;
int numOfCells;

int topRank;
int botRank;

MPI_Datatype cells_struct;

int numberOfIterationsDone;

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int startInd;
  int endInd;
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int func_initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** func_accelerate_flow(), func_propagate(), func_rebound() & func_collision()
*/
int func_timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int func_accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int func_propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int func_rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int func_collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int func_write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
float func_gatherVelocity(const t_param params,  t_speed *cells, int* obstacles);
void func_gatherData(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);


/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);
float av_velocity_withoutDiv(const t_param params, t_speed* cells, int* obstacles);
float av_velocity_forAll(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
bool inLocalRows(int myStartInd, int myEndInd, int globalPos);
int getLocalRows(int myStartInd, int myEndInd, int globalPos);
int getHaloCellsForY(int attempt);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the average velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */




 int items = 1;
 int block_lengths = {NSPEEDS};



  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }


  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Datatype this_type = {MPI_FLOAT};
  const MPI_Aint offset = {
      offsetof(t_speed,speeds)
  };
  MPI_Type_create_struct(items,&block_lengths,&offset,&this_type,&cells_struct);
  MPI_Type_commit(&cells_struct);


  /* initialise our data structures and load values from file */
  func_initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);


  if(rank == MASTER){
   gettimeofday(&timstr, NULL);
   tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
   printf("--------Number of workers being run : %d--------\n", size);
 }

  numberOfIterationsDone = 0;
  /* iterate for maxIters timesteps */
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //printf("Worker %d is doing iteration %d \n",rank, tt);
    func_timestep(params, cells, tmp_cells, obstacles);
    float this_avgV = func_gatherVelocity(params,cells,obstacles);
    ++numberOfIterationsDone;
    av_vels[tt] = this_avgV;
    if(rank == MASTER){
        #ifdef DEBUG
          printf("==timestep: %d==\n", tt);
          printf("av velocity: %.12E\n", av_vels[tt]);
          printf("tot density: %.12E\n", total_density(params, cells));
          #endif
    }
  }

  func_gatherData(params,cells,tmp_cells,obstacles);

  MPI_Finalize();

  //if(rank == MASTER){ printf("MASTER HAS FINISHED TIMESTEPS \n");}
  //else{ printf("WORKER %d HAS FINISHED TIMERSTEPS!\n",rank);}

  if(rank != MASTER){
   // printf("---Worker %d is leaving---\n",rank );
   return EXIT_SUCCESS;
  }


  if(rank == MASTER){
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  func_write_values(params, cells, obstacles, av_vels);
 }
  /* write final values and free memory */


  // finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}


void func_haloExchange(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles){


 // printf("Worker %d sent ind %d to worker %d. Worker %d received into ind %d from worker %d\n",rank,myStartInd,botRank,rank,haloTop,topRank);
  int val = MPI_Sendrecv(&cells[0 + myStartInd*params.nx], params.nx, cells_struct,botRank,0,&cells[0 + haloTop*params.nx],params.nx,cells_struct, topRank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
 // printf("Worker %d sent ind %d to worker %d Worker %d received into ind %d from worker %d\n",rank,myEndInd,topRank,rank,haloBottom,botRank);
  int valTwo = MPI_Sendrecv(&cells[0 + (myEndInd-1)*params.nx], params.nx, cells_struct,topRank,0,&cells[0 + haloBottom*params.nx],params.nx,cells_struct,botRank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  // printf("Worker %d Finished Halo Exchange \n",rank);

}


float func_gatherVelocity(const t_param params,  t_speed *cells, int* obstacles){

    int collect_cells;
    float av = av_velocity_withoutDiv(params, cells, obstacles);
    float collect = 0;

    /*
    int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm)
    */

    //if(rank == MASTER){printf("Collect has value : %f before reduce\n",collect);}
    MPI_Reduce(&av, &collect, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&numOfCells, &collect_cells , 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    //if(rank == MASTER){printf("Collect has value : %f after reduce\n",collect);}



    if(rank == MASTER){
        // collect_cells += numOfCells;
        // printf("Master has av as : %f\n",collect);
        collect = (collect/(float) collect_cells);
        av = collect;
        return collect;
    }

    // printf("Workers %d have collect stuff %d \n",rank,numOfCells);
    // printf("Workers %d have average velocity %f \n",rank,av);
    return (av/numOfCells);
}



int getLimitsFromRankLower(int rank){
    int offset = floor(bigY/size);
    int lowerLim;
    // printf("inFunctionGetLimitsFromRank\n");
    if(rank == 0){
        lowerLim = 0;
        // printf("Returned values for rank %d from getLimitsFromRank \n",rank);
        return lowerLim;
    }
    if(rank == size-1){
        lowerLim = (offset * rank) + 1;
        // printf("Returned values for rank %d from getLimitsFromRank \n",rank);
        return lowerLim;
    }
    else{
        lowerLim = (rank * offset) + 1;
        // printf("Returned values for rank %d from getLimitsFromRank \n",rank);
        return lowerLim;
    }

}

int getLimitsFromRankUpper(int rank){
    int offset = floor(bigY/size);
    int upperLim;
    // printf("inFunctionGetLimitsFromRank\n");
    if(rank == 0){
        upperLim = offset+1;
        // printf("Returned values for rank %d from getLimitsFromRank \n",rank);
        return upperLim;
    }
    if(rank == size-1){
        upperLim = bigY;
        // printf("Returned values for rank %d from getLimitsFromRank \n",rank);
        return upperLim;
    }
    else{
        //lowerLim = (rank * offset) +1; which is why the bottom is this
        upperLim = ((rank * offset) + 1) + offset + 1;
        // printf("Returned values for rank %d from getLimitsFromRank \n",rank);
        return upperLim;
    }


}


// Currently implementing Idea 1 due to ease.
void func_gatherData(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles){
    if(rank == MASTER){
        // printf("Master Starting to Gather data \n");
        for(int i = 1; i < size; i ++){

            int this_lowerLim = getLimitsFromRankLower(i); // Basically the y limit lower
            int this_upperLim = getLimitsFromRankUpper(i); // Basically the y limit higher

//
            // printf("Worker %d's limits are lower : %d, upper : %d \n",i,this_lowerLim,this_upperLim);


            void* recvPointer = &cells[0 + this_lowerLim*params.nx];
            int recieveSize = params.nx * abs(this_lowerLim - this_upperLim);

            // printf("Master is trying to get work from %d \n",i);

            MPI_Recv(recvPointer, recieveSize, cells_struct, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // printf("Finished gatherData for rank %d\n",i);
        }
	// printf("MASTER HAS FINISHED GATHERING \n");
    }
    else{
        // printf("Worker %d trying to send \n",rank);

        void* sendbuffer = &cells[0 + myStartInd*params.nx];
        int sendSize = params.nx * abs(myStartInd - myEndInd);

        // printf("Worker %d lower : %d upper : %d  \n",rank, myStartInd, myEndInd);

        MPI_Send(sendbuffer, sendSize, cells_struct, 0, 1, MPI_COMM_WORLD);

        // printf("Worker %d has SENT THE DATA! \n",rank);
    }
    // printf("Leaving the gatherData Function! \n");
}

int func_timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    // printf("Worker %d starts timestep\n", rank);
    if(rank == size-1){
    //printf("Worker %d starts func_accelerate_flow\n", rank);
    func_accelerate_flow(params, cells, obstacles);
    }
    // printf("Worker %d starts Propogate\n", rank);
    func_propagate(params, cells, tmp_cells);
    // printf("Worker %d starts func_rebound\n", rank);
    func_rebound(params, cells, tmp_cells, obstacles);
    // printf("Worker %d starts func_colli\n", rank);
    func_collision(params, cells, tmp_cells, obstacles);
    // printf("Worker %d starts halo\n", rank);
    func_haloExchange(params,cells,tmp_cells,obstacles);
    // printf("Worker %d starts gathering data\n ",rank);
    // func_gatherData(params,cells,tmp_cells,obstacles);
    // printf("Worker %d finishes timestep\n", rank);
    return EXIT_SUCCESS;
}

int func_accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = myEndInd - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx] && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
      && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[ii + jj*params.nx].speeds[1] += w1;
      cells[ii + jj*params.nx].speeds[5] += w2;
      cells[ii + jj*params.nx].speeds[8] += w2;

      /* decrease 'west-side' densities */
      cells[ii + jj*params.nx].speeds[3] -= w1;
      cells[ii + jj*params.nx].speeds[6] -= w2;
      cells[ii + jj*params.nx].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int getHaloCellsForY(int attempt){
    if(attempt > myEndInd){
      return haloTop;
    }
    if(attempt < myStartInd){
      return haloBottom;
    }
  return attempt;
}

int func_propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
  // This is the function that requries making sure that the loops look at Halo'd cells.

  /* loop over _all_ cells */
  for (int jj = myStartInd; jj < myEndInd; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = getHaloCellsForY(jj + 1);
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1); //getHaloCellsForY(jj); //(jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);



      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

int func_rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  /* loop over the cells in the grid */
  for (int jj = myStartInd; jj < myEndInd; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
        cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
        cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
        cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
      }
    }
  }

  return EXIT_SUCCESS;
}

int func_collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = myStartInd; jj < myEndInd; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity_forAll(const t_param params, t_speed* cells, int* obstacles)
{
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;


    /* loop over all non-blocked cells */
    for (int jj = 0; jj < local_rows; jj++)
    {
      for (int ii = 0; ii < params.nx; ii++)
      {
        /* ignore occupied cells */
        if (!obstacles[ii + jj*params.nx])
        {
          /* local density total */
          float local_density = 0.f;

          for (int kk = 0; kk < NSPEEDS; kk++)
          {
            local_density += cells[ii + jj*params.nx].speeds[kk];
          }

          /* x-component of velocity */
          float u_x = (cells[ii + jj*params.nx].speeds[1]
                        + cells[ii + jj*params.nx].speeds[5]
                        + cells[ii + jj*params.nx].speeds[8]
                        - (cells[ii + jj*params.nx].speeds[3]
                           + cells[ii + jj*params.nx].speeds[6]
                           + cells[ii + jj*params.nx].speeds[7]))
                       / local_density;
          /* compute y velocity component */
          float u_y = (cells[ii + jj*params.nx].speeds[2]
                        + cells[ii + jj*params.nx].speeds[5]
                        + cells[ii + jj*params.nx].speeds[6]
                        - (cells[ii + jj*params.nx].speeds[4]
                           + cells[ii + jj*params.nx].speeds[7]
                           + cells[ii + jj*params.nx].speeds[8]))
                       / local_density;
          /* accumulate the norm of x- and y- velocity components */
          tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
          /* increase counter of inspected cells */
          ++tot_cells;
        }
      }
    }

    return tot_u / (float)tot_cells;
}

float av_velocity_withoutDiv(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  int tot_obs = 0;
  /* loop over all non-blocked cells */
  for (int jj = myStartInd; jj < myEndInd; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
      else if(obstacles[ii + jj*params.nx]){
          ++tot_obs;
      }
    }
  }

  numOfCells = tot_cells;
  numOfObstacles = tot_obs;


  return tot_u;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;


  /* loop over all non-blocked cells */
  for (int jj = myStartInd; jj < myEndInd; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

/*
** Allocate memory.
**
** Remember C is pass-by-value, so we need to
** pass pointers into the initialise function.
**
** NB we are allocating a 1D array, so that the
** memory will be contiguous.  We still want to
** index this memory as if it were a (row major
** ordered) 2D array, however.  We will perform
** some arithmetic using the row and column
** coordinates, inside the square brackets, when
** we want to access elements of this array.
**
** Note also that we are using a structure to
** hold an array of 'speeds'.  We will allocate
** a 1D array of these structs.
*/

int func_initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */


    /* open the parameter file */
    fp = fopen(paramfile, "r");
    if (fp == NULL)
    {
      sprintf(message, "could not open input parameter file: %s", paramfile);
      die(message, __LINE__, __FILE__);
    }
    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));
    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->ny));
    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->maxIters));
    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
    if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->density));
    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->accel));
    if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->omega));
    if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);
    /* and close up the file */
    fclose(fp);


    bigX = params->nx;
    bigY = params->ny;
    int offset = floor(bigY/size);

    if(rank == 0){ // First worker --> MASTER
      myStartInd = 0;
      myEndInd = offset+1;

      haloBottom = bigY; // bottom overflows to top.
      haloTop = myEndInd; // one above my upper limit

      topRank = rank+1; // 1
      botRank = size-1; // last one.
    }


    if(rank != MASTER){
        if(rank == size-1){
            myStartInd = (offset * rank) + 1;
            myEndInd = bigY;

            haloTop = 0;
            haloBottom = myStartInd - 1;

            topRank = 0;
            botRank = rank-1;
        }
        else{
            myStartInd = (offset * rank) + 1;
            myEndInd = myStartInd + offset;

            haloTop = myEndInd;
            haloBottom = myStartInd -1;

            topRank = rank+1;
            botRank = rank-1;
        }
    }
      local_cols = params->nx;
      local_rows = params->ny;


    //printf("Rank: %d, startInd = %d, endInd : %d, haloTop : %d, haloBottom: %d, topRank :%d, botrank :%d \n",rank,myStartInd,myEndInd,haloTop,haloBottom,topRank,botRank);

    /* main grid */
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (bigY * params->nx));
    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (bigY * params->nx));
    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int) * (bigY * params->nx));
    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    if( yy < 0 || yy > bigY -1){
      printf("Obstacle y coord out of range! %d for worker %d with (%d,%d) \n",yy,rank,params->startInd, params->endInd);
    }

    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > bigY - 1){
	 die("obstacle y-coord out of range", __LINE__, __FILE__);
	}
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int func_write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
