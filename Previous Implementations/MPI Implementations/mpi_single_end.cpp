/* ------------
 * The code is adapted from the XSEDE online course Applications of Parallel Computing. 
 * The copyright belongs to all the XSEDE and the University of California Berkeley staff
 * that moderate this online course as well as the University of Toronto CSC367 staff.
 * This code is provided solely for the use of students taking the CSC367 course at 
 * the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * -------------
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <vector>
using namespace std;


//
//  tuned constants (taken from common.cpp)
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

// Get the bin number for a particular particle
int get_bin_number(particle_t &p, int bins_per_row){
	return floor(p.x / cutoff) * bins_per_row + floor(p.y / cutoff);
}

int get_row(particle_t &p, int bins_per_row){
  int current_bin = get_bin_number(p, bins_per_row);
  int row = current_bin / bins_per_row;
  return row;
}

// int get_row(particle_t &p){
//   return floor(p.x / cutoff);
// }

int get_processor_number_for_particle(particle_t &p, int rowsofbin_per_proc, int bins_per_row){
	int row = get_row(p, bins_per_row);
  int proc = row / rowsofbin_per_proc;
  return proc;
}

double get_size( int n )
{
  double size = sqrt( density * n );
  return size;
}

int determin_move_proc(particle_t &p, int rank, int  rowsofbin_per_proc, int bins_per_row){
  int proc = get_processor_number_for_particle(p, rowsofbin_per_proc, bins_per_row);
  if (proc < rank){
    return -1;
  }
  if (proc > rank){
    return 1;
  }
  // CHECKER, if the proc is differ from the rank by more than 1
  /*
  if ((proc < (rank - 1)) || (proc > (rank + 1))){
    exit(1);
  }
  */
  return 0;
}

int determin_edge(particle_t &p, int rank, int rowsofbin_per_proc, int n_proc, int bins_per_row){
	int row = get_row(p, bins_per_row);
  int upper_edge = rank * rowsofbin_per_proc;
  int lower_edge = ((rank+1) * rowsofbin_per_proc) - 1;
  // printf("upper_edge: %d, lower_edge: %d, row: %d\n", upper_edge, lower_edge, row);
  // last processor's lower edge does not matter here as not needed
  if ((rank+1) == n_proc){
    lower_edge = bins_per_row;
  }
  

  if (row == upper_edge){
    return -1;
  }
  if (row == lower_edge){
    return 1;
  }
  // CHECKER, more than edge
  /*
  if ((row < upper_edge) || (row > lower_edge)){
    exit(1);
  }
  */
  return 0;
}

int determin_send(particle_t &p, int rank, int rowsofbin_per_proc, int n_proc, int bins_per_row){
	int row = get_row(p, bins_per_row);
  int upper_edge = rank * rowsofbin_per_proc;
  int lower_edge = ((rank+1) * rowsofbin_per_proc) - 1;
  // printf("upper_edge: %d, lower_edge: %d, row: %d\n", upper_edge, lower_edge, row);
  // last processor's lower edge does not matter here as not needed
  if ((rank+1) == n_proc){
    lower_edge = bins_per_row;
  }
  

  if (row <= upper_edge){
    return -1;
  }
  if (row >= lower_edge){
    return 1;
  }
  // CHECKER, more than edge
  /*
  if ((row < upper_edge) || (row > lower_edge)){
    exit(1);
  }
  */
  return 0;
}

// return 0 if in range
int determin_in_range(particle_t &p, int rank, int rowsofbin_per_proc, int n_proc, int bins_per_row){
	int row = get_row(p, bins_per_row);
  int upper_edge = rank * rowsofbin_per_proc;
  int lower_edge = ((rank+1) * rowsofbin_per_proc) - 1;
  // printf("upper_edge: %d, lower_edge: %d, row: %d\n", upper_edge, lower_edge, row);
  // last processor's lower edge does not matter here as not needed
  if ((rank+1) == n_proc){
    lower_edge = bins_per_row;
  }
  

  if (row >= upper_edge){
    return 0;
  }
  if (row <= lower_edge){
    return 0;
  }
  // CHECKER, more than edge
  /*
  if ((row < upper_edge) || (row > lower_edge)){
    exit(1);
  }
  */
  return 1;
}

// Clear all particle bins
void clear_bins(vector<particle_t> *particle_bins, int number_of_bins){
	for(int i = 0; i < number_of_bins; i++){
		particle_bins[i].clear();
	}
}

// Place particles in their respective bin's
void place_particles(vector<particle_t> *particle_bins, particle_t *particles, int number_of_particles, int bins_per_row){
	int particle_bin_number;
	for(int i = 0; i < number_of_particles; i++){
		particle_bin_number = get_bin_number(particles[i], bins_per_row);
		particle_bins[particle_bin_number].push_back(particles[i]);
	}
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    // printf("n_proc: %d\n", n_proc);
    // printf("rank: %d\n", rank);
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ )
        partition_offsets[i] = min( i * particle_per_proc, n );
    
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];
    
    //
    //  allocate storage for local partition
    //
    // int nlocal = partition_sizes[rank];
    // particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
    int nlocal;
    vector<particle_t> local;
    vector<particle_t> cur_particles;
    vector<particle_t> tmp_particles;

    vector<particle_t> new_local_up;
    vector<particle_t> new_local_down;
    vector<particle_t> move_up;
    vector<particle_t> move_down;

    vector<particle_t> send_up;
    vector<particle_t> send_down;
    vector<particle_t> receive_up;
    vector<particle_t> receive_down;

    vector<particle_t> edge_up;
    vector<particle_t> edge_down;
    vector<particle_t> ghost_from_up;
    vector<particle_t> ghost_from_down;

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );

    // We partition the 2d simulation space into bins per the ProjectHelper.pdf suggestions
    // Reading move() from common.cpp we deduce that the simulation space is size * size
    // Using A2 as motivation, we decide to store our bins that represent a 2d space in a 1d manner
    // using vectors as they are dynamically allocated and work well for our purpose
    
    int bins_per_row = ceil(sqrt(density * n) / cutoff);					// We first calculate how many bins per row of the simulation space there are of size cutoff
    // printf("bins_per_row %d\n", bins_per_row);
    int number_of_bins = bins_per_row * bins_per_row;					// Since we are dealing with a square simulation space
    vector<particle_t> *particle_bins = new vector<particle_t>[number_of_bins];	// Container for all particle bins in the space

    int rowsofbin_per_proc = (bins_per_row + n_proc - 1) / n_proc;
    // printf("rowsofbin_per_proc %d\n", rowsofbin_per_proc);
    // int start_row = rank*rowsofbin_per_proc;
    // int end_row = (rank+1)*rowsofbin_per_proc;
    // if ((rank+1) == n_proc) {
    //   end_row = bins_per_row;
    // }


    // printf("start initialiation\n");
    // initial distribution of particles to different processors
    int recieve_amount;
    int total_vefore_distribute = 0;
    if( rank == 0 ) {
      int proc_num;
      vector<particle_t> *processors_vectors = new vector<particle_t>[n_proc];	// Container for particles for different processors
      // put particles in to processor arries
      // printf("gate 0\n");
      for (int i = 0; i<n; i++){
        proc_num = get_processor_number_for_particle(particles[i], rowsofbin_per_proc, bins_per_row);
        // printf("proc_num: %d\n", proc_num);
        processors_vectors[proc_num].push_back(particles[i]);
      }
      // printf("gate 1\n");

      // save local for master
      local = processors_vectors[0];
      // printf("gate 2\n");
      
      for (int i = 0; i<n_proc; i++){
        // printf("for processor %d, the amount of particles is : %d\n", i, processors_vectors[i].size());
        total_vefore_distribute += processors_vectors[i].size();
      }
      // printf("total_vefore_distribute: %d\n", total_vefore_distribute);
      
      // mast save for it self, so start from 1
      for (int i = 1; i<n_proc; i++){
        MPI_Send(&(processors_vectors[i]).front(), processors_vectors[i].size(), PARTICLE, i, 0, MPI_COMM_WORLD);
      }
      // printf("gate 3\n");
    }
    // receive, if not master, dynamic receive amount by using MPI_Probe
    else{
      MPI_Status m_status;
      MPI_Probe(0, 0, MPI_COMM_WORLD, &m_status);
      MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
      local.resize(recieve_amount);
      MPI_Recv(&local[0], recieve_amount, PARTICLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // printf("finish initialiation\n");

    

    //
    //  simulate a number of time steps
    //
		int bin_row, bin_col, neighbor_row, neighbor_col, current_bin, neighbor_bin, move_result, edge_result, send_result;
    double simulation_time = read_timer( );
    int total;

    for( int step = 0; step < NSTEPS; step++ )
    // for( int step = 0; step < 100; step++ )
    {
      printf("step %d\n", step);
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        nlocal = local.size();
        // MPI_Reduce(&nlocal,&total,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
        // printf("total at the start of the loop: %d\n", total);

        // cur_particles = local;

        // go over local, remove not in range, move and send
        // for (int i = 0; i<cur_particles.size(); i++) {
        //   // printf("i %d\n", i);
        //   // send_result = determin_send(cur_particles[i], rank, rowsofbin_per_proc, n_proc, bins_per_row);
        //   send_result = determin_edge(local[i], rank, rowsofbin_per_proc, n_proc, bins_per_row);
        //   if (send_result == -1){
        //     send_up.push_back(cur_particles[i]);
        //   }
        //   else if (send_result == 1) {
        //     send_down.push_back(cur_particles[i]);
        //   }
          
        //   send_result = determin_move_proc(local[i], rank, rowsofbin_per_proc, bins_per_row);
        //   if (send_result == -1){
        //     send_up.push_back(cur_particles[i]);
        //   }
        //   else if (send_result == 1) {
        //     send_down.push_back(cur_particles[i]);
        //   }
        // }

        for (int i = 0; i<local.size(); i++) {
          // printf("i %d\n", i);
          move_result = determin_move_proc(local[i], rank, rowsofbin_per_proc, bins_per_row);
          if (move_result == -1){
            send_up.push_back(local[i]);
            ghost_from_up.push_back(local[i]);
            local.erase(local.begin()+i);
          }
          else if (move_result == 1) {
            send_down.push_back(local[i]);
            ghost_from_down.push_back(local[i]);
            local.erase(local.begin()+i);
          }
        }
        for (int i = 0; i<local.size(); i++) {
          // printf("i: %d\n", i);
          edge_result = determin_edge(local[i], rank, rowsofbin_per_proc, n_proc, bins_per_row);
          if (edge_result == -1){
            send_up.push_back(local[i]);
          }
          else if (edge_result == 1) {
            send_down.push_back(local[i]);
          }
        }
        printf("send_up.size(): %d, send_down.size(): %d\n", send_up.size(), send_down.size());
        // local.clear();

        // a barrier before all send and receive start
        MPI_Barrier(MPI_COMM_WORLD);

        if (n_proc > 1){
          if ( rank % 2 == 0 ) {  // even
            // if first
            if (rank == 0) {
              MPI_Send(&send_down.front(), send_down.size(), PARTICLE, rank+1, 0, MPI_COMM_WORLD);
              // receive from below
              MPI_Status m_status;
              MPI_Probe(rank+1, 0, MPI_COMM_WORLD, &m_status);
              MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
              receive_down.resize(recieve_amount);
              MPI_Recv(&receive_down[0], recieve_amount, PARTICLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              cur_particles.insert(cur_particles.end(), receive_down.begin(), receive_down.end());
              // CHECKER, if the receive_down.size() == recieve_amount
            }
            // if last
            else if ((rank+1) == n_proc){
              MPI_Send(&send_up.front(), send_up.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD);
              // receive from above
              MPI_Status m_status;
              MPI_Probe(rank-1, 0, MPI_COMM_WORLD, &m_status);
              MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
              receive_up.resize(recieve_amount);
              MPI_Recv(&receive_up[0], recieve_amount, PARTICLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              cur_particles.insert(cur_particles.end(), receive_up.begin(), receive_up.end());
            }
            // else
            else {
              MPI_Send(&send_up.front(), send_up.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD);
              MPI_Send(&send_down.front(), send_down.size(), PARTICLE, rank+1, 0, MPI_COMM_WORLD);
              // receive from above
              MPI_Status m_status;
              MPI_Probe(rank-1, 0, MPI_COMM_WORLD, &m_status);
              MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
              receive_up.resize(recieve_amount);
              MPI_Recv(&receive_up[0], recieve_amount, PARTICLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              cur_particles.insert(cur_particles.end(), receive_up.begin(), receive_up.end());
              // receive from below
              MPI_Probe(rank+1, 0, MPI_COMM_WORLD, &m_status);
              MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
              receive_down.resize(recieve_amount);
              MPI_Recv(&receive_down[0], recieve_amount, PARTICLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              cur_particles.insert(cur_particles.end(), receive_down.begin(), receive_down.end());
            }

          } 
          else {  // odd
          // if first
          if (rank == 0) {
            // receive from below
            MPI_Status m_status;
            MPI_Probe(rank+1, 0, MPI_COMM_WORLD, &m_status);
            MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
            receive_down.resize(recieve_amount);
            MPI_Recv(&receive_down[0], recieve_amount, PARTICLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cur_particles.insert(cur_particles.end(), receive_down.begin(), receive_down.end());

            MPI_Send(&send_down.front(), send_down.size(), PARTICLE, rank+1, 0, MPI_COMM_WORLD);
          }
          // if last
          else if ((rank+1) == n_proc){
            // receive from above
            MPI_Status m_status;
            MPI_Probe(rank-1, 0, MPI_COMM_WORLD, &m_status);
            MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
            receive_up.resize(recieve_amount);
            MPI_Recv(&receive_up[0], recieve_amount, PARTICLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cur_particles.insert(cur_particles.end(), receive_up.begin(), receive_up.end());
            
            MPI_Send(&send_up.front(), send_up.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD);
          }
          // else
          else {
            // receive from above
            MPI_Status m_status;
            MPI_Probe(rank-1, 0, MPI_COMM_WORLD, &m_status);
            MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
            receive_up.resize(recieve_amount);
            MPI_Recv(&receive_up[0], recieve_amount, PARTICLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cur_particles.insert(cur_particles.end(), receive_up.begin(), receive_up.end());
            // receive from below
            MPI_Probe(rank+1, 0, MPI_COMM_WORLD, &m_status);
            MPI_Get_count(&m_status, PARTICLE, &recieve_amount);
            receive_down.resize(recieve_amount);
            MPI_Recv(&receive_down[0], recieve_amount, PARTICLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cur_particles.insert(cur_particles.end(), receive_down.begin(), receive_down.end());

            MPI_Send(&send_up.front(), send_up.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD);
            MPI_Send(&send_down.front(), send_down.size(), PARTICLE, rank+1, 0, MPI_COMM_WORLD);
          }
        }
          // printf("finish send new local\n");
        }

        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
        tmp_particles = cur_particles;
        cur_particles.insert(cur_particles.end(), local.begin(), local.end());

        for(int i = 0; i < tmp_particles.size(); i++){
          if (determin_in_range(tmp_particles[i], rank, rowsofbin_per_proc, n_proc, bins_per_row) == 0) {
            local.push_back(tmp_particles[i]);
          }
        }



        // As per ProjectHelper.pdf algorithm outline method 1, we clear the bins
        // at each time stamp
        clear_bins(particle_bins, number_of_bins);
        place_particles(particle_bins, &cur_particles[0], cur_particles.size(), bins_per_row);


        nlocal = local.size();
        MPI_Reduce(&nlocal,&total,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
        printf("total before start calculation: %d\n", total);

        // printf("Start calculation.\n");
        // Serial O(n) implementation
        for(int i = 0; i < nlocal; i++){
          local[i].ax = local[i].ay = 0;
          current_bin = get_bin_number(local[i], bins_per_row);
          bin_row = current_bin / bins_per_row;
          bin_col = current_bin % bins_per_row;
          for(int j = -1; j <= 1; j++){
            for(int k = -1; k <= 1; k++){
              neighbor_row = bin_row + k;
              neighbor_col = bin_col + j;
              if ((neighbor_row>=0) && (neighbor_col>=0) && (neighbor_row<bins_per_row) && (neighbor_col<bins_per_row)){
                neighbor_bin = bins_per_row*neighbor_row + neighbor_col;
                for(int p = 0; p < particle_bins[neighbor_bin].size(); p++){
                  apply_force(local[i], particle_bins[neighbor_bin][p], &dmin, &davg, &navg);
                  // printf("%f %f %f\n", dmin, davg, navg);
                }
              }
            }
          }
        }

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        // printf("Finish MPI_Reduce.\n");


        //
        //  move particles
        //
        for( int i = 0; i < nlocal; i++ )
            move( local[i] );
        // printf("Finish move.\n");

        
        // clear
        cur_particles.clear();
        new_local_up.clear();
        new_local_down.clear();
        move_up.clear();
        move_down.clear();
        ghost_from_up.clear();
        ghost_from_down.clear();
        edge_up.clear();
        edge_down.clear();
        send_up.clear();
        send_down.clear();
        receive_up.clear();
        receive_down.clear();
        // printf("Finish the loop.\n");
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -the minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    // free( local );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
