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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <vector>
using namespace std;

int bins_per_row;


//
//  tuned constants (taken from common.cpp)
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

// Get the bin number for a particular particle
int get_bin_number(particle_t &p){
	return floor(p.x / cutoff) * bins_per_row + floor(p.y / cutoff);
}

// Clear all particle bins
void clear_bins(vector<particle_t> *particle_bins, int number_of_bins){
    #pragma omp for 
	for(int i = 0; i < number_of_bins; i++){
		particle_bins[i].clear();
	}
}

// Place particles in their respective bin's
void place_particles(vector<particle_t> *particle_bins, particle_t *particles, int number_of_particles){
	int particle_bin_number;
	for(int i = 0; i < number_of_particles; i++){
		particle_bin_number = get_bin_number(particles[i]);
		particle_bins[particle_bin_number].push_back(particles[i]);
	}
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

	// We partition the 2d simulation space into bins per the ProjectHelper.pdf suggestions
	// Reading move() from common.cpp we deduce that the simulation space is size * size
	// Using A2 as motivation, we decide to store our bins that represent a 2d space in a 1d manner
	// using vectors as they are dynamically allocated and work well for our purpose

	bins_per_row = ceil(sqrt(density * n) / cutoff);					// We first calculate how many bins per row of the simulation space there are of size cutoff
	int number_of_bins = bins_per_row * bins_per_row;					// Since we are dealing with a square simulation space
	vector<particle_t> *particle_bins = new vector<particle_t>[number_of_bins];	// Container for all particle bins in the space
	// vector<particle_t> *particle_bins = (vector<particle_t> *) malloc(number_of_bins * sizeof(vector<particle_t>));
    // // new vector<particle_t>[number_of_bins];	// Container for all particle bins in the space

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

	int bin_row, bin_col, neighbor_row, neighbor_col, current_bin, neighbor_bin;


    #pragma omp parallel private(dmin) 
    {
    numthreads = omp_get_num_threads();
    for( int step = 0; step < 1000; step++ )
    {
        navg = 0;
        davg = 0.0;
	dmin = 1.0;

		// As per ProjectHelper.pdf algorithm outline method 1, we clear the bins
		// at each time stamp
		clear_bins(particle_bins, number_of_bins);
        #pragma omp barrier


		// Then we place all the particles in their respective bins
        #pragma omp single
        {
		    place_particles(particle_bins, particles, n);
        }

        #pragma omp barrier
        
        //
        //  compute all forces
        //

        #pragma omp for reduction (+:navg) reduction(+:davg) \
        private(current_bin, bin_row, bin_col, neighbor_row, neighbor_col, neighbor_bin)
		for(int i = 0; i < n; i++){
			particles[i].ax = particles[i].ay = 0;
			current_bin = get_bin_number(particles[i]);
			bin_row = current_bin / bins_per_row;
			bin_col = current_bin % bins_per_row;
			for(int j = -1; j <= 1; j++){
				for(int k = -1; k <= 1; k++){
					neighbor_row = bin_row + k;
					neighbor_col = bin_col + j;
					if ((neighbor_row>=0) && (neighbor_col>=0) && (neighbor_row<bins_per_row) && (neighbor_col<bins_per_row)){
						neighbor_bin = bins_per_row*neighbor_row + neighbor_col;
						for(int p = 0; p < particle_bins[neighbor_bin].size(); p++){
							apply_force(particles[i], particle_bins[neighbor_bin][p], &dmin, &davg, &navg);
						}
					}
				}
			}
		}

        // #pragma omp for reduction (+:navg) reduction(+:davg)
        // for( int i = 0; i < n; i++ )
        // {
        //     particles[i].ax = particles[i].ay = 0;
		// // 	current_bin = get_bin_number(particles[i]);
        //     for (int j = 0; j < n; j++ )
        //         apply_force( particles[i], particles[j],&dmin,&davg,&navg);
        // }
        
		
        //
        //  move particles
        //
        #pragma omp for
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );
  
        if( find_option( argc, argv, "-no" ) == -1 ) 
        {
          //
          //  compute statistical data
          //
          #pragma omp master
          if (navg) { 
            absavg += davg/navg;
            nabsavg++;
          }

          #pragma omp critical
	  if (dmin < absmin) absmin = dmin; 
		
          //
          //  save if necessary
          //
          #pragma omp master
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    }
}
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

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
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
