#include "library.h"

//STL
#include <cstring>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>


double read_timer( )
{
  static bool initialized = false;
  static struct timeval start;
  struct timeval end;
  if( !initialized )
    {
      gettimeofday( &start, NULL );
      initialized = true;
    }

  gettimeofday( &end, NULL );

  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void fill( double *p, int n ) {
  for( int i = 0; i < n; i++ )
    p[i] = 2*drand48()-1;
}

void printMatrix( double *A, int n ) {
  for( int i = 0; i < n; i++ ) {
    for( int j = 0; j < n; j++ )
      printf("%7.2g ", A[i+j*n]);
    printf("\n");
  }
}

int find_option( int argc, char **argv, const char *option )
{
  for( int i = 1; i < argc; i++ )
    if( strcmp( argv[i], option ) == 0 )
      return i;
  return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
  int iplace = find_option( argc, argv, option );
  if( iplace >= 0 && iplace < argc-1 )
    return atoi( argv[iplace+1] );
  return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
  int iplace = find_option( argc, argv, option );
  if( iplace >= 0 && iplace < argc-1 )
    return argv[iplace+1];
  return default_value;
}
