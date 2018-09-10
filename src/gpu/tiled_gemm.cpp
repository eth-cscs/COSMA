// include libraries
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda.h>

#define nstreams 4

void gpu_dgemm_(double* a, double* b, double* c,
          double* a_device, double* b_device, double* c_device,
          int m, int n, int k,
          double alpha, double beta) {

    // echo device data
    int idevice = 0;
    cudaSetDevice(idevice);
    cudaDeviceProp dprops;
    cudaGetDeviceProperties( &dprops, idevice );
    printf ("\nDevice name = %s, with compute capability %d.%d \n",
            dprops.name, dprops.major, dprops.minor);

    // define parameters
    int tile = 4096;

    // create communcations arrays
    double *pa;
    double *pb;
    double *pc;
    cudaMallocHost ( &pa, tile*tile*sizeof(double)*nstreams );
    cudaMallocHost ( &pb, tile*tile*sizeof(double)*nstreams );
    cudaMallocHost ( &pc, tile*tile*sizeof(double)*nstreams );

    // create a handle to cuBlas
    cublasHandle_t cublasHandle;
    cublasCreate( &cublasHandle );

    // allocate space on device - 3 tiles for a, b, c
    double *d_a;
    double *d_b;
    double *d_c;
    cudaMalloc ( &d_a, nstreams*tile*tile*sizeof(double) );
    cudaMalloc ( &d_b, nstreams*tile*tile*sizeof(double) );
    cudaMalloc ( &d_c, nstreams*tile*tile*sizeof(double) );

    int offset = tile*tile;

    int longest_dim = std::max(m, std::max(n, k));
    int ntiles = (longest_dim - 1) / tile + 1;

    cudaStream_t myStreams[nstreams];
    for ( int i=0; i<nstreams; i++ ) {
        cudaStreamCreate( &myStreams[i] );
    }

    cudaEvent_t bufferfilled[nstreams];
    for ( int i=0; i<nstreams; i++ ) {
        cudaEventCreate ( &bufferfilled[i] );
    }

    // record start time
    cudaEvent_t t_start;
    cudaEvent_t t_end;
    cudaEventCreate (&t_start);
    cudaEventCreate (&t_end);
    cudaEventRecord (t_start,0);

    // caches for indices of previous tiles in streams
    int prowtile[nstreams];
    int pcoltile[nstreams];

    // PERFORM MULTIPLICATION
    {
        int ibuff = 0;
        int itile = 0;

        // loop over inner tile dimension
        for ( int iktile = 0; iktile < ntiles; iktile++ ) {

            // loop over row tiles
            for ( int irowtile = 0; irowtile < ntiles; irowtile++ ) {

                // loop over column tiles
                for ( int icoltile = 0; icoltile < ntiles; icoltile++ ) {

                    if ( itile >= nstreams ) {

                        // block the host until this streams buffers are available
                        // (that is, all previous operations in this stream have completed)
                        cudaEventSynchronize ( bufferfilled[ibuff] );

                        // copy result in pinned buffer back to global matrix
// # pragma omp parallel for
                        for ( int i=0; i<tile; i++ ) {
                            for ( int j=0; j<tile; j++ ) {
                                c[(prowtile[ibuff]*tile+i)*n+pcoltile[ibuff]*tile+j] = pc[ibuff*offset+i*tile+j];
                            }
                        }
                    }

                    // copy next tile to pinned buffer
// # pragma omp parallel for
                    for ( int i=0; i<tile; i++ ) {
                        for ( int j=0; j<tile; j++ ) {
                            pa[ibuff*offset+i*tile+j] = a[(irowtile*tile+i)*n+iktile*tile+j];
                            pb[ibuff*offset+i*tile+j] = b[(iktile*tile+i)*n+icoltile*tile+j];
                            pc[ibuff*offset+i*tile+j] = c[(irowtile*tile+i)*n+icoltile*tile+j];
                        }
                    }

                    // copy tile data to device
                    cudaMemcpyAsync ( &d_a[ibuff*offset], &pa[ibuff*offset], tile*tile*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );
                    cudaMemcpyAsync ( &d_b[ibuff*offset], &pb[ibuff*offset], tile*tile*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );
                    cudaMemcpyAsync ( &d_c[ibuff*offset], &pc[ibuff*offset], tile*tile*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );

                    // tell cuBLAS which stream to use
                    cublasSetStream( cublasHandle, myStreams[ibuff] );


                    // perform dgemm
                    cublasDgemm ( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, tile, tile, tile, &alpha, &d_a[ibuff*offset], tile, &d_b[ibuff*offset], tile, &beta, &d_c[ibuff*offset], tile );
                    prowtile[ibuff] = irowtile;
                    pcoltile[ibuff] = icoltile;

                    // copy result back to host
                    cudaMemcpyAsync ( &pc[ibuff*offset], &d_c[ibuff*offset], tile*tile*sizeof(double), cudaMemcpyDeviceToHost, myStreams[ibuff] );

                    // this event will signal when the D2H copy of the result has completed
                    cudaEventRecord ( bufferfilled[ibuff], myStreams[ibuff] );

                    // update buffer / stream
                    ibuff++;
                    ibuff = ibuff%nstreams;
                    itile++;

                }
            }
        }

        for ( itile=0; itile < nstreams; itile ++ ) {

            // make sure that buffers are free
            cudaStreamSynchronize ( myStreams[itile] );

            // copy result in pinned buffer back to source
# pragma omp parallel for
            for ( int i=0; i<tile; i++ ) {
                for ( int j=0; j<tile; j++ ) {
                    c[(prowtile[itile]*tile+i)*n+pcoltile[itile]*tile+j] = pc[itile*offset+i*tile+j];
                }
            }

        }

    }

    // record end time
    cudaEventRecord (t_end,0);
    cudaEventSynchronize(t_end);
    float et;
    cudaEventElapsedTime (&et, t_start, t_end);

    // check results
    printf ("\nchecking results: ");
    bool correct = true;
    double abs_error, sum_abs_errors = 0;
// # pragma omp parallel for
    for ( int row = 0;  row < n; row++ ) {
        for ( int col = 0; col < n; col++ ) {

            abs_error = fabs(c[row * n + col] - a[row * n + col] );
            sum_abs_errors += abs_error;
            if (  abs_error > 10e-5 ) {
                printf ("FAILED\n\nerror: c[%d]: %f != a[%d]: %f",
                        row * n + col,  c[row * n + col], row * n + col,  a[row * n + col]);
                correct = false;
                break;
            }
        }
    }

    // report results
    if ( correct ) {
        printf ("SUCCESS");
        printf ("\nSum abs errors: %f", sum_abs_errors);
        printf("\nelapsedTime        = %4.4f seconds\n", (double)et/1000.);     // cudaEventElapsedTime is in milliseconds
        printf(  "gigaflops achieved = %4.4f Gflops/s\n\n\n", 2.0e-6*n*n*n/et); // 2( * and + ) *n (inner dimension)*n^2(result size)/(time in ms.)
    } else {
        printf ("\nResult not correct, check your code !\n");
    }

    // clean up
    cublasDestroy ( cublasHandle );
    cudaEventDestroy ( t_start  );
    cudaEventDestroy ( t_end );

    cudaFreeHost ( pa );
    cudaFreeHost ( pb );
    cudaFreeHost ( pc );

    cudaFree ( d_a );
    cudaFree ( d_b );
    cudaFree ( d_c );

    free (a);
    free (b);
    free (c);
}
