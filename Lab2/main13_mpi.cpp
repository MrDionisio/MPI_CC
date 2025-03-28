#include <iostream>
#include <math.h>
#include <mpi.h>
#include <chrono>
#include <random>


using namespace std;
using namespace chrono;





int main(int argc, char** argv){

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);  


    int n = 1e6;

    double *x = nullptr;
    int *sendcount{nullptr},  *displace{nullptr};
    int local_n = n/size;

    if (rank==0){
        srand(1);
        x = new double[n];
        for(int i=0; i<n; i++){
            x[i]=static_cast<double>(rand()%100);
        }
        sendcount = new int[size];
        displace = new int[size];
        for(int i=0; i<size; i++){
            sendcount[i] = local_n;
            if(i==size-1){
                sendcount[size-1]=n - local_n * (size-1);
            }
        }
        displace[0] = 0;
        for(int i=1; i<size; i++){
            displace[i] = displace[i-1]+sendcount[i-1];
        }
    }

    if(rank==size-1){
        local_n = n - local_n * rank;
    }

    double* local_x = new double[local_n];

    
    MPI_Scatterv(x, sendcount, displace, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_mean=0.0;
    for(int i=0; i<local_n; i++){
        local_mean+=local_x[i];
    }
    
    double mean;
    MPI_Reduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank==0){
        mean/=n;
        cout <<"Mean = "<< mean<<endl;
    }
    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto start_mpi = high_resolution_clock::now();
    double std_dev_local;
    for(int k=0; k<10000; k++){
        std_dev_local=0.0;
        for(int i=0; i<local_n; i++){
            std_dev_local += (local_x[i]-mean)*(local_x[i]-mean);
        }
    }


    double std_dev;
    MPI_Reduce(&std_dev_local, &std_dev, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank==0){
        std_dev = sqrt(std_dev/n);
        cout <<"Standart Deviation = "<< std_dev <<endl;
    }

    auto end_mpi = high_resolution_clock::now();

    double mpi_time = duration_cast<microseconds>(end_mpi - start_mpi).count() / 1e6;

    if(rank==0){
        cout << "Execution Time: " << mpi_time << endl; 
        delete[] x, sendcount, displace;
    }
    delete[] local_x;


    MPI_Finalize();
    return 0;
}