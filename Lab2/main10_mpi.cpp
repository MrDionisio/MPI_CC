#include <iostream>
#include <math.h>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace chrono;

const double pi = 4*atan(1.0);

double f(double x){
    if(x==0){x+=1e-8;}
    
    return x*log(sin(x));

}



int main(int argc, char** argv){

    MPI_Init(&argc, &argv);
    double a=0;
    double b=pi;

    double res = -pi*pi/2*log(2.0);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);  

    int n = 1000000000;


    int local_n;
    double h = (b-a)/n;
    double local_a, local_b;

    local_n = n/size;
 
    if (rank==size-1){
        local_n = n - local_n*rank;
    }

    local_a = a + rank*local_n*h;
    local_b = local_a + h;

    //Формула Котеса
    auto start_mpi = high_resolution_clock::now();
    double local_sum = 0;
    double x;
    for (int i = 1; i < local_n; i++) {
        x = local_a + (i-0.5) * h;
        local_sum += f(x);
    }
    local_sum *= h;


    double mpi_sum=0.0;
    MPI_Reduce(&local_sum, &mpi_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    auto end_mpi = high_resolution_clock::now();

    double mpi_time = duration_cast<microseconds>(end_mpi - start_mpi).count() / 1e6;

    if(rank==0){
        cout << "Execution time: " << mpi_time << " seconds\n";
        cout << "Numerical integral = " << mpi_sum << "\n";
        cout << "Theoretical integral = " << res << "\n";
    }


    MPI_Finalize();
    return 0;
}