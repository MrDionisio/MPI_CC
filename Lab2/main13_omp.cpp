#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;
using namespace chrono;

int main() {
    const int n = 1e6;
    const int iterations = 10000;
    
    // Инициализация данных
    double* x = new double[n];
    srand(1);
    for(int i = 0; i < n; i++) {
        x[i] = static_cast<double>(rand() % 100);
    }

    // Вычисление среднего значения

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++) {
        sum += x[i];
    }
    double mean = sum / n;
    cout << "Mean = " << mean << endl;

    // Вычисление стандартного отклонения
    int i;
    auto start = high_resolution_clock::now();    
    double variance = 0.0;

    for(int k=0; k<10000;k++){
        variance = 0.0;
        #pragma omp parallel for reduction(+:variance)
        for(i = 0; i < n; i++) {
            variance += (x[i] - mean) * (x[i] - mean);
        }
    }

    double std_dev = sqrt(variance / n);
    cout << "Standard Deviation = " << std_dev << endl;

    auto end = high_resolution_clock::now();
    double exec_time = duration_cast<microseconds>(end - start).count() / 1e6;
    cout << "Execution Time: " << exec_time << " seconds" << endl;

    // Очистка памяти
    delete[] x;

    return 0;
}