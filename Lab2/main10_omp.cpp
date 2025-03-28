#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

const double pi = 4*atan(1.0);

// Подынтегральная функция
double f(double x) {
    if(x == 0) { x += 1e-8; }
    return x * log(sin(x));
}

int main() {
    const double a = 0;
    const double b = pi;
    const double res = -pi*pi/2*log(2.0);  
    const int n = 1000000000;               

    const double h = (b - a) / n;           
    double sum = 0.0;

    auto start = high_resolution_clock::now();


    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= n; i++) {
        double x = a + (i - 0.5) * h;
        sum += f(x);
    }
    sum *= h;

    auto end = high_resolution_clock::now();
    double exec_time = duration_cast<microseconds>(end - start).count() / 1e6;


    cout << "Execution time: " << exec_time << " seconds\n";
    cout << "Numerical integral = " << sum << "\n";
    cout << "Theoretical integral = " << res << "\n";

    return 0;
}