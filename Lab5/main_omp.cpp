#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

struct COOMatrix {
    vector<size_t> rows;
    vector<size_t> cols;
    vector<double> values;

    size_t nrows = 0;
    size_t ncols = 0;

    void add_val(size_t r_idx, size_t c_idx, double val) {
        rows.push_back(r_idx);
        cols.push_back(c_idx);
        values.push_back(val);
    }

    void random(size_t rows_, size_t cols_, size_t nnz) {
        nrows = rows_;
        ncols = cols_;

        rows.clear();
        cols.clear();
        values.clear();

        std::srand(std::time(nullptr));
        std::unordered_set<size_t> used;

        while (rows.size() < nnz) {
            size_t r = rand() % nrows;
            size_t c = rand() % ncols;
            size_t hash = r * ncols + c;
            if (used.find(hash) == used.end()) {
                used.insert(hash);
                double val = (rand()) % 10 + 1;
                add_val(r, c, val);
            }
        }
    }

    void print() {
        cout << "COO Matrix:\n";
        for (size_t i = 0; i < rows.size(); i++) {
            cout << "[ " << rows[i] << "\t" << cols[i] << "\t" << values[i] << " ]\n";
        }
    }
};

int main() {
    COOMatrix A;
    vector<double> x;

    size_t rows = 100000, cols = 100000, nnz = 50000000;

    A.random(rows, cols, nnz);
    //A.print();

    x.resize(cols);
    for (auto &el : x) {
        el = (rand()) % 10 + 1;
    }
    // Замер общего времени
    double start_time = omp_get_wtime();

    // Замер времени без инициализации
    double start_compute_time = omp_get_wtime();

    // Умножение матрицы на вектор с OpenMP
    vector<double> result(rows, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < nnz; ++i) {
        size_t r = A.rows[i];
        size_t c = A.cols[i];
        double v = A.values[i];

        #pragma omp atomic
        result[r] += v * x[c];
    }

    double end_compute_time = omp_get_wtime();
    double end_time = omp_get_wtime();

    // Вывод времени
    cout << "Execution time: " << (end_time - start_time) << " seconds\n";
    cout << "Execution time (compute only): " << (end_compute_time - start_compute_time) << " seconds\n";

    return 0;
}
