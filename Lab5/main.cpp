#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <ctime>

using namespace std;

struct COOMatrix {
    vector<size_t> rows;
    vector<size_t> cols;
    vector<double> values;

    size_t nrows = 0;
    size_t ncols = 0;

    COOMatrix() {}

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    COOMatrix A;
    vector<double> x;
    size_t rows = 100000, cols = 100000, nnz = 50000000;

    if (rank == 0) {
        A.random(rows, cols, nnz);
        //A.print();

        x.resize(cols);
        for (auto &el : x) {
            el = (rand()) % 10 + 1;
        }

        // cout << "X Vector:\n";
        // for (auto el : x) cout << el << "\t";
        // cout << "\n";
    }

    // Замер времени: старт
    double start_time = MPI_Wtime();

    // Широковещание размеров
    MPI_Bcast(&rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Широковещание вектора x
    if (rank != 0) x.resize(cols);
    MPI_Bcast(x.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Широковещание COO
    size_t nnz_total = A.values.size();
    MPI_Bcast(&nnz_total, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    vector<size_t> all_rows, all_cols;
    vector<double> all_vals;

    if (rank == 0) {
        all_rows = A.rows;
        all_cols = A.cols;
        all_vals = A.values;
    } else {
        all_rows.resize(nnz_total);
        all_cols.resize(nnz_total);
        all_vals.resize(nnz_total);
    }

    MPI_Bcast(all_rows.data(), nnz_total, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_cols.data(), nnz_total, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_vals.data(), nnz_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Разбиение по вертикальным полосам
    size_t stripe_width = cols / size;
    size_t col_start = rank * stripe_width;
    size_t col_end = (rank == size - 1) ? cols : col_start + stripe_width;
    vector<size_t> local_idx;
    size_t c;
    for (int i = 0; i<nnz_total; i++) {
        c = all_cols[i];
        if (c >= col_start && c < col_end) {
            local_idx.push_back(i);
        }
 
    }
    // Замер времени: старт
    double start_time_loc = MPI_Wtime();
    // Локальное умножение
    vector<double> local_result(rows, 0.0);
    for (auto idx : local_idx) {
        size_t r = all_rows[idx];
        size_t c = all_cols[idx];
        double v = all_vals[idx];
            local_result[r] += v * x[c];
    }
    // Замер времени: конец
    double end_time_loc = MPI_Wtime();

    // Сбор локальных результатов
    vector<double> final_result(rows, 0.0);
    MPI_Reduce(local_result.data(), final_result.data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Замер времени: конец
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    double elapsed_loc = end_time_loc - start_time_loc;

    // Вывод результатов и времени
    if (rank == 0) {
        // cout << "Result Ax:\n";
        // for (auto val : final_result) {
        //     cout << val << "\t";
        // }
        // cout << "\n";

        cout << "Execution time: " << elapsed << " seconds\n";
        cout << "Execution time without communication: " << elapsed_loc << " seconds\n";

    }

    MPI_Finalize();
    return 0;
}
