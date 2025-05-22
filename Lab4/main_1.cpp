#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 20000;  // Размер матрицы N x N и вектора N
    std::vector<int> matrix;
    std::vector<int> vector(N);
    std::vector<int> result;

    // Генерация данных в корневом процессе
    if (rank == 0) {
        matrix.resize(N * N);
        std::srand(std::time(nullptr));
        
        for (int i = 0; i < N * N; ++i)
            matrix[i] = std::rand() % 9 + 1;
        
        for (int i = 0; i < N; ++i)
            vector[i] = std::rand() % 9 + 1;
    }

    // Замер времени выполнения (включая коммуникации)
    

    // Распространение вектора
    MPI_Bcast(vector.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    // Распределение матрицы
    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    int send_counts[size], displs[size];
    for (int r = 0; r < size; ++r) {
        send_counts[r] = (rows_per_proc + (r < remainder ? 1 : 0)) * N;
        displs[r] = (r > 0) ? displs[r-1] + send_counts[r-1] : 0;
    }

    std::vector<int> local_matrix(local_rows * N);
    MPI_Scatterv(matrix.data(), send_counts, displs, MPI_INT,
                 local_matrix.data(), send_counts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    // Локальное умножение
    std::vector<int> local_result(local_rows);
    for (int i = 0; i < local_rows; ++i) {
        local_result[i] = 0;
        for (int j = 0; j < N; ++j)
            local_result[i] += local_matrix[i * N + j] * vector[j];
    }

    // Сбор результатов
    int recv_counts[size], offsets[size];
    for (int r = 0; r < size; ++r) {
        recv_counts[r] = rows_per_proc + (r < remainder ? 1 : 0);
        offsets[r] = (r > 0) ? offsets[r-1] + recv_counts[r-1] : 0;
    }
    double end_time = MPI_Wtime();
    if (rank == 0) result.resize(N);
    MPI_Gatherv(local_result.data(), local_rows, MPI_INT,
                result.data(), recv_counts, offsets, MPI_INT,
                0, MPI_COMM_WORLD);

    // Расчет и вывод времени
    
    double duration = end_time - start_time;
    double max_duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Вывод результатов
    if (rank == 0) {
        /*
        std::cout << "Matrix:" << std::endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                std::cout << matrix[i * N + j] << " ";
            std::cout << std::endl;
        }
        
        std::cout << "\nVector:\n";
        for (int x : vector) std::cout << x << " ";
        std::cout << "\n\nResult:\n";
        for (int x : result) std::cout << x << " ";
        */

        std::cout << "\n\nExecution time: " << max_duration << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}