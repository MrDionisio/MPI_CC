#include <mpi.h>
#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rows = 4; // Число строк
    const int cols = size; // Число столбцов (должно быть равно числу процессов)

    // Указатель на матрицу
    double* matrix = nullptr;

    // Нулевой процесс выделяет память и инициализирует матрицу
    if (rank == 0) {
        matrix = new double[rows * cols]; // Выделяем память только на нулевом процессе
        cout << "Исходная матрица:\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i * cols + j] = i * cols + j + 1;
                cout << setw(4) << matrix[i * cols + j] << " ";
            }
            cout << endl;
        }
    }

    // Буфер для каждого процесса
    double local_column[rows];

    // Рассылаем столбцы матрицы
    MPI_Scatter(matrix, rows, MPI_DOUBLE, local_column, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Изменяем элементы (например, умножаем на 2)
    for (int i = 0; i < rows; i++) {
        local_column[i] *= 2;
    }

    // Собираем обратно измененные данные
    MPI_Gather(local_column, rows, MPI_DOUBLE, matrix, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Выводим обновленную матрицу на нулевом процессе
    if (rank == 0) {
        cout << "\nМодифицированная матрица:\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << setw(4) << matrix[i * cols + j] << " ";
            }
            cout << endl;
        }

        // Освобождаем память, выделенную для матрицы
        delete[] matrix;
    }

    MPI_Finalize();
    return 0;
}
