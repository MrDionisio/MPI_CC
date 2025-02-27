#include <mpi.h>
#include <iostream>


using namespace std;
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Определяем соседей в кольце
    int left_neighbor = (rank - 1 + size) % size;
    int right_neighbor = (rank + 1) % size;

    // Буферы для отправки и получения данных
    int send_buffer = rank;  // Отправляем свой ранг
    int recv_buffer_left, recv_buffer_right;

    // Запросы для неблокирующих операций
    MPI_Request send_request_left, send_request_right;
    MPI_Request recv_request_left, recv_request_right;

    // Неблокирующая отправка и прием
    MPI_Isend(&send_buffer, 1, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, &send_request_left);
    MPI_Isend(&send_buffer, 1, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD, &send_request_right);

    MPI_Irecv(&recv_buffer_left, 1, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, &recv_request_left);
    MPI_Irecv(&recv_buffer_right, 1, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD, &recv_request_right);

    // Ожидание завершения операций приема
    int recv_done_left = 0, recv_done_right = 0;
    while (!recv_done_left || !recv_done_right) {
        MPI_Test(&recv_request_left, &recv_done_left, MPI_STATUS_IGNORE);
        MPI_Test(&recv_request_right, &recv_done_right, MPI_STATUS_IGNORE);
    }

    // Ожидание завершения операций отправки
    MPI_Wait(&send_request_left, MPI_STATUS_IGNORE);
    MPI_Wait(&send_request_right, MPI_STATUS_IGNORE);

    // Вывод полученных данных
    std::cout << "Process " << rank << " received from left neighbor: " << recv_buffer_left << std::endl;
    std::cout << "Process " << rank << " received from right neighbor: " << recv_buffer_right << std::endl;

    MPI_Finalize();
    return 0;
}
