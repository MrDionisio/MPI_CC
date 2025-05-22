//Перемножеие матриц
// Алгоритм Фокса умножения матриц– блочное представление данных
//  Условия выполнения программы: все матрицы квадратные, размер блоков и их
//  количество по горизонтали и вертикали одинаковово, процессоры образуют
//  квадратную решетку

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>




int ProcNum = 0; 
int ProcRank = 0;
int GridSize; // Size of virtual processor grid 
int GridCoords[2]; // координыты процессора в решетке
MPI_Comm GridComm; // коммуникатор сетки
MPI_Comm ColComm; // коммуникатор столбцов 
MPI_Comm RowComm; // коммуникатор строк
 
 
// функция заполнения матриц
void RandomDataInitialization (double* pAMatrix, double* pBMatrix, int Size) {
int i, j; // Loop variables
srand(unsigned(10));
for (i=0; i<Size; i++)
for (j=0; j<Size; j++) {
pAMatrix[i*Size+j] = rand()%10;
pBMatrix[i*Size+j] = rand()%10;
}
}
 
 
 
// собираем блоки в результат
void ResultCollection (double* pCMatrix, double* pCblock, int Size, int BlockSize) { 
double * pResultRow = new double [Size*BlockSize];
for (int i=0; i<BlockSize; i++) {
/*Функция MPI_Gather производит сборку блоков данных, посылаемых всеми процессами группы, в один массив процесса с номером root
&pCblock[i*BlockSize] - адрес начала размещения посылаемых данных
BlockSize - число посылаемых элементов
&pResultRow[i*Size] - адрес начала буфера приема 
BlockSize - число элементов, получаемых от каждого процесса
0 - номер процессора получатедля*/
		MPI_Gather( &pCblock[i*BlockSize], BlockSize, MPI_DOUBLE, &pResultRow[i*Size], BlockSize, MPI_DOUBLE, 0, RowComm);
}
if (GridCoords[1] == 0) {
/*Функция MPI_Gather производит сборку блоков данных, посылаемых всеми процессами группы, в один массив процесса с номером root*/
	MPI_Gather(pResultRow, BlockSize*Size, MPI_DOUBLE, pCMatrix, BlockSize*Size, MPI_DOUBLE, 0, ColComm);
}
delete [] pResultRow;
}
// Умножение матричных блоков
void BlockMultiplication(double *pAblock, double *pBblock, double *pCblock, int BlockSize) { 
// вычисление произведения матричных блоков
for (int i=0; i<BlockSize; i++) { 
for (int j=0; j<BlockSize; j++) { 
double temp = 0; 
for (int k=0; k<BlockSize; k++ ) 
temp += pAblock [i*BlockSize + k] * pBblock [k*BlockSize + j];
pCblock [i*BlockSize + j] += temp; 
} 
} 
} 
 
 
/*в начале каждой итерации iter алгоритма для каждой строки процессной решетки выбирается процесс, который будет рассылать свой блок матрицы А*/
void ABlockCommunication(int iter, double *pAblock, double* pMatrixAblock, int BlockSize) { 
// определяем ведущий процессор
int Pivot = (GridCoords[0] + iter) % GridSize; 
// копируем передаваемый блок в буфер
if (GridCoords[1] == Pivot) { 
for (int i=0; i<BlockSize*BlockSize; i++) 
pAblock[i] = pMatrixAblock[i]; 
} 
// раскидываем блок
MPI_Bcast(pAblock, BlockSize*BlockSize, MPI_DOUBLE, Pivot, RowComm); 
} 
 
 
// циклический сдвиг
void BblockCommunication(double *pBblock, int BlockSize) { 
MPI_Status Status; 
int NextProc = GridCoords[0] + 1; 
if ( GridCoords[0] == GridSize-1 ) NextProc = 0; 
 
int PrevProc = GridCoords[0] - 1; 
if ( GridCoords[0] == 0 ) 
PrevProc = GridSize-1; 
/*обмен данными одного типа с замещением посылаемых данных на принимаемые
pBblock - адрес начала расположения посылаемого и принимаемого сообщени
BlockSize*BlockSize - число передаваемых элементов
NextProc - номер процесса-получателя
0 - идентефикатор посылаемого сообщения
PrevProc - номер процессора отправителя
0 - идентефикатор принимаемого сообщения
*/
MPI_Sendrecv_replace( pBblock, BlockSize*BlockSize, MPI_DOUBLE, NextProc, 0, PrevProc, 0, ColComm, &Status); 
} 
 
 
void ParallelResultCalculation(double* pAblock, double* pMatrixAblock, double* pBblock, double* pCblock, int BlockSize) { 
for (int iter = 0; iter < GridSize; iter ++) { 
/*рассылка блока матрицы А по строке процессорной решетки*/
ABlockCommunication (iter, pAblock, pMatrixAblock, BlockSize); 
// умножение
BlockMultiplication(pAblock, pBblock, pCblock, BlockSize); 
/*циклический сдвиг блоков матрицы В вдоль столбца процессорной решетки*/
BblockCommunication(pBblock, BlockSize); 
} 
}
 
 
 
 
 
// функция разложения матрицы на сетку
void CheckerboardMatrixScatter(double* pMatrix, double* pMatrixBlock, int Size, int BlockSize) {
double * pMatrixRow = new double [BlockSize*Size];
/*делим матрицу горизонтальными полосами между процессорами нулевого столбца решетки*/
if (GridCoords[1] == 0) {
/*pMatrix - адрес начала размещения блоков распределяемых данных
BlockSize*Size - число элементов, посылаемых каждому процессу
pMatrixRow - адрес приема
BlockSize*Size - число получаемых элементов
0 - номер процессора отправителя
colComm - коммуникатор*/
MPI_Scatter(pMatrix, BlockSize*Size, MPI_DOUBLE, pMatrixRow, BlockSize*Size, MPI_DOUBLE, 0, ColComm);
}
/*распределение каждой строки горизонтальной полосы матрицы вдоль строк процессорной решетки*/
for (int i=0; i<BlockSize; i++) {
MPI_Scatter(&pMatrixRow[i*Size], BlockSize, MPI_DOUBLE, &(pMatrixBlock[i*BlockSize]), BlockSize, MPI_DOUBLE, 0, RowComm);
}
delete [] pMatrixRow;
}
// функция распределения данных между процессорами
void DataDistribution(double* pAMatrix, double* pBMatrix, double* pMatrixAblock, double* pBblock, int Size, int BlockSize) {
CheckerboardMatrixScatter(pAMatrix, pMatrixAblock, Size, BlockSize);
CheckerboardMatrixScatter(pBMatrix, pBblock, Size, BlockSize);
}
 
 
// функция инициализации памяти и данных
void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix, double* &pAblock, double* &pBblock, double* &pCblock, double* &pTemporaryAblock, int &Size, int &BlockSize ){ 
MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD); 
BlockSize = Size/GridSize; 
pAblock = new double [BlockSize*BlockSize]; 
pBblock = new double [BlockSize*BlockSize]; 
pCblock = new double [BlockSize*BlockSize]; 
pTemporaryAblock = new double [BlockSize*BlockSize]; 
for (int i=0; i<BlockSize*BlockSize; i++) { 
pCblock[i] = 0; 
} 
if (ProcRank == 0) { 
pAMatrix = new double [Size*Size]; 
pBMatrix = new double [Size*Size]; 
pCMatrix = new double [Size*Size]; 
RandomDataInitialization(pAMatrix, pBMatrix, Size); 
} 
} 
 
 
 
/*функция создает коммуникатор в виде двумерной квадратной решетки, определяет координаты каждого процесса в этой решетке*/
void CreateGridCommunicators() { 
int DimSize[2]; // число процессоров в каждом измерении сетки
int Periodic[2]; // =1, если размерность должна быть динамична
int Subdims[2]; // =1, если размерность должна быть фиксированна
DimSize[0] = GridSize; 
DimSize[1] = GridSize; 
Periodic[0] = 0; 
Periodic[1] = 0; 
// создание коммуникатора
/*2 - число измерений
DimSize - -	 массив, в котором задается число процессов вдоль каждого измерения
Periodic - логический массив размера ndims для задания граничных условий
1 - логическая переменная, указывает, производить перенумерацию процессов (true) или нет (false)
GridComm - новый коммуникатор*/
MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm); 
//определение координат каждого процессора в решетке
MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords); 
// создание коммуникатора для строк 
Subdims[0] = 0; // фиксация размерности
Subdims[1] = 1; // наличие данной размерности в подсетке
MPI_Cart_sub(GridComm, Subdims, &RowComm); 
// создание коммуникатора для столбцов
Subdims[0] = 1; 
Subdims[1] = 0; 
// создание множества коммуникаторов для каждой строки и каждого столбца решетки в отдельности
MPI_Cart_sub(GridComm, Subdims, &ColComm); 
} 
 
 
int main ( int argc, char * argv[] ) { 
double* pAMatrix; // первая матрица
double* pBMatrix; // вторая матрица
double* pCMatrix; // результат
int Size = 2000; // размер матрицы 
int BlockSize; // размер матричного блока на процессоре
double *pAblock; // начальный блок матрицы А
double *pBblock; // начальный блок матрицы В 
double *pCblock; // результирующий блок
double *pMatrixAblock; 
double Start, Finish, Duration; 
/*Функция setvbuf() позволяет программисту задать буфер, его размер и режим работы с указанным потоком*/
setvbuf(stdout, 0, _IONBF, 0); 
MPI_Init(&argc, &argv); 
MPI_Comm_size(MPI_COMM_WORLD, &ProcNum); 
MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank); 
GridSize = sqrt((double)ProcNum); 
if (ProcNum != GridSize*GridSize) { 
if (ProcRank == 0) { 
printf ("Number of processes must be a perfect square \n"); 
} 
} 
else { 
if (ProcRank == 0) 
printf("Parallel matrix multiplication program\n"); 
// создание процессорной сетки
CreateGridCommunicators(); 
// выделение памяти и инициализация матриц
ProcessInitialization ( pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock, pCblock, pMatrixAblock, Size, BlockSize ); 

 
 
DataDistribution(pAMatrix, pBMatrix, pMatrixAblock, pBblock, Size, BlockSize); 
double start_time = MPI_Wtime();
ParallelResultCalculation(pAblock, pMatrixAblock, pBblock, pCblock, BlockSize); 
double end_time = MPI_Wtime();
ResultCollection(pCMatrix, pCblock, Size, BlockSize); 
 


if (ProcRank == 0) {
    /*
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j)
            std::cout << pAMatrix[i * Size + j] << " ";
        std::cout << std::endl;
    }
    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j)
            std::cout << pBMatrix[i * Size + j] << " ";
        std::cout << std::endl;
    }
    std::cout << "Matrix C:" << std::endl;
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j)
            std::cout << pCMatrix[i * Size + j] << " ";
        std::cout << std::endl;
    }
    
    */

    std::cout << "\n\nExecution time: " << end_time- start_time << " seconds" << std::endl;
}





MPI_Finalize();
}
return 0;
}