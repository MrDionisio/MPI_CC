// Полная реализация JacobiMPI на C++ с использованием MPI
#include <mpi.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include <algorithm>

using std::vector;

double xbound(double y) {
    return 2.0 * std::sqrt(y);
}

double ybound(double x) {
    return 0.25 * x * x;
}

double f(double x, double y) {
    return 0.25;
}

double phi(double x, double y) {
    return std::sqrt(0.25 * x * x + y * y);
}

void calc_bounds(vector<vector<double>>& u, const vector<vector<double>>& x, const vector<vector<double>>& y,
                 vector<int>& Imin, vector<int>& Imax, vector<int>& Jmin,
                 int ni, int nj, double hx, double hy) {
    Jmin.assign(ni, nj);
    for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj - 1; ++j) {
            if (y[i][j] > ybound(x[i][j])) {
                Jmin[i] = j;
                break;
            } else {
                u[i][j] = 0.0;
            }
        }
    }
    Imin[0] = (ni + 1) / 2;
    Imin[nj - 1] = 0;
    for (int j = 1; j < nj - 1; ++j) {
        for (int i = 0; i < ni; ++i) {
            if (x[i][j] > -xbound(y[i][j])) {
                Imin[j] = i;
                break;
            }
        }
    }
    Imax[0] = (ni + 1) / 2;
    Imax[nj - 1] = ni - 1;
    for (int j = 1; j < nj - 1; ++j) {
        for (int i = ni - 2; i >= 0; --i) {
            if (x[i][j] < xbound(y[i][j])) {
                Imax[j] = i;
                break;
            }
        }
    }
}

void InterComm(const vector<double>& u_up, const vector<double>& u_low,
               vector<double>& up, vector<double>& low,
               int ni, int pid, int procs) {
    MPI_Request reqs[4];
    MPI_Status stats[4];
    if (pid > 0) {
        MPI_Isend(u_low.data(), ni, MPI_DOUBLE, pid - 1, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(low.data(), ni, MPI_DOUBLE, pid - 1, 1, MPI_COMM_WORLD, &reqs[1]);
    }
    if (pid < procs - 1) {
        MPI_Isend(u_up.data(), ni, MPI_DOUBLE, pid + 1, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(up.data(), ni, MPI_DOUBLE, pid + 1, 0, MPI_COMM_WORLD, &reqs[3]);
    }
    if (pid == 0) {
        MPI_Waitall(2, &reqs[2], &stats[2]);
    } else if (pid == procs - 1) {
        MPI_Waitall(2, &reqs[0], &stats[0]);
    } else {
        MPI_Waitall(4, reqs, stats);
    }
}

void Sweep(vector<double>& uline, const vector<double>& up, const vector<double>& low,
           int ni, int j, const vector<int>& Imin, const vector<int>& Imax, const vector<int>& Jmin,
           double hx, double hy, double Xmin, double Ymin) {
    vector<double> u0 = uline;
    for (int i = Imin[j] + 1; i < Imax[j]; ++i) {
        if (j == Jmin[i]) continue;
        double x = Xmin + i * hx;
        double y = Ymin + j * hy;
        uline[i] = (hy * hy * (u0[i + 1] + u0[i - 1]) +
                    hx * hx * (up[i] + low[i]) -
                    hx * hx * hy * hy * f(x, y)) /
                   (2.0 * (hx * hx + hy * hy));
    }
}

void print_probe_values(const vector<vector<double>>& x, const vector<vector<double>>& y, const vector<vector<double>>& u, int ni, int nj) {
    int i1 = (ni + 1) / 2;
    int j1 = (nj + 1) / 2;
    int j2 = (nj - 1) * 9 / 10 + 1;
    std::cout << " x =" << std::setw(8) << std::setprecision(5) << x[i1][j1]
              << "; y = " << std::setw(8) << y[i1][j1]
              << "; u = " << std::setw(11) << std::setprecision(7) << u[i1][j1] << "\n";
    std::cout << " x =" << std::setw(8) << x[i1][j2]
              << "; y = " << std::setw(8) << y[i1][j2]
              << "; u = " << std::setw(11) << u[i1][j2] << "\n";
}

void init(vector<vector<double>>& u, const vector<vector<double>>& x, const vector<vector<double>>& y, int ni, int nj) {
    for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nj - 1; ++j)
            u[i][j] = 1.0;
    for (int i = 0; i < ni; ++i)
        u[i][nj - 1] = phi(x[i][nj - 1], y[i][nj - 1]);
}

void bcond(vector<vector<double>>& u, const vector<int>& Imin, const vector<int>& Imax, const vector<int>& Jmin,
           double hx, double hy, int ni, int nj, double Xmin, double Ymin,
           int jp1, int jp2, const vector<double>& up) {
    for (int i = std::max(Imin[jp2], 1); i <= std::max(Imin[jp1], 1); ++i) {
        int jr = Jmin[i] - jp1;
        if (jr >= 0 && jr < jp2 - jp1 + 1) {
            double xm = Xmin + i * hx;
            double ym = Ymin + (Jmin[i]) * hy;
            double delta = ym - ybound(xm);
            double unode = (jr == jp2 - jp1) ? up[i] : u[i][jr + 1];
            u[i][jr] = (hy * phi(xm, ybound(xm)) + delta * unode) / (hy + delta);
        }
    }
    for (int i = std::min(Imax[jp1], ni - 2); i <= std::min(Imax[jp2], ni - 2); ++i) {
        int jr = Jmin[i] - jp1;
        if (jr >= 0 && jr < jp2 - jp1 + 1) {
            double xm = Xmin + i * hx;
            double ym = Ymin + (Jmin[i]) * hy;
            double delta = ym - ybound(xm);
            double unode = (jr == jp2 - jp1) ? up[i] : u[i][jr + 1];
            u[i][jr] = (hy * phi(xm, ybound(xm)) + delta * unode) / (hy + delta);
        }
    }
    for (int j = jp1; j <= jp2; ++j) {
        int jj = j - jp1;
        if (Jmin[Imin[j]] < j) {
            double xm = Xmin + Imin[j] * hx;
            double ym = Ymin + j * hy;
            double delta = xm + xbound(ym);
            u[Imin[j]][jj] = (hx * phi(-xbound(ym), ym) + delta * u[Imin[j] + 1][jj]) / (hx + delta);
        }
        if (Jmin[Imax[j]] < j) {
            double xm = Xmin + Imax[j] * hx;
            double ym = Ymin + j * hy;
            double delta = xbound(ym) - xm;
            u[Imax[j]][jj] = (hx * phi(xbound(ym), ym) + delta * u[Imax[j] - 1][jj]) / (hx + delta);
        }
    }
}

void Jacobi(vector<vector<double>>& u, const vector<int>& Imin, const vector<int>& Imax, const vector<int>& Jmin,
            double hx, double hy, int ni, int nj, double Xmin, double Ymin, int jp1, int jp2,
            double& error1, vector<double>& up, int pid, int procs) {
    vector<vector<double>> u0 = u;
    bcond(u, Imin, Imax, Jmin, hx, hy, ni, nj, Xmin, Ymin, jp1, jp2, up);

    for (int j = jp1 + 1; j < jp2; ++j) {
        int jj = j - jp1;
        for (int i = Imin[j] + 1; i < Imax[j]; ++i) {
            if (j == Jmin[i]) continue;
            double x = Xmin + i * hx;
            double y = Ymin + j * hy;
            u[i][jj] = (hy * hy * (u0[i + 1][jj] + u0[i - 1][jj]) +
                        hx * hx * (u0[i][jj + 1] + u0[i][jj - 1]) -
                        hx * hx * hy * hy * f(x, y)) /
                       (2.0 * (hx * hx + hy * hy));
        }
    }

    if (procs > 1) {
        vector<double> low(ni);
        InterComm(u0[jp2 - jp1], u0[0], up, low, ni, pid, procs);
        if (jp1 == jp2 && jp1 > 0 && jp2 < nj - 1) {
            Sweep(u[jp2 - jp1], up, low, ni, jp1, Imin, Imax, Jmin, hx, hy, Xmin, Ymin);
        } else {
            if (jp1 > 0)
                Sweep(u[0], u0[1], low, ni, jp1, Imin, Imax, Jmin, hx, hy, Xmin, Ymin);
            if (jp2 < nj - 1)
                Sweep(u[jp2 - jp1], up, u0[jp2 - jp1 - 1], ni, jp2, Imin, Imax, Jmin, hx, hy, Xmin, Ymin);
        }
    }

    double errloc = 0.0;
    for (int j = 0; j < jp2 - jp1 + 1; ++j)
        for (int i = 0; i < ni; ++i)
            errloc = std::max(errloc, std::abs(u0[i][j] - u[i][j]));

    MPI_Allreduce(&errloc, &error1, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void decomp(const vector<vector<double>>& u, vector<vector<double>>& uloc, int ni, int nj, int pid, int procs,
            vector<int>& disp, vector<int>& scount, int& jp1, int& jp2, const vector<int>& Imin, const vector<int>& Imax) {
    vector<int> jpu(procs), jpd(procs);
    if (pid == 0) {
        int opt = (1 + ni * (nj - 1) + 2 * (nj - 2 - std::accumulate(Imin.begin() + 1, Imin.end() - 1, 0))) / procs;
        int sp = 0, i = 0;
        jpd[0] = 0;
        disp[0] = 0;
        for (int j = 0; j < nj; ++j) {
            sp += Imax[j] - Imin[j] + 1;
            if (sp >= opt || j == nj - 1 || (procs - i - 1) == (nj - j - 1)) {
                jpu[i] = j;
                if (i < procs - 1) {
                    jpd[i + 1] = j + 1;
                    disp[i + 1] = (j + 1) * ni;
                }
                scount[i] = (jpu[i] - jpd[i] + 1) * ni;
                ++i;
                sp = 0;
            }
        }
    }
    MPI_Scatter(jpd.data(), 1, MPI_INT, &jp1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(jpu.data(), 1, MPI_INT, &jp2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    uloc.resize(ni, vector<double>(jp2 - jp1 + 1, 0.0));
    vector<double> flat_u;
    if (pid == 0) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i)
                flat_u.push_back(u[i][j]);
    }
    vector<double> recvbuf((jp2 - jp1 + 1) * ni);
    MPI_Scatterv(flat_u.data(), scount.data(), disp.data(), MPI_DOUBLE,
                 recvbuf.data(), recvbuf.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int j = 0; j < jp2 - jp1 + 1; ++j)
        for (int i = 0; i < ni; ++i)
            uloc[i][j] = recvbuf[j * ni + i];
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int pid, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int ni, nj, maxit;
    double Xmin = -std::sqrt(8.0), Xmax = std::sqrt(8.0);
    double Ymin = 0.0, Ymax = 2.0;

    if (pid == 0) {
        std::ifstream fin("input.txt");
        fin >> maxit >> ni >> nj;
        fin.close();
    }
    MPI_Bcast(&maxit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ni, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nj, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double hx = (Xmax - Xmin) / (ni - 1);
    double hy = (Ymax - Ymin) / (nj - 1);
    vector<int> Imin(nj), Imax(nj), Jmin(ni);
    vector<int> disp(procs), scount(procs);
    vector<vector<double>> x, y, u, uloc;
    int jp1 = 0, jp2 = 0;

    if (pid == 0) {
        x.resize(ni, vector<double>(nj));
        y.resize(ni, vector<double>(nj));
        u.resize(ni, vector<double>(nj));
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i) {
                x[i][j] = Xmin + i * hx;
                y[i][j] = Ymin + j * hy;
            }
        init(u, x, y, ni, nj);
        calc_bounds(u, x, y, Imin, Imax, Jmin, ni, nj, hx, hy);
    }
    MPI_Bcast(Imin.data(), nj, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Imax.data(), nj, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Jmin.data(), ni, MPI_INT, 0, MPI_COMM_WORLD);

    decomp(u, uloc, ni, nj, pid, procs, disp, scount, jp1, jp2, Imin, Imax);
    vector<double> up = uloc[jp2 - jp1];

    double t1 = MPI_Wtime();
    double error1 = 1.0, errmax = 1e-7;
    int l;
    for (l = 1; l <= maxit; ++l) {
        Jacobi(uloc, Imin, Imax, Jmin, hx, hy, ni, nj, Xmin, Ymin, jp1, jp2, error1, up, pid, procs);
        if (l % 100 == 0 && pid == 0)
            std::cout << "iter: " << l << " error: " << error1 << "\n";
        if (error1 < errmax) break;
    }
    double t2 = MPI_Wtime();

    vector<double> flat_uloc((jp2 - jp1 + 1) * ni);
    for (int j = 0; j < jp2 - jp1 + 1; ++j)
        for (int i = 0; i < ni; ++i)
            flat_uloc[j * ni + i] = uloc[i][j];

    vector<double> gathered;
    if (pid == 0) gathered.resize(ni * nj);
    MPI_Gatherv(flat_uloc.data(), flat_uloc.size(), MPI_DOUBLE,
                gathered.data(), scount.data(), disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        u.resize(ni, vector<double>(nj));
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i)
                u[i][j] = gathered[j * ni + i];

        std::ofstream fout("output.plt");
        fout << "Variables = \"x\", \"y\", \"u\"\n";
        fout << "Zone i = " << ni << ", j = " << nj << "\n";
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i)
                fout << x[i][j] << " " << y[i][j] << " " << u[i][j] << "\n";
        fout.close();

        std::cout << "time: " << (t2 - t1) << "; l = " << l
                  << "; l/t = " << l / (t2 - t1)
                  << "; N = " << ni + nj * (ni - 2) - std::accumulate(Jmin.begin() + 1, Jmin.end() - 1, 0) + 1 << "\n";
        print_probe_values(x, y, u, ni, nj);
    }

    MPI_Finalize();
    return 0;
}
