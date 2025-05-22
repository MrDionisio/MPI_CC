#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <numeric>

using namespace std;

double xbound(double y) {
    return 2.0 * sqrt(y);
}

double ybound(double x) {
    return 0.25 * x * x;
}

double f(double x, double y) {
    return 0.25;
}

double phi(double x, double y) {
    return sqrt(0.25 * x * x + y * y);
}

void init(vector<double>& u, int ni, int nj, const vector<double>& x, const vector<double>& y) {
    fill(u.begin(), u.end(), 1.0);
    for (int i = 0; i < ni; ++i) {
        int j = nj - 1;
        u[i + j * ni] = phi(x[i + j * ni], y[i + j * ni]);
    }
}

void decomp(const vector<double>& u, vector<double>& uloc, int ni, int nj, int pid, int procs,
            vector<int>& disp, vector<int>& scount, int& jp1, int& jp2,
            const vector<int>& Imin, const vector<int>& Imax) {
    vector<int> jpd(procs), jpu(procs);
    if (pid == 0) {
        int opt = (1 + ni * (nj - 1) + 2 * (nj - 2 - accumulate(Imin.begin() + 1, Imin.end() - 1, 0))) / procs;
        int sp = 0, i = 0;
        jpd[0] = 1;
        disp[0] = 0;
        for (int j = 1; j <= nj; ++j) {
            sp += Imax[j-1] - Imin[j-1] + 1;
            if (sp >= opt || j == nj || (procs - i - 1) == (nj - j)) {
                jpu[i] = j;
                if (i < procs - 1) {
                    jpd[i+1] = j + 1;
                    disp[i+1] = j * ni;
                }
                scount[i] = (jpu[i] - jpd[i] + 1) * ni;
                i++;
                sp = 0;
                if (i >= procs) break;
            }
        }
    }

    MPI_Scatter(jpd.data(), 1, MPI_INT, &jp1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(jpu.data(), 1, MPI_INT, &jp2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    jp1--;
    jp2--;
    int local_j = jp2 - jp1 + 1;
    uloc.resize(ni * local_j);

    if (pid == 0) {
        MPI_Scatterv(u.data(), scount.data(), disp.data(), MPI_DOUBLE,
                     uloc.data(), ni * local_j, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     uloc.data(), ni * local_j, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void bcond(vector<double>& u, const vector<int>& Imin, const vector<int>& Imax, const vector<int>& Jmin,
           double hx, double hy, int ni, int nj, double Xmin, double Ymin,
           int jp1, int jp2, vector<double>& up) {
    int local_j = jp2 - jp1 + 1;
    for (int i = max(Imin[jp2], 1); i <= max(Imin[jp1], 1); ++i) {
        int j_global = Jmin[i] - 1;
        int jr = j_global - jp1;
        if (jr >= 0 && jr < local_j) {
            double xm = Xmin + i * hx;
            double ym = Ymin + j_global * hy;
            double delta = ym - ybound(xm);
            double unode = (jr == local_j - 1) ? up[i] : u[i + (jr + 1) * ni];
            u[i + jr * ni] = (hy * phi(xm, ybound(xm)) + delta * unode) / (hy + delta);
        }
    }

    for (int i = min(Imax[jp1], ni - 2); i <= min(Imax[jp2], ni - 2); ++i) {
        int j_global = Jmin[i] - 1;
        int jr = j_global - jp1;
        if (jr >= 0 && jr < local_j) {
            double xm = Xmin + i * hx;
            double ym = Ymin + j_global * hy;
            double delta = ym - ybound(xm);
            double unode = (jr == local_j - 1) ? up[i] : u[i + (jr + 1) * ni];
            u[i + jr * ni] = (hy * phi(xm, ybound(xm)) + delta * unode) / (hy + delta);
        }
    }

    for (int j = jp1; j <= jp2; ++j) {
        if (Jmin[Imin[j]] - 1 < j) {
            int i = Imin[j];
            double xm = Xmin + i * hx;
            double ym = Ymin + j * hy;
            double delta = xm + xbound(ym);
            int jr = j - jp1;
            u[i + jr * ni] = (hx * phi(-xbound(ym), ym) + delta * u[i + 1 + jr * ni]) / (hx + delta);
        }

        if (Jmin[Imax[j]] - 1 < j) {
            int i = Imax[j];
            double xm = Xmin + i * hx;
            double ym = Ymin + j * hy;
            double delta = xbound(ym) - xm;
            int jr = j - jp1;
            u[i + jr * ni] = (hx * phi(xbound(ym), ym) + delta * u[i - 1 + jr * ni]) / (hx + delta);
        }
    }
}

void InterComm(vector<double>& u_up, vector<double>& u_low, vector<double>& up, vector<double>& low,
               int ni, int pid, int procs) {
    MPI_Request reqs[4];
    MPI_Status stats[4];

    if (pid > 0) {
        MPI_Isend(u_low.data(), ni, MPI_DOUBLE, pid-1, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(low.data(), ni, MPI_DOUBLE, pid-1, 1, MPI_COMM_WORLD, &reqs[1]);
    }
    if (pid < procs-1) {
        MPI_Isend(u_up.data(), ni, MPI_DOUBLE, pid+1, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(up.data(), ni, MPI_DOUBLE, pid+1, 0, MPI_COMM_WORLD, &reqs[3]);
    }

    if (pid == 0)
        MPI_Waitall(2, reqs+2, stats+2);
    else if (pid == procs-1)
        MPI_Waitall(2, reqs, stats);
    else
        MPI_Waitall(4, reqs, stats);
}




void Jacobi(vector<double>& u, const vector<int>& Imin, const vector<int>& Imax, const vector<int>& Jmin,
            double hx, double hy, int ni, int nj, double Xmin, double Ymin,
            int jp1, int jp2, double& error1, vector<double>& up, int pid, int procs) {
    int local_j = jp2 - jp1 + 1;
    vector<double> u0(u);
    vector<double> low(ni, 0.0);

    bcond(u, Imin, Imax, Jmin, hx, hy, ni, nj, Xmin, Ymin, jp1, jp2, up);

    for (int j = jp1 + 1; j <= jp2 - 1; ++j) {
        for (int i = Imin[j] + 1; i <= Imax[j] - 1; ++i) {
            if (j == Jmin[i] - 1) continue;
            int jr = j - jp1;
            double x = Xmin + i * hx;
            double y_val = Ymin + j * hy;
            u[i + jr * ni] = (hy*hy*(u0[i+1 + jr*ni] + u0[i-1 + jr*ni]) +
                             hx*hx*(u0[i + (jr+1)*ni] + u0[i + (jr-1)*ni]) -
                             hx*hx*hy*hy*f(x, y_val)) / (2*(hx*hx + hy*hy));
        }
    }

    if (procs > 1) {
        vector<double> u_up(ni), u_low(ni);
        copy(u.begin() + (local_j - 1)*ni, u.begin() + local_j*ni, u_up.begin());
        copy(u.begin(), u.begin() + ni, u_low.begin());
        InterComm(u_up, u_low, up, low, ni, pid, procs);
    }

    double errloc = 0.0;
    for (size_t i = 0; i < u.size(); ++i)
        errloc = max(errloc, abs(u[i] - u0[i]));
    MPI_Allreduce(&errloc, &error1, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}


void Sweep(vector<double>& uline, vector<double>& up, vector<double>& low,
           int ni, int nj, int j, const vector<int>& Imin, const vector<int>& Imax,
           const vector<int>& Jmin, double hx, double hy, double Xmin, double Ymin) {
    vector<double> u0 = uline;
    for (int i = Imin[j] + 1; i <= Imax[j] - 1; ++i) {
        if (j == Jmin[i] - 1) continue;
        double x = Xmin + i * hx;
        double y_val = Ymin + j * hy;
        uline[i] = (hy*hy*(u0[i+1] + u0[i-1]) + hx*hx*(up[i] + low[i]) -
                   hx*hx*hy*hy*f(x, y_val)) / (2*(hx*hx + hy*hy));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int procs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    double Xmin = -sqrt(8.0), Xmax = sqrt(8.0), Ymin = 0.0, Ymax = 2.0;
    int ni = 0, nj = 0, maxit = 0;
    vector<double> x, y, u;
    vector<int> Imin, Imax, Jmin;

    if (pid == 0) {
        ifstream fin("input.txt");
        fin >> maxit >> ni >> nj;
        fin.close();

        x.resize(ni * nj);
        y.resize(ni * nj);
        u.resize(ni * nj);

        double hx = (Xmax - Xmin) / (ni - 1);
        double hy = (Ymax - Ymin) / (nj - 1);

        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                x[i + j * ni] = Xmin + i * hx;
                y[i + j * ni] = Ymin + j * hy;
            }
        }

        init(u, ni, nj, x, y);

        Jmin.resize(ni);
        fill(Jmin.begin(), Jmin.end(), nj - 1);
        for (int i = 0; i < ni; ++i) {
            for (int j = 0; j < nj - 1; ++j) {
                double y_val = Ymin + j * hy;
                if (y_val > ybound(Xmin + i * hx)) {
                    Jmin[i] = j;
                    break;
                } else {
                    u[i + j * ni] = 0.0;
                }
            }
        }

        Imin.resize(nj);
        Imax.resize(nj);
        Imin[0] = (ni - 1) / 2;
        Imin[nj - 1] = 0;
        for (int j = 1; j < nj - 1; ++j) {
            double y_val = Ymin + j * hy;
            for (int i = 0; i < ni; ++i) {
                double x_val = Xmin + i * hx;
                if (x_val > -xbound(y_val)) {
                    Imin[j] = i;
                    break;
                }
            }
        }

        Imax[0] = (ni - 1) / 2;
        Imax[nj - 1] = ni - 1;
        for (int j = 1; j < nj - 1; ++j) {
            double y_val = Ymin + j * hy;
            for (int i = ni - 1; i >= 0; --i) {
                double x_val = Xmin + i * hx;
                if (x_val < xbound(y_val)) {
                    Imax[j] = i;
                    break;
                }
            }
        }
    }

    MPI_Bcast(&ni, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nj, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxit, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double hx = (Xmax - Xmin) / (ni - 1);
    double hy = (Ymax - Ymin) / (nj - 1);

    Imin.resize(nj);
    Imax.resize(nj);
    Jmin.resize(ni);
    MPI_Bcast(Imin.data(), nj, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Imax.data(), nj, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Jmin.data(), ni, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> disp(procs), scount(procs);
    int jp1, jp2;
    vector<double> uloc;

    decomp(u, uloc, ni, nj, pid, procs, disp, scount, jp1, jp2, Imin, Imax);

    vector<double> up(ni, 0.0), low(ni, 0.0);
    double error1 = 1.0, errmax = 1e-7;
    double t1 = MPI_Wtime();

    for (int l = 1; l <= maxit; ++l) {
        Jacobi(uloc, Imin, Imax, Jmin, hx, hy, ni, nj, Xmin, Ymin, jp1, jp2, error1, up, pid, procs);

        if (pid == 0 && l % 100 == 0)
            cout << "iter: " << l << " error: " << error1 << endl;

        if (error1 < errmax) {
            if (pid == 0) 
                cout << "last iter: " << l << " error: " << error1 << endl;
            break;
        }
    }

    double t2 = MPI_Wtime();

    if (pid == 0) {
        u.resize(ni * nj);
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, u.data(), scount.data(), disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        ofstream fout("output.plt");
        fout << "Variables = \"x\", \"y\", \"u\"" << endl;
        fout << "Zone i=" << ni << ", j=" << nj << endl;
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                fout << x[i + j*ni] << " " << y[i + j*ni] << " " << u[i + j*ni] << endl;
            }
        }
        fout.close();

        cout << "Time: " << t2 - t1 << "; iterations: " << maxit << endl;
    } else {
        MPI_Gatherv(uloc.data(), uloc.size(), MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}