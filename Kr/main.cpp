#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <numeric>

using namespace std;

const int maxit = 100000;
const double errmax = 1e-7;

inline double xbound(double y) {
    return 2.0 * sqrt(y);
}

inline double ybound(double x) {
    return 0.25 * x * x;
}

inline double f(double x, double y) {
    return 0.25;
}

inline double phi(double x, double y) {
    return sqrt(0.25 * x * x + y * y);
}

void init(vector<vector<double>> &u,
          const vector<vector<double>> &x,
          const vector<vector<double>> &y,
          int ni, int nj) {
    for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nj - 1; ++j)
            u[i][j] = 1.0;

    for (int i = 0; i < ni; ++i)
        u[i][nj - 1] = phi(x[i][nj - 1], y[i][nj - 1]);
}

void bcond(vector<vector<double>> &u,
           const vector<vector<double>> &x,
           const vector<vector<double>> &y,
           const vector<int> &Imin,
           const vector<int> &Imax,
           const vector<int> &Jmin,
           double hx, double hy,
           int ni, int nj) {

    for (int i = 1; i < ni - 1; ++i) {
        double xm = x[i][Jmin[i]];
        double ym = y[i][Jmin[i]];
        double delta = ym - ybound(xm);
        u[i][Jmin[i]] = (hy * phi(xm, ybound(xm)) + delta * u[i][Jmin[i] + 1]) / (hy + delta);
    }

    for (int j = 0; j < nj; ++j) {
        if (Jmin[Imin[j]] < j) {
            double xm = x[Imin[j]][j];
            double ym = y[Imin[j]][j];
            double delta = xm + xbound(ym);
            u[Imin[j]][j] = (hx * phi(-xbound(ym), ym) + delta * u[Imin[j] + 1][j]) / (hx + delta);
        }
        if (Jmin[Imax[j]] < j) {
            double xm = x[Imax[j]][j];
            double ym = y[Imax[j]][j];
            double delta = xbound(ym) - xm;
            u[Imax[j]][j] = (hx * phi(xbound(ym), ym) + delta * u[Imax[j] - 1][j]) / (hx + delta);
        }
    }
}

void Jacobi(vector<vector<double>> &u,
            const vector<vector<double>> &x,
            const vector<vector<double>> &y,
            const vector<int> &Imin,
            const vector<int> &Imax,
            const vector<int> &Jmin,
            double hx, double hy,
            int ni, int nj,
            double &error1,
            int iter) {

    vector<vector<double>> u0 = u;

    for (int k = 0; k < iter; ++k) {
        bcond(u, x, y, Imin, Imax, Jmin, hx, hy, ni, nj);
        u0 = u;

        for (int j = 1; j < nj - 1; ++j) {
            for (int i = Imin[j] + 1; i < Imax[j]; ++i) {
                if (j == Jmin[i]) continue;

                u[i][j] = (hy * hy * (u0[i + 1][j] + u0[i - 1][j]) +
                           hx * hx * (u0[i][j + 1] + u0[i][j - 1]) -
                           hx * hx * hy * hy * f(x[i][j], y[i][j])) /
                          (2 * (hx * hx + hy * hy));

                if (k == iter - 1)
                    error1 = max(error1, fabs(u0[i][j] - u[i][j]));
            }
        }
    }
}

int main() {
    int iter, ni, nj;
    double Xmin, Xmax, Ymin = 0.0, Ymax = 2.0;
    double error1 = 1.0;
    int l = 0;

    ifstream input("input.txt");
    input >> iter >> ni >> nj;
    input.close();

    Xmax = sqrt(8.0);
    Xmin = -Xmax;

    double hx = (Xmax - Xmin) / (ni - 1);
    double hy = (Ymax - Ymin) / (nj - 1);

    vector<vector<double>> x(ni, vector<double>(nj));
    vector<vector<double>> y(ni, vector<double>(nj));
    vector<vector<double>> u(ni, vector<double>(nj, 0.0));
    vector<int> Imin(nj, 0), Imax(nj, ni - 1), Jmin(ni, nj - 1);

    for (int j = 0; j < nj; ++j)
        for (int i = 0; i < ni; ++i) {
            x[i][j] = Xmin + i * hx;
            y[i][j] = Ymin + j * hy;
        }

    init(u, x, y, ni, nj);

    for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nj - 1; ++j) {
            if (y[i][j] > ybound(x[i][j])) {
                Jmin[i] = j;
                break;
            } else {
                u[i][j] = 0.0;
            }
        }

    for (int j = 1; j < nj - 1; ++j) {
        for (int i = 0; i < ni; ++i) {
            if (x[i][j] > -xbound(y[i][j])) {
                Imin[j] = i;
                break;
            }
        }
    }

    for (int j = 1; j < nj - 1; ++j) {
        for (int i = ni - 2; i >= 0; --i) {
            if (x[i][j] < xbound(y[i][j])) {
                Imax[j] = i;
                break;
            }
        }
    }

    double t1 = omp_get_wtime();
    while (l <= maxit && error1 > errmax) {
        error1 = 0.0;
        Jacobi(u, x, y, Imin, Imax, Jmin, hx, hy, ni, nj, error1, iter);
        l += iter;
        cout << "iter: " << l << " error: " << error1 << endl;
    }
    double t2 = omp_get_wtime();

    ofstream output("output.plt");
    output << "Variables = \"x\", \"y\", \"u\"\n";
    output << "Zone i = " << ni << ", j = " << nj << "\n";
    for (int j = 0; j < nj; ++j)
        for (int i = 0; i < ni; ++i)
            output << fixed << setprecision(5)
                   << x[i][j] << " " << y[i][j] << " " << u[i][j] << "\n";
    output.close();

    cout << "time: " << t2 - t1 << "; l = " << l
         << "; l/t = " << l / (t2 - t1)
         << "; N = " << ni + nj * (ni - 2)
         - accumulate(Jmin.begin() + 1, Jmin.end() - 1, 0) << endl;

    int i_center = (ni - 1) / 2;
    int j_center = (nj - 1) / 2;
    int j_high = ((nj - 1) * 9) / 10;

    cout << " x = " << x[i_center][j_center]
         << "; y = " << y[i_center][j_center]
         << "; u = " << u[i_center][j_center] << endl;

    cout << " x = " << x[i_center][j_high]
         << "; y = " << y[i_center][j_high]
         << "; u = " << u[i_center][j_high] << endl;

    return 0;
}
