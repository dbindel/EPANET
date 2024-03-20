#include <stdarg.h>
#include <stdlib.h>

#include <cholmod.h>
#include <SuiteSparse_config.h>
#include <vector>
#include <algorithm>
#include <unordered_set>

#define XDTYPE (CHOLMOD_REAL + CHOLMOD_DOUBLE)

static cholmod_common common;

/* Exported Functions. */
extern "C" int linsolve_cholmod(int n, int ncoeffs, int *XLNZ, int *NZSUB, int *LNZ, double *Aii, double *Aij, double* B);
extern "C" void linsolve_init();
extern "C" void linsolve_finalize();

static void create_tril_csc(int n, int ncoeffs, const int *XLNZ, const int *NZSUB, const int *LNZ,
                   const double *Aii, const double *Aij,
                   cholmod_sparse **out);
static int linsolve_printf(const char* s, ...);

static void create_tril_csc(int n, int ncoeffs, const int *XLNZ, const int *NZSUB, const int *LNZ,
                   const double *Aii, const double *Aij,
                   cholmod_sparse **out) {
    int max_nz = ncoeffs + n; // max non-zeroes in the full matrix
    cholmod_sparse *sp = cholmod_allocate_sparse(n, n, max_nz, 0, 1, -1, XDTYPE, &common);
    int *col_ptrs = reinterpret_cast<int*>(sp->p);
    int *row_idx = reinterpret_cast<int*>(sp->i);
    double *values = reinterpret_cast<double*>(sp->x);

    // adjust XLNZ to be 0-indexed
    ++XLNZ;

    col_ptrs[0] = 0; // first column starts at index 0
    int k = 0;

    for (int col = 0; col < n; ++col) {
        // diagonal value
        row_idx[k] = col;
        values[k] = Aii[col + 1];
        k++;

        // off-diagonal values in column col
        for (int i = XLNZ[col]; i < XLNZ[col + 1]; i++) {
            row_idx[k] = NZSUB[i] - 1;
            values[k] = Aij[LNZ[i]];
            k++;
        }

        col_ptrs[col+1] = k;
    }

    *out = sp;


    if (!cholmod_check_sparse(sp, &common)) {
        cholmod_print_sparse(sp, "A", &common);
        fprintf(stderr, "Error: Sparse matrix is invalid\n");
        exit(1);
    }
}

extern "C" int linsolve_cholmod(int n, int ncoeffs, int *XLNZ, int *NZSUB, int *LNZ, double *Aii, double *Aij, double* B) {
    cholmod_sparse *A;
    create_tril_csc(n, ncoeffs, XLNZ, NZSUB, LNZ, Aii, Aij, &A);
    
    cholmod_factor* L = cholmod_analyze(A, &common);
    cholmod_factorize(A, L, &common);
    if (common.status & CHOLMOD_NOT_POSDEF) {
        fprintf(stderr, "Matrix is not positive definite\n");
        // TODO: do something here
    }

    // copy over B
    cholmod_dense *b = cholmod_allocate_dense(n, 1, n, XDTYPE, &common);
    memcpy(b->x, B + 1, n * sizeof(double));

    cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, b, &common);
    memcpy(B + 1, x->x, n * sizeof(double));

    cholmod_free_factor(&L, &common);
    cholmod_free_dense(&b, &common);
    cholmod_free_dense(&x, &common);
    cholmod_free_sparse(&A, &common);

    return 0;
}

extern "C" void linsolve_init() {
    cholmod_start(&common);
    common.precise = true;
    common.print = 5;
}

extern "C" void linsolve_finalize() {
    cholmod_finish(&common);
}
