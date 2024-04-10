#include <stdarg.h>
#include <stdlib.h>

#include <SuiteSparse_config.h>
#include <algorithm>
#include <cholmod.h>
#include <vector>

#define XDTYPE (CHOLMOD_REAL + CHOLMOD_DOUBLE)

static cholmod_common common;

extern "C" struct SolverScratch {
  cholmod_sparse *A;

  SolverScratch(int n, int ncoeffs);
  ~SolverScratch();
};

/* Exported Functions. */
extern "C" void alloc_scratch(SolverScratch **scratch, int n, int ncoeffs) {
  *scratch = new SolverScratch(n, ncoeffs);
}
extern "C" void free_scratch(SolverScratch **scratch) { delete *scratch; *scratch = nullptr; }
extern "C" int linsolve_cholmod(SolverScratch *scratch, int n, int ncoeffs,
                                const int *XLNZ, const int *NZSUB,
                                const int *LNZ, const double *Aii,
                                const double *Aij, double *B);
extern "C" void linsolve_init();
extern "C" void linsolve_finalize();

SolverScratch::SolverScratch(int n, int ncoeffs) {
  int max_nz = ncoeffs + n;
  A = cholmod_allocate_sparse(n, n, max_nz, 0, 1, -1, XDTYPE, &common);
}

SolverScratch::~SolverScratch() {
  cholmod_free_sparse(&A, &common);
}

static void create_tril_csc(int n, int ncoeffs, const int *XLNZ,
                            const int *NZSUB, const int *LNZ, const double *Aii,
                            const double *Aij, cholmod_sparse *sp) {
  int *col_ptrs = reinterpret_cast<int *>(sp->p);
  int *row_idx = reinterpret_cast<int *>(sp->i);
  double *values = reinterpret_cast<double *>(sp->x);

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

    col_ptrs[col + 1] = k;
  }

  if (!cholmod_check_sparse(sp, &common)) {
    cholmod_print_sparse(sp, "A", &common);
    fprintf(stderr, "Error: Sparse matrix is invalid\n");
    exit(1);
  }
}

extern "C" int linsolve_cholmod(SolverScratch *scratch, int n, int ncoeffs,
                                const int *XLNZ, const int *NZSUB,
                                const int *LNZ, const double *Aii,
                                const double *Aij, double *B) {
  auto A = scratch->A;

  create_tril_csc(n, ncoeffs, XLNZ, NZSUB, LNZ, Aii, Aij, A);

  cholmod_factor *L = cholmod_analyze(A, &common);
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

  return 0;
}

extern "C" void linsolve_init() {
  cholmod_start(&common);
  common.precise = true;
  common.print = 5;
}

extern "C" void linsolve_finalize() { cholmod_finish(&common); }
