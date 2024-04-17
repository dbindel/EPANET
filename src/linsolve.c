#include <stdlib.h>
#include <math.h>

#include <SuiteSparse_config.h>
#include <cholmod.h>

#define XDTYPE (CHOLMOD_REAL + CHOLMOD_DOUBLE)

static cholmod_common common;

typedef struct SolverScratch {
    cholmod_sparse *A;
    cholmod_factor *L;
    cholmod_dense *b;
    cholmod_dense *x;
    cholmod_dense *r;
    cholmod_dense *Y;
    cholmod_dense *E;
} SolverScratch;


/* Exported Functions. */
void alloc_scratch(SolverScratch **scratch, int n, int ncoeffs);
void free_scratch(SolverScratch **scratch);
int linsolve_cholmod(SolverScratch *scratch, int n, int ncoeffs,
                     const int *XLNZ, const int *NZSUB,
                     const int *LNZ, const double *Aii,
                     const double *Aij, double *B);
void linsolve_init();
void linsolve_finalize();


void alloc_scratch(SolverScratch **scratch, int n, int ncoeffs)
{
    SolverScratch *s = (SolverScratch*) malloc(sizeof(SolverScratch));
    s->A = cholmod_allocate_sparse(n, n, ncoeffs+n, 0, 1, -1, XDTYPE, &common);
    s->b = cholmod_allocate_dense(n, 1, n, XDTYPE, &common);
    s->x = cholmod_allocate_dense(n, 1, n, XDTYPE, &common);
    s->r = cholmod_allocate_dense(n, 1, n, XDTYPE, &common);
    s->L = NULL;
    s->Y = NULL;
    s->E = NULL;
    *scratch = s;
}

void free_scratch(SolverScratch **scratch)
{
    SolverScratch *s = *scratch;
    cholmod_free_sparse(&(s->A), &common);
    cholmod_free_factor(&(s->L), &common);
    cholmod_free_dense(&(s->b), &common);
    cholmod_free_dense(&(s->x), &common);
    cholmod_free_dense(&(s->r), &common);
    cholmod_free_dense(&(s->Y), &common);
    cholmod_free_dense(&(s->E), &common);
    free(*scratch);
    *scratch = NULL;
}

static
double check_resid(int n, int ncoeffs,
                   const int* XLNZ,
                   const int* NZSUB,
                   const int* LNZ,
                   const double* Aii,
                   const double* Aij,
                   double* B,
                   double* X,
                   double* R)
{
    memcpy(R+1, B+1, n*sizeof(double));
    for (int j = 1; j <= n; ++j) {
        R[j] -= Aii[j]*X[j];
        for (int k = XLNZ[j]; k < XLNZ[j+1]; ++k) {
            int i = NZSUB[k];
            double aij = Aij[LNZ[k]];
            R[i] -= aij*X[j];
            R[j] -= aij*X[i];
        }
    }
    double rnorm2 = 0.0;
    double bnorm2 = 0.0;
    for (int j = 1; j <= n; ++j) {
        rnorm2 += R[j]*R[j];
        bnorm2 += B[j]*B[j];
    }
    return sqrt(rnorm2/bnorm2);
}

static
void create_tril_csc(int n, int ncoeffs, const int *XLNZ,
                     const int *NZSUB, const int *LNZ, const double *Aii,
                     const double *Aij, cholmod_sparse *sp)
{
    int *col_ptrs = (int*) sp->p;
    int *row_idx = (int*) sp->i;
    double *values = (double*) sp->x;

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

int linsolve_cholmod(SolverScratch *s, int n, int ncoeffs,
                     const int *XLNZ, const int *NZSUB,
                     const int *LNZ, const double *Aii,
                     const double *Aij, double *B)
{
    create_tril_csc(n, ncoeffs, XLNZ, NZSUB, LNZ, Aii, Aij, s->A);

    if (s->L == NULL)
        s->L = cholmod_analyze(s->A, &common);

    cholmod_factorize(s->A, s->L, &common);
    if (common.status & CHOLMOD_NOT_POSDEF) {
        fprintf(stderr, "Matrix is not positive definite\n");
        // TODO: do something here
    }
    
    // copy from B, solve, and copy back
    memcpy(s->b->x, B + 1, n * sizeof(double));
    cholmod_solve2(CHOLMOD_A, s->L, s->b, NULL,
                   &(s->x), NULL,
                   &(s->Y), &(s->E), &common);
    memcpy(B + 1, s->x->x, n * sizeof(double));

    // Check residuals
    double rresid =
        check_resid(n, ncoeffs, XLNZ, NZSUB, LNZ, Aii, Aij,
                    (double*) (s->b->x)-1,
                    (double*) (s->x->x)-1,
                    (double*) (s->r->x)-1);
    fprintf(stderr, "Relative resid: %g\n", rresid);
    
    return 0;
}

void linsolve_init()
{
    cholmod_start(&common);
    common.precise = true;
    common.print = 5;
}

void linsolve_finalize()
{
    cholmod_finish(&common);
}

