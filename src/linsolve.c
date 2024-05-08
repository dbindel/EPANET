#include "types.h"
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
void linsolve_update_structs(Smatrix* sm, int nlinks, int njuncs);
void linsolve_init();
void linsolve_finalize();


void alloc_scratch(SolverScratch **scratch, int n, int ncoeffs)
{
    SolverScratch *s = (SolverScratch*) malloc(sizeof(SolverScratch));
    s->A = cholmod_allocate_sparse(n, n, ncoeffs+n+1, 1, 1, -1, XDTYPE, &common);
    s->L = NULL;
    s->b = cholmod_allocate_dense(n, 1, n, XDTYPE, &common);
    s->x = cholmod_allocate_dense(n, 1, n, XDTYPE, &common);
    s->Y = cholmod_allocate_dense(n, 1, n, XDTYPE, &common);
    s->E = cholmod_allocate_dense(n, 1, n, XDTYPE, &common);
    *scratch = s;
}

void free_scratch(SolverScratch **scratch)
{
    SolverScratch *s = *scratch;
    cholmod_free_sparse(&(s->A), &common);
    cholmod_free_factor(&(s->L), &common);
    cholmod_free_dense(&(s->b), &common);
    cholmod_free_dense(&(s->x), &common);
    cholmod_free_dense(&(s->Y), &common);
    cholmod_free_dense(&(s->E), &common);
    free(s);
    *scratch = NULL;
}

static
double resid(int n, const int* A, const int* p, const int* i,
            double* B, double* X, double* R) {
    memcpy(R+1, B+1, n*sizeof(double));
    for (int j = 0; j < n; ++j) {
        for (int k = p[j]; k < p[j+1]; ++k) {
            int r = i[k];
            double aij = A[k];

            R[r+1] -= aij*X[j+1];
            if (r == j) continue;
            R[j+1] -= aij*X[r+1];
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

/* Maps indices of Aij (which doesn't contain the main diagonal) 
 * to indices of the full sparse matrix A (which does contain the main diagonal).
 */
static
void create_old_to_new_map(int* map, const int* Ndx, const int* XLNZ, int nlinks) {
    int col = 1;
    for (int i = 0; i < nlinks; i++) {
        while (i + 1 >= XLNZ[col+1]) {
            col++;
        }

        map[i] = col + i;
    }
}

/* Creates map that maps from entries of the partial sparse matrix Aij to the full
 * sparse matrix A. maxi refers to the maximum index of an entry in A, which
 * is used as the "trash" entry.
 */
static 
void create_maps(int* map, const int* Ndx, const int* XLNZ, const int* LNZ, 
                        int njuncs, int nlinks, int ncoeffs, int maxi) {
    int* old_to_new = (int*) malloc((nlinks + 1) * sizeof(int));
    int* inv_lnz = (int*) malloc(ncoeffs * sizeof(int));

    old_to_new[nlinks] = maxi;
    for (int i = 0; i < ncoeffs; i++) {
        old_to_new[i] = map[i] = maxi;
        inv_lnz[i] = nlinks;
    }

    // create inverse LNZ map
    for (int i = 1; i <= XLNZ[njuncs]; i++) {
        inv_lnz[LNZ[i] - 1] = i - 1;
    }

    // create old to new map
    create_old_to_new_map(old_to_new, Ndx, XLNZ, nlinks);

    for (int i = 0; i <= ncoeffs; i++) {
        map[i] = old_to_new[inv_lnz[i]];
    }

    free(inv_lnz);
    free(old_to_new);
}

static
void create_sp_structure(int n, int ncoeffs, const int *XLNZ,
                      const int *NZSUB, cholmod_sparse *sp)
{
    int *col_ptrs = (int*) sp->p;
    int *row_idx = (int*) sp->i;

    // adjust XLNZ to be 0-indexed
    ++XLNZ;
    
    col_ptrs[0] = 0; // first column starts at index 0
    int k = 0;
    
    for (int col = 0; col < n; ++col) {
        // diagonal value
        row_idx[k] = col;
        k++;
        
        // off-diagonal values in column col
        for (int i = XLNZ[col]; i < XLNZ[col + 1]; i++) {
            row_idx[k] = NZSUB[i] - 1;
            k++;
        }
        
        col_ptrs[col + 1] = k;
    }
}

void linsolve_update_structs(Smatrix* sm, int nlinks, int njuncs) {
    sm->map = (int*) malloc((sm->Ncoeffs) * sizeof(int));

    create_maps(sm->map, sm->Ndx, sm->XLNZ, sm->LNZ, njuncs, nlinks,
                 sm->Ncoeffs, sm->scratch->A->nzmax - 1);
    create_sp_structure(njuncs, sm->Ncoeffs, sm->XLNZ, sm->NZSUB, sm->scratch->A);

    sm->A = sm->scratch->A->x;
    sm->row_ptrs = (int*) sm->scratch->A->p;

    sm->scratch->L = cholmod_analyze(sm->scratch->A, &common);
}

int linsolve_cholmod(SolverScratch *s, int n, int ncoeffs,
                     const int *XLNZ, const int *NZSUB,
                     const int *LNZ, const double *Aii,
                     const double *Aij, double *B) {
    // if (!cholmod_check_sparse(s->A, &common)) {
    //     cholmod_print_sparse(s->A, "A", &common);
    //     fprintf(stderr, "Error: Sparse matrix is invalid\n");
    //     exit(1);
    // }

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

    // // Check residuals
    double rresid = resid(n, (int*) s->A->x, (int*) s->A->p, (int*) s->A->i,
                         (double*) (s->b->x)-1, B, (double*) (s->Y->x)-1);
    fprintf(stderr, "Relative resid: %g\n", rresid);

    return 0;
}

void linsolve_init()
{
    cholmod_start(&common);
    common.precise = true;
    common.print = 5;
    common.supernodal = CHOLMOD_SIMPLICIAL;
    common.final_ll = true;
    common.final_monotonic = false;
}

void linsolve_finalize()
{
    cholmod_finish(&common);
}

