# EPANET structures

The purpose of this note is to walk through the fundamental data
structures in EPANET in order to make it easier to replace the current
linear solver with something more modern (and potentially speed up
some other things as well).

As a general note, despite writing in C, the EPANET developers use
1-based array indexing by default (and allocate an extra dummy entry
to compensate).

## Network topology

A water network consists of nodes (junctions, reservoirs, tanks)
connected by links (pipes, pumps, and various types of valves).
There are two separate data structures that represet the network
topology.

We note that the network topology is generally very "2D" in nature,
which means that the direct factorization of the associated system
matrices should be quite fast.

### Reference arrays

The `Node` and `Link` arrays in the `Network` object represent all
nodes and links in the network.  A link has start and end nodes
(indices `N1` and `N2` in the `Slink` struct).  Links and nodes are
both listed in the "natural" input order.

The `Node` and `Link` arrays are the reference data structure.  They
are typically traversed when assembling contributions to the system
matrix and the right hand side vector; they are also traversed to
create the `Adjlist` structure.

### Adjlist

The `Adjlist` array is an adjacency list representation of the network
connectivity.  The `Adjlist[i]` entry is a pointer to the head of a
linked list with entry types `Sadjlist`.  Each entry in the linked
list has an index to the other node for the link (i.e. `entry.node` in
the `i`th list corresponds to a link connecting nodes `i` and `j`) and
an index for the entry in the `Link` array.  Note that each link is
referenced twice in this data structure (once for each end point).

There are three flavors of adjacency lists in EPANET:

- The adjacency lists constructed in `project.c`.  This has one entry
  for each end point for each relevant link.
- The "local" adjacency lists constructed in `smatrix.c`.  This is
  similar to the initial adjacency lists, but removes items associated
  with parallel links (those that have the same start/end points).
- The augmented adjacency lists constructed in `smatrix.c`.  This is
  the local adjacency lists plus entries for possible structural
  nonzeros created during factorization.  The augmented adjacency
  lists are computed via a symbolic Cholesky factorization.

All three types of lists are referred to through the `Adjlist`
structure.  However, the local adjacency lists and augmented adjacency lists
are only used in the routines called from `createsparse`; at the end
of the `createsparse` call, the main adjacency lists are
reconstructed.

*NB*: In the local adjacency list computation, links `i` and `j` are
merged only if `Link[i].N1 == Link[j].N1 && Link[i].N2 == Link[j].N2`.
The code does not check 
for `Link[i].N1 == Link[j].N2 && Link[i].N2 == Link[j].N1`.
This seems like a bug, as it would result in some missing
contributions to off-diagonal entries of the system matrix in this
case.  If this is a bug, it is not an easy-to-find one -- it would
affect the solver convergence rates, but would not necessarily make it
so that the solver would fail to converge to the correct answer.

## Matrix representations

For the system matrix and the right-hand side vectors, there are two
relevant orderings of nodes: the "natural" input ordering, and the
optimized ordering used to minimize fill in the factorization step.
The `Row` array in `Smatrix` maps from the natural order to the index
order (i.e. `Row[n]` is the optimized index associated with node
number `n` in the natural order).  The `Order` array maps from the
index order back to the natural order.

The matrix coefficients are stored in two vectors, `Aii` and `Aij`.
The `Aii` array stores the diagonal entries of the system matrix in
the optimized order.  The `Aij` matrix stores the off-diagonal entries
in parallel to the links array (with one exception for parallel links,
noted below).  The right hand side vector `F` for linear systems is
also stored in optimized order.

In the case of a network with no parallel links, the `Ndx` array
corresponds to an identity mapping, i.e. `Ndx[k] == k`.  When a
network has parallel links `k1`, ..., `km` (in increasing order),
then `Ndx[kj] == k1` for each parallel link.

The `LNZ`, `NZSUB`, and `XLNZ` arrays in `SMatrix` form a compressed
sparse column representation of the strictly lower-triangular part of
the system matrix.  Here `Aij[LNZ[k]]` is the kth entry of the
strictly lower-triangular part in column-major order; `NZSUB[k]` is
the corresponding row index; and `XLNZ` is the start position of each
column in `LNZ` and `NZSUB` Note that `Aij` is longer than the number
of links -- it is extended to include not only the nonzeros associated
with the original matrix, but also those associated with potential
fill in the factorization.

## Solver flow

The basic stages in the solver algorithm are:

- Initially
  - Computation of node order (`genmmd.c`)
  - Symbolic factorization and construction of the index structure (`smatrix.c`)
- For each time step and each solver step underneath
  - Construction of the system matrix and right hand side (`hydcoeffs.c`)
  - Numerical factorization and solve (`smatrix.c`)

## Recommended improvement

Though there is probably room for performance improvement in the
beginning, the main improvements are likely to come from the numerical
solver.  My recommended strategy for speeding this up would be:

- Change the initialization to create the compressed sparse column
  representation used by CHOLMOD, and use CHOLMOD's reordering and
  symbolic factorization routines.
- Change the assembly to directly target the CHOLMOD input data
  structure.
- Use CHOLMOD's factorization and solve routines

One could try to change the higher-level algorithm to re-use the
system matrix factorization for multiple steps, but this seems like a
separate project.
