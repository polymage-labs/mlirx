
This information goes together with this article on high-performance code
generation in MLIR:
https://github.com/bondhugula/llvm-project/blob/hop/mlir/docs/HighPerfCodeGen.md

### On running the DGEMM benchmark for experimentation

See the benchmark/ directory in mlir/.

To execute the included benchmark:

```
$ mlir-opt -hopt='vect=true copy=true unroll=true scalrep=true' benchmark/dgemm-tiled-benchmark.mlir -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm='use-aligned-alloc=1'  -canonicalize  | mlir-cpu-runner  -O3  -e main -time -reps=5   -entry-point-result=void    -shared-libs=lib/libmlir_runner_utils.so > /dev/null
```

The generated MLIR can be inspected by running the following while
adding/removing individual flags:

```
$ mlir-opt -hopt='vect=true,copy=true,unroll=true,scalrep=true' benchmark/dgemm-tiled-benchmark.mlir
```

### Command-line flags

**-hopt**: Customized matmul optimization sequence (based on the BLIS schedule)
           where the following opts can be enabled incrementally.

**-hopt-vect**: Enable auto-vectorization.

**-hopt-copy**: Enable packing of memrefs.

**-hopt-unroll**: Enable unroll-and-jam and unroll.

**-hopt-scalrep**: Perform scalar replacement.

Any combination of these could be used. The only optimization step not included
here is of loop tiling: as such, we start from an already tiled loop nest in
dgemm-tiled-benchmark.mlir (albeit with no other optimizations on it).
Performing the tiling via the existing utilities (mlir::tile and
mlir::interchange) is left as an exercise to the reader. :)

**To try some of these optimizations standalone**:

**-affine-vectorize**: Auto-vectorization (entirely different from the -affine-vectorize/"super vectorizer" in MLIR tree).

**-affine-scalrep**: Perform scalar replacement on subscripted memref accesses

```
$ mlir-opt -affine-vectorize -affine-scalrep benchmark/dgemm-tiled-benchmark.mlir
```

Please raise an issue at https://github.com/polymage-labs/mlirx/ if you find
something unexpected.
