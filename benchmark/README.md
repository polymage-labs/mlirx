
This information goes together with this article on high-performance code
generation in MLIR:
https://github.com/bondhugula/mlir/blob/hop/g3doc/HighPerfCodeGen.md

*On running the DGEMM benchmark for experimentation*

See the benchmark/ directory in mlir/

```
$ ../../../build.release/bin/mlir-opt -hopt -hopt-vect -hopt-unroll -hopt-copy -hopt-scalrep benchmark/dgemm-tiled-benchmark.mlir  -convert-linalg-to-loops   -convert-linalg-to-llvm -convert-std-to-llvm  -canonicalize  | /home/uday/llvm-project-bondhugula/build.release/bin/mlir-cpu-runner  -O3  -e main -time -reps=5   -entry-point-result=void    -shared-libs=/home/uday/llvm-project-bondhugula/build.release/lib/libmlir_runner_utils.so > /dev/null
```

Take a look at the generated MLIR by running this and adding/removing flags one
by one:

```
$ mlir-opt -hopt -hopt-vect -hopt-copy -hopt-unroll -hopt-scalrep benchmark/dgemm-tiled-benchmark.mlir
```

Command-line flags

-hopt: Customized matmul optimization sequence (based on the BLIS schedule)
       where you can enable the following opts incrementally.

-hopt-vect: Enable auto-vectorization

-hopt-copy: Enable packing of memrefs

-hopt-unroll: Enable unroll-and-jam and unroll

Any combination of these could be used.

To try standalone:

-affine-vect: auto-vectorization

-affine-scalrep: scalar replacement

$ mlir-opt -affine-vect -affine-scalrep benchmark/dgemm-tiled-benchmark.mlir

Please raise an issue at https://github.com/bondhugula/mlir/ if you find
something unexpected.
