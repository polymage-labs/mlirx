# MLIRX

MLIRX is MLIR with extensions. This repository is always meant to be in sync
with upstream LLVM, and includes non-trivial extensions to its sub-project MLIR
that are meant for advanced research and experimentation. LLVM project
components in this repository other than MLIR almost never deviate from
upstream. They currently do not.

## Why does MLIRX exist?

We believe that MLIR is still in really early stages of its evolution as far as
its code generation and optimization infrastructure goes, and there isn't yet a
community large enough to provide timely and meaningful reviews upstream to
allow rapid progress and experimentation. Most of the extensions in MLIRX are
thus planned for future contribution upstream when the time is right. MLIRX does
not deviate from MLIR on the core IR support infrastructure part, but primarily
provides additional ops and 'mid-level' optimization abstractions, utilities,
and passes supporting its polyhedral form.

## Upstream LLVM README

[The LLVM Complier Infastructure
README](https://github.com/llvm/llvm-project/blob/main/README.md)
