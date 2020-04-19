# RFC - affine.execute_region op

This proposal is on adding a new op named *affine.execute_region* to MLIR's [affine
dialect](https://github.com/llvm/llvm-project/blob/master/mlir/docs/Dialects/Affine.md).
The op allows the polyhedral form to be used without the need for outlining to
functions, and without the need to turn affine ops such as affine.for, affine.if
into standard unrestricted for's and if's respectively. In particular, with an
*affine.execute_region*, it is possible to represent *every* load and store operation
using an *affine.load* and *affine.store* respectively.

## Op Description

1. The *affine.execute_region* op introduces a new symbol context for affine
   operations.  It holds a single region,  which can be a list of one or more
   blocks. The op's region can have zero or more arguments, each of which can
   only be a *memref*.  The operands bind 1:1 to its region's arguments.  The op
   can't use any memrefs defined outside of it, but can use any other SSA values
   that dominate it. Its region's blocks can have terminators the same way as
   current MLIR functions (FuncOp) can.  Control from any *return* ops from the
   top level of its region returns to right after the *affine.execute_region* op.  Its
   control flow thus conforms to the control flow semantics of regions, i.e.,
   control always returns to the immediate enclosing (parent) op. The results of
   an affine.execute_region op match 1:1 with the return values from its region's blocks.

2. The requirement for an SSA value to be a valid symbol
   ([mlir::isValidSymbol](https://github.com/llvm/llvm-project/blob/3671cf5558a273a865007405503793746e4ddbb7/lib/Dialect/AffineOps/AffineOps.cpp#L128))
   is now sensitive to a 'scope', which is identified by the closest surrounding
   op that is either an *affine.execute_region* or another FuncLikeOp (whichever
   is closer) - let such an op be called a scope op. Given a scope op, valid
   symbols also includes (a) symbols at the top-level of such a scope op, (b)
   those values that dominate the scope op. Symbol validity is sensitive to the
   enclosing execute_region/FuncLikeOp.  As such, there has to be an additional
   method: mlir::isValidSymbol(Value \*v, Operation \*op) to check for symbol
   validity for use in the specific op.

3. The op is not [isolated from
   above](https://github.com/llvm/llvm-project/blob/be6746fc3b9e6bd0527cc961256e362f44130cd4/include/mlir/IR/OperationSupport.h#L76)
   for typical SSA purposes, but effectively isolated for polyhedral purposes
     and affine passes (see further below on ''affine walks'', which do not walk
     into affine.execute_region's when walking from above).  All SSA values
   (other than
   memrefs) that dominate the op can be used in the execute_region. Constants
   and
   other replacements can freely propagate into it. Since memrefs from outside
   can't be used in the execute_region and have to be explicitly provided as
   operands,
   canonicalization or replacement for them will only happen through rewrite
   patterns registered on *affine.execute_region* op. More on this further below.

4. The op is eventually discarded by -lower-affine, with
   its region being inlined. The inlined IR will not violate any symbol
   restrictions since all affine ops will have been lowered to
   [standard](https://github.com/llvm/llvm-project/blob/master/mlir/docs/Dialects/Standard.md)
   ones. The existing
   [Inliner](https://github.com/llvm/llvm-project/blob/5b7d9bb465c0e86b2c8e506889daf5ae2619736d/include/mlir/Transforms/InliningUtils.h#L189)
   could be extended to inline *affine.execute_region* and used from the
   LowerAffinePass as its final step.

## Syntax

```mlir {.mlir}
// Custom form syntax.
(ssa-id `=`)? `affine.execute_region` `[` memref-region-arg-list `]` `=` `(` memref-use-list `)`
                 `:` memref-type-list-parens `->` function-result-type `{`
    block+
`}`
```

## Terminology

An *affine scope* is the set of all ops that have the same closest enclosing
*affine.execute_region* op or function op (if one or both of them
have no enclosing *affine.execute_region*). Every MLIR op is always part of a
unique *affine scope*.

## Goals

An *affine.execute_region* op's goal is to start a new polyhedral scope, i.e.,
an [*affine scope*](#Terminology). IR that would have otherwise been considered
non-affine and failed verification will be affine with affine.execute_region's
inserted at the right places. In addition, the design choices made herein ensure
that:

* nearly all existing pattern rewrites work across execute_region op boundaries,
* all existing affine passes work correctly as is in the execute_region ops while
  allowing affine passes to work seamlessly at an affine.execute_region-local
  affine level,
* everything that forced function outlining due to symbol restrictions will no
  longer require such outlining.

## Examples

Here are three examples: one related to non-affine control flow, one to
non-affine loop bounds, and the third to non-affine data accesses that can be
represented via affine.execute_region's without having to outline the function
or
without having to use standard (unrestricted) load/stores or loops: the latter
are the current ways of representing them.

### Example 1

This example was used in the Rationale document to show how
outlining to a separate function allowed representation using existing
affine constructs:
https://github.com/llvm/llvm-project/blob/master/mlir/docs/Rationale.md#Examples

```
// A simple linear search in every row of a matrix.
for (i = 0; i < N; i++) {
  for (j = 0; j < N; j++) {
    // Dynamic/non-affine control flow
    if (a[i][j] == key) {
      s[i] = j;
      break;
    }
  }
}
```

All affine.load/store and nested CFG with an affine.execute_region. There are two *affine
regions* here - the outer one containing four ops (including the return
terminator), and the inner one being the region of the *affine.execute_region*.

```mlir {.mlir}
// CHECK-LABEL: func @search
func @search(%A : memref<?x?xi32>, %S : memref<?xi32>, %key : i32) {
  %ni = dim %A, 0 : memref<?x?xi32>
  %c1 = constant 1 : index
  // This loop can be parallelized.
  affine.for %i = 0 to %ni {
    affine.execute_region [%rA, %rS] = (%A, %S) : memref<?x?xi32>, memref<?xi32> {
      %c0 = constant 0 : index
      %nj = dim %rA, 1 : memref<?x?xi32>
      br ^bb1(%c0 : index)

    ^bb1(%j: index):
      %p1 = cmpi "slt", %j, %nj : index
      cond_br %p1, ^bb2(%j : index), ^bb5

    ^bb2(%j_arg : index):
      %v = affine.load %rA[%i, %j_arg] : memref<?x?xi32>
      %p2 = cmpi "eq", %v, %key : i32
      cond_br %p2, ^bb3(%j_arg : index), ^bb4(%j_arg : index)

    ^bb3(%j_arg2: index):
      %j_int = index_cast %j_arg2 : index to i32
      affine.store %j_int, %rS[%i] : memref<?xi32>
      br ^bb5

    ^bb4(%j_arg3 : index):
      %jinc = addi %j_arg3, %c1 : index
      br ^bb1(%jinc : index)

    ^bb5:
      return
    }
  }
  return
}
```

### Example 2

Loop bounds that were originally non-affine.

```
for (i = 0; i < N; i++)
  for (j = 0; j < N; j++)
    // Non-affine loop bound for k loop
    for (k = 0; k < pow(2, j); k++)
       for (l = 0; l < N; l++)
         // block loop body
        ...
```

All affine.for with an affine.execute_region.

```mlir {.mlir}
  %c2 = constant 2 : index
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      affine.execute_region [] = () {
        %pow = call @powi(%c2, %j) : (index, index) ->  index
        affine.for %k = 0 to %pow {
          affine.for %l = 0 to %n {
            ...
          }
        }
        return
      }  // execute_region end
    }  // %j
  }  // %i
```

### Example 3

Data accesses that were originally non-affine.

```
for (i = 0; i < N; ++i) {
  A[B[i]]++;
}
```

```mlir {.mlir}
  %cf1 = constant 1.0 : f32
  affine.for %i = 0 to 100 {
    %v = affine.load %B[%i] : memref<100xf32>
    affine.execute_region [%rA] = (%A) : memref<100xf32> {
      // %v is now a symbol here.
      %s = affine.load %rA[%v] : memref<100xf32>
      %o = addf %s, %cf1 : f32
      affine.store %o, %rA[%v] : memref<100xf32>
      return
    }
  }
```


## Helpers, Utilities, and Passes

* **Hoist or eliminate affine.execute_region's**

   There will be a function pass that will hoist or eliminate unnecessary
   *affine.execute_region* ops, i.e., when an *affine.execute_region* can be eliminated or
   hoisted without violating the dimension and symbol requirements. The
   propagation of constants and other simplification that happens in scalar
   optimizations / canonicalization often helps get rid of
   affine.execute_region's.  As
   such it's useful to have non-memref SSA values be implicitly captured and not
   have the op completely isolated from above --- this is another reason it is
   not a blackbox.

* **Walkers: GrayBoxes are isolated from above for polyhedral passes**

   There has to be a new walkAffine method that behaves like
   [walk](https://github.com/llvm/llvm-project/blob/e4e6ec350536905f0eeb85c658f87da848ed7658/include/mlir/IR/Operation.h#L511)
   except that it does not walk regions of an affine.execute_region.  Most polyhedral/affine passes will use this and thus see
   *affine.execute_region* as opaque *for any walks from above*.

   An affine pass' run should be changed to run on its function op
   as well as every *affine.execute_region* op in the function op. Unfortunately, these
   runs can only be done sequentially only because the "declaration" of the
   *affine.execute_region* and the "imperative" call to it are one thing - the affine
   affine.execute_region's in a function are otherwise disjoint and can otherwise be
   processed in parallel. In summary, there can be an AffineFunctionPass that,
   instead of providing
   [runOnFunction](https://github.com/llvm/llvm-project/blob/8aadfe58a5deea85656a8d8318e6b40d9de350c4/include/mlir/Pass/Pass.h#L267)
   needs to only implement an runOnOperation(op) where op is either a FuncOp or
   an AffineGrayBoxOp.

   Some of the current polyhedral passes/utilities can continue using walk (for
   eg. [normalizeMemRefs](https://github.com/llvm/llvm-project/blob/331c663bd2735699267abcc850897aeaea8433eb/include/mlir/Transforms/Utils.h#L89), while many will just have to be changed to use walkAffine.

* **Simplification / Canonicalization**

   There has to be a simplification that drops unused block arguments from
   an *affine.execute_region*'s region. This is easily implemented as a
   canonicalization on the op, and will allow removal of dead memrefs that would
   otherwise have operand uses in affine.execute_region ops
   with the corresponding region arguments not having any uses inside the
   execute_region. Canonicalization patterns would be needed to drop duplicate
   region
   arguments and unused region arguments.
   [MemRefCastFold](https://github.com/llvm/llvm-project/blob/ef77ad99a621985aeca1df94168efc9489de95b6/lib/Dialect/StandardOps/Ops.cpp#L228)
   is another canonicalization pattern that the *affine.execute_region* has to
   implement, and this is easily/cleanly done (by replacing the argument and its
   uses with a memref of a different type). Overall, having memrefs as explicit
   arguments is a good middle ground to make it easier to let standard SSA
   passes / scalar optimizations / canonicalizations work unhindered in
   conjunction with polyhedral passes, and with the latter not worrying about
   explicitly checking for escaping/hidden memref accesses. More discussion on
   this is further below in the Rationale section.

*  **Memref replacement across execute_region's**
   There are situations/utilities where one can consistently perform
   rewriting/transformation/analysis cutting across affine.execute_region's. One example is
   [normalizeMemRefs](https://github.com/llvm/llvm-project/blob/331c663bd2735699267abcc850897aeaea8433eb/include/mlir/Transforms/Utils.h#L89),
   which turns all non-identity layout maps in memrefs into identity
   ones. Having memrefs explicitly captured is a slight hindrance here, but
   [mlir::replaceAllMemrefUsesWith](https://github.com/llvm/llvm-project/blob/331c663bd2735699267abcc850897aeaea8433eb/include/mlir/Transforms/Utils.h#L69)
   can be extended to transparently perform the
   replacement inside any affine.execute_region's encountered if the caller says
   so.
   In other cases like scalar replacement, memref packing / explicit copying,
   DMA generation, pipelining of DMAs, transformations are supposed to be
   blocked by those boundaries because the accesses inside the execute_region can't be
   meaningfully analyzed in the context of the surrounding code. As such, the
   memrefs there are treated as escaping / non-dereferencing.

*  **Inlining**
   In the presence of affine constructs, the inliner can now simply inline
   functions by putting the callee inside an affine.execute_region, without having to
   worry about symbol restrictions.

* *op.getParentOfType\<AffineGrayBoxOp\>()* can already be used to return the
  closest enclosing *affine.execute_region* op or null if it hits a function op.


## Other Benefits and Implications

1. The introduction of this op allows arbitrary control flow (list of basic
   blocks with terminators) to be used within and mixed with affine.fors/ifs
   while staying in the same function. Such a list of blocks will be carried by
   an *affine.execute_region* op whenever it's not at the top level.

2. Non-affine data accesses can now be represented through
   *affine.load/affine.store* without the need for outlining.

3. Symbol restrictions for affine constructs will no longer restrict inlining:
   any function can now be inlined into another by enclosing the just inlined
   function into an affine.execute_region.

4. Any access to a memref can be represented with an *affine.load/store*. This
   is quite useful in order to reuse existing passes more widely (for eg. to
   perform scalar replacement on affine accesses) --- there is no reason
   memref-dataflow-opt won't work on / shouldn't be reused on straightline code,
   which is always a valid *affine.execute_region* region (at the limit, the innermost
   loop body is a valid *affine.execute_region* under all circumstances.

5. Any countable C-style 'for' without a break/continue can be represented as an
   affine.for (irrespective of bounds). Any if/else without a continue/break
   inside can be represented as an affine.if. The rest are just represented as a
   list of basic blocks with an enclosing *affine.execute_region* if not at the
   top-level.

6. SSA scalar opts work across "affine boundaries" without having to be
   interprocedural.

## Rationale and Design Alternatives: What to explicitly capture?

We discuss here the trade-offs for the various design points that differ in what
is explicitly captured by an *affine.execute_region*, i.e., which SSA values should be
operands, and provide the rationale for explicitly capture memref
uses alone (i.e., implicitly capturing all other SSA values). The competing
design point is that of allowing all SSA values including memrefs to be
implicitly captured, i.e., zero operands and arguments for *affine.execute_region*.

Consider the following representation where an affine.execute_region is used since the
access to the output memref would otherwise be non-affine.

```mlir {.mlir}
func @foo(%A : ..., %X : ..., %Y : ..., %Z : ...) {
  affine.for %i = 0 to %N {
    affine.store %cf0, %Y[%i] : memref<100xf32>
  }

  affine.for %i = 0 to %N {
    affine.for %j = 0 to %N {
      %lhs = affine.load %A[%i, %j] : memref<100xf32>
      %rhs = affine.load %X[%j] : memref<100xf32>
      %p = mulf %lhs, %rhs : f32
      %idx.i32 = affine.load %Z[%j] : memref<100 x i32>
      %idx = index_cast %idx.i32 : i32 to index
      affine.execute_region [%rY] = (%Y) : memref<100xf32> {
        %in = affine.load %rY[%idx] : memref<100xf32>
        %out = addf %p, %in : f32
        affine.store %out, %rY[%idx] : memref<100xf32>
        return
      }
    }
  }
  return
}
```

In the above IR, %idx is a valid symbol for the execute_region and thus the load and
store on %Y in the execute_region is affine. For most polyhedral analysis and
transformation, the execute_region is treated opaquely; the inside of the execute_region
itself is all locally affine. There is no way to analyze or represent
dependences precisely between the store on %Y outside the execute_region and the
load/store on it inside the execute_region. Depedence analysis relies on symbolic
context which is unique to a specific [*affine scope*](#Terminology).

Note that all scalars within their dominance scope are used freely in the
execute_region region. As an example, any pass that computes memref regions for the
purpose of packing, explicit copying, or estimating memory footprints will only
be able to compute regions for %A, %X, %Z while traversing the outer affine
region.  There is no way to meaningful represent the region of %Y accessed from
the outer context since %idx is not a symbol for the outer region.
%Y for the outer region is as good as an argument that escaped as an argument to
a call.  Passes like DMA generation or explicit copying are expected to do
nothing to %Y when running on the function (outer region). That said, when
conservative analysis is needed in the future, summaries on such nested regions
can be constructed, but that would have to be handled the same way for call ops.

### Explicitly Capture only MemRefs

Capturing everything implicitly (i.e., no arguments/operands on the
*affine.execute_region* op turns out inconvenient for all polyhedral transformations
and analyses. The latter would have to check and scan any
affine.execute_region's
encountered to see if any memrefs are being used therein, and if so, they would
in most cases have to treat them as if the memrefs were being passed to a
function call. This would be the case with dependence analysis, memref region
computation/analysis, fusion, explicit copying/packing, DMA generation,
pipelining, scalar replacement and anything depending on the former analyses
(like tiling, unroll and jam).

Having memrefs explicitly as operands means that the inside of
affine.execute_region's will not have to be inspected in walks from above (which
is also why they are called affine.execute_region's), effectively maintaining
isolation for polyhedral and affine passes. As an example, consider a pass
that's computing memref regions and generating packing code for a memref. The
scan of uses that currently happens via methods like getUses(),
replaceAllMemRefUsesWith() will all just work transparently and do the work: the
non-dereferencing uses of that memref on an affine.execute_region op just makes
things like double buffering, data copy generation, etc. all bail out on those
(just because it isn't polyhedrally analyzeable unless the execute_region can be
eliminated and you get a larger encompassing [*affine scope*](#Terminology)) --
the same way they currently bail out on any call ops taking memrefs as arguments
or return ops returning memrefs.  The same is true for memref dependence
analysis: there isn't a way to represent dependences between an affine access
and another one that is inside another execute_region dominated by it - for all these
purposes, the latter access is like one happening on a memref that has escaped
at the execute_region op boundary. With explicit capture of memref's, an
affine.execute_region gets treated as any other op (for eg. like a 'call' op
that takes memref as an operand), and so for an affine pass, you don't even have
to know that the affine.execute_region exists, and one won't even have to modify
any of these passes. *walkAffine* will simply not walk their regions, and
"operand uses" consistently have all that the affine pass needs for *every* op.

The isolation of an affine.execute_region's region for polyhedral passes running from above is
necessary (this is why it's not a white box), and to do so cleanly, explicitly
capturing all those memrefs that are used inside as operands helps.

### Maintaining MemRef Operands/Arguments

The discussion here is on the additional complexity of maintaining memrefs as
operands/arguments on *affine.execute_region*. Note that other SSA values are still
implicitly captured. The trade-offs are very different for memrefs vis-a-vis
non-memref SSA values.

**Additional canonicalization patterns needed on affine.execute_region**

Explicitly capturing memrefs necessitates additional pattern rewrites on
*affine.execute_region*  that wouldn't have been necessary had they been implicitly
captured. We identify a list of pattern rewrites needed below. The first one
already exists and can simply be registered. The second one is trivial and has
to anyway exist and is reusable/necessary for any op holding one or regions. The
third one is non-trivial and is a good example to explain the burden explicit
capture entails.

* [**MemRefCastFolder**](https://github.com/llvm/llvm-project/blob/ef77ad99a621985aeca1df94168efc9489de95b6/lib/Dialect/StandardOps/Ops.cpp#L228):
  this pattern already exists and can merely be
  registered on the affine.execute_region op.

* **RegionArgumentCanonicalizer**: drop any unused or duplicate region arguments
  for ops with one or more regions. Other than FuncOp, nearly every other op
  with a region would want to have such a pattern registered. Unlike a FuncOp,
  an *affine.execute_region*'s declaration and invocation is one thing (it's
  not "callable" from elsewhere like a FuncOp); as such, an update to its
  arguments is easy.

* **SimplifyDeadDealloc**: This pattern removes any deallocs on memrefs if
  all other uses of the memref are dealloc's and if the memref is the result of
  an alloc. With the explicit capture of the memref, note that this pattern by
  itself not sufficient to remove all dead deallocs since memref are going to be
  passed via the region operands and its arguments to the ops within the region.
  However, this will have to be addressed by adding an extra pattern on
  affine.execute_region just to make sure the simplification works across region
  boundaries. The pattern will have to work as follows: if all uses of the
  memref operand on the affine.execute_region inside its region are dealloc's
  (including recursively any other nested *affine.execute_region* ops), all those
  dealloc's can be removed and the operand memref can be dropped on the
  execute_region.
  Likewise any other patterns that are added and that have to do with memrefs
  will have to be implemented on *affine.execute_region* to make them work across
  boundaries.

Other than these three, it does not look like any more pattern rewrites are
needed on *affine.execute_region* due to the explicit capture of memrefs. It is clear
that not having any operands/arguments and capturing everything
implicitly is the best thing for pattern matching and rewrites to work. Explicit
capture only necessitates additional code for pattern rewrites, which in some
cases is trivial, in some has to be anyway available / is already available, and
in other cases, leads to addition of non-trivial canonicalization patterns on
*affine.execute_region* (like SimplifyDeadDealloc). These downsides have to be
weighed against the benefits explicit capture brings for the polyhedral/affine
passes, none of which are based on pattern rewrites (but are aided by pattern
rewrites before/after polyhedral passes). On this note, one should also consider
the fact these additional patterns are probably not specific to *affine.execute_region*
to many other ops  that have a region and choose to explicitly capture (either
everything or specific things).

Overall, we have made two arguments here:

(a) having to inspect, scan, gather memrefs accessed within execute_region's
from
above
within all affine passes is not worth the special casing needed just for the
*affine.execute_region* op in polyhedral passes.

(b) maintaining the memref arguments is *often* lightweight, and
while there is benefit in having them implicitly captured for pattern rewrites,
it doesn't appear to be worth the trouble of the special casing/handling
described in (a). The additional canonicalization patterns on
*affine.execute_region* will take care of this, and such patterns may anyway be
needed / shared with for other ops that choose to do explicit capture for
other reasons.


### Other Design Points

Another design point could be of requiring symbols associated with the
affine constructs used in an affine.execute_region, but defined outside, to be explicitly
listed as operands/arguments; this is in addition to the memrefs used. This
makes isValidSymbol really simple: because one won't need isValidSymbol to be
context sensitive. Anything that is at the top-level of an *affine.execute_region* op
or its region argument will become a valid symbol. However, other than this, it
doesn't simplify anything else. Instead, it adds/duplicates a bookkeeping with
respect to propagation of constants, similar, to some extent, to the argument
rewriting done for interprocedural constant propagation. Similarly, the other
extreme of requiring everything from the outside used in an *affine.execute_region* to
be explicitly listed as its operands and region arguments is even worse on this
front.

### Conclusion
In summary, it appears that the requirement to explicitly
capture only the memrefs used inside an *affine.execute_region*'s region is a good
middle ground to make it easier to let standard SSA passes / scalar optimization
/ canonicalization work unhindered in conjunction with polyhedral passes, and
with the latter not worrying about  explicitly checking for escaping/hidden
memref accesses in execute_region ops. In the whole scheme of things, a design that
implicitly captures everything is not radically different that it'd be hard to
switch from one to the other at a later stage.
