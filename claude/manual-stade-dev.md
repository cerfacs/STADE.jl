# A developer's manual to `src/STADE.jl`

This manual is for a developer who will change `src/STADE.jl` and who has
never worked on automatic differentiation before. It assumes you can read
Julia. It assumes nothing about derivatives beyond first-year calculus.

Read Part I if the words *tangent*, *adjoint*, and *tape* are new to you.
Read Part II for the shape of the program. Read Part III to learn how to
read STADE's own output, which is the main debugging tool. Part IV walks
each pipeline stage. Part V is a glossary of every domain-specific term
the source code uses. Part VI covers the build and the test discipline.

Line numbers refer to `src/STADE.jl` at the revision this manual
documents. They will drift. The `# ==== prefix_* ====` banners will not.

---

## Table of contents

- [Part I. Automatic differentiation from zero](#part-i-automatic-differentiation-from-zero)
- [Part II. What STADE is](#part-ii-what-stade-is)
- [Part III. Reading generated code](#part-iii-reading-generated-code)
- [Part IV. The pipeline, stage by stage](#part-iv-the-pipeline-stage-by-stage)
- [Part V. Glossary](#part-v-glossary)
- [Part VI. Working on STADE](#part-vi-working-on-stade)

---

# Part I. Automatic differentiation from zero

## I.1 The problem

A numerical kernel is a function. It reads some numbers and writes some
numbers. Here is one:

```julia
function affine_loss(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
```

Mathematically this is a function `f`. It maps the inputs `u`, `a`, `b`
to the outputs `v` and `loss`. Many applications need the derivatives of
`f`. Gradient-based optimization needs them. Sensitivity analysis needs
them. Data assimilation needs them. Neural network training needs them.

You could compute a derivative by hand and write a second kernel. That is
slow work and it goes stale every time the first kernel changes.

You could use finite differences. Evaluate `f(x + h)`, evaluate
`f(x - h)`, and divide the difference by `2h`. This is easy but it has two
defects. It costs one full evaluation of `f` per input direction, and it
is only accurate to about 8 significant digits at best.

Automatic differentiation (AD) is the third option. An AD tool reads the
kernel and produces a second program. The second program computes exact
derivatives, up to floating-point rounding. STADE is such a tool.

## I.2 A program is a chain of elementary operations

The key insight of AD is small. Every kernel decomposes into elementary
operations. Each elementary operation has a known derivative. The chain
rule composes them.

Take one line of the kernel above:

```julia
v[i_x] = a[i_x] * u[i_x] + b[i_x]
```

This is a multiply followed by an add. The derivative of `x * y` with
respect to `x` is `y`. The derivative with respect to `y` is `x`. The
derivative of `x + y` is `1` with respect to each argument. These are the
**local partial derivatives**. STADE keeps a table of them, one entry per
allowed operator. The table is in the `der_*` section.

Nothing about AD is more complicated than that. Everything else is
bookkeeping. The bookkeeping is what fills 8600 lines.

## I.3 Two directions

The chain rule composes in two orders. Both give correct answers. They
cost different amounts, and they need different amounts of memory.

### Forward mode, also called tangent mode

Pick one input direction. Call it `Xd`. Push it through the program in
the same order the program runs. Each variable gains a companion called
its **shadow**, holding that variable's derivative along the chosen
direction. STADE names the shadow of `u` as `ud`.

For the line above, the tangent line is:

```julia
vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
```

Read it as the product rule, then the sum rule. The shadow line is placed
**before** the primal line, so it reads pre-statement values.

One forward pass gives you `J * Xd`, one column-combination of the
Jacobian. Getting a full gradient of a scalar loss with respect to `n`
inputs costs `n` forward passes. Forward mode is cheap when the number of
inputs is small.

### Reverse mode, also called adjoint mode

Pick one output direction. Call it `Yb`. Push it through the program in
**reverse** order. Each variable gains a shadow holding the sensitivity of
the seeded output to that variable. STADE names the adjoint shadow of `u`
as `ub`.

The reverse of the same line is:

```julia
ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
bb[i_x] = bb[i_x] + vb[i_x]
vb[i_x] = 0.0
```

Read it as follows. The sensitivity that arrived at `v` is `vb`. Each
input of the statement receives `vb` multiplied by that input's local
partial. The contributions accumulate with `+=` because one variable can
feed many later statements. The last line clears `vb`, because `v`'s own
sensitivity is now fully distributed and the slot must be clean before an
earlier statement writes into it.

One reverse pass gives you `J' * Yb`. For a scalar loss, one reverse pass
gives the **whole gradient**, whatever the number of inputs. This is why
reverse mode dominates in machine learning and in large inverse problems.

### The cost of reverse mode

Reverse mode is not free. It runs backwards, so it needs values the
forward run has already overwritten. Consider:

```julia
for i_x = 2:i_n
    u[i_x] = c * u[i_x - 1]
end
```

Each iteration overwrites `u[i_x]`. The reverse sweep needs the value that
`u[i_x]` held **before** the write, for every iteration. That value is
gone.

The fix is to save it. Classical AD tools save values on a stack, often
called a **tape**. STADE does the same, and calls the saved values
**snapshots**. The generated code pushes before the write and pops during
the reverse sweep:

```julia
for i_x = 2:i_n
    push!(u_stack, u[i_x])       # forward sweep, saves the old value
    u[i_x] = c * u[i_x - 1]
end
# ... other forward code ...
for i_x = i_n:-1:2               # backward sweep, note the reversed range
    u[i_x] = pop!(u_stack)       # restore before differentiating
    cb = cb + u[i_x - 1] * ub[i_x]
    ub[i_x - 1] = ub[i_x - 1] + c * ub[i_x]
    ub[i_x] = 0.0
end
```

An adjoint program therefore has two halves. The **forward sweep** replays
the original computation and pushes snapshots. The **backward sweep**
runs the statements in reverse, pops snapshots, and accumulates
derivatives. Both halves live in one generated function.

Memory is now the central engineering problem of reverse-mode AD. Saving
every value is always correct and often too expensive. Most of STADE's
complexity exists to save less. Two analyses do this work, and both are
described in Part IV.

## I.4 Second derivatives

A Hessian-vector product (HVP) is `H * v`, where `H` is the matrix of
second derivatives. You rarely want the full Hessian, because it has `n^2`
entries. You often want its action on a vector, because Newton-type
optimizers and uncertainty analyses need exactly that.

STADE computes an HVP by **forward-over-reverse**. It takes the adjoint
program, which computes the gradient, and applies forward mode to it. The
derivative of the gradient in direction `v` is `H * v`.

This composes cleanly because reverse-mode output is straight-line replay
code. It has no data-dependent recursion left to confuse a second pass.
The `hvp_*` stage therefore walks the already-generated adjoint code
directly, and does not repeat the earlier analyses.

## I.5 Why source transformation

There are two families of AD implementation.

**Operator overloading** replaces `Float64` with a custom number type that
records operations while the program runs. This is easy to build and works
on almost any code. The derivative program exists only at run time. You
cannot read it, and you usually cannot run it without the AD library.

**Source transformation** reads the source code and writes new source
code. This is harder to build, and it works only on a restricted subset of
the language. In exchange you get a plain source file.

STADE is a source transformation tool. The three motivations are in the
project README. The output is auditable, because it is Julia you can read.
The output is reproducible, because it runs with no dependency on STADE.
The output is portable to GPUs, because a source-level loop can be turned
into a device kernel by a further source-level pass.

The price is the restricted input language. That restriction is a real
contract, and it is documented separately in `skill-stade`. A kernel that
breaks it gets a parse error rather than a wrong derivative.

---

# Part II. What STADE is

## II.1 The input contract in one page

STADE accepts a narrow subset of Julia. The full rule list lives in
`skill-stade`. The rules that shape STADE's own design are these.

- Every variable has one of four shapes. They are `Float64`, `Int64`,
  `Array{Float64}`, and `Array{Int64}`. There are no Bools, tuples,
  strings, structs, or ranges in variables.
- A kernel never allocates an array. Every array comes from the caller.
  A scalar result travels in a length-1 array.
- Arguments are positional. Keyword arguments are refused.
- Indirect indexing (`a[b[i]]`) is refused. Read the index into a scalar
  first.
- Broadcasting is refused. Write the loop.
- Only whitelisted intrinsics may be called. The list is the 27 keys of
  `der_table()`.
- Integer division is written `div(a, b)` and nothing else.
- `while` is not supported. Only `for`, `if`, and assignment exist.

Type annotations are permitted and inert. STADE infers each shape from
usage and drops the annotation.

## II.2 The four outputs

For a kernel named `foo`, STADE can generate:

| Output | Function name | Meaning |
|---|---|---|
| Primal | `foo` | The original kernel, copied into the output file |
| Tangent | `foo_d` | `J * Xd`, forward mode |
| Adjoint | `foo_b` | `J' * Yb`, reverse mode |
| HVP | `foo_hv` | `H * v`, forward-over-reverse |

An adjoint or an HVP also comes with a companion `initstacks_foo_b`. That
function allocates the snapshot storage and returns it. The caller calls
it once and passes the result into `foo_b`.

A further pass ports any of these to a GPU. The GPU function is named
`foo_b_cuda` and its device kernels are named `cuda_kernel_foo_b_1!` and
so on.

## II.3 The public API

Seven functions are exported.

```julia
stade_tangent_file(in_path, out_path; keep_push_pop=true, fuse_ii_loops=false)
stade_adjoint_file(in_path, out_path; keep_push_pop=true, fuse_ii_loops=false)
stade_hvp_file(in_path, out_path;     keep_push_pop=true, fuse_ii_loops=false)

stade_cuda_file(in_path, out_path;   precision=nothing, reduction_threshold=32768, warp_atomics=true, sync_free_reduce=true, hoist_atomics=true, gather_stencil=2, shared_private=true, coarsen=1, coarsen_min=256, coarsen_fill=65536, coarsen_occ=0.6, inbounds=true, dead_scalars=true, split_reduce=16, split_min_inner=256, split_max_out=32768, contraction=256)
stade_amdgpu_file(in_path, out_path; precision=nothing, reduction_threshold=32768)
stade_metal_file(in_path, out_path;  precision=nothing, reduction_threshold=32768)
stade_jacc_file(in_path, out_path;   precision=nothing, reduction_threshold=32768)
```

Each `_file` function has an `Expr`-in, `Expr`-out sibling without the
suffix, and a `_corpus` sibling that takes a `Dict{Symbol,Expr}` of
several kernels. The `_file` wrappers are the only code allowed to touch
the filesystem, through the `io_*` stage.

The differentiation entry points take **exactly two** flags. Do not add a
third without a strong argument. Both flags are explained in Part IV.

## II.4 The pipeline

Data flows through eighteen stages. Each stage has a name prefix. The
prefix is the whole namespacing mechanism, because the file has no
submodules.

```
  .jl file
     |  io_          read the file, find the function definitions
     v
  raw Expr
     |  inl_         splice callee bodies into callers (multi-kernel only)
     v
  flat Expr
     |  parse_       validate and convert to (sig, body)
     |  shape_       infer every variable's kind
     v
  kernel = (sig, body)
     |  norm_        insert dead loop-entry resets
     |  act_         which variables carry a derivative
     |  snap_        which writes need a snapshot
     |  ii_          which loops can skip snapshots entirely
     |  der_         local partial derivative rules
     |  lin_         build the derivative tree per statement
     v
  analyses
     |  tgen_        forward-mode codegen
     |  agen_        reverse-mode codegen + initstacks_
     |  hvp_         forward-over-reverse codegen
     |  emit_        shared Expr builders used by all three
     v
  generated Expr
     |  cgen_        CUDA / AMDGPU / Metal port
     |  jgen_        JACC port
     v
     |  val_         correctness oracles
     |  io_          write the file
     v
  .jl file
```

Two prefixes do not appear in the diagram. `stade_` is the public API that
wires the rest together. `norm_` is a small normalization pass called
from `stade_adjoint` and `stade_hvp` before analysis begins.

## II.5 The frozen data shapes

STADE uses no `struct` and no `@enum`. Records are plain `NamedTuple`s
with a fixed key set. Changing a key is a breaking change for every stage
that touches it. The shapes are declared in the file header, at lines
69 to 107.

```julia
kernel        :: (sig, body)
kernel_sig    :: (name::Symbol, args::Vector{Symbol},
                  kinds::Dict{Symbol,Symbol},
                  independents::Vector{Symbol}, dependents::Vector{Symbol})
statement     :: (kind=:assign, lhs, rhs)
               | (kind=:for, var, lo, hi, step, body)
               | (kind=:if, cond, then, els)
statement_list:: Vector{NamedTuple}
active_map    :: Dict{Symbol,Bool}
snapshot_site :: (kind=:value|:array|:branch|:tripcount, array::Symbol, at::Int)
snapshot_plan :: Vector{snapshot_site}
lin_node      :: (kind=:leaf|:op, expr, op, args, children, partials, active)
der_rule_pair :: (tangent::Function, adjoint::Function)
cuda_plan     :: (host::Expr, kernels::Vector{Expr})
cgen_kernel   :: (name, args, body, ret)
ii_plan       :: Dict{site_key => :independent|:reduction|:mixed|:recompute}
```

Expressions inside these records are raw Julia `Expr`, `Symbol`, and
`Number`. STADE has no custom AST type. Everything a stage builds is
something `Base.string` can print back as valid Julia.

`independents` and `dependents` are derived by `parse_infer_indep_dep`.
They are every float-kinded argument. A caller never supplies them. The
`parse_override_indep_dep` escape hatch exists for the rare exclusion
case and is not part of the public API.

## II.6 House rules you must follow

These come from `skill-stade-dev`. They are the contract that lets several
people work on separate stages without a merge conflict in the middle of
the analysis code.

1. **One file, top level, no submodules.** The file has one `module STADE`
   wrapper for packaging. It has no other module block. Organize with
   `# ==== prefix_* ====` banners.
2. **No `struct`, no `@enum`, no global `const`.** Lookup tables are built
   fresh inside the function that reads them. `der_table()` builds its
   whole dictionary on every call, on purpose.
3. **Every function name carries its stage prefix.** A cross-prefix name
   collision is a bug. Fix it before you merge.
4. **Only `io_*` touches the filesystem.**
5. **Every stage function is pure**, given its documented inputs, except
   `io_*`. This purity is what lets you build `agen_*` against a
   hand-written fake `snap_plan` output.
6. **A stage consumes documented upstream shapes only.** It does not reach
   into another stage's private helpers.
7. **Comments are one line and sparse.** Put design rationale in comments
   only where the code cannot show it.
8. **No corpus kernel is ever named in `STADE.jl`.**

### The duplicated pairs

Rule 5 forces one surprising pattern. Several analyses exist **twice**,
once under `snap_` and once under `agen_`. `snap_value_needed_sites` and
`agen_value_needed_sites` are one such pair. `snap_ii_plan` and
`agen_ii_plan` are another.

The duplication is deliberate. Two independently written implementations
of the same rule catch each other's mistakes. `stade_site_level_tbr_check`
and `stade_ii_plan_check` assert exact equality between the two copies on
every call.

If the two disagree, you made a one-sided edit. Fix both. Do not relax the
assertion. Edit a duplicated pair identically or not at all.

---

# Part III. Reading generated code

The fastest way to understand STADE is to read what it writes. Every
example below is real output from the corpus.

## III.1 A tangent

Input:

```julia
function affine_loss(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
```

Output of `stade_tangent_file`:

```julia
function affine_loss_d(loss, lossd, u, ud, a, ad, b, bd, v, vd, i_n)
    for i_x = 1:i_n
        vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x2 = 1:i_n
        lossd[1] = lossd[1] + (2 * v[i_x2]) * vd[i_x2]
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    return nothing
end
```

Observations to carry with you:

- Each float argument gained a shadow argument right after it. `i_n` is
  `Int64`, so it gained nothing.
- The loop structure is untouched. Forward mode never reverses anything.
- The shadow line precedes its primal line. A tangent never needs the new
  value of its own left-hand side, so reading pre-statement values is
  always safe.
- There is no stack anywhere. Forward mode needs no memory, which is why
  both flags are no-ops for `stade_tangent`.

## III.2 An adjoint with no stack

```julia
function affine_loss_b(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n)
    for i_x = 1:i_n                       # --- forward sweep ---
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    for i_x2 = i_n:-1:1                   # --- backward sweep ---
        vb[i_x2] = vb[i_x2] + (2 * v[i_x2]) * lossb[1]
    end
    for i_x = i_n:-1:1
        ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
        ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
        bb[i_x] = bb[i_x] + vb[i_x]
        vb[i_x] = 0.0
    end
    return nothing
end
```

Observations:

- The two halves are visible. The forward sweep replays the primal exactly.
  The backward sweep comes after it.
- Statement order reverses. Loop **direction** reverses too, from `1:i_n`
  to `i_n:-1:1`.
- Every adjoint update accumulates with `+`. The variable's own slot is
  cleared once its sensitivity is fully distributed. That is the
  `vb[i_x] = 0.0` line.
- No snapshot was needed here. `v[i_x]` is written once, from a linear
  expression, and never read before its write inside a sequential loop.
  The snapshot analysis proved that and emitted nothing.

## III.3 An adjoint that needs a stack

```julia
function geomrecur(loss, u, c, i_n)
    for i_x = 2:i_n
        u[i_x] = c * u[i_x - 1]        # value-carrying recurrence
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
end
```

Output with `keep_push_pop = true`:

```julia
function initstacks_geomrecur_b()
    u_stack = Vector{Float64}()
    return u_stack
end

function geomrecur_b(loss, lossb, u, ub, c, cb, i_n, u_stack)
    for i_x = 2:i_n
        push!(u_stack, u[i_x])
        u[i_x] = c * u[i_x - 1]
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * lossb[1]
    end
    for i_x = i_n:-1:2
        u[i_x] = pop!(u_stack)
        cb = cb + u[i_x - 1] * ub[i_x]
        ub[i_x - 1] = ub[i_x - 1] + c * ub[i_x]
        ub[i_x] = 0.0
    end
    return cb
end
```

The recurrence forced a snapshot. `u[i_x]` is read at `i_x - 1` by the
next iteration, so the reverse sweep must restore each element before it
differentiates the write.

Note the return value. `c` is a scalar float argument, so its adjoint `cb`
cannot escape through mutation. It escapes through `return`. Array
adjoints mutate in place and need no return.

## III.4 The same adjoint without `push!`

Now with `keep_push_pop = false`:

```julia
function initstacks_geomrecur_b(i_n)
    u_stack = Vector{Float64}(undef, max(0, div(i_n - 2, 1) + 1))
    return u_stack
end

function geomrecur_b(loss, lossb, u, ub, c, cb, i_n, u_stack)
    for i_x = 2:i_n
        u_stack[(i_x - 2) + 1] = u[i_x]
        u[i_x] = c * u[i_x - 1]
    end
    ...
    for i_x = i_n:-1:2
        u[i_x] = u_stack[(i_x - 2) + 1]
        ...
    end
    return cb
end
```

Two things changed. `initstacks_` now takes `i_n` and pre-sizes the
vector. Every `push!` became an indexed write, and every `pop!` became an
indexed read, at a **closed-form position** derived from the loop nest.

This matters for one reason. A shared stack pointer has no meaning once a
loop runs across GPU threads. A running counter is no substitute, because
a kernel launch guarantees no visit order. Only a closed-form,
thread-local index survives parallel execution. `keep_push_pop = false` is
therefore the only GPU-portable mode.

## III.5 A branch snapshot

```julia
function branchsel(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
end
```

```julia
function branchsel_b(loss, lossb, x, xb, y, yb, branch_stack)
    if x > y
        push!(branch_stack, 1)
        loss[1] = x ^ 2 - y
    else
        push!(branch_stack, 0)
        loss[1] = y ^ 2 - x
    end
    __branch = pop!(branch_stack)
    if __branch == 1
        xb = xb + (2x) * lossb[1]
        yb = yb + -(lossb[1])
        lossb[1] = 0.0
    else
        yb = yb + (2y) * lossb[1]
        xb = xb + -(lossb[1])
        lossb[1] = 0.0
    end
    return (xb, yb)
end
```

The backward sweep must take the same branch the forward sweep took. It
cannot re-evaluate `x > y`, because the reverse sweep may already have
disturbed `x` or `y`. So the forward sweep records the branch taken as an
integer flag, and the backward sweep replays the recorded choice. Every
`if` gets one of these. That is the `:branch` snapshot kind.

## III.6 A trip-count snapshot

```julia
cur = n
for i_l = 1:levels
    for i = 1:cur
        ...
    end
    cur = div(cur + 1, 2)     # the bound shrinks each pass
end
```

The inner loop's bound changes on every outer pass. The backward sweep
must run each inner loop with the bound that pass actually used, not with
whatever `cur` holds at the end. So the forward sweep pushes `cur` before
entering, and the backward sweep pops it. That is the `:tripcount`
snapshot kind.

## III.7 Loop fusion under `fuse_ii_loops`

Take a loop whose local scalar is needed for its derivative:

```julia
function fuse_demo(loss, u, i_n)
    for i_x = 1:i_n
        s = u[i_x] * u[i_x]
        loss[1] = loss[1] + s * s
    end
end
```

With `fuse_ii_loops = false`:

```julia
function fuse_demo_b(loss, lossb, u, ub, i_n, s_stack)
    s = 0.0
    sb = 0.0
    for i_x = 1:i_n
        push!(s_stack, s)
        s = u[i_x] * u[i_x]
        loss[1] = loss[1] + s * s
    end
    push!(s_stack, s)
    s = pop!(s_stack)
    for i_x = i_n:-1:1
        sb = sb + s * lossb[1]
        sb = sb + s * lossb[1]
        s = pop!(s_stack)
        ub[i_x] = ub[i_x] + u[i_x] * sb
        ub[i_x] = ub[i_x] + u[i_x] * sb
        sb = 0.0
    end
    return nothing
end
```

With `fuse_ii_loops = true`:

```julia
function fuse_demo_b(loss, lossb, u, ub, i_n)
    s = 0.0
    sb = 0.0
    for i_x = 1:i_n
        s = u[i_x] * u[i_x]
        loss[1] = loss[1] + s * s
        sb = sb + s * lossb[1]
        sb = sb + s * lossb[1]
        ub[i_x] = ub[i_x] + u[i_x] * sb
        ub[i_x] = ub[i_x] + u[i_x] * sb
        sb = 0.0
    end
    return nothing
end
```

The stack disappeared. So did `initstacks_`'s allocation and the
`s_stack` argument. The iterations of this loop do not interact, so the
derivative of iteration `i` can run inside iteration `i` while `s` still
holds the right value. There is nothing to remember and nothing to
reverse.

This is the entire point of `fuse_ii_loops`. It is named after Tapenade's
`$AD II-LOOP` directive, where the user asserts that a loop is
iteration-independent. STADE **proves** it instead. That caps coverage at
what the analysis can prove, and the trade is deliberate.

## III.8 A GPU port

`stade_cuda_file` applied to the adjoint of `affine_loss` gives:

```julia
function cuda_kernel_affine_loss_b_4!(a, ab, bb, i_n, u, ub, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_x = i_n + (__tid - 1) * -1
    ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
    ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
    bb[i_x] = bb[i_x] + vb[i_x]
    vb[i_x] = 0.0
    return nothing
end

function affine_loss_b_cuda(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = ... cuda_kernel_affine_loss_b_1!(a, b, i_n, u, v)
    ...
end
```

Each loop that the analysis proves parallel becomes one device kernel plus
one launch. The loop variable is recovered from the thread index. The
bounds check makes out-of-range threads return early.

One loop in the same file got different treatment:

```julia
function cuda_kernel_affine_loss_b_2!(i_n, loss, v)
    ...
    CUDA.@atomic loss[1] += v[i_x2] ^ 2
    return nothing
end
```

`loss[1]` has an index that does not depend on the thread variable. Every
thread writes the same slot. That is a race. The write is an accumulation,
so an atomic add fixes it. If the write had been a plain replacement,
`cgen_device_assign` would raise an error instead, because no atomic can
fix a replacement race.

Note that the GPU stage is **not** a derivative stage. It is a loop-nest
transform. It ingests already-generated code and knows nothing about
derivatives.
---

# Part IV. The pipeline, stage by stage

Each subsection below states what the stage consumes, what it produces,
and what a new contributor most often gets wrong there.

## IV.1 `io_*` — file access

**Consumes** a path. **Produces** `Expr` values, or writes a file.

This is the only stage allowed to touch the filesystem. Every other stage
works on `Expr` in memory. That separation is what makes the rest of the
pipeline testable without temporary directories.

`io_read_kernel` expects exactly one `function ... end` in the file and
errors otherwise. `io_read_kernel_corpus` reads many and keys them by
name. `io_read_kernel_bundle` reads many and keeps file order, for a
generated file that holds `initstacks_foo_b`, `foo_b`, and a copy of
`foo` together.

`io_write_kernel_file` writes the generated definitions and then appends
the primal. That is why every generated file ends with the original
kernel. The file runs on its own, with no dependency on STADE.

## IV.2 `inl_*` — inlining

**Consumes** `Dict{Symbol,Expr}` of raw kernel definitions.
**Produces** the same dictionary with every call spliced away.

This is the only stage that runs **before** `parse_kernel`, so it works on
raw `Expr` and not on the frozen statement shape.

A call is a bare `callee_name(args...)` statement with symbol-only
arguments. `inl_build_call_graph` finds them, `inl_topo_sort` orders them,
and `inl_inline_body` splices each callee body into its call sites. A call
cycle is a hard error.

The **root kernel** is the one no other kernel calls. That is the kernel
the public API differentiates.

Local variables of a callee are renamed per call site by
`inl_rename_map`, so two call sites cannot collide. After this stage, a
multi-kernel corpus looks exactly like a single flat kernel, and no later
stage needs to know that calls ever existed.

## IV.3 `parse_*` — validation and conversion

**Consumes** one raw `Expr`. **Produces** `kernel = (sig, body)`.

This stage enforces the input contract. Every rejection is a loud error
with a message naming the rule. Keyword arguments, stored comparisons,
indirect indexing, broadcasting, `÷`, `fld`, `Int(...)` conversions, and
any call outside the intrinsic whitelist all fail here.

`parse_desugar_compound` turns `+=`, `-=`, `*=`, `/=`, and `^=` into plain
assignments before anything else sees them. `÷=`, `%=`, `\=`, and `.=` are
not covered and remain errors.

A `for` header of the plain `lo:hi` form parses with `step` set to the
literal `1`. The `step` field always exists, so no consumer decides
whether to look for it.

`parse_infer_indep_dep` derives independents and dependents. They are
every float-kinded argument, and nothing else. Never add a public flag
that asks a caller for them.

## IV.4 `shape_*` — kind inference

**Consumes** `sig` and `body`. **Produces** `kinds::Dict{Symbol,Symbol}`.

Every variable gets one of four kinds: `:scalar_float`, `:scalar_int`,
`:array_float`, `:array_int`.

Kinds come from **usage**, never from a type annotation. A declared
`::Float64` is stripped and discarded. Two questions decide the kind.

- Is the variable ever indexed? If yes it is an array.
- Is there evidence of integer use? A loop variable, a range bound, a
  `div` argument, a subscript, or a comparison all count as evidence.

Anything not provably `Int64` defaults to `Float64`. Integer evidence
propagates through int-preserving operators, and the propagation loop runs
to a fixed point, capped at 8 passes.

This is why `skill-stade` insists on `div(a, b)`. A literal `div` call is
one of the signals shape inference reads. The other spellings never reach
this stage, because `parse_*` refuses them first.

## IV.5 `der_*` — the derivative rule table

**Consumes** an operator symbol and its arguments. **Produces** local
partial derivatives, and from them a tangent or an adjoint contribution.

`der_table()` maps 27 operators to a `(tangent, adjoint)` pair. It is
rebuilt on every call, because house rule 2 forbids a global constant.

The real content is `der_partials`. For each operator it returns one
partial derivative per argument. Everything else is generic:

- `der_tangent_generic` computes `sum_i partials[i] * dargs[i]`.
- `der_adjoint_generic` returns one contribution per argument, for
  `agen_*` to accumulate.

Both drop zero terms and simplify multiplication by one, so the generated
source stays readable. That simplification is not an optimization for
speed. It is what keeps the output auditable.

To add an operator you write `der_partials_<name>`, add a two-line
tangent/adjoint pair, add the table entry, and add the symbol to
`parse_intrinsic_whitelist`. All four steps are required.

## IV.6 `act_*` — activity analysis

**Consumes** `kernel`. **Produces** `active_map::Dict{Symbol,Bool}`.

A variable is **active** when its value depends on an independent input.
An inactive variable has no derivative to carry, so no shadow line is
generated for it. This is the first and cheapest way STADE avoids emitting
dead code.

The analysis is a forward taint propagation from the independents, swept
to a fixed point. It is whole-variable and monotonic. Once a variable is
active it stays active, even past a later overwrite.

One pass is not enough. A loop can carry taint backwards into its own
earlier statements, so `act_analyze` repeats the pass until nothing
changes, bounded by the variable count plus one.

Integer variables are never active. That fact does real work elsewhere: it
is what excludes loop index variables from the fusion analysis.

## IV.7 `norm_*` — dead loop-entry resets

**Consumes** `kernel`. **Produces** a `kernel` with extra reset statements.

This small pass runs from `stade_adjoint` and `stade_hvp` before analysis.

Consider a scalar assigned only inside a nested block, always written
before anything reads it. Its value from the previous iteration reaches
the top of the block but nothing uses it. The adjoint's pre-write snapshot
still reads that value. To `cgen_use_before_def` the loop then looks
loop-carried, and the loop stays on the host.

`norm_insert_dead_entry_resets` writes `var = 0.0` at the top of the
block. That states the kill explicitly. Suppressing the snapshot instead
does not work, because `cgen_use_before_def` refuses the loop regardless.

`norm_first_touch` decides whether the first touch along **every** path is
a write. A write counts only when it is guaranteed, which means both arms
of an `if`, or a loop whose literal bounds prove at least one iteration.
Anything weaker and the entry value can survive to a later read.

A read here is **any occurrence** of the variable, linear or not. The
question is whether the primal sees the entry value, not whether the
adjoint needs it. `y[i] = s + 1.0` and `y[i] = 0.5 * s` both read `s`. A
`while` proves no write, because its body may not run, and its condition
and body are reads. Until v0.4.27 the test used the value-needed rule
(`agen_var_value_needed!`), which skips linear reads. A carried scalar read
through a sum before a branch that assigns it in both arms then got a
reset at the top of the body, and the primal changed: the adjoint was wrong
by a relative 0.8 in the corpus kernel `lin_carried_sum`, and by 0.3 in
`while_carried_cond` for a `while` condition. Both kernels are in the
corpus now, with sabotage cases in `test/validate_linear_scale.jl`.

## IV.8 `snap_*` — snapshot planning

**Consumes** `kernel` and `active_map`. **Produces**
`snapshot_plan::Vector{snapshot_site}`, and separately a per-site decision
dictionary.

This is the memory analysis. It answers one question per write: does the
reverse sweep need the value this write is about to destroy?

The classical name for this question is **TBR analysis**, short for
to-be-recorded. STADE implements the equivalent.

### The whole-variable layer

`snap_var_value_needed!` walks an expression from the root, threading a
`needed` flag. The flag starts false and stays false through nested linear
calls (`snap_linear_call`), because a partial derivative with no variable
in it never needs its operand's value. A call is linear when it is `+` or
`-`, a product with exactly one operand that is not a literal (`0.25 * x`,
`2.0 * 3.0 * x`), or a quotient with a literal denominator (`x / 4.0`). Any
other call is nonlinear, so `needed` flips true and stays true inside it.

The consequence is worth internalizing. A copy, a sum, or a scaling by a
literal needs nothing saved. A product of two variables (`a * x`, `x * x`),
a quotient by a variable, a power, or a `sin` needs its operands saved. A
product with a scalar coefficient that is a variable stays nonlinear even
when that coefficient is not differentiated. Using the activity map there
would save more, and is not done.

Before v0.4.27 only `+` and `-` were linear. A stencil written
`u[i] + 0.25 * (u[i-1] - 2.0 * u[i] + u[i+1])` therefore marked `u` as
needed, and an in-place time loop over it stored one snapshot per pass for
nothing. The rule exists in three copies that must agree: `snap_linear_call`,
`agen_linear_call` and `ii_linear_call`. `test/validate_linear_scale.jl`
checks the three on the same expressions.

A change to the value-needed rule can move a refusal pin. `idm_refuse_scalar_read`
used `z[i] = s * 3.0`; once a linear use stopped saving `s`, the GPU stage no
longer met the scalar gate and refused the loop on the array alone. The gate was
not weakened, the control kernel stopped reaching it. `idm_refuse_scalar_read_nl`
(`s * s`) keeps a witness, and `validate_offload.jl` pins both. When a pin
breaks, read the report of the kernel before editing the pin, and add a twin
that still reaches the gate.

### The site-level layer

`snap_fwd_walk!` refines the whole-variable answer to a per-write answer.
A write `w` to `var` needs its old value if and only if one of two things
holds. Either `w` self-references `var` nonlinearly, or a nonlinear read
of `var` occurred strictly before `w` in forward order.

Loops need a fixed point here. A read near the bottom of a loop body
precedes a write near the top on every iteration except the first. So
`snap_fwd_walk_loop!` repeats the walk until the seen-set stops growing,
then walks once more to record decisions at the fixed point.

Site-level decisions must be a **subset** of the whole-variable set. They
refine it and never diverge from it. `stade_site_level_tbr_check` asserts
that on every call.

### The four site kinds

| Kind | What it saves | Why |
|---|---|---|
| `:value` | A scalar's pre-write value | The reverse sweep needs the old value |
| `:array` | One array element's pre-write value | Same, for an indexed write |
| `:branch` | An integer flag, 1 or 0 | The reverse sweep must take the same branch |
| `:tripcount` | A loop bound | The reverse sweep must run the same iteration count |

Every `if` gets a `:branch` site. There is no analysis to skip it.

### Sites that need nothing

`snap_plan` emits no site for a write with all three of these properties:
no sequential-loop ancestor, exactly one assignment site, and no
read-before-write. This is the common case, and it is why the
`affine_loss` adjoint in Part III has no stack at all.

`agen_exempt_vars` computes a related set. An **exempt** variable is
value-needed but written once, outside any loop, with no earlier read. It
needs no stack.

### Block-boundary pushes

`snap_boundary_snapshot_vars` handles a different shape. A scalar written
inside a nested block, and read outside it, can be covered by one push at
the block boundary instead of one push per iteration. That collapse
matters for GPU coverage: a per-iteration push inside a loop makes
`cgen_contains_stackop` refuse to split the loop.

`snap_boundary_kill_vars` retires exactly the sites the boundary push
replaces, so the push and its matching pop stay balanced.

The emission walks (`agen_block_boundary_vars` with `ii_plan` and `cross`)
drop two more cases, because their stack would have no reader. A scalar
whose writes the `ii_plan` covers needs no boundary stack, and neither does
a loop-local scalar of the cross-iteration rule below: the head of each
reverse iteration rebuilds it. When this empties a stack,
`agen_finalize_stacks` prunes it. That pass runs when `ii_plan` is set or
when `force` is set, and `force` is set for a cross scalar. The layout walks
(`agen_layout_*`, `agen_tier_b_sizing_walk`) call the function without
those two arguments and so size a superset of the stacks. Never pass them
there: a layout smaller than an emission would index outside its stack.

## IV.9 `lin_*` — linearization

**Consumes** `kernel` and `active_map`. **Produces** `lin_plan`.

A `lin_node` mirrors one primal sub-expression. It is either a `:leaf`, a
variable read or a literal, or an `:op`, a rebuilt call carrying its
`partials` and one child node per argument. Every node carries
`active::Bool`.

An array-element read is a **leaf**. The array's own name carries
activity, and indices are always `Int64` and never differentiated.

`lin_stmt` parallels the frozen statement shape. Only `:assign` gains a
`tree`. The `:for` and `:if` fields mirror the primal exactly, which is
what lets the backward sweep walk `lin_plan` and the primal body in
lockstep.

The same `lin_plan` feeds both codegen directions. Tangent generation
sweeps it one way, adjoint generation the other.

## IV.10 `tgen_*` — tangent codegen

**Consumes** `kernel` and `lin_plan`. **Produces** one `Expr`.

This is the simplest codegen stage, and a good first place to contribute.

One sweep, original order, no stacks. Every active statement gets a shadow
line placed before its primal line, built from current pre-statement
values. The shadow is emitted **even when it collapses to `0.0`**, so a
later active read sees the reset rather than a stale value.

`tgen_shadow` appends `d` to a name. Every float argument gains a shadow
argument right after it.

A reassigned scalar argument's shadow escapes through `return`, because a
scalar cannot mutate in place. `emit_return_scalars` builds that return.

Both public flags are documented no-ops here. Tangent code emits no
`push!` at all, so neither the storage strategy nor the fusion has
anything to apply to. The flags exist so a caller can iterate uniformly
over the three modes.

## IV.11 `agen_*` — adjoint codegen

**Consumes** `kernel`, `lin_plan`, `snapshot_plan`, and optionally a
`layout` and an `ii_plan`. **Produces** `(adjoint, initstacks)`.

This is the largest stage, roughly 1900 lines. It builds two sweeps and
one companion allocator.

### The forward sweep

`agen_forward_body` replays the primal and inserts a push before any write
the snapshot analysis marked. It also pushes branch flags and trip counts.

### The backward sweep

`agen_backward_body` walks `lin_plan` in reverse. For each active
assignment it seeds the left-hand side's shadow and calls
`agen_distribute!`, which recursively pushes `target = target +
contribution` at every active leaf.

Loop direction reverses. `agen_negate_step` builds the reversed header.

A **pure accumulation** is special. In `res[i] = res[i] + aux` the
left-hand side's own slot must not be reset, because its incoming
sensitivity is still needed. `agen_is_pure_accumulation` detects the
shape, and `agen_distribute!`'s `skip_expr` argument suppresses the
self-contribution. Everything else resets the shadow to `0.0` after
distributing.

### Branch scalars and branch flags

A scalar assigned in an `if` arm and read elsewhere gets back its value in one of two places. The ordinary pre-write restore runs at the
`if`'s own reverse position. When the scalar is a **block-boundary** scalar (assigned in a nested `if` or loop of the body), the pop at the
head of the reverse body already gives it the value it had at the end of the body. That is the post-`if` value whenever no later statement
of the block writes it again.

Through 0.4.9 a third mechanism, the **branch-scalar hoist**, popped the branch flag at the head of the reverse body and recomputed the
scalar from its right-hand side in the arm that the flag chose. It had three defects, and it was removed in 0.4.10:

1. The recompute ran after the head-of-body restore and overwrote the restored value. When the right-hand side read an array that the
   sweep rewrites (`left = u[i_j - 1]` in an in-place stencil) it read the array in a later state, and the adjoint was wrong in both
   `keep_push_pop` settings (`inplace_stencil_if`; the tangent and HVP checks did not see it).
2. Under `keep_push_pop = true` it popped a flag that was not on top of the shared stack whenever another `if` or a loop holding an `if`
   followed in the block (`if_scalar_then_if`, `if_scalar_then_loop_if`).
3. A later write to the scalar in the same block made the recompute overwrite that write's value (`if_scalar_rewrite`, fixed in 0.4.6 by a guard).

After the head-of-body restore was given priority, no corpus kernel, no mutated corpus (each loop body wrapped one level deeper, in a
one-trip loop or an always-true `if`) and none of the reduced kernels reached the hoist, so the code was deleted and generated output was
byte-identical on all of them. `if_top_scalar`, `if_top_two` and `if_in_outer_loop` keep the shapes that used to reach it in the corpus.

**Flag order under `keep_push_pop = true`.** The forward sweep pushes an `if`'s flag after the body of the arm, and the backward sweep
pops it before replaying the arm. A flag pushed at the start of the arm sat above the flags of any `if` nested in it, so every `if` inside
an `if` popped the inner flag first (through 0.4.9; `nest_if`, `nest_if_scalar`). Indexed mode addresses every flag by its own site key and
is unaffected.

Push and pop sites are matched by a **structural key** and never by a running counter: the two sweeps visit sites in different orders, and a
device launch guarantees no visit order at all.

`validate_if_nesting.jl` holds all of the above and the captured size terms of Tier A (witnesses in the four flag combinations, the refusal
described under Tier B, four sabotage runs).

### Unsafe integer variables

`agen_unsafe_int_vars` finds `Int64` variables that are evolving state
rather than recomputable values. The set is seeded by **self-reference**,
not by mere multiplicity, so a fresh per-loop variable stays hoistable.
Anything depending on the set joins it. Members are never hoisted or
recomputed. Their loop bounds are already restored through `:tripcount`
sites.

### `initstacks_*`

`agen_init_emit` builds the companion allocator. Under
`keep_push_pop = true` it allocates empty growable vectors and takes no
arguments. Under `keep_push_pop = false` it pre-sizes every vector, and
its signature grows to accept the free variables its size expressions
reference.

The signature grows to the **minimal** set, not to the kernel's full
argument list. Scalars derived in the kernel body, such as `n_d = n * d`,
have their defining assignments hoisted into the allocator's body in
dependency order. The invariant is guarded: `initstacks_*` must have no
free variable outside its own parameter list.

### Post-generation cleanup

`agen_finalize_stacks` prunes any stack with zero remaining references in
the generated body. It checks the actual generated code and never predicts
the outcome ahead of time, because a variable's fusion coverage does not
by itself guarantee its stack became unused.

### Integer state in the reverse sweep

The reverse sweep re-runs, at the top of its body, every integer assignment that
is not marked unsafe (scalars and integer-array writes), in forward order. A
scalar assigned once per iteration from values that do not change is then correct
at every iteration. Three mechanisms cover what that recompute cannot rebuild.
Each one lives in a `snap_*`/`agen_*` pair that is edited identically.

- `:ivalue` stacks restore an integer scalar at the block boundary (v0.4.18).
- `snap_itbr_arrays`, `snap_itbr_vars` and `agen_unsafe_int_vars` restore an
  integer array that the kernel reads and then writes again, and mark every
  scalar that depends on it unsafe (v0.4.19).
- `:tripcount` sites push a reassigned loop bound before the loop and pop it
  before the reverse loop.

v0.4.21 closed four ways in which a restore was still missing. All four gave a
silent wrong gradient.

1. **A scalar assigned at more than one site of a loop body**
   (`jj = idx[i]; ...; jj = idx[j]`). The recompute keeps the last write, so a
   read before it sees the wrong value. `agen_seed_unsafe_multi_site!` marks such
   a scalar unsafe when a read sits before its last write. The same name in two
   different loops does not count: each loop body is its own scope.
2. **A loop bound that several loops share, or that a loop rewrites.**
   `agen_multisite_bound_vars` and `agen_collect_reassigned_all` widen the
   `reassigned` test, which used to require an assignment inside a loop. A bound
   assigned between two loops at the top level now gets its own `:tripcount` push.
3. **A bound read from an array element** (`for i = 1:cnt[3]`).
   `parse_hoist_array_bounds` rewrites it to a scalar that is assigned just
   before the loop, so it goes through the same sites as any scalar bound.
4. **An integer copy of an array element that is used only as a number**
   (`c = cnt[i]; s += c * x[i]`). `shape_expr_int_status` now returns `:int` for a
   `:ref` to an integer array. Before, `c` stayed float and kept the value of the
   last iteration in the reverse sweep.

Two smaller changes. `agen_propagate_unsafe!` counts the index expressions of an
array left-hand side, so a write at an index that is itself unsafe is not
replayed. `agen_init_derived_stmts` skips a derived scalar that reads an array
the kernel writes, because `initstacks_*` sees the arguments as they are on
entry, not as they are at that point of the kernel.

**`keep_push_pop = false` refuses what it cannot size.** A loop bound that
`initstacks_*` cannot rebuild from the arguments (an array element, a value
derived from a written array, a scalar assigned at several sites) used to give
`UndefVarError` when the allocator ran. `agen_refuse_unsized` now raises a
generation-time error. The text names the free variables and the two ways out:
pass the bound as an integer argument, or use `keep_push_pop = true`.

### The differential fuzzer (`test/fuzz_int_state.jl`)

`julia fuzz_int_state.jl FIRST LAST [OUTDIR]` builds one random integer-state
kernel per seed and checks it against several oracles: the tangent and the
adjoint against central differences, the exact tangent-vs-adjoint dot-product
identity, and `keep_push_pop = false` against `true`. A refusal whose text
contains "must be restored", "no trip count" or "cannot size the stacks" is a
legal answer. The run counts it and never counts it as a pass. A failure writes
`OUTDIR/fail_SEED.jl` with the kernel, the inputs and the verdict of each
oracle. `STADE_SRC=path/to/STADE.jl` runs the fuzzer against another version of
the source.

- Seeds 1 to 999: general shapes.
- Seeds 1000 to 1999: a restricted mode. Every loop has a bound that
  `initstacks_*` can compute, so `keep_push_pop = false` is accepted and runs.
- Seeds 2000 to 2999: integer accumulators, integer-conditioned statements, and
  strided and reversed loops.
- Seeds 3000 and up: ragged range loops (`fz_range_kernel`). The kernel takes
  `rp` (row pointers) and `n`, and each outer iteration runs an inner loop from
  `rp[v]` to `rp[v + 1] - 1`. The loop has two bounds that are reassigned.

The generator pins the type of every integer array (`fz_int_typed`). A literal
that has only array-valued pairs promotes an integer array to `Float64`, and the
finite-difference harness then perturbs it. Such a failure belongs to the
harness, not to STADE.

Against v0.4.19 the fuzzer found six defects, four of them silent wrong gradients
that the corpus and the earlier validators never reached. On the v0.4.21 tree,
600 general seeds, 250 restricted seeds and 300 seeds from 2001 to 2300 gave no
failure. Run the fuzzer after any change to `snap_itbr_*`, `agen_unsafe_int_vars`,
the `:tripcount` sites or `initstacks_*`.

The ragged mode found a seventh defect in v0.4.21 and earlier. All `:tripcount`
sites of one kernel share a single stack, so the push order and the pop order
must be last-in first-out. The code took the bounds from a `Set{Symbol}`, whose
iteration order is not defined, and used the same order for the push and for
the pop. With two reassigned bounds in one loop, the reverse sweep read the
values swapped. The gradient was wrong without any error. `agen_tripcount_bound_vars`
now returns the bounds sorted by name, and the pop side walks them in reverse.
Seeds 3001 to 3030 give 30 failures on v0.4.21 and none on v0.4.22.

## IV.12 `keep_push_pop = false` in detail

This flag replaces every `push!` and `pop!` with an indexed write and read
into a pre-sized vector, addressed by a closed-form, thread-local
position.

`keep_push_pop = true` must stay **byte-identical** to the output with
`ii_plan = nothing`. Both flags off is an exact rollback. That property is
what makes a two-flag public surface defensible for this much machinery.

### The index formula

A site's index is `base_offset + local_position`.

- `agen_local_position` gives the 1-based position within the site's own
  loop nest. Each frame carries `par`, the layout stage's verdict on whether
  the GPU stage turns that loop into the thread. The first frame with
  `par` set is the fastest-varying digit. The other frames follow it
  row-major in nest order. With no such frame the whole order is
  row-major, innermost fastest (the order up to v0.4.22). With the
  outermost frame as thread, the position is `p_1 + T_1 * (sum over i >= 2
  of p_i * stride_i) + 1`, where `p_i` is the 0-based position in frame `i`
  and `T_1` the trip count of frame 1. Every case is a bijection onto
  `1:multiplicity`, so sizes and offsets do not depend on the order.
  On the GPU the thread is the loop that `cgen_body` makes a kernel, so
  adjacent threads touch adjacent slots.
- `agen_loop_par_map(kernel)` computes `par` for every loop of the primal
  kernel. It runs `cgen_body` on the primal (`cgen_from_kernel`) and reads
  the decision records, which carry `body_id = objectid(loop body)`. A loop
  with a verdict other than `:host` is parallel. The table is built once
  in `agen_layout`, stored as `layout.par`, and read by `agen_frame` in the
  forward and the backward sweep. The backward sweep keys on the primal
  loop (`primal_body[idx]`), because its own statement is a linearization
  node with a different body object. A loop with no entry is an error, so
  a sweep that disagrees with the other fails loudly. Tier B blocks use
  row-major order for their inner frames.
- The verdict is a guess about the adjoint. The adjoint's loops can fail a
  gate that the primal's pass. A wrong guess costs speed only. A throw in
  the walk makes every loop sequential.
- `agen_stride(loop_ctx, i)` is the product of trip counts of every frame
  strictly more nested than frame `i`.
- `agen_local_multiplicity` is the product of all frames' trip counts, so
  one occurrence's total slot demand.
- `base_offset` is the running sum of earlier occurrences' multiplicities
  on the same stack.

`agen_size_trip_count` clamps a trip count at zero when it is used inside
a **size sum**. A loop whose bound fell below its start runs zero times,
but `div(hi - lo, step) + 1` goes negative once `hi <= lo - 2`. Inside a
sum a negative term shrinks the allocation and undersizes the stack. The
raw unclamped form stays in `agen_stride`, where a stride is only ever
evaluated for a site that actually ran.

### Tier A and Tier B

**Tier A** is the good case. Every trip count is a closed-form expression
of kernel arguments and the site's own loop nest. One offset formula and
one size formula per stack.

**Tier B** is the ragged case. A loop whose bound is reassigned inside
**any** ancestor loop has no closed-form trip count. A multigrid solver's
level-halving sequence is the reference instance. The ancestor does not
have to carry a value between its own iterations. Even `w = m0 + i` in an
otherwise independent outer loop varies the inner trip count.

`agen_tier_b_offender` detects the shape. `agen_ragged_block` resolves it
into per-ancestor-iteration tables instead of refusing the kernel:

- a **prefix table**, holding each ancestor iteration's starting offset,
- a **total**, the sum over all iterations,
- a **value table**, holding each iteration's value of a block-local
  scalar that an offset or size formula depends on.

A real example from `mg_vcycle` shows all three:

```julia
prefix_u_stack_1[(i_level - 1) + 1] = __tot_u_stack_1
val_n_1[(i_level - 1) + 1] = n
__tot_u_stack_1 = __tot_u_stack_1 + max(0, div(n - 1, 1) + 1)
```

and the matching site index inside the body:

```julia
__idx_branch_stack_1_0 =
    prefix_branch_stack_1[(i_level - 1) + 1] +
    (((i_k - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
```

Read it as `base_offset` from the prefix table, plus a local position that
looks up `n`'s value for this level in the value table.

Two rules govern Tier B index expressions.

1. They must stay **pure**. An embedded mutation double-evaluates under
   `hvp_double_stmt`.
2. A block-local scalar a formula depends on must go through its own value
   table. It must never be read as a bare in-scope reference, because
   `agen_backward_body` reverses per-statement order and a boundary
   occurrence's forward-last statement can run first in reverse, before
   any recompute of that scalar.

**Where the tables are written.** The sizing loop in `initstacks_*` writes the value
tables and adds to the totals at one point in the ancestor's body: after every
statement that comes before the first **retiring** assignment. A retiring
assignment writes a scalar that a size formula uses, after some loop has already
read that scalar as a bound. It belongs to the next ancestor iteration, so the
tables must be written before it. `agen_tier_b_split_point` finds the split from the **program-order events** of the whole body: each
loop header (a `:consume` of the scalars it reads) and each assignment, at every depth, including inside `if`s and nested loops. A
statement retires when it holds a write to a size-relevant scalar that a header already consumed, earlier in the body or earlier inside the
same statement. Through 0.4.8 the pass read only the headers of top-level statements (fixed in 0.4.9). A body such as `for i_ ... for i_cell = 1:nc ... end
end; nc = nc - 1` then recorded each level from the value after the `nc = nc - 1`. Every block was one cell short and overran its successor.
With the consuming loop inside an `if` the result was silently wrong gradients (0.59 relative error), not an index error. Through 0.4.9 the
same pass still asked only whether a top-level assignment retires, so a retiring write nested in an `if` was missed (`retire_in_if`).

The tables hold one value per ancestor iteration. When a loop header reads a size-relevant scalar **after** an assignment to it in the
stream that starts at the split statement, the second read would need the new value and the table holds the old one. The sizing pass stops
with an error that says so (`keep_push_pop = true` is unaffected). `validate_if_nesting.jl` checks the message and sabotages it.
`retire_nested`, `retire_if` and `retire_in_if` are the corpus witnesses, next to `coarsen_retire`, which has the consumer at the top level.

**Size terms of the stages before a block (`__mult_<k>`).** The slots of a ragged block start after the slots of every earlier stage, and
the offset of each later site holds the size of those stages. A stage size is a product of trip counts, and a trip count can read a scalar
that a later statement reassigns (`nc = nl - 1` in the level loop, after a smoother that ran over `1:nc`). The forward sweep read the live
`nc`. The backward sweep did not recompute it, so it read a stale one: the two sweeps used different indexes, giving a silent wrong gradient
or a `BoundsError` (through 0.4.10 before this fix; `early_stage_offset`). `agen_layout` now walks twice for a kernel with ragged blocks. The
second walk, with a set of the scalars assigned later in program order, replaces such a size term by `__mult_<k>`. The replay of
`initstacks_*` (`agen_tier_b_kernel_skeleton`) sets `__mult_<k>` at the position of the site, and `initstacks_*` returns it with the
`__tot_*` values, so the adjoint and the HVP take it as an argument. A kernel without a ragged block has no replay, and its output is
unchanged. The position of the site in its own loop (`(i_k - 1) * nc + (i_j - 1)`) still reads the live bound: the sweep restores that bound
from the tripcount stack, so both sweeps agree.

A stack that no block can resolve is **tainted**. It falls back to
growable `push!`/`pop!` regardless of the flag, per stack and never per
occurrence.

### Rematerialized restores

A register-tiled convolution rewrites each operand scalar once per tap, and the
adjoint saves the old value before every rewrite. With `keep_push_pop = false`
those saves are per-thread global stores, and they dominate: on the V100 the
forward re-execution kernels of the wide-128 U-Net used 255 registers, spilled,
and took two thirds of the adjoint's time.

`agen_remat_plan` removes a save when the old value is the right-hand side of an
earlier write in the same block and that right-hand side can be evaluated again
at the backward position with the same result. The restore then reads
`v = rhs_d` instead of popping a slot (`agen_backward_assign`, through
`ectx.remat`), and `agen_remat_apply` takes the site out of the site-level push
map before layout, stack sizing and emission, so no slot is allocated. Applies
only under `keep_push_pop = false`; `true` is untouched (byte-identical to
before on every old corpus kernel).

The right-hand side may use:

- literals and `+ - * / div mod ÷ %`;
- a scalar the kernel never assigns, or an enclosing loop variable;
- an int defined by exactly one top-level assignment in this or an enclosing
  block and by no nested statement of that block, not in `agen_unsafe_int_vars`
  (the backward sweep recomputes such an int up front, so its value at the
  restore equals its value at the load). Names reused across inlined calls do
  not matter: the test is per block, not per kernel;
- a load from an array with no write that can run after the load: later in
  program order, or anywhere inside a loop that encloses the load
  (`agen_remat_index!` numbers statements and records enclosing loops).

The walk does not enter `:if`/`:while` bodies (`agen_branch_pushed` mirrors the
forward gate there), and any nested write to the scalar clears the earlier
definition. A float scalar standing in the right-hand side blocks it.

`test/validate_remat.jl` pins the sites chosen for `remat_shapes` (nine loops,
each breaking one condition) and `so_tile`, checks that a tiled U-Net-shaped
kernel loses most of its stack references, and sabotages five guards in fresh
processes; `remat_shapes` must then fail its adjoint and dot-product oracles.
One guard is not distinguishable by any shape tried: the enclosing-loop rule for
array writes. Every array this analysis can load from was also snapshotted at
its overwrite, so dropping the rule changed nothing. It stays as a conservative
bound and is marked as such in the sabotage list.

### Cross-iteration restores (`agen_remat_cross_*`)

The block-local rule needs an earlier write in the same block. The **first**
write of an operand scalar in a loop body has none: its save holds the value
the previous iteration (or the code before the loop) left behind. On the
U-Net tiles those first-tap saves and the cross-iteration `x` saves were what
was left, 264 of 904 sites on tile A.

For a **loop-local** scalar nobody reads that old value. The first write then
saves nothing, and each reverse iteration starts by rebuilding the
*end-of-iteration* value from the pure right-hand side of the body's last
top-level write (`last_def` of the walk, the same purity rule as above). Inside
the iteration the earlier values come from the block-local restores. The
`ectx.remat` map carries two new entry kinds: `Expr(:cross, v)` at the dropped
site (`agen_backward_assign` emits nothing there), and an `Expr(:block)` of
`v = rhs` assignments under `agen_site_key(body, 0)` that `agen_backward_body`
appends right after its up-front int recompute. `agen_remat_apply` leaves the
head keys out of the push map.

A scalar is loop-local when `agen_remat_cross_scan!` finds all of:

1. every write of it is a top-level statement of a loop body (a branch-local
   or root-level write would lean on restores the transformed loops no longer
   perform);
2. no statement nested in a body that writes it at its top level writes it
   again (a nested loop's own first save brings back the enclosing body's
   first value for the read between them);
3. every read of it follows a write in the **same** body: an enclosing body's
   earlier write does not cover it, since the first iteration would read that
   write and later ones the previous iteration's value.

And it is all-or-nothing per scalar: if one loop that needs the first save
cannot rebuild its end value (its right-hand side goes through a float local),
no loop transforms. A loop that kept its save relies on the value the reverse
sweep leaves in the scalar after its neighbours, and a transformed neighbour
would leave it altered. Measured on the U-Net: stack references 1732 to 1212
(tile A), 832 to 512 (C), 502 to 282 (D); atomics and offload counts unchanged.

Sabotage coverage (`validate_remat.jl`): the nested-write rule (`cross_nested`),
the coverage rule (`cross_argread`), both together for `cross_readbefore` and
`cross_entry` (either guard alone catches those two), the all-or-nothing rule
(`cross_mixed`, whose loops sit at the top level because inside an enclosing
loop a block-boundary restore supplies the live value anyway) and the head
restore itself (`remat_cross`). Two guards are not distinguishable: the
placement rule (a write inside an `if` is recovered by the branch machinery's
own recompute) and the after-the-loop half of the coverage rule (served by the
block-boundary restore). Both stay as conservative bounds and print as `info`.

### Array snapshot (`agen_asnap_*`)

`ttgc` smooths a nodal field `mup` inside a Jacobi loop: zero it, scatter cell
contributions into it through `i_cell_to_node`, then read it in a nonlinear
node update. The site-level TBR pushes the old value of `mup` at all four write
sites, because the read at the bottom of the loop body reaches the writes at the
top through the back edge. A push at a scatter reads the old value, so the
scatter is a read-modify-write that the offload gate refuses (`rmw`). The four
cell loops inside the Jacobi loops stayed on the host.

No derivative reads a partial sum of `mup`. The reverse sweep needs only the
value the body left. So, for a loop `L` with body `B` and an array `A`:

1. the forward sweep copies the whole of `A` into slot `n` of `<A>_snap_stack`
   at the head of iteration `n` of `L`;
2. no write site to `A` in `B` pushes or pops;
3. the reverse sweep restores `A` from slot `n` at the tail of the reverse of
   iteration `n`.

At the head of the reverse of iteration `n`, memory holds the value `B` left (the
live value for the last iteration, or what the restore of iteration `n+1`
wrote). That is exactly what every final read needs, for every element, so no
argument about which elements `B` covers is needed. This matters: with an
indirect index no covering-zero-loop argument can be proved.

`agen_asnap_plan(kernel, active_map, sites; ii_plan)` is pure. `agen_asnap_apply`
returns the site map without the write sites, plus one
`agen_site_key(B, 0, A) => true` entry per snapshot, and the plan.
`agen_asnap_site(A)` is the pseudo-site `(kind=:array, array=<A>_snap)`, appended
to the snapshot plan so that the stack name, the signature, the allocation, the
HVP shadow stack (`<A>_snap_stack_d`) and the CSE float names follow without
extra code. `agen_layout_walk_top!` registers the head of each snapshot body and
`agen_layout` multiplies that stack's size by `__len_A`. `agen_forward_body`
emits the copy loop first, `agen_backward_body` emits the restore loop last.
Both are ordinary `for` loops over the linear index `1:__len_A`, so the code is
independent of the rank of `A`, and `hvp_double_stmt` doubles them like any
other loop. `initstacks_*` takes `A`, evaluates `__len_A = length(A)`, and
returns `__len_A` after the stacks; the adjoint and the HVP take it as a trailing
argument, the channel Tier B tables use. A stack whose every site was removed has
no layout entry: `agen_init_emit` leaves it out, and `agen_finalize_stacks` runs
with `force` so the unused name and its HVP shadow leave the signatures.
`cgen_is_length_call` lets the generated-code parser take `__len_A = length(A)`
in `initstacks_*`; the cgen parser still refuses `length` anywhere else.

Guards. Each refuses when it cannot prove its condition.

| Guard | Condition | Kind |
|---|---|---|
| G1 | `A` is an active float kernel argument | scope |
| G3 | at least one write site to `A` in `B` is still pushed | scope |
| G4 | every top-level statement of `B` that touches `A` at or before the last statement writing `A` only reads `A` as the self term of a pure accumulation | correctness |
| G5 | a read in an `if` condition counts as a read | cautious (no oracle can fail it: the reverse sweep replays a branch from its saved flag) |
| G6 | outermost qualifying loop only | cost |
| G7 | neither `L`, nor an ancestor, nor a loop in `B` that writes `A` is classified by `ii_plan` | correctness |
| G9 | no ragged-block ancestor (Tier B) | correctness |
| G10 | `B` writes `A` under an inner loop whose bounds are not both literals | cost |
| G12 | the kernel has no variable named `<A>_snap` or `__len_<A>` | correctness |
| G13 | no write index to `A` follows the loop variable (through scalars, inner loop bounds, or written integer arrays) | cost |
| G14 | a pushed scatter site exists: a read-modify-write of `A` through an index an integer array supplies | cost |

G10, G13 and G14 exist because the copy costs `length(A)` per iteration. Without
them the first version snapshotted `mpnn` (a per-edge slice of a large scratch
array: quadratic memory, and the loops over edges and nodes left the GPU),
`transformer` (stack slots 4900 to 19924) and `advection` (8 to 16). With them
the corpus outputs differ from v0.4.7 only for the scatter kernels and `ttgc`;
`keep_push_pop = true` is byte-identical everywhere.

Measured (`validate_offload.jl` ceilings, host loops left, adjoint and HVP
alike): `ttgc` 18 to 6; the scatter witnesses `asnap_ok`, `_2d`, `_ifwrite`,
`_twoarrays`, `_zerocover`, `_zerotrip`, `_outside`, `_ii` 7 (or 9) to 3,
`asnap_nested` 10 to 6. Stack slots: `ttgc` 605 to 475 (`fuse_ii_loops`: 345 to
215), `asnap_ok` 60 to 24; a snapshot costs more when the body writes few
elements of a large array (`asnap_zerotrip` 9 to 21, `asnap_2d` 117 to 207). The
mechanism buys offload, not memory.

`test/validate_array_snapshot.jl` checks the plan of every corpus kernel, the
effect on `ttgc`, the final state of every array and adjoint against
`keep_push_pop=true` (exactly equal, with a negative control that removes the
restore loop), the ii guards with a synthetic plan, and sabotages each guard in
a fresh process. G5 prints as `info`.

## IV.13 `ii_*` and `fuse_ii_loops` — loop fusion

**Consumes** `kernel`. **Produces**
`ii_plan::Dict{site_key => classification}`.

The flag eliminates snapshots for provably safe loops. Part III.7 shows
the effect. This subsection covers the rules a contributor must not break.

### The four classifications

| Kind | Forward position | Backward position |
|---|---|---|
| `:independent` | Replaced by one fused un-reversed loop | Already emitted |
| `:reduction` | Ordinary loop, `vn_red` excluded from push | `agen_emit_ii_loop(recompute=true)` |
| `:mixed` | Ordinary loop, `vn_local` excluded | Same, and **never split** across positions |
| `:recompute` | Ordinary loop, `vn_local` excluded | Same as `:reduction` |

`vn_local` is the set of value-needed local scalars in the loop body.
`vn_red` is its reduction-accumulator part and `vn_ind` the rest. The
classifier checks each half against its own safety condition, so a loop
holding both a reduction accumulator and an independent chain still
classifies.

`:recompute` exists for a loop containing an **escaping array write**. The
three fusing kinds move that array's adjoint to the forward position,
which is too early. `:recompute` moves nothing and replaces snapshots with
re-execution.

### The walker is post-order

Nested loops are classified before the loop containing them, so the
enclosing loop can see that they are covered. Reversing the order silently
disables nested classification and no oracle notices.

### Load-bearing constraints

Each item below is a real defect that was found and fixed. Do not weaken
any of them.

- `ii_body_has_surviving_snapshot` must key on **site-level** decisions
  and not on whole-variable `value_needed`. A pure accumulation such as
  `res[i] = res[i] + auxres` is value-needed as a variable yet emits no
  push. Treating it as one refuses every scatter-accumulate loop.
- `agen_emit_ii_loop`'s `fwd` half is the real primal at the forward
  position, and only a **recompute** at the backward one. At the backward
  position it must be filtered to scalar assignments. Re-executing array
  writes applies every accumulation twice, and **gradients stay
  bit-identical while this happens**. Only an array-state comparison
  catches it.
- `ii_recomputable`'s scalar rule is **live-in**, not "assigned
  elsewhere". A scalar reassigned in a later unrelated region is fine.
- `ii_array_intact` is the exact dual of
  `ii_body_has_escaping_array_write`. One asks whether a *read* is
  reachable after the loop, the other whether a *write* is. The
  **wraparound pass**, which walks a repeating body's earlier statements
  after its later ones, is mandatory in both.
- Classification must check the candidate loop's **own header**, not only
  its body. A `for i = 1:cur` header with `cur` retired each pass is
  re-run at the backward position against the wrong value.
- `agen_ii_covered_write_check`'s covered-kind tuple must list **every**
  kind. Missing one is the difference between "classified" and "the stack
  actually disappears".
- A local assigned two or more times with a read between the first and
  last assignment (`ii_versions_read`) has several live VERSIONS per
  iteration. Fusing or recomputing such a loop rebuilds the scalar from its
  last assignment only, so every read sees the final version. Both
  classifiers refuse it, except a self-accumulation `v = v + t`, which has
  one version. A register-tiled convolution reloads its weight scalar per
  tap; with the flag on, the gradient of its input came out wrong (relative
  error about 2) until this check. `ii_multiversion` is the witness.

- **Live-out scalars.** `:independent` fuses a loop's adjoint into the FORWARD
  position, so every adjoint it consumes must already exist there. The adjoint
  of an active scalar the loop assigns and later code reads is seeded only by
  that code's backward sweep, which runs after the forward position. Such a
  loop previously produced a wrong adjoint (about 1.5 relative error) in every
  version up to and including 0.4.5: a reduction loop whose accumulator is read
  after it, with a nonlinear local in its body (`a = a + xv * xv * w[ci]`
  followed by `z[t] = a`). `snap_ii_classify` and `agen_ii_classify` now
  compute the live-out set with `ii_escapes_nested` over every active local and
  accumulator. An accumulator that is live out is deferred to the backward
  position (`:mixed`, with the surviving-snapshot and own-array-write gates
  that every deferred kind carries). Any other live-out local keeps only its
  last iteration's value, so every fusing kind would hand its adjoint to the
  FIRST iteration of an un-reversed loop, and the loop is refused.
  `test/validate_liveout.jl` pins the four shapes (`liveout_*`) and sabotages
  both halves. No corpus kernel and none of the U-Net tiles changed class.

### Escape detection tracks program order

A flat "read anywhere else" test is unsound. A variable name can be reused
in an unrelated later loop, and a flat check reads that loop's own fresh
overwrite as an escaping read of the earlier value. A fresh overwrite
kills the dependency. `ii_kill_and_collect!` threads an alive-set in
forward order for exactly this reason, and it treats a kill inside a
possibly-empty loop as **not** guaranteed. The `ii_kill` corpus kernel
guards that rule and fails by about 40 percent if it is relaxed.

### Measured effect

Peak stack slots, with the flag off then on: `ttgc` 504 to 240,
`transformer` 2406 to 1095, `unet` 432 to 324. Every other corpus kernel
is unchanged, because it is dominated by loops with no value-needed local
scalar. Do not justify an extension to this machinery on breadth it will
not deliver.

## IV.14 `hvp_*` — Hessian-vector products

**Consumes** `kernel`, `active_map`, `lin_plan`, `sites`. **Produces**
`(hvp, initstacks)`.

This stage applies forward mode to `agen_*`'s own generated `Expr`. There
is no second `lin_plan`. `hvp_double_body` walks the generated statements
and inserts a tangent line before each one.

The new mechanic is stack shadowing. A `push!` of an active value gets a
paired push on a shadow stack named `<stack>_d`. A `pop!` gets a paired
pop.

`hvp_drop_unused_shadow_stack_allocs` removes a shadow whose primal stack
turned out unused. `agen_finalize_stacks` with `is_hvp = true` keeps the
two functions' signatures consistent, which matters because validation
code shares one `initstacks_*` across both the adjoint and the HVP call.

Naming reaches four levels here. `u` is the primal, `ud` its tangent, `ub`
its adjoint, and `ubd` the tangent of its adjoint.

## IV.15 `cgen_*` — GPU codegen

**Consumes** a plain kernel or a STADE-generated function, plus a backend
record. **Produces** `cuda_plan = (host, kernels)`.

This stage computes no derivatives. It is a loop-nest transform. It proves
which loops are data-parallel and splits each one into a device kernel
plus a launch call.

`cgen_ingest` accepts either input form. `cgen_from_kernel` takes a
validated kernel. `cgen_parse_generated` takes STADE's own output, which
has a trailing `return` and a wider vocabulary.

### Two hard safety gates

**Stack safety.** A loop containing a `push!` or `pop!` anywhere, at any
depth, is never split. LIFO order cannot survive concurrent threads. This
is why only `keep_push_pop = false` output is a GPU target: growable
stacks are host-only by construction.

**Race safety.** `cgen_device_assign` classifies each write.

- The index depends on the thread variable, possibly through a chain of
  scalar bindings, and the chain is injective. The write is safe as is.
- The index is thread-invariant or non-injective, and the write is an
  accumulation. It becomes an atomic add.
- The index is thread-invariant and the write is a plain replacement. This
  is refused with an error. No atomic fixes a replacement race.

- The index is injective in the thread variable for each fixed inner-loop
  value but the SITE collides with itself across threads: `xb[t + ci - 1]`
  under an inner serial loop over `ci` (the adjoint of a convolution).
  `cgen_race_write_arrays` only paired different sites, so a single site kept a
  plain read-modify-write and lost updates on a V100 (relative error about
  1.5 in the adjoint and 2 in the HVP). `cgen_write_self_overlaps` now pairs a
  site with itself. It counts only positive evidence: every dimension must be
  affine in the thread variable with a nonzero integer coefficient and an
  offset that moves with the loop through integer-scaled sums. A symbolic
  stride (`(ci - 1) * hw + t`, a channel-major layout whose threads cover one
  slice each), a gather, or an invariant offset leaves the verdict to
  `cgen_expr_injective_ok` as before. `test/validate_selfrace.jl` pins both
  directions and sabotages the check.

`cgen_expr_injective_ok` refuses `div`, `mod`, `rem`, `fld`, `cld`, `÷`,
and `%` outright. These collapse several thread indices onto one result by
construction. A two-times upsample's `div(oim1, scale)` maps adjacent
output positions to the same source row, and a plain write through it
silently loses a fraction of the accumulation on real hardware.

### Proving a loop parallel

Two gates decide whether a loop is independent. They share their parts and
differ in one place.

The parts:

1. `cgen_loop_scalar_synth` walks the body with `cgen_use_before_def` and
   refuses any read of a locally-assigned scalar before its own definition
   inside this body. This subsumes a self-assignment check, because a
   cross-iteration dependency can exist with no self-reference on the
   left-hand side. A local scalar that fails may still be safe if it provably
   converges to the same literal on every path, **and** that literal matches
   the value known coming into the loop. `known_consts` carries that entry
   from a preceding top-level literal assignment. The proof then injects
   `var = c` as the device kernel's first statement.
2. `cgen_array_ok_strict` accepts one written array. Any structural mismatch
   between a write index and a read index disqualifies it, unless one of
   three relaxations applies.
3. `cgen_array_ok_idiom` accepts one written array if it is **accumulate-only**
   or **consume-only** (next section).

The gates:

- `cgen_reduction_only_loop` is the **strict** gate: part 1 and part 2. The
  `fuse_ii_loops` classifiers (`snap_ii_classify`, `agen_ii_classify`) call it,
  and they must never see part 3. A wider answer there changes the AD output
  and breaks the equality that `stade_ii_plan_check` asserts. `idm_src_consume`
  and the ceiling in `validate_ii_coverage.jl` hold this line.
- `cgen_split_loop_gate` is the **GPU** gate: part 1, then part 2 or part 3
  for each written array, then the overlap and read-modify-write checks below.
  `cgen_body` and `jgen_body` call it. It returns `(synth, reason, array, idiom,
  saveonly)`; `reason` is `:ok`, `:scalar`, `:array`, `:overlap` or `:rmw`.

### The two idioms, and the contract for plain writes

Accumulate-only: every write to the array is `A[e] = A[e] + ...` and the array
occurs nowhere else in the loop. Iterations commute, and the emitter turns
every site into an atomic add. Consume-only: every occurrence is a read
`old = A[e]` closed by a write `A[e] = c` with nothing touching `A` between
them. An HVP twin statement between the two is allowed.

STADE does not prove that two threads plain-write different elements. It trusts
the source. The contract: **iterations of an offloaded loop own disjoint
elements for plain writes.** Four controls make the trust explicit.

1. The idiom classifiers work on source text alone, so they cover generated and
   hand-written code with no metadata.
2. `test/validate_idiom_oracle.jl` classifies the arrays of each independent
   PRIMAL loop (read-only, write-only) and requires the matching REVERSE loop
   to have the matching shadow class. It skips and reports a loop it cannot pair.
3. `test/validate_ownership.jl` runs every generated CUDA plan on the CPU
   through `test/ownership_shim.jl`, which records which thread read, wrote or
   atomically added each element of each launch. It reports `multi_writer`,
   `write_vs_other` and `read_vs_atomic`, independent of thread order, and it
   repeats the run with the threads shuffled. `idm_dup_write` must be flagged.
4. `test/validate_atomic_guarantee.jl` reads the emitted kernels and requires an
   atomic add at every site of an accumulate-only array that has two or more
   different index expressions.

`stade_gpu_report_file(path, backend)` returns one record per loop:
`(owner, loop_var, lo, hi, step, depth, verdict, reason, array, via_idiom)`.
`verdict` is `:kernel`, `:reduction` or `:host`. `reason` also takes
`validate_offload.jl` pins the refusals with it.

### Two refusals the trust needs

The contract above is trust, and two shapes break it provably. Both live in
`cgen_split_loop_gate` and both were found by `validate_ownership.jl` on
unmodified 0.4.4.

- `cgen_overlap_proven_arrays` (`:overlap`). Two plain writes to one array
  whose indices differ by a constant (`v[i]` and `v[i+1]`) land on one element
  from two threads. `cgen_offset_difference` decides this exactly; an index it
  cannot analyse stays trusted, so mpnn's three slices of one buffer still
  offload. Two additive writes are exempt, since they become atomic adds.
  `idm_refuse_cond_read` is the witness.
- `cgen_rmw_hazard_arrays` (`:rmw`). Reached through an index that is itself
  read from an array (a gather), a read-modify-write that no atomic add repairs
  (`m[n] = m[n] * 0.5 + x`), an undo from a stack, and an array that is both
  accumulated and plainly written. `ttgc`'s scatter and its stack undo are
  the witnesses; its host loops went from 6 to 18, deliberately.

CSE hides an accumulation behind a scalar (`t = A[e]; A[e] = t + x`).
`cgen_fold_accumulates` folds that back to `A[e] = A[e] + x` when `t` is used
once, so the loop offloads as an atomic add instead of being refused as a
read-modify-write. The kernel emitters fold with the same function.

`test/validate_plain_write_gate.jl` pins the verdicts, checks that the guards do
not touch mpnn, stencil_loss, transformer and the others, and sabotages each
guard in a fresh process: the ownership check must then flag the kernel the
guard protects.

### Save-only scalars (register-tiled loops)

A tile loop reassigns operand scalars once per tap, reading between the
assignments. The adjoint saves each scalar's old value before every
reassignment, so the first save of an iteration reads the value the scalar held
on ENTRY. `cgen_use_before_def` refuses that, and the whole forward
re-execution of a tiled convolution stayed on the host (reason `:scalar`).

`cgen_saveonly_synth` accepts such a scalar when four things hold, and seeds
it with `0.0` in `synth`:

1. every other read in the loop follows an assignment
   (`cgen_use_before_def(...; save_ok = true)`, where `save_ok` skips a bare
   `X_stack[i] = v`);
2. the loop body ends by pushing `v` after its last write
   (`cgen_ends_with_push`), so the reverse iteration starts from its own slot;
3. no later statement reads `v` before redefining it
   (`ii_escapes_nested(...; save_ok = true)`, called from
   `cgen_liveout_is_zeroed`);
4. every non-save read of `v` anywhere in the program sits inside a loop that
   writes `v` and is covered there (`cgen_saveonly_closed`).

Condition 2 is what separates the tile loop (offloaded) from the tap loop
inside it (never): a tap iteration's entry save carries the previous tap's
value, which the reverse of that previous tap reads. The seeded value fills only
slots nothing reads. An HVP keeps its twin scalar's saves in `<var>_stack_d`,
so `cgen_is_stack_name` accepts both spellings.

Controls: `so_tile` offloads completely in adjoint and HVP; `so_inner`,
`so_hostread` and `so_liveout` each break one condition and must stay on the
host. `validate_plain_write_gate.jl` sabotages conditions 2, 3 and 4 and the
ownership check, which runs the emulated kernels with a fresh scalar per thread
and compares against the CPU adjoint, must then fail (relative error 0.9 to 1.5).

### Snapshot elision gates all three relaxations

`cgen_array_private_to_loop` sees a snapshot save's read of an array as an
ordinary consumer unless `cgen_snapshot_save_dead` proves the element is dead
on arrival: overwritten later in the same statement list with nothing touching
the array in between. Getting this wrong does not produce wrong numbers. It
produces correct numbers computed on the host, which no oracle notices.

The scan cannot be an adjacency test. An adjoint puts the overwrite immediately
after its own save, but `hvp_double_stmt` interleaves every statement's shadow
twin with its primal, so in an HVP the overwrite sits two statements later.
mpnn is the witness: its adjoint offloaded every loop while its HVP left the
loops over graph edges and over graph nodes on the host, launching one kernel
per edge and per node.

One measured caveat, so nobody over-trusts this guard. Replacing
`cgen_snapshot_save_dead` with an unconditional `return true` -- eliding every
snapshot save, however live -- changes **nothing** across the corpus: all 47
kernels in both modes offload exactly as before, `halo_assembly` still refuses,
and every oracle stays green. Dropping only the intervening-touch check is
likewise inert. So elision *precision* is not what keeps an unsafe loop off the
device; `cgen_array_private_to_loop`'s index arithmetic is, and elision only
moves the reader COUNT that pattern 3 keys on. The conservatism in
`cgen_stmt_touches_array` is belt-and-braces against shapes the corpus does not
yet contain, not a load-bearing safety check -- treat a change there as a
correctness question to reason about, not one the test suite will catch for you.

That gives a load-bearing invariant, and `validate_offload.jl` now asserts it:
**an HVP must offload exactly as well as its adjoint**, because it is the
adjoint with every statement doubled -- same loops, same bounds, same snapshot
sites. Measuring only the adjoint hid this defect for as long as that file
existed.

### The three aliasing relaxations

**Array privacy** (`cgen_array_private_to_loop`). The pairwise check
treats every access to an array as one undifferentiated set, so it cannot
tell disjoint per-sub-loop slices assembled into a whole apart from a real
value recurrence. This proof accepts an array whose every access
decomposes into an outer-invariant base plus an inner loop's additive
index, matching one of three patterns: no pure-reader group at all, every
group covering the same sub-region, or cumulative assembly with one reader
and all writers strictly on one side of it.

**Loop-variable slicing** (`cgen_array_sliced_by_loopvar`). Every access
pins one fixed dimension to the candidate loop's own variable, so
iteration `i` touches only slice `[.., i, ..]`. This is the
multi-dimensional counterpart of an index reducing to `base + i`.

**Literal disagreement** (`cgen_indices_cannot_alias`). Two index vectors
that disagree at a position where both are integer literals address
different memory. `cb[1, i_node]` and `cb[2, i_node]` are separate
components of one vector field and can never alias.

### Backends

`cgen_backend_cuda`, `cgen_backend_amdgpu`, and `cgen_backend_metal`
return one `gpu_backend` record each. The record holds a suffix, a launch
macro, thread and block keyword names, a thread-index expression, an
atomic macro, an `allowscalar` macro, a source preamble, a default
precision, and a precision lock with its reason. Adding a fourth
launch-macro vendor means writing one such function and nothing else.

## IV.16 `jgen_*` — JACC codegen

JACC.jl is not a `gpu_backend` value. Its programming model differs: a
plain indexed function plus `JACC.@parallel_for range=N`, with the vendor
chosen later through Preferences.jl. So it gets its own prefix, and it
reuses `cgen_*`'s backend-agnostic front end directly.

Two consequences follow from deferring the vendor choice.

- Precision is not locked, because STADE cannot know which backend will
  run the output.
- Stacks allocate through `JACC.zeros`, the only documented portable
  allocator. There is no `undef`-style JACC allocator, so this zero-fills.
  That is harmless in practice, because every element is overwritten by a
  push before its first read.

JACC splats its launch arguments into the kernel. A kernel with more than
`JGEN_MAX_SPLAT_ARGS` (32, Julia's `MAX_TUPLE_SPLAT`) free variables fails to
compile (`jl_f__apply_iterate`; `ttgc`'s HVP loop takes 42, measured on a
V100: 25 compile, 42 do not). Until v0.4.16 `jgen_body` kept such a loop on the
host. That was correct only by accident: a scalar adjoint promoted to a
one-element device array (`dtbd`) was still updated by host code, which died with
`+(CuArray, Float64)`. Now `jgen_kernel_def` packs the free variables into one
tuple parameter (`__jargs`), unpacks them first, and `jgen_launch_expr` builds
the tuple in its own host statement. A kernel at or below the limit is emitted
exactly as before. `validate_backend_agreement.jl` compares the unpacked count
with the CUDA kernel's argument count.

JACC also has no `allowscalar` escape hatch, so a reduction result cannot
be written back to a device array with a plain host assignment.
`jgen_reduction_writeback_kernel` synthesizes a `range=1` kernel instead.

## IV.17 `val_*` — the oracles

**Consumes** a kernel, its generated code, and a baseline. **Produces**
`(ok, max_rel_err, trials)`.

Four oracles run per kernel.

1. **Tangent.** A direct JVP check against central finite differences of
   the primal.
2. **Adjoint.** The dot-product identity `<y, J*x> == <J'*y, x>`,
   evaluated with `val_finite_diff_check` on a scalar closure.
3. **HVP.** The same JVP oracle applied one layer out, against the adjoint
   instead of the primal.
4. **Exact tangent-versus-adjoint identity.** The same
   `<Yb, J*Xd> == <J'*Yb, Xd>`, computed from the two generated codes
   directly with no finite differences at all.

The first three carry an epsilon and cap out near `1e-8`. The fourth
agrees to about `1e-15` and carries no epsilon. It catches systematic
adjoint errors two orders of magnitude below what finite differences can
see.

The fourth oracle does **not** validate the tangent. A bug shared by both
generated codes cancels in the identity. It complements the other three
and replaces none of them.

### What a GPU parity pass has to mean

`stade_validate_gpu_file` and `stade_validate_jacc_file` take
`mode = :adjoint | :hvp`. The HVP argument layout -- the whole adjoint list,
then one `(tangent, tangent-of-shadow)` pair per float argument -- cannot be
derived from `sig.kinds`, so `val_gpu_call_args` and `validate_corpus_gpu.jl`'s
`gval_build_call` both reimplement it. Two arrays of equal length swapped
between those slots raises nothing anywhere; `validate_gpu_arglayout.jl` pins
the two against the generated signature, position for position, with no GPU.

The parity script compares the device result to a **BigFloat truth**, not
to the Float64 CPU result. The Float64 CPU code is itself off the exact value
(3e-12 on the U-Net HVP, up to 1e-10 on the transformer HVP, measured with
192-bit BigFloat), so a device-vs-CPU difference adds two roundings and used to
pass or fail by draw, on stock v0.4.10 as well. The script runs the same CPU
code in BigFloat on the same inputs. The run gets one verdict. It passes when
the worst device error against that truth, over the trials, is at most 20 times
the worst CPU error over the same trials, or at most an absolute ceiling of 1e-10,
and never below the atomic-count `rtol`. The device does not round like the host
(its libm, fused operations, the order of an atomic sum), so its error is not a
multiple of the host's. Over 64 V100 runs (8 draws of four large kernels, adjoint
and HVP, 192 trials) the per-trial ratio ran from 0.01 to 87. Judging each trial
against its own CPU error failed one transformer HVP draw deterministically
(3.7e-12 against a worst CPU error of 7.7e-13). With the verdict above, 63 of the
64 runs pass on the 20x bound and the other passes on the ceiling. The ceiling is
a single constant, `truth_ceiling`, in `val_gpu_parity_script`. A gap above it
is no longer rounding. If the BigFloat run throws, that trial falls back to the
device-vs-CPU comparison with `rtol`. The stack comparison stays exact. The JSON line carries `truth_trials`
(trials that had a truth), `tr_cpu` and `tr_gpu` (worst errors against it).
`validate_gpu_truth.jl` pins the criterion without a GPU: it makes the device
side a copy of the host side, so the script must pass, and then changes one
element, so the script must fail. It also pins the verdict itself with seven checks: a lucky CPU draw,
a heavy draw below the ceiling, a gap of 1e-8, one catastrophic trial, the sharp edge of
the ceiling, a kernel whose own CPU error is 1e-10, and empty trial data.

The parity script also reports `stack_max_rel_err`, the worst difference between
the host stacks and the device stacks. It is informational and is not part of `ok`.
The script zero-fills the float stacks before each run. Without that, a slot that
nothing writes held allocator garbage, and the value changed between runs (a fused
`ttgc` HVP gave 1.0 in one run and 5e-16 in the next, on 0.4.7 as well). A nonzero
value now means a written slot differs. A nonzero value does not mean a wrong
gradient. `retire_if` reads 1.0 on a V100 with an exact gradient (adjoint and HVP):
62 of 184 slots of `t_stack` differ. The device forward kernel runs each
(`i_l`, `i_r`) pair in its own thread, and `t` starts at 0.0 there, while the CPU
carries `t` over from the earlier pair. `t` is overwritten before any derivative
expression reads it, so the restored value is not read. This is the same class as
`cellscatter` (a slot written from a host scalar that the device did not keep).
The exact gradient on every trial is the evidence, not a proof. Treat such a value
as a finding to read, not as a failure. `validate_gpu_stack_compare.jl` checks the
comparison with no GPU, with a sabotage run.

A parity run also has to have done something. Counting elements *compared* is
not enough: an all-zero integer draw leaves every array at its baseline length
while collapsing every loop bound to zero trips, so the arrays are compared,
they match, and a run that executed no device code reports a pass -- measured
live at 416 elements compared and `max_rel_err = 0.0`. What is counted instead
is elements the CPU reference actually **wrote**, and a run that wrote none is
reported `vacuous` rather than `ok`.

### Baselines

`val_generate_baseline` draws random integer arguments and random float
values, then writes them to a YAML file so a run is reproducible. The
generator gates each candidate draw on the `keep_push_pop = false`
`initstacks_*` call, so a dimensionally incoherent draw is redrawn
automatically. This is generic and needs no per-kernel annotation.

It catches only the loud class of bad draw. A valid-but-wrong shape, such
as `d != h * dk` in an attention kernel, has to be prevented by deriving
the relation **in the kernel body** instead, which makes the bad state
unrepresentable.

---

# Part V. Glossary

Terms are grouped by the part of the system that uses them. Within a
group they run roughly in the order a reader meets them.

## V.1 Automatic differentiation vocabulary

**Automatic differentiation (AD)**
A method that computes exact derivatives of a program by composing the
known derivatives of its elementary operations. It is not symbolic
differentiation and it is not finite differences.

**Source transformation**
An AD implementation strategy that reads source code and writes new source
code. The alternative is operator overloading, which builds the derivative
program at run time inside a custom number type. STADE is a source
transformation tool.

**Primal**
The original, underived kernel and its values. In generated code the
"primal line" is the copy of the original statement, and the "primal
value" is the value the original program held at that point.

**Independent**
An input the derivative is taken with respect to. `parse_infer_indep_dep`
derives this set as every float-kinded argument.

**Dependent**
An output the derivative is taken of. STADE derives the same set, because
a kernel writes its outputs into its float-kinded arguments.

**Active / activity**
A variable is active when its value depends on an independent. An inactive
variable carries no derivative and gets no generated shadow line.
`act_analyze` computes the whole map.

**Taint**
The propagation mechanism activity analysis uses. Activity spreads forward
from the independents through assignments, the way a taint spreads.

**Shadow variable**
The companion variable holding a primal variable's derivative. STADE uses
three suffixes. `ud` is a tangent shadow, `ub` an adjoint shadow, and
`ubd` the tangent of an adjoint shadow, which appears only in HVP code.
The generated names come from `tgen_shadow` and `agen_shadow`.

**Tangent, forward mode, JVP**
Three names for the same thing. Propagate a chosen input direction forward
through the program. One pass yields `J * Xd`. Generated function suffix
`_d`.

**Adjoint, reverse mode, VJP**
Propagate a chosen output direction backward through the program. One pass
yields `J' * Yb`, which for a scalar output is the whole gradient.
Generated function suffix `_b`.

**Seed**
The vector you start a sweep with. `Xd` seeds a tangent sweep and `Yb`
seeds an adjoint sweep. Inside `agen_distribute!` the word also means the
sensitivity arriving at one node of the derivative tree.

**Sweep**
One traversal of the program. An adjoint function contains two: a forward
sweep that replays the primal and saves values, then a backward sweep that
accumulates derivatives in reverse order.

**Local partial derivative**
The derivative of one elementary operation with respect to one of its own
arguments, treating the others as constants. `der_partials` returns the
list for each operator.

**Linear operation**
An operation whose local partials hold no variable, so it never needs its
operands' values saved. STADE treats addition, subtraction, a product with
exactly one non-literal operand, and a quotient with a literal denominator
this way. Everything else is nonlinear for snapshot purposes.

**Accumulation, pure accumulation**
A write of the form `x = x + something`, where the left-hand side appears
only under linear operators. Its derivative with respect to its own old
value is exactly 1, so it needs no snapshot and its shadow must not be
reset. `agen_is_pure_accumulation` detects it.

**Hessian-vector product (HVP)**
`H * v`, the action of the second-derivative matrix on a vector. Generated
function suffix `_hv`.

**Forward-over-reverse**
The way STADE computes an HVP. Apply forward mode to the already-generated
reverse-mode code. This composes because reverse-mode output is
straight-line replay code.

**Oracle**
A check that decides whether generated code is correct. STADE has four,
described in section IV.17.

## V.2 Kernel and intermediate-representation vocabulary

**Kernel**
The frozen record `(sig, body)`. In loose speech it also means the input
Julia function, and in GPU sections it means a device kernel. Context
disambiguates.

**`sig`**
The record `(name, args, kinds, independents, dependents)`.

**`kinds`**
`Dict{Symbol,Symbol}` mapping every variable to one of `:scalar_float`,
`:scalar_int`, `:array_float`, `:array_int`. Inferred from usage by
`shape_infer`, never from a type annotation.

**Kind, shape**
Used interchangeably in the source for a variable's inferred category.
"Shape" never means array dimensions in STADE.

**Statement, statement list**
The frozen statement shape: `:assign`, `:for`, or `:if`. A statement list
is `Vector{NamedTuple}`. `:while` is deliberately unsupported.

**Stage prefix**
The two-to-four-letter prefix every function name carries, such as `snap_`
or `agen_`. It replaces namespacing, because the file has no submodules. A
cross-prefix name collision is a bug.

**Root kernel**
In a multi-kernel file, the kernel no other kernel calls. `inl_*` inlines
everything into it, and it is the one STADE differentiates.

**Inlining**
Splicing a callee's body into its call site. `inl_*` does this on raw
`Expr` before parsing, so every later stage sees one flat function.

**`lin_node`, `lin_plan`**
The derivative tree. A node is a `:leaf`, a variable read or literal, or
an `:op`, a rebuilt call with its partials and children. `lin_plan`
mirrors the statement list with a tree attached to each `:assign`.

**Trip count**
The number of iterations a loop runs, written `div(hi - lo, step) + 1`.
`cgen_trip_count` builds the raw form and `agen_size_trip_count` builds a
zero-clamped form for use inside size sums.

**Zero-trip loop**
A loop whose bound has fallen at or below its start, so it runs zero
times. Six separate STADE analyses have mishandled this shape and produced
silently wrong gradients. `validate_zerotrip.jl` exists to force it.

**Retire, retired**
A loop bound that shrinks across passes until the loop stops running.
Multigrid coarsening is the canonical source of retired bounds.

**Ragged**
A loop whose trip count varies across executions of the same loop, because
its bound is reassigned by an ancestor. The opposite of rectangular.

## V.3 Reverse-mode memory vocabulary

**Snapshot**
A saved copy of a value that the forward sweep is about to destroy and the
backward sweep will need. STADE's term for what other AD literature calls
a tape entry.

**Snapshot site**
The record `(kind, array, at)` naming one place a snapshot happens. The
four kinds are `:value`, `:array`, `:branch`, and `:tripcount`.

**Stack**
The vector a snapshot kind's values live in, named `<var>_stack`,
`branch_stack`, or `tripcount_stack`. Allocated by `initstacks_*`.

**Tape**
The classical AD name for the same storage. STADE's source prefers
"stack". The two mean the same thing here.

**Push, pop**
Writing a snapshot during the forward sweep, and reading it back during
the backward sweep. Under `keep_push_pop = false` both become indexed
array accesses, and the words survive as the names of the operations.

**TBR analysis**
To-be-recorded analysis. The classical name for deciding which values a
reverse sweep needs. STADE's `snap_*` stage is the equivalent, and the
source calls it "the TBR-equivalent check".

**`value_needed`**
The whole-variable layer of TBR. A variable is value-needed when its value
feeds some local partial derivative anywhere in the kernel.

**Site-level TBR**
The refinement of `value_needed` to a per-write decision. It is
unconditional, not flag-gated. Site-level decisions must always be a
subset of the whole-variable set.

**Site key**
`agen_site_key(body, idx[, bv])`, a structural identity for one snapshot
site, built from the body object's identity and the statement index. Push
and pop sites are matched by this key and **never** by a running counter,
because the two sweeps visit sites in different orders.

**Exempt variable**
A value-needed variable that still needs no stack, because it is written
once, outside any loop, with no earlier read. `agen_exempt_vars` computes
the set.

**Block-boundary push**
One push at the entry of a nested block, covering a scalar that block
writes and outer code reads, replacing one push per iteration. It reduces
stack traffic and, more importantly, keeps a loop splittable for the GPU
stage.

**Boundary kill**
The retirement of the individual sites a block-boundary push replaces.
`snap_boundary_kill_vars` and `agen_boundary_kill_vars` must agree exactly
or push and pop counts desynchronize.

**Branch flag**
The integer 1 or 0 a `:branch` snapshot stores, recording which arm of an
`if` the forward sweep took. The backward sweep replays the record instead
of re-evaluating the condition. Under `keep_push_pop = true` it is pushed after the arm's body.

**Unsafe integer variable**
An `Int64` variable that is evolving state rather than a recomputable
value. Seeded by self-reference and closed under dependency. Members are
never hoisted or recomputed.

**Dead loop-entry scalar**
A scalar whose value from the previous iteration reaches the top of a
block but is always overwritten before any read. `norm_*` inserts an
explicit reset for it, so the GPU stage does not mistake the block for
loop-carried.

**`ectx`**
The emission context `NamedTuple` threaded through `agen_forward_body` and
`agen_backward_body`, carrying `keep_push_pop`, `loop_ctx`, `layout`,
`push_pop`, and `ii_plan`. Any new snapshot-exclusion mechanism must go
through `agen_ii_override_ectx`, because `agen_push_pop_source` reads
`ectx.push_pop` and modifying `value_needed` alone is silently ignored.

**`loop_ctx`**
The stack of enclosing loop frames at an emission point, outermost first.
Each frame carries `var`, `lo`, `hi`, and `step`.

**Stack mode, indexed mode**
The two storage strategies. Stack mode is `keep_push_pop = true`, growable
vectors with `push!`/`pop!`. Indexed mode is `keep_push_pop = false`,
pre-sized vectors with closed-form indices. Only indexed mode is a GPU
target.

**Local position**
The 1-based offset of one occurrence within its own loop nest, built by
`agen_local_position`. The outermost frame is the fastest-varying digit.

**Stride**
The product of trip counts of every loop frame more nested than a given
frame. It is that frame's stride among the inner frames.

**Local multiplicity**
The product of all enclosing frames' trip counts, so the total number of
slots one occurrence needs.

**Base offset**
The running sum of earlier occurrences' multiplicities on the same stack.
A site's index is `base_offset + local_position`.

**Tier A**
A stack whose index is computable in closed form from kernel arguments and
the site's own loop nest.

**Tier B**
A stack whose index is not, because some loop's bound is reassigned inside
an ancestor loop. Resolved into prefix, total, and value tables rather
than refused.

**Ragged block**
The unit Tier B resolution works on: one ancestor loop whose iterations
each contribute a different number of slots.

**Prefix table**
`prefix_<stack>_<block>`, holding each ancestor iteration's starting
offset into the stack.

**Total**
`__tot_<stack>_<block>`, the sum of all iterations' contributions, so the
stack's full size.

**Value table**
`val_<scalar>_<block>`, holding each ancestor iteration's value of a
block-local scalar that an offset or size formula reads. Required because
a bare in-scope read is unsafe: reverse-order emission can place a lookup
before the scalar's own recompute.

**Tainted stack**
A stack no block could resolve. It falls back to growable `push!`/`pop!`
regardless of the flag. Taint is per stack and never per occurrence.

**`initstacks_*`**
The generated companion function that allocates every stack and table and
returns them as a tuple. Its body must have no free variable outside its
own parameter list. That invariant is guarded.

## V.4 Loop-fusion vocabulary

**II-loop, iteration-independent loop**
A loop whose iterations do not interact, so each iteration's derivative
can be computed inside that same iteration. Named after Tapenade's
`$AD II-LOOP` directive, where the user asserts the property. STADE proves
it instead.

**`fuse_ii_loops`**
The flag enabling fusion. Off by default. When on, `snap_ii_plan` and
`agen_ii_plan` each build a plan independently and
`stade_ii_plan_check` asserts they match.

**`ii_plan`**
`Dict{site_key => :independent | :reduction | :mixed | :recompute}`. A
loop absent from the plan is unclassified and keeps its snapshots.

**`vn_local`**
The value-needed local scalars in a candidate loop's body. Fusion has no
purpose when this set is empty, which is why most corpus kernels are
unaffected by the flag.

**`vn_red`, `vn_ind`**
The reduction-accumulator and independent parts of `vn_local`. Each half
is checked against its own safety condition, so a mixed loop is not
refused wholesale.

**`:independent`**
The strongest classification. The forward loop is replaced by one fused,
un-reversed loop that does forward and backward work together.

**`:reduction`**
The loop keeps its ordinary forward form with the accumulator excluded
from pushes, and the backward position re-executes the body.

**`:mixed`**
Both halves present. It must never be split across forward and backward
positions.

**`:recompute`**
Nothing moves. Snapshots are replaced by re-execution. This is the only
classification safe for a loop with an escaping array write.

**Escaping array write**
A write to an array inside the loop whose value is read after the loop. It
rules out the three fusing kinds, because they would move that array's
adjoint to the forward position.

**Escape**
More generally, a read of a loop-local value that happens outside the
loop. Escape detection tracks program order, because a fresh overwrite in
an unrelated later region kills the dependency and a flat read-anywhere
test would wrongly flag it.

**Live-in**
A scalar that carries a value into the loop from before it. A live-in
scalar cannot be rebuilt by a recompute, so the ordinary restore machinery
must supply it. `ii_recomputable`'s rule is live-in, not "assigned
elsewhere".

**Wraparound pass**
Walking a repeating body's statements from after the target, then from the
beginning up to the target. It models the loop back edge. It is mandatory
in both `ii_array_intact` and `ii_body_has_escaping_array_write`.

**Post-order walk**
The classification order. Nested loops are classified before their
enclosing loop, so the enclosing loop can see that they are covered.
Reversing this silently disables nested classification.

**Covered**
A write already handled by some classification, so no separate stack entry
is needed. `agen_ii_covered_write_check` decides it, and its covered-kind
tuple must list every kind.

## V.5 GPU vocabulary

**Offload**
Moving a loop from the host to the device. The measure of success is
**loops offloaded**, never device-kernel count, because adjacent
splittable loops merge and a lower kernel count can mean more offloaded.

**Device kernel**
A generated function that runs one loop iteration per GPU thread, named
`cuda_kernel_<owner>_<n>!` or `jacc_kernel_<owner>_<n>!`.

**Launch**
The host-side call that starts a device kernel, built from the backend's
launch macro with computed thread and block counts.

**Thread variable**
The loop variable a split loop's iterations map onto, recovered from the
thread index at the top of the device kernel.

**`thread_dep`**
The set of variables that depend on the thread variable, possibly through
a chain of same-body scalar bindings. It is a real occurs-check, tracked
per variable and removed on a reassignment that breaks the chain.

**Injective, `injective_dep`**
A stricter version of `thread_dep`. A chain stays injective only while
every hop is pure arithmetic on already-injective symbols with no array
read. `div`, `mod`, `rem`, `fld`, `cld`, `÷`, and `%` destroy injectivity
by construction, because they map several thread indices onto one result.

**Thread-invariant write**
A write whose index does not depend on the thread variable at all. Every
thread targets the same slot. An accumulation becomes an atomic add. A
plain replacement is refused outright.

**Atomic**
A hardware-serialized read-modify-write, emitted as `CUDA.@atomic` or the
backend equivalent, used to make a racing accumulation correct.

**`reduction_threshold`**
A GPU keyword (default 32768). A matched reduction loop gets both
lowerings and a run-time test on its trip count: below the threshold the
synthesized atomic kernel runs, at or above it one idiomatic reduction
call. It replaced `keep_all_atomic`, removed in v0.3.0.

**Warp aggregation** (`warp_atomics`)
CUDA only. An array-element atomic add is emitted as a call to a
generated helper (`__stade_wa_add!`). With all 32 lanes active and one
shared index it adds the lane values by a shuffle tree and lane 0 issues
the single atomic; otherwise it issues the plain atomic. The summation
order changes, so the last bits of a result differ.

**Atomic hoisting** (`hoist_atomics`)
All GPU backends (default on; `false` restores the old output exactly). In a
device loop, an atomic add whose index is the same on every iteration is
replaced by an add into a local accumulator, plus one atomic after the loop.
A site qualifies only when its index has no array read, no variable the loop
assigns and is not the loop variable; every use of its array in the loop is such
a site; it is not inside a nested `for`; and the loop has at most 16 sites. A
flag set in the loop guards the flush, so a loop that runs zero times flushes
nothing (an index can be 0 exactly then). The order of the float sum changes.

**Kernel finishing passes** (`inbounds`, `dead_scalars`)
All GPU backends (CUDA, AMDGPU, Metal, JACC), `stade_gpu`, `stade_cuda`, `stade_jacc`, the `_file` and the `validate` forms; both default `true`.
`inbounds = false, dead_scalars = false` gives exactly the v0.4.31 text (hash-identical on the 700 plans of the corpus and the kit).
The passes run last, on the kernels only: first `dead_scalars`, then (CUDA) any variant that is built from the kernel, then `inbounds`.
`cgen_finish_kernels(kernels; inbounds, dead_scalars)` applies them to a list (JACC uses it); `cgen_dead_scalars(kernel, refs_ok)` and
`cgen_inbounds_wrap(kernel)` are the two passes.

`dead_scalars`: a backward liveness walk per kernel. `cgen_dead_walk` keeps the set of scalars read later in the thread. A plain
assignment `v = rhs` whose `v` is not live and whose `rhs` is pure is dropped and adds no reads. A `for` loop is iterated to its least
fixpoint, so a carried `s = s + x` that nothing reads is dead (the dead accumulation in the adjoint of `mlp3`); a branch unions its arms,
and a branch without `else` also keeps what is live after it. A statement of any other form is opaque: it keeps every symbol it mentions live
and kills nothing. A call is pure only if it is in `CGEN_DEAD_PURE_CALLS`; an array read is pure only when `refs_ok` is true, which is
`inbounds`, because with bounds checks on a deleted read would delete the error it can raise.

`inbounds`: wraps the whole body in `@inbounds begin ... end`, the thread guard (`if __tid > n_iter; return nothing; end`) inside the block.
Cost: a wrong index is a silent wrong read or write instead of an exception. Pass `inbounds = false` to debug a kernel.
Measured on a V100: geometric mean 1.18 (primal) and 1.06 (adjoint) over 39 cases; `dead_scalars` alone 1.00. Three cases are slower
(`ufv_scatter_step` 4e6 adjoint 0.83, 1e6 adjoint 0.94; `transformer_block` d512 primal 0.94); the cause is not known.
`test/validate_inbounds.jl` (19 sources, corpus pass, 11 sabotage cases) tests both passes. The text-inspecting validators pass
`inbounds = false, dead_scalars = false`; `validate_ownership` runs with `OWN_FINISH=checked` (default: dead on, `@inbounds` off), `full` or `off`.

**Split reduction** (`split_reduce`, `split_min_inner`, `split_max_out`)
CUDA only (the keywords are accepted, and ignored, by the other backends' `stade_gpu` calls). `split_reduce = L` (default 16; 0 or 1 is off) builds a
second kernel `<name>_s!` beside a plain kernel that has the shape below, and the host launches it instead of the plain kernel when a test holds.
The motivating case is the weight gradient of a dense layer: one thread per output, each running a long inner loop over the input, which uses
a V100 at a few percent. The twin gives each output `L` lanes: a block of `32 x L` threads, 32 outputs on `threadIdx().x` and the lanes on
`threadIdx().y`. `split_reduce = 0` gives exactly the text without it (with `inbounds = false, dead_scalars = false` also exactly v0.4.31: 700 of 700
plans hash-identical; with the finishing passes at their defaults, exactly v0.4.32).

*Shape.* The kernel is one thread per output (`if __tid > n_iter` guard) whose body is a list of items: a scalar assignment; an *accumulate-only loop*
whose body is scalar assignments, at least one an accumulator `s = s + t`, `s = t + s` or `s = s - t`; a top-level store or atomic add; or a *loop with
stores* whose body is scalar assignments and stores or atomic adds (v0.4.39: accumulators are allowed in it, see *A loop that stores and accumulates*). Anything else (a branch in a loop, a carried scalar that is not a sum, a bound
that depends on the output) leaves the kernel without a twin.

*Mechanism.* In an accumulate-only loop each lane takes iterations `lo + (ly-1), lo + (ly-1) + step*L, ...` and sums its own partial; the partials go to
dynamic shared memory (`CuDynamicSharedArray`, `8 * 32 * L * nslots` bytes), a barrier follows, and every lane adds the `L` partials in the same order,
so all lanes hold the same total. A top-level store runs on lane 1 only. A loop with stores is partitioned over the lanes the same way (each lane stores
at its own iterations, atomic adds go through unchanged). A barrier (`sync_threads()`) separates a group of items that read what another group wrote. Threads past
`n_iter` are clamped to `n_iter` so that every load stays in bounds and every thread reaches every barrier; their stores are guarded. The sum is formed
in another order, so the last bits differ.

*Soundness conditions* (each refused case is a unit of `test/validate_split.jl`): an accumulator is read by no other statement of its loop (a store included) and starts from a
non-integer scalar assigned before the loop; a temporary is defined before use and dead after the loop; an array stored in a loop appears nowhere else (v0.4.39: or only in loops that store it with the same index, start and step),
only at one index, affine in the loop variable with a stride that is a kernel argument, which the host tests to be nonzero; an array that gets atomic adds is never read.

*Launch test* (the plain kernel runs otherwise): the number of outputs is in `1:split_max_out` (default 32768), the longest inner trip count is at
least `split_min_inner` (default 256), every stride is nonzero, and the block of `32 x L` threads fits the register limit of the compiled twin
(`CUDA.maxthreads` of a `launch = false` build, read before the launch; `L = 32` does not fit `mlp3`: 72 registers).

*Measured* on a V100 (adjoint, `fuse_ii_loops = true`, medians): `mlp3` primal and adjoint 14 and 10.9 times faster at 3M parameters (0.891 to 0.0635 ms, 3.27 to 0.301 ms), 11.9 and 9.0 at 13M, 5.9 and 4.65 at 79M (4.32 to 0.734 ms, 16.27 to 3.50 ms). Lanes 4, 8, 16 at 79M: adjoint 5.24, 3.70, 3.50 ms. `mpnn_gather`, `unet2` and `transformer_block` are unchanged within 2% with the
defaults. With the twin forced on every launch (`split_min_inner = 0`, `split_max_out = 10^9`) they are slower (transformer d1024 adjoint 427 to 552 ms at 16 lanes,
`mpnn_gather` 4.0 to 5.9 ms at 4 lanes; 16 lanes do not fit `mpnn_gather`), which is why the two limits exist; with `split_max_out` unlimited and the default `split_min_inner` the transformer
is 25% slower, with `split_max_out = 32768` it is not. `test/validate_split.jl` (17 sources, 8 that split and 9 that must not, 2 register-limit checks, 15 sabotage cases) and `OWN_SPLIT=L` for `validate_ownership` (forces
the twin on, `split_min_inner = 0`) test the transformation on the CPU shim, which models the blocks of `32 x L` threads, the barriers and the dynamic shared memory
and reports a conflict between threads no barrier orders.

*Comparison with the CPU in `validate_ownership`.* The twin changes the order of a sum, so a result can differ from the CPU Float64 result by more than the tolerance of the draw while being closer to the exact value (the `unet` HVP at `OWN_SPLIT=16`: 7.7e-12 from the CPU result, tolerance 5.6e-12; the CPU result is 9.5e-12 from the BigFloat value and the twin 1.8e-12). A draw that fails the tolerance is therefore run again against the BigFloat truth (`gval_check_backend(...; truth = true)`, 192 bits), with the rule of the GPU validators: at most 20 times the CPU error, or 1e-10, and never below the tolerance. The row says when it passed this way. A device result perturbed by 1e-7 still fails. `validate_inbounds` generates both of its plans with `split_reduce = 0`: the dead-scalar pass lets one more kernel split, and the pass under test is not the split.

**Contraction** (`contraction`)
CUDA only (accepted, and ignored, by the other backends' `stade_gpu` calls). `contraction = n` (default 256; 0 is off; any other value scales the thresholds of the launch test: `n` replaces 256 in them, see the rule of v0.4.37 below) builds a
second kernel `<name>_t!` beside a plain kernel that computes a matrix product, and the host launches it instead of the plain kernel when a test holds. The plain kernel runs one thread per output element
and reads every operand element from global memory once for every output element it takes part in; the twin computes a tile of 64 x 64 outputs per block of 16 x 16 threads (4 x 4 outputs per thread),
reads a 64 x 16 piece of the left operand and a 16 x 64 piece of the right operand into shared memory (`CuDynamicSharedArray`, 2064 doubles, 16512 bytes) for each step of 16 along the inner index, and
runs the statements before the loop (the start value of the sum) and after it (activation, residual, stores) once per output element, on the same text as the plain kernel.
`contraction = 0` gives exactly the text without it (with `split_reduce = 0` also exactly v0.4.33).

*Shape.* After the thread guard: `idx = 1 + (__tid - 1)`; scalar statements, among them the decode of the flat index into a column `o = mod(idx - 1, N) + 1` and a row `t = div(idx - 1, N) + 1` (also through a
variable that holds `idx - 1`, assigned once); the start value of the sum, a scalar assigned once before the loop; ONE flat loop over a step of +1 whose body is the one accumulation `acc = acc + A[ia] * B[ib]`
(`+`, `-`, either order of the factors; the factors are plain array references); then scalar statements and plain stores. `ia` reads only kernel parameters, the row and the loop variable; `ib` only parameters,
the column and the loop variable; the loop bounds are parameters; neither operand array is stored in the kernel; the loop variable, the column and the row are not read or assigned elsewhere.

*Mechanism.* The inner sum keeps its order (index ascending) and is added to the start value at the end, so the result differs from the plain kernel in the last bits only. The tile reads are guarded by the sizes
and filled with zeros outside them. Threads outside the matrix take part in every `sync_threads()` and store nothing.

*Launch test* (the plain kernel runs otherwise; the rule is that of v0.4.37, see *Tile count rule* below): `M`, `N` and `K` are at least `min(contraction, 16)`, `N > 0`, the number of tiles `cld(M, 64) * cld(N, 64)` is at least `cld(base * contraction, 256)` (`base` 8, 32 or 64 by layout), the loop covers whole rows (`mod(n_iter, N) == 0`: a loop of 7 elements over rows of 3 has a partial last row that the tiles would not compute), and the block of 256 threads fits the register limit of the compiled twin (`CUDA.maxthreads` of a `launch = false` build, at least 256).

*Measured* on a V100 (`transformer_block`, forward product, medians): d1024 56.3 to 20.7 ms (2.72 times), d512 7.63 to 3.13 ms (2.44), d256 1.23 to 0.80 ms (1.54); below that the twin is slower (d128 0.73 times,
d64 0.56, d32 0.78, `small` 0.86, `tiny` 0.79), so the default is 256. The adjoint of the same kernel is unchanged (ratio 1.00 to 1.02); `unet2` has one kernel that matches (the output convolution), and with the twin forced (`contraction = 1`) its primal changes by at most 1% at the three bench sizes (0.99 to 1.00 times; `small` 0.93 and `tiny` 0.88); at the default it never launches it there (the channel count is below 256). `test/validate_contraction.jl`
(8 sources that must tile, 15 that must not, direct recognizer cases for guards that STADE's own gates make unreachable from source, threshold, zero-length and register-limit checks, and sabotage cases, each of which one case must catch)
and `OWN_CONTRACT=n` for `validate_ownership` (forces the twin on every launch that has one, with `n` as the smallest side) test the transformation on the CPU shim, which models 2D grids and blocks.

**Adjoint of a product** (`contraction`, from v0.4.35)
The same keyword; the launch test of a group is described under *Launch*. The adjoint kernel of `out[t, o] = b[o] + sum_c x[t, c] * w[c, o]` is one thread per output element `(t, o)` that holds the seed of
the element in a scalar `g` and scatters it, in a loop over `c`, into the two input adjoints with atomic adds (`xb[ix] += w[iw] * g`, `wb[ix2] += x[iw2] * g`). Each of the two is a matrix
product in the other direction (`xb[t, c] += sum_o g[t, o] * w[c, o]`). The plain kernel, the twin and a launch test are emitted as follows.

*Forward kernels inside an adjoint plan.* The forward sweep of an adjoint plan holds the forward kernels again (`b_3!`, `b_7!`, ...), and the common-subexpression pass has replaced `idx - 1` in them by a variable (`__icse_2 = idx - 1`). From v0.4.35 the forward recognizer accepts a decode through such a variable (assigned once; a variable that is changed between the two decodes is refused), so these kernels get the forward twin as well. In the adjoint of `transformer_block` that is 4 forward products (`b_3`, `b_7`, `b_10`, `b_11`) next to 4 adjoint products (`b_12`, `b_13`, `b_16`, `b_20`: 9 tiled kernels, one of them a forward stage).

*Kernels.* (1) `<name>_a!`: the plain kernel with the loop of atomic adds replaced by a store of `g` into a scratch array `__ct_g` at the thread's own flat index (the zeroing of the seed, the bias
adjoint and every other statement stay). (2) `<name>_t1!`, `<name>_t2!`, ...: one tiled kernel per atomic statement, the tile of the forward twin (64 x 64 outputs per block of 16 x 16 threads, 16512 bytes of shared
memory), which reads `__ct_g` and the operand array, forms the sum and adds it into its target with ONE atomic add per output element, so no proof is needed that the targets of different elements are distinct.
(3) When the plain kernel recomputes the forward product before the loop of adds (the value that its epilogue differentiates, for example a `tanh`), a forward stage `<name>_f_t!` computes that sum with a
tiled kernel, stores it in a second scratch array `__cf_s`, and the first kernel reads it in place of the loop. The sum over the other index is formed in tile order, so the last bits differ from the plain kernel.

*Shape.* After the thread guard: `idx = 1 + (__tid - 1)` (or `idx = X + (__tid - 1) * -1` for a reversed loop with `X` the thread count); statements that are scalar assignments or stores, among them
the decode of the flat index into a column `o = mod(idx - 1, N) + 1` and a row `t = div(idx - 1, N) + 1` (also through a variable that holds `idx - 1`, assigned once, and repeated once for a recomputed
forward value if every repeat names the same variable and the same size `N`, which is a kernel parameter); ONE flat loop whose body is atomic adds `arr[ix] += F[iw] * g` and scalar temporaries (common subexpressions of
operand reads, inlined in order into the later statements, and read or assigned nowhere outside the loop); `g` is a thread-local scalar assigned before the loop. `ix` may read parameters, `c` and one of `t`, `o`; `iw`
parameters, `c` and the other. The operand arrays `F` are never stored and the target arrays never read in the kernel. A kernel that does not fit any of this keeps the plain kernel.

*Launch.* The twin branch of the host allocates the scratch arrays with `CUDA.zeros(Float64, count)` on every call (the device memory pool serves them), runs the kernels in the order above, and the plain kernel
runs otherwise. The branch is taken when every tiled kernel of the group can run (`M`, `N`, `K` at least `min(contraction, 16)`, whole rows, the block of 256 threads fits `CUDA.maxthreads` of a `launch = false` build with the scratch array replaced by an array of the same type and rank), at least one of them has the tile count of its layout and every one has at least a quarter of it (v0.4.37, see *Tile count rule* below); otherwise the whole group runs on the plain kernel.

*Not covered.* (v0.4.40: the backward loops of the HVP are tiled, see *Tiled kernels for the HVP*; its forward loops are not.) `mpnn_gather` has two loops of atomic adds in its adjoint kernel and only the first is tiled (the second keeps the plain atomic loop).

*Measured* on a V100 (`transformer_block`, adjoint, medians, `contraction = 0` against the default): d1024 426.8 to 103.9 ms (4.11 times), d512 36.3 to 15.8 ms (2.30), d256 5.67 to 3.86 ms (1.47); the primal
is that of v0.4.34. The plan's target (adjoint at most 100 ms at d1024) is missed by 4%. The adjoint is now 5.95 times the JAX time (17.46 ms; 24.4 times before). `test/validate_adjoint_contraction.jl` (sources that must tile and sources that must not, direct recognizer cases, threshold and register-limit checks, sabotage cases) and `OWN_CONTRACT=n` (which also forces the adjoint twins on the CPU shim) test the transformation.

*Batched product* (v0.4.36). A product over heads (the attention scores) is one flat loop over `nb * m * n` outputs whose index is decoded into three variables: the column `o = mod(X, N) + 1`, the row `t = mod(div(X, N), M) + 1` and the batch `h = div(X, N * M) + 1` (`X` is `idx - 1`, or a variable that holds it; every size is an expression of kernel parameters; the divisor of the batch may be written `N * M` or `M * N`; a repeated decode names the same variable and the same size). The operands read the batch through `h`. The twin has the tile of the forward twin and a third grid dimension, `blockIdx().z = h - 1`, which sets `h`; the decode, the start value and the epilogue run once per output on the same text as the plain kernel. Launch test (the plain kernel runs otherwise): the number of batches `Bh` is in `1:65535` (the limit of the grid dimension), the rows reach `contraction` and the columns and the inner length are at least `min(contraction, 8)` (a head is 16 to 128 wide, so a threshold of 256 on every side would never fire), `N > 0`, the loop covers whole batches (`mod(n_iter, N * M) == 0`), and the block of 256 threads fits the register limit of the compiled twin. The adjoint of such a product has the form of v0.4.35 with batched tiled kernels. The recognizer refuses a batch size that is not rows times columns, a row decode with another size than the column's, a batch that is changed after its decode, a loop variable with the batch's name, a partial last batch, and a product of three factors.

*Measured* (v0.4.36, V100, `transformer_block`, medians; the attention kernels are the batched products): primal d1024 20.7 to 10.0 ms (2.07 times), adjoint 103.9 to 52.6 ms (1.97; the plan's target of 100 ms is met); d512 primal 3.38 to 1.95 ms, adjoint 15.8 to 9.6 ms; d2048 primal 57.8 to 35.2 ms, adjoint 264 to 151 ms; gap to JAX at d1024 adjoint 6.0 to 3.0, primal 3.4 to 1.6. The product whose head is a part of the column index (the weighted sum of the value rows) is not recognized by this twin; in the profile it took 31% of the adjoint and 29% of the primal.

*Head in the column* (v0.4.37). A product whose head is a part of the column index (the weighted sum of the value rows) has a two-variable decode: a column `c = mod(X, D) + 1`, a row `t = div(X, D) + 1` and a head `h = div(c - 1, dh) + 1`, which is assigned once, is not the loop variable and has no kernel parameter of its name, and which an operand index reads. The twin batches its tiles over the heads: `blockIdx().z` is `h - 1`, the grid has `D / dh` heads, one tile covers `dh` columns starting at `(h - 1) * dh + 1`, and the stores use the flat index `row * D + (h - 1) * dh + col`. The adjoint of such a product (the gradient with respect to the weights over the rows, and the one with respect to the values over the head) follows v0.4.35 with the head read from a base index. Launch test (the plain kernel runs otherwise): `dh` and the inner length are at least `min(contraction, 8)`, the rows or `dh` reach `contraction`, `dh > 0`, the number of heads `D / dh` is in `1:65535`, the loop covers whole rows (`mod(n_iter, D) == 0`), `D` is a multiple of `dh`, and the block fits the register limit.

*Tile count rule* (v0.4.37; replaces the rule of v0.4.34 for the twin that is not batched). The launch test of the forward twin and of each tiled kernel of an adjoint group is: `M`, `N` and `K` are at least `min(contraction, 16)`, `N > 0`, the loop covers whole rows, the block fits the register limit, and the number of tiles `cld(M, 64) * cld(N, 64)` is at least `cld(base * contraction, 256)`. The layout comes from the operand indices and the inner variable (a read is "unit" if its index moves by 1 when the inner variable does: the variable is a bare summand of a chain of `+` and `-`, as in `(t - 1) * k + c`, and appears nowhere else in the index): both operands unit (`x * w'`): `base = 8`; only the left one unit (`x * w`): 32; the left one not unit (`x' * g`, the weight gradient, also with a strided right operand): 64. An adjoint group runs when every tiled kernel can run (the sizes and whole rows), at least one of them has the tile count of its layout (`want`), and every one has at least a quarter of that count (`okt`); a kernel that does not want the tiles and has less than a quarter of them means that the plain kernel runs for the group. The batched twin and the head-in-column twin keep their own tests (above). Why: the twin is slower than the plain kernel when there are few tiles, because a tile that sums over many rows runs on one block (the weight gradient of a graph layer with 16 columns and 1e5 rows is one tile with an inner length of 1e5).

*Measured* (v0.4.37, V100, medians): the probe (forced twin against the plain kernel; 50 shapes for each layout) gives, at or above the counts, 1.06 to 15.5 times with medians of 4.08 (both unit, 42 shapes), 2.39 (left unit, 19 shapes) and 1.59 (weight gradient, 13 shapes); below the counts 0.94 to 2.46, 0.87 to 1.48 and 0.41 to 1.08. `transformer_block` primal 10.0 to 7.75 ms and adjoint 52.6 to 36.1 ms at d1024 (gap to JAX 1.63 to 1.26 and 3.0 to 2.07), d512 1.95 to 1.78 and 9.59 to 7.38 ms, d2048 35.2 to 26.5 and 151.4 to 113.8 ms. The first version of the group rule (every kernel can run and at least one has the tile count of its layout) launched the `mpnn_gather` weight-gradient twin with one tile and made its adjoint 7.4 times slower (4.00 to 29.6 ms at 1e5 nodes); the quarter rule declines it and the adjoint is back at 4.00 ms.

*Reuse of a stored value* (v0.4.38; CUDA; keyword `reuse_stored`, default 1; 0 is off). Function `cgen_reuse_stored(host, kernels, backend; inbounds)`, called in `stade_gpu` after `cgen_emit` and `cgen_dead_scalars` and before the contraction and split passes, when `dead_scalars` is true. It works on launch statements of the host body (`cgen_reuse_launch_info`: a plain launch group of one kernel family, not a coarse `_c!` variant) and on a reduction block of a backward kernel (`cgen_reuse_blocks`: `v = init` followed by a loop of scalar statements). For each block it walks back over the host statements before the launch (scalar assignments and launches only; a host scalar that the block, its scalars, the thread index or the guard read must not be assigned) and stops at the first forward kernel that matches (`cgen_reuse_match`): same thread index and guard, the same block text and scalar definitions after `cgen_reuse_subst` replaces the kernel parameters by the host arguments, one top-level store `h[idx] = F` with `idx` a plain variable and `F` reading only the sum and pure, no assignment of the sum after the block, no other write to `h` in the forward kernel, and no write by the forward kernel to an array the block reads. A kernel launched between the two must not write `h` or an array the block reads. `cgen_reuse_rewrite` replaces `F` in every kernel of the backward group (plain and `_p!`) by `h[idx]`, requires that every use of `F` is after the block and before a later assignment to the sum, that the group does not write `h`, runs the dead-scalar pass again, drops loops left empty, and refuses when the block is still there. Limits: the HVP backward kernel also needs the tangent of the sum, which no forward kernel stores, so the sum stays; the forward kernel must be a plain launch in the same host body; a coarse variant is never described. `test/validate_reuse_stored.jl` (29 unit checks on the CPU shim with `reuse_stored` 1 against 0 and against the source function, 3 direct recognizer checks, 17 sabotage cases) covers the guards. One guard, that the forward kernel has exactly one store reading the sum, has no sabotage case: without it the case "a second store that reads the sum" is still refused by another guard that I did not isolate.

*Measured* (v0.4.38, V100): `mlp3` adjoint at 79M, profile, 3.49 to 3.01 ms; kernels b_4 and b_5 1.0 to 0.76 ms; primal unchanged. Hand edits that came first: reading the stored value in layers 2 and 1 gave 3.01 ms; a write-only weight gradient (not implemented, it changes the result when the gradient starts non-zero) is bounded by about 2.14 ms.

*A loop that stores and accumulates* (v0.4.39; the split reduction, same keywords). `cgen_split_kernel` accepts a loop with stores (`:A`) whose body also holds accumulate statements. In the loop with stores, `accv` is the set of scalars every assignment of which in the body is an accumulate statement (`cgen_split_accum`: `v = v + e`, `v = e + v`, `v = v - e`, `v` not in `e`); several such statements for one scalar are allowed there (a reduction loop still needs exactly one). Guards, in order: (1) no statement of the body other than the accumulate statements of `v` reads `v`, tested separately for scalar statements and for stores; (2) the start value is the last scalar assignment before the loop and is not an integer literal (`start_ok`, shared with reduction loops); (3) the other scalars of the loop follow the old rules (not self-reading, defined before use, dead after the loop), and the reads of the accumulate statements and stores are tested with the accumulators exempt because guard (1) has already refused any read of one. Emission is shared with reduction loops: `cgen_split_partials(lp)` returns the accumulators (a scalar whose statements are all accumulate statements and which no other statement reads) and the body with `v = v + e` turned into `__p_v = __p_v + e`; the partials start at `zero(v)` before the loop (for a loop with stores, before the `if __act` that guards its stores), each lane writes its partial to `__shr`, a barrier follows outside the guard, and every lane adds the `L` partials in order (`finish_partials`). `nslots` counts one slot per accumulator. An array stored by several loops (`plain_sites` holds one entry per loop): the index must be equal after the loop variable is renamed (`cgen_reuse_subst`), and the loops need the same start expression and step, so that lane `ly` stores the same elements in each loop and no barrier is needed between them; the array still appears nowhere outside those loops and is read in them only at its own index. `test/validate_split.jl` has 4 sources that must split (one accumulator and a store; the layer-norm backward with two accumulate statements and the same array stored in two loops; a subtracted accumulator with stores to two arrays; one array stored in two loops) and 10 refusals, each run with 4 and 16 lanes through the CPU shim with the number of elements changed counted, and 21 sabotage cases; `test/validate_split_real.jl` runs the real `transformer_block` adjoint and HVP (generated with `fuse_ii_loops = true`, `OWN_FUSE=1` of `validate_ownership.jl`) with the twins forced, and requires the twins of `b_15`, `b_22`, `hv_15`, `hv_22` to run. Two guards have no sabotage case: the one of v0.4.33 that an array stored in loops appears nowhere else (every shape tried is refused by another gate), and the test that a scalar assigned by an accumulate statement and by another statement is not an accumulator (the rule that a loop scalar must not read itself, or be read after the loop, refuses every such scalar first). Limits: the reduction loop rule is unchanged (one accumulate statement per scalar); the launch test needs `CUDA.maxthreads` of the compiled twin to be at least `32 * L`, and it is 256 for the HVP layer-norm twins on the V100, so the HVP runs the plain kernel with the default 16 lanes.

*Measured* (v0.4.39, V100): `transformer_block` d1024_T1024 adjoint, profile, 39.36 to 36.31 ms; `b_15` 1.80 to 0.247 ms and `b_22` 1.82 to 0.251 ms; kernels not named here are the same text (the kernel `b_18_s` reads 2.50, 2.52 and 2.60 ms in three profiles of the same text).

*Tiled kernels for the HVP* (v0.4.40; the adjoint contraction, same keyword `contraction`). `cgen_adjoint_contraction` parses the value of each atomic add as one product or as the sum of two products (`product(v)`: a thread-local scalar times a plain array element, either order). The scalars of all statements (`gs_list`) are at most two; each is assigned before the loop and is neither the loop variable nor a parameter; the first is stored in `__ct_g`, the second in `__ct_g2` (`scratch_of`). The kernel that replaces the loop (`_a`) stores every seed at the thread's own flat index and takes both scratch arrays; the host allocates both (`A.a.scratch`) and each tiled kernel takes the scratch arrays of its own terms (`d.scratch`). A statement with two terms gives a descriptor with `terms` (a list of `(A, B)` pairs); `cgen_contraction_emit` loads one tile of each `A_i` and `B_i` into its own part of the dynamic shared memory (`nt * ((TM + 1) * TK + TM * TK)` values, `shmem = 8 * nt * TS`), and the accumulators add `sum_i av_i * bv_i`. The orientation test is made on every term (`all(tm -> ..., s.terms)`) and so is the test that an operand is never stored in the kernel and is no target (`for tm in s.terms`). With one term and one seed the generated text is that of v0.4.39. The register test of the launch (`CUDA.maxthreads >= 256` on a probe build) is made for each tiled kernel as before.

*Forward loops of the HVP.* (v0.4.41: tiled, see *Tiled kernels for the forward loops of the HVP*.)

*Measured* (v0.4.40, V100): `transformer_block` HVP, median of 30 runs: d512_T512 79.2 to 32.7 ms, d1024_T1024 0.975 to 0.359 s, d2048_T1024 4.63 to 1.54 s. A hand-written tiled kernel for `hv_12` alone took 23.9 ms against 200.9 ms for the plain kernel (the generated twins for `hv_12` take 6.0, 2.5, 6.3 and 2.7 ms, 17.5 ms in all).

*Tiled kernels for the forward loops of the HVP* (v0.4.41; the forward twin of v0.4.34 and the forward stage of the adjoint twin, same keyword `contraction`). `cgen_contraction_accums` reads the statements of the loop: a scalar assignment that no statement outside the loop reads and that is no parameter and no accumulator is a temporary and is inlined, in order, into the later statements; every other statement must be an accumulation `s = s + P` (or `s - P`) with `P` one product of two array elements or the sum of two such products (`cgen_contraction_factors`). Up to two accumulators with distinct names. The descriptor of `cgen_contraction_kernel` has `accs` (for each accumulator, a list of `(ia, ib)` slot indices of its terms), `alist` and `blist` (the distinct A and B operands: one tile each, so equal operands share a tile) and `nt = max(nA, nB)` slots, at most 2, which is `shmem = 8 * nt * TS` = 33 KB. `cgen_contraction_emit` keeps one register accumulator per accumulator for each of the 4 x 4 outputs of a thread (`acc_ij(a, i, j)`), adds `sum over its terms av_ia * bv_ib` per k, and runs the epilogue (the statements after the loop, with the sums) once per output element. A loop of one accumulator and one product keeps the descriptor and the text of v0.4.40 (`single`). The forward stage of an adjoint group (`_f_t`, in `cgen_adjoint_contraction`) accepts up to two stored scalars: `fnames` lists them, `fstores` writes them to the scratch arrays `__cf_s` and `__cf_s2` (`fstage.scratch`), and the tiled backward kernels read them back (`repl`); `cgen_contraction_apply` allocates them (`fsc`), passes them as arguments and includes them in the register probe.

*Launch of the two-sum twin.* A descriptor with `accs` has one more condition, `N >= min(contraction, 64)`: the launch needs a full tile of columns. With 16 columns (`mpnn_gather`, the projection of the nodes) three quarters of the products of a tile are wasted and the twin ran 21 to 27% slower than the plain kernel on the V100 (primal), 1.8% for the adjoint. A kernel of one sum keeps `min(contraction, 16)`. The guard is tested in `validate_contraction.jl` (40 columns run the twin at contraction 32 and 24 do not; one sum with 24 columns runs it) and by two sabotage cases.

*Measured* (v0.4.41, V100): `transformer_block` HVP, median of 30 runs: d512_T512 22.3 ms, d1024_T1024 0.125 s, d2048_T1024 0.410 s (0.359 s and 1.54 s in v0.4.40 at the last two). The largest kernels at d1024_T1024 are `hv_11_t` 11.7 ms, `hv_10_t` 9.8 ms, `hv_13_f_t` 9.6 ms and `hv_3_t` 7.4 ms. Parity on the V100 (twins forced at small sizes, each run counts the launches of its twins and the changed elements): `transformer_block` HVP at three sizes, `mpnn_gather` HVP and adjoint at two sizes each, and the corpus `transformer` HVP (540 twin launches, 108 of them forward kernels, 10,709 changed elements).

*Baseline redraws* (v0.4.42). `val_generate_baseline` draws integer arguments, grows the array sizes and runs a tangent self-check, and redraws the whole triple when any step fails; `attempts` (default 256) bounds the redraws. A kernel with a low acceptance needs many: `transformer` accepts about 6% of the draws in the range 3 to 6, so 16 redraws failed on 21 of 60 seeds, and which seeds fail depends on the Julia version (the random stream consumed by a failed attempt differs between 1.10 and 1.11). `test/validate_baseline_attempts.jl` keeps two such seeds. A kernel that fails after 256 redraws has a real incoherence in its integer arguments, not bad luck.

*Fallback twin of the split reduction* (v0.4.43, keyword `split_fallback`, default 8). `cgen_split_apply` builds, for each kernel that has a split twin of `split_reduce` lanes, a second twin with `split_fallback` lanes (name `<kernel>_s8!`; the same text with the lane count changed; `cgen_split_kernel` is called twice) when `2 <= split_fallback < split_reduce`. The launch is `if cond && fits16 -> twin16 else (if cond && fits8 -> twin8 else plain)`, where `fits` reads `CUDA.maxthreads` of a `launch = false` build of that twin and compares it with `32 * lanes`. The point is the HVP: its split kernels carry the shadow twin of every statement and need more registers, so a block of 32 x 16 threads does not fit on the V100 and the plain kernel ran; 8 lanes fit. `split_fallback = 0` gives the v0.4.42 text exactly. `test/validate_split.jl` checks, on the CPU shim with a lowered `maxthreads`: the twin of 8 lanes runs when 16 do not fit, the plain kernel runs with the fallback off or when neither fits, the twin of 16 lanes runs when it fits, no second twin is built when `split_fallback >= split_reduce`; three sabotage cases (never launched, register limit of the fallback not checked, fallback built with the lanes of the first).

*Dead stack stores* (v0.4.44, keyword `dead_stack`, CUDA only, default 2). `cgen_dead_stack_apply(host, kernels, backend; inbounds, init_hosts, compact_ok)` runs in `stade_gpu` after `cgen_dead_scalars` and `cgen_reuse_stored` and before contraction, split, inbounds and the precision conversion. Candidates are the `*_stack` arguments of the kernel launches and the `*_stack_d` shadow stacks of an HVP host, kept only if an `initstacks_*` host allocates them (or, for a shadow stack, its base stack); `init_hosts` is that dictionary, and without it nothing is deleted. The index formula of an access is its index with the `__icse_*` and `__idx_*` scalars inlined, kernel parameters replaced by the host arguments, and loop and thread variables left free (a thread variable is any symbol assigned, at any depth, from an expression with `__tid`; the `_p!` kernels assign it inside an `if`). A host scalar may stand in a formula only if its definition is unique (recursively); a local scalar, an array element or any other name makes the formula opaque (`nothing`), and an opaque store is kept. Rules the pass depends on: the sites of a stack own disjoint regions; every kept read repeats the formula of some store; a deleted store has a pure index and value; the stack appears in the host only as a launch argument, in `length(X)` and in its allocation. Level 1 deletes the stores whose formula no read repeats, repeats until nothing changes, runs the dead-scalar pass again, and allocates a fully dead stack with one slot. Level 2 also compacts: when every access splits as `P + local` with one `P` (`cgen_dstack_split`), and after inlining the local definitions of both sides `P` equals the sum of the first `j` terms of the left-nested allocation in the init host, the accesses are indexed by `local` and the allocation is the suffix sum of the terms `j+1..n` (the suffix, not the site's own region size, so a wrong region assumption cannot make the array too small). Region 1 has no prefix and is never compacted. `stade_gpu` returns `dead_stacks`, `compact_stacks` and `stack_init` (the name of the init host); `cgen_shrink_stacks(host, dead, compact)` rewrites the init host, and `stade_gpu_file` and `stade_validate_gpu_file` apply it. A bundle with two functions that take stacks (`nclaim > 1`) passes no init hosts. `dead_stack = 0` is the exact v0.4.43 text. `test/validate_dead_stack.jl` has the unit and compaction cases, the refusals and 8 sabotage cases (keyword not tested, no init evidence, a read need not repeat a store, a deletable store a read repeats, a local scalar in a formula, a prefix that differs between accesses, an allocation prefix that differs, a host scalar assigned twice); `test/validate_dead_stack_real.jl` (one size per process, `DS_SIZE=tiny|small`, `STADE_KIT_KERNELS` and `STADE_KIT_TOOLS`) checks the plan of the real `unet2` adjoint and HVP and runs `validate_ownership` with `baseline_fn`. Known limits: the device-versus-host stack comparison of the GPU parity script differs in deleted slots (informational); the compaction needs the init allocation to match the formula text; the forward kernels that lose all stores are still launched; `mpnn_gather`'s `tripcount_stack` is untouched.

**Gather form of affine stencil adjoints** (`gather_stencil`)
All GPU backends. The keyword is a level: `0` or `false` restores the v0.4.27 output exactly, `1` gives the gather form alone (the v0.4.28 to v0.4.30 output, byte for byte), `2` or `true` (the default) adds the zero-fold described below. The
adjoint of `y[i] = f(x[i-1], x[i], x[i+1])` scatters: iteration `i` adds to
`xb[i-1]`, `xb[i]` and `xb[i+1]`, so threads collide and every add is atomic.
The rewrite, run on the ingested adjoint before any other GPU pass, turns such a
`for` loop into two: a *gather* loop in which thread `j` owns `xb[j]` and sums the
terms of the iterations `j - c` (one block per offset `c`, each guarded to the
iteration range, each replaying the scalars its term needs at the shifted
index), writing `xb[j]` once; and the *residual* loop with everything else,
dropped when it only holds generated temporaries. A loop qualifies when
its step is +1 or -1; its body is plain assignments (no branch, nested loop or
stack operation); every index of a receiving array is the loop variable plus a
constant; no scalar is read before it is defined in the iteration; the
receiving arrays are only accumulated into and never read; every other array
written is owned (touched only at the loop variable) and is kept in a scalar in
the replay; there are at most 8 offsets and at most 24 replayed statements per
offset. Loops that carry a value (a running scalar adjoint, an in-place sweep),
stride-2 or mirrored indices, and loops that read the receiving array stay as
they were. The sum is formed in another order, so the last bits differ.
`test/validate_gather_stencil.jl` runs the rewritten code on the host against
the original for n from 0 to 20, runs the adjoint and the HVP of the whole
corpus through the finite-difference oracles with the rewrite applied, and
sabotage-tests each guard and refusal. `test/validate_write_overlap.jl` checks the
property a gathered kernel has in place of atomics: exactly one plain write to the
receiving array, at the thread's own index, and no access at any other index; its
checks of the atomic repair of a scatter run with `gather_stencil = false`.

*Zero-fold* (level 2). The adjoint of a time-stepping loop zeroes each received array after
it has been used (`o = ub[i]; ub[i] = 0.0`), which the gather rewrite leaves in its residual
loop, and that loop is a launch of its own. `cgen_gather_fold_pair` moves such a residual `R`
into the gather loop `G` that follows it, as a guarded prologue (`if jv >= amin; if jv <= amax;
<body of R at jv>`): each thread zeroes its own element and then adds its terms. The
fold needs all of these: `R` is an owned loop (step +1 or -1, assignments only, every array
reference at exactly the loop variable, no carried scalar, no stack operation) that writes at
least one array; no scalar of `R` is read after `R` (`ii_escapes_nested` is empty); `G` is a
gather loop (`__gj_` variable, step 1) that touches every array `R` writes only at `G`'s own
variable; the range of `R` lies inside the range of `G` (an affine comparison that accepts a
constant difference only); and the two loops are adjacent in one statement list. A residual
cannot fold into the gather *before* it, because neighbour threads still read the old values.
So in `heat1d_multi` the zero of `ub` folds into the next gather and the zero of `vb` (the last
one of the time step, with no gather after it) stays a launch: 4 backward launches become 3 per
step. Dead temporaries of the folded statements are removed by backward liveness and the
remaining scalars get the suffix `_z<tag>`; the renaming is defence in depth, since the escape
test already guarantees no clash, so no sabotage case can see it. The folded loop is
bit-identical to the unfolded pair (the unit checks compare them with `isequal`).

**Coarsened kernel variant** (`coarsen`, `coarsen_min`, `coarsen_fill`, `coarsen_occ`)
CUDA only, opt-in (`coarsen = 1`, the default, emits exactly the v0.4.28 code; 700 CUDA and
JACC plans hash identical). With `coarsen = K` (2 to 4) a kernel whose atomic adds can
be merged also gets a *variant* `<name>_c!` beside the plain kernel; the host launches
the variant only when a run-time test passes, so the plain kernel is never edited. In the
variant one thread runs K iterations of the offloaded loop, 32 apart (the lanes of a warp
still read consecutive elements), and the copies are jammed: an inner `for` or `if`
whose header or condition does not mention a per-copy scalar becomes one shared loop or
branch, and an atomic add whose address is thread-invariant (`:inv`) or depends on the
thread only through a division by a thread-invariant value above 32 (`:key`) is summed
over the copies in registers and issued once when the K addresses agree at run time
(one add per copy when they do not). Anything else is replicated per copy. The last warp
group, where a lane may hold fewer than K iterations, runs its iterations one at a time
through the original body. Only the order of a sum changes. The body is refused (no
variant) for: a `return`, a name starting with `__jm`, a scalar that is also a loop
variable, called as a function, a kernel argument, or read before it is assigned, a body
over 4000 nodes, and a body with no mergeable site under a loop. The host test is
`D >= coarsen_min && n_iter >= coarsen_fill`: `D` is the number of threads that add to one
address (the trip count for an `:inv` site, the divisor for a `:key` site, written in
kernel arguments with the body's invariant scalars substituted; a site whose count
cannot be written that way contributes nothing; several sites take the maximum), and
`n_iter` is the plain trip count. A launch that privatizes (`shared_private`) keeps the
private kernel first. `coarsen_min = 0` turns every test off and builds a variant
wherever one can be built (the validators use it). Evidence (Tesla V100, adjoints,
`fuse_ii_loops = true`): `unet2` takes 45.66, 46.94 and 54.00 ms at c16_n256, c64_n64 and
c128_n32 without the variant and 37.32, 39.34 and 44.51 ms with `coarsen = 2` and the
default thresholds (-18%, -16%, -18%); the per-kernel profile shows what the thresholds
are for: a backward kernel with `D = 64` and 32768 threads got 21% slower, one with
12288 threads 30% slower, and every kernel with `D >= 256` and at least 65536 threads got
15 to 35% faster. `mlp3` (2048 threads) and `mpnn_gather` (`D = 16`) build a variant that
the tests decline (times identical to the plain kernel); forced, they take +28% and
+112%. **Occupancy gate (v0.4.30, `coarsen_occ = 0.6`)**: the transformer lost 17% in v0.4.29 because
the variant takes 1.6 to 2 times the registers (88 against 48 for `b_12`), so 2 blocks of 256
threads fit on an SM where 5 did. Measured on a V100 with one job that read `CUDA.registers`,
the local memory and the block limit of the plain kernel, the variant and a variant without
the register merge: no kernel spills (32 bytes of local memory everywhere), the merge helps in
every kernel (variant faster than the no-merge variant), and the ratio of blocks per SM by
registers (65536 per SM, 256-thread blocks, registers rounded up to 8), variant over plain,
separates all 15 kernels measured: 0.50 or less gave +17 to +39% (transformer `b_12`, `b_13`,
`b_16`, `b_17`, `b_20`), 0.60 or more gave -1 to -35% (transformer `b_19` -1%, every `unet2`
kernel). Merged share, register count alone and body size do not. So when `coarsen_occ > 0` the
host condition ends with a `let` that compiles both kernels with `@cuda launch=false`, reads
`CUDA.registers` of each and requires `100 * blocks(variant) >= 100 * coarsen_occ *
blocks(plain)` (`cgen_coarsen_occ_gate`); it runs last, so a launch that the thread-count tests
decline probes nothing (`mlp3` probes 0 times), and the probe costs a few microseconds per
launch otherwise. `coarsen_occ = 0` removes the gate and gives the v0.4.29 output. With the
gate (`coarsen = 2`, defaults): `unet2` 45.60, 46.90, 53.99 -> 37.40, 39.33, 44.58 ms (-18, -16,
-17%), transformer 38.18 -> 38.20 ms (d512_T512) and 439.4 -> 435.1 ms (d1024_T1024), `mlp3` and
`mpnn_gather` unchanged. Two families on one GPU model: treat 0.6 as measured, not proven, and
time your own kernel before turning `coarsen` on. The 65536-register file is hard-coded. A K of 3
was slower than 2 in the spike (register pressure). A plain owner-computes rewrite (one thread
per received element) also lost. `test/validate_coarsen.jl`: unit
kernels, shim execution at K = 2, 3, 4 in two thread orders with zero-trip and ragged
sizes, "forced launches only variants, unreachable threshold launches only plain
kernels", the gate on 11 register pairs (the shim models `launch=false` and `CUDA.registers`),
corpus byte-identity of every plain kernel, and sabotage of every guard, the merge, the
selector and the gate. `validate_ownership` runs with `OWN_COARSEN=2` (forced variants
of every corpus plan against the CPU adjoint).

**Block-private accumulators** (`shared_private`)
CUDA only (default on; `false` restores the old output exactly, an integer sets
the threads-per-element ratio, 0 turns the launch tests off). A kernel whose
atomic accumulations go to a few small arrays is emitted twice: unchanged, and
as a `_p` variant in which each block adds into a shared-memory copy and issues
one global atomic per element at the end. An array qualifies when the kernel
only accumulates into it, every index is thread-invariant or depends on the
thread through `mod` by an invariant, some index varies in the kernel, and it
is a kernel argument. The host launches the variant only when every
qualifying array fits 8192 bytes, there are at least 32 threads per element,
and an array accumulated under no loop has four block threads per element.
One array that does not fit keeps the whole kernel on the original. A warp-helper
site keeps the helper on the private array. The CPU ownership shim has no shared
memory, so `validate_ownership` runs with `shared_private = false`.

**Sync-free reduction** (`sync_free_reduce`)
CUDA only. The large-trip branch of an array-element reduction is a
device-side broadcast add on a one-element view, with no scalar copied to
the host.
This branch is not capturable in a CUDA graph (the dims reduction of
CUDA.jl fails during capture). A captured call needs
`reduction_threshold = typemax(Int)`, which always takes the atomic kernel.

**Array privacy**
The proof that a written array's whole access pattern is confined to a
private, non-overlapping per-iteration region, so concurrent iterations
cannot collide. It is an additive proof layered on top of the pairwise
write-versus-read index check, which cannot by itself tell disjoint
assembled slices apart from a real value recurrence.

**Use-before-def**
A read of a locally-assigned scalar before its own definition inside the
loop body. It signals a cross-iteration dependency and disqualifies the
loop, unless the convergent-constant proof recovers it.

**Convergent constant**
A local scalar that provably ends every path at the same literal. When
that literal also matches the value known coming into the loop, induction
makes every iteration start there, so injecting it as the device kernel's
first statement reproduces sequential semantics.

**`known_consts`**
The map of variables known to hold a specific literal on entry to a loop,
carried from a preceding top-level literal assignment and threaded down
through nesting levels.

**`reduce_vars`, boxing**
Free scalar variables a device kernel accumulates into. `cgen_emit` boxes
each into a one-element device array before any launch, so the
accumulation can become an atomic add against index 1.

**`gpu_backend`**
The vendor record holding a suffix, launch macro, thread and block keyword
names, thread-index expression, atomic macro, `allowscalar` macro,
preamble, default precision, and precision lock. Adding a launch-macro
vendor means writing one such function.

**`allowscalar`**
A GPU library's escape hatch permitting a host-side scalar index into a
device array. CUDA, AMDGPU, and Metal have one. JACC does not, which is
why `jgen_*` synthesizes a `range=1` kernel for a reduction writeback.

**Precision lock**
A backend's declaration that it supports only one float precision. Metal
locks. JACC does not lock, because the vendor is chosen after generation
and STADE cannot know which one runs.

## V.6 Validation vocabulary

**Corpus**
`test/val-corpus/`, holding 46 primal kernels. Every syntactic shape the
analyses can encounter should have a kernel here. A shape absent from the
corpus is a path that never executes, and an unexecuted path is where
defects survive.

**Baseline**
A YAML file of random integer arguments and float values for one kernel,
written once and reused, so a validation run is reproducible.

**Dot-product identity**
`<Yb, J*Xd> == <J'*Yb, Xd>`. The exact oracle, agreeing to about `1e-15`
with no epsilon.

**Floor, ceiling**
The two coverage-test shapes. `validate_ii_coverage.jl` records a floor
per kernel: classifying more passes with a note, classifying fewer fails.
`validate_offload.jl` records a ceiling on loops left on the host, the
same idea mirrored. Both exist because the correctness oracles are silent
about coverage.

**Sabotage test**
Deliberately breaking a guard to confirm it fails. A guard never seen to
fail is one you do not know you have.

---

# Part VI. Working on STADE

## VI.1 Getting Julia

Julia is not on apt, and JuliaLang's binary host is not reachable from
this environment. The project mirrors the official 1.10.11 LTS tarball as
a GitHub release asset.

```bash
curl -sL https://github.com/luciano-drozda/julia-tar/releases/download/julia-1.10.11/julia-1.10.11-linux-x86_64.tar.gz -o /home/claude/julia.tar.gz
echo "fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf  /home/claude/julia.tar.gz" | sha256sum -c -
tar -xzf /home/claude/julia.tar.gz -C /home/claude/
export PATH="/home/claude/julia-1.10.11/bin:$PATH"
```

The sha256 matches the official JuliaLang checksum. If the check fails,
stop. Treat the tarball as untrusted and report it. Re-verify if the URL
or the tag ever changes.

## VI.2 Generating something by hand

```bash
export PATH="/home/claude/julia-1.10.11/bin:$PATH"
cd STADE.jl
julia -e '
include("src/STADE.jl")
STADE.stade_adjoint_file("test/val-corpus/geomrecur.jl", "/tmp/g_b.jl";
                         keep_push_pop = false, fuse_ii_loops = true)
println(read("/tmp/g_b.jl", String))
'
```

Read the output. That habit catches more defects than any aggregate
counter, and Part III exists to make the output readable.

## VI.3 Running the tests

```bash
cd test

julia validate_corpus.jl                 # all four oracles, default flags
julia validate_corpus.jl affine_loss.jl  # one kernel
julia validate_corpus_flags.jl           # all four flag combinations
julia validate_ii_coverage.jl            # fusion coverage floor
julia validate_offload.jl                # GPU coverage ceiling, no GPU needed
julia validate_zerotrip.jl               # integer draws that include zero
julia validate_gpu_arglayout.jl          # adjoint/HVP call layout, no GPU needed
julia validate_gpu_stack_compare.jl      # stack comparison in the parity script, no GPU needed
julia validate_if_nesting.jl             # nested ifs, branch scalars, nested retiring writes, early-stage size terms, with sabotage
julia validate_elision_coverage.jl       # snapshot-elision branches, no GPU needed
julia validate_gpu_truth.jl              # BigFloat truth criterion of the parity script, no GPU needed, with sabotage
julia validate_int_state.jl              # integer-state restores: values, refusals, with sabotage
julia fuzz_int_state.jl 1 300            # differential fuzzer for integer-state kernels
julia validate_corpus_gpu.jl             # live device parity -- needs a real GPU
```

`validate_zerotrip.jl` deletes the `.yaml` baselines it wrote. It draws every
integer from `[0, 2]`, and `.yaml` is gitignored, so leaving them behind would
hand the next script a corpus in which nearly every loop is empty. GPU parity
keeps its own `.gpu.yaml` namespace for the same reason.

Run the corpus in halves. `validate_corpus.jl` accepts kernel names as
arguments. Backgrounded runs get reaped in this environment.

**Pass the flags you generated with through to the validator**, not only
to generation. The validator's defaults otherwise check a different mode's
math than the one under test, and the flagged path goes unexercised.

## VI.4 The development discipline

These come from `skill-stade-dev`. They are the accumulated cost of past
defects.

1. Write adversarial test kernels **before** implementing.
2. A minimal synthetic case passing is not sufficient. Validate on the
   corpus.
3. Check generated code directly, not just `ok = true`.
4. Run the corpus after every change, with matching flags.
5. Add a permanent regression test using the exact shape that caught the
   defect.
6. When a guard test breaks, the gate is wrong, not the test.
7. Sabotage-test every new guard.
8. **Measure the property, not a proxy.** Device-kernel count is not
   offload coverage. A liveness walker that cannot kill is not liveness. A
   single-kernel fingerprint is not default-path drift. Print the per-item
   detail beside any aggregate and check they agree.
9. Verify a mechanism before you write it down in a comment. A confidently
   wrong comment is worse than no comment.
10. Keep a corpus kernel for every syntactic shape the analyses can
    encounter. When you add a mechanism, enumerate the shapes it can meet
    and check that one of each is represented.

## VI.5 Self-check before returning STADE code

- [ ] No `module`, `struct`, `@enum`, top-level `const`, or shared state
- [ ] Correct stage prefix on every function, and no filesystem access
      outside `io_*`
- [ ] `NamedTuple`s match the frozen shapes, no `:while`, no ad hoc keys
- [ ] Plain `Expr`, `Symbol`, `Number`, never a custom type
- [ ] Only documented upstream shapes as inputs
- [ ] No caller-supplied independents or dependents
- [ ] Comments one line, only where the code is not self-explanatory
- [ ] No corpus kernel named in `STADE.jl`
- [ ] Loop-nest analysis walks actual `for` structure, with no synthetic
      nests reconstructed from body-level `div`/`mod`
- [ ] Duplicated `snap_*`/`agen_*` pairs edited identically, and the
      `stade_*_check` assertions pass
- [ ] Tier B index expressions pure, block-local scalars through value
      tables
- [ ] `initstacks_*` has no free variable outside its parameter list
- [ ] Any `keep_push_pop = false` claim went through
      `stade_validate_*_file` with the flag explicitly passed

## VI.6 Two notes on the current source

Two details differ from the letter of `skill-stade-dev`. Both are worth
knowing before you read the file.

**There is one `module STADE` block**, at line 110, closing at line 8598.
Hard rule 1 says no module blocks. The wrapper exists for packaging, and
there are no nested modules inside it. Treat the rule as forbidding
*internal* modules.

**There is a `norm_` prefix** that the skill's prefix list does not
mention. It holds four functions and runs from `stade_adjoint` and
`stade_hvp`. Section IV.7 describes it. Add it to your mental prefix list.

## VI.7 Prefix reference

| Prefix | Stage | Section |
|---|---|---|
| `inl_` | Inline multi-kernel call graphs | IV.2 |
| `parse_` | Validate and convert to `(sig, body)` | IV.3 |
| `shape_` | Infer variable kinds | IV.4 |
| `der_` | Local derivative rule table | IV.5 |
| `emit_` | Shared `Expr` builders | II.4 |
| `act_` | Activity analysis | IV.6 |
| `norm_` | Dead loop-entry resets | IV.7 |
| `snap_` | Snapshot planning, TBR | IV.8 |
| `lin_` | Derivative trees | IV.9 |
| `tgen_` | Tangent codegen | IV.10 |
| `agen_` | Adjoint codegen and `initstacks_` | IV.11 |
| `ii_` | Loop-fusion eligibility | IV.13 |
| `hvp_` | Forward-over-reverse codegen | IV.14 |
| `cgen_` | CUDA / AMDGPU / Metal codegen | IV.15 |
| `jgen_` | JACC codegen | IV.16 |
| `val_` | Correctness oracles | IV.17 |
| `io_` | File access, the only such stage | IV.1 |
| `stade_` | Public API and cross-checks | II.3 |

## VI.8 Where to start

Three entry points, in rising order of difficulty.

**Add a derivative rule.** Write `der_partials_<name>`, add the
tangent/adjoint pair, add the `der_table` entry, add the symbol to
`parse_intrinsic_whitelist`, and add a corpus kernel that calls it. This
touches one stage and teaches you the rule table.

**Improve tangent codegen.** `tgen_*` is under 100 lines and has no
memory model. Anything you break there fails loudly in the tangent oracle.

**Work on the snapshot analysis.** This is where the real difficulty
lives. Before you change anything in `snap_*` or `agen_*`, read section
IV.8, then read `stade_site_level_tbr_check`, then write a kernel that
exercises the shape you care about and confirm the current code handles it
wrongly. Only then change the code, and change both copies of any
duplicated pair.
