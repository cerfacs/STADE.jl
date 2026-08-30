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

stade_cuda_file(in_path, out_path;   precision=nothing, keep_all_atomic=true)
stade_amdgpu_file(in_path, out_path; precision=nothing, keep_all_atomic=true)
stade_metal_file(in_path, out_path;  precision=nothing, keep_all_atomic=true)
stade_jacc_file(in_path, out_path;   precision=nothing, keep_all_atomic=true)
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
`needed` flag. The flag starts false and stays false through nested `+`
and `-`, because a constant partial derivative never needs its operand's
value. Any other call is nonlinear, so `needed` flips true and stays true
inside it.

The consequence is worth internalizing. A copy or a sum needs nothing
saved. A product, a power, or a `sin` needs its operands saved.

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

### Branch-scalar hoisting

A scalar assigned in only one arm of an `if` needs its restore hoisted
above the whole `if`, not placed inside an arm. `agen_if_branch_scalar_vars`
finds them and `skip_restore` stops the arm from popping a second time.
`agen_branch_scalar_fallback` supplies the constant for the arm that does
not assign it, and trusts only a literal number, because a recompute must
not depend on a variable the reverse sweep may already have disturbed.

This hoisting is why push and pop sites are matched by a **structural
key** and never by a running counter. Hoisting reorders pops, and a
counter silently desynchronizes.

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

## IV.12 `keep_push_pop = false` in detail

This flag replaces every `push!` and `pop!` with an indexed write and read
into a pre-sized vector, addressed by a closed-form, thread-local
position.

`keep_push_pop = true` must stay **byte-identical** to the output with
`ii_plan = nothing`. Both flags off is an exact rollback. That property is
what makes a two-flag public surface defensible for this much machinery.

### The index formula

A site's index is `base_offset + local_position`.

- `agen_local_position` gives the 1-based row-major position within the
  site's own loop nest. For a nest of depth `d` it sums
  `position_in_frame * stride_of_frame` over all frames and adds one
  global `+1`.
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

A stack that no block can resolve is **tainted**. It falls back to
growable `push!`/`pop!` regardless of the flag, per stack and never per
occurrence.

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

`cgen_expr_injective_ok` refuses `div`, `mod`, `rem`, `fld`, `cld`, `÷`,
and `%` outright. These collapse several thread indices onto one result by
construction. A two-times upsample's `div(oim1, scale)` maps adjacent
output positions to the same source row, and a plain write through it
silently loses a fraction of the accumulation on real hardware.

### Proving a loop parallel

`cgen_reduction_only_loop` is the gate. It runs three checks.

1. `cgen_use_before_def` walks in program order and refuses any read of a
   locally-assigned scalar before its own definition inside this body.
   This subsumes a self-assignment check, because a cross-iteration
   dependency can exist with no self-reference on the left-hand side.
2. Convergent-constant recovery. A local scalar that fails check 1 may
   still be safe if it provably converges to the same literal on every
   path, **and** that literal matches the value known coming into the
   loop. `known_consts` carries that entry from a preceding top-level
   literal assignment. The proof then injects `var = c` as the device
   kernel's first statement.
3. Array aliasing. For each written array, any structural mismatch between
   a write index and a read index disqualifies the loop, unless one of
   three relaxations applies.

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
An operation whose local partials are constants, so it never needs its
operands' values saved. Addition and subtraction are the only ones STADE
treats this way. Everything else is nonlinear for snapshot purposes.

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
because branch-scalar hoisting reorders pops.

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
of re-evaluating the condition.

**Branch-scalar hoisting**
Moving a scalar's restore above an entire `if`, because only one arm
assigns it. It is the reason site keys must be structural.

**`skip_restore`**
The argument that stops an `if` arm from popping a value an enclosing
hoist already popped.

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
The 1-based row-major offset of one occurrence within its own loop nest,
built by `agen_local_position`.

**Stride**
The product of trip counts of every loop frame more nested than a given
frame. It is that frame's row-major stride.

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

**`keep_all_atomic`**
A GPU flag. When true a matched reduction loop keeps its synthesized
per-element atomic kernel. When false it is replaced by one idiomatic
reduction call.

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
julia validate_elision_coverage.jl       # snapshot-elision branches, no GPU needed
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
