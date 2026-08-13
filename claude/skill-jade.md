---
name: skill-jade
description: >
  Use whenever writing, reviewing, or refactoring Julia code for numerical
  kernels in scientific computing (stencils, linear algebra, integrators,
  elementwise/reduction kernels, etc.) intended to run on a single
  processor. Trigger any time the user asks for a Julia function or kernel
  to crunch numbers, even without naming "skill-jade" or the rules
  themselves. Enforces a strict house style: snake_case names (sequential
  loop variables prefixed `i_seq_`), loop headers written with `=` and
  always naming the iteration variable (never a throwaway `_`), untyped
  positional-only arguments, loops over branching, iteration-independent
  loops over loop-carried dependencies, fusing nested iteration-independent
  loops into one flattened loop when the nest is rectangular, no broadcasting, no indirect
  indexing, no in-function array allocation, no compound assignment, only
  Fortran-intrinsic-equivalent calls (plus user-supplied helpers),
  `#`-comment input headers instead of docstrings, and functions that
  always end with an explicit `return nothing` statement.
---

# skill-jade: Julia numerical kernel style

This skill encodes a house style for **monoprocessor Julia numerical kernels** —
small, serial, allocation-free, Fortran-subroutine-flavored functions meant
to be called in a hot loop elsewhere (finite-difference stencils, linear
algebra kernels, time-integrator steps, reductions, etc.). Kernels do their
work by writing into arguments the caller already allocated, then end with
an explicit `return nothing`. The style favors code that is simple to
read, cheap to
reason about, and easy to later port or transpile, because it avoids the
constructs that make those things hard: hidden allocations, data-dependent
control flow, data-dependent memory access, and Julia-only conveniences
that don't map onto a plain imperative language.

Apply every rule below to any Julia numerical-kernel code you write in this
conversation, even if the user's prompt doesn't repeat the rules. When
revising existing code, bring it into compliance rather than leaving
violations in place.

## Installing Julia in this environment

Julia isn't available via apt, and JuliaLang's official binary host
(`julialang-s3.julialang.org`) isn't reachable from this network. The user
has mirrored the official Julia 1.10.11 (LTS) linux-x86_64 tarball as a
GitHub release asset on their own account, on hosts that are reachable
(`github.com`, `release-assets.githubusercontent.com`). Its sha256 has been
verified to match the official JuliaLang checksum for
`julia-1.10.11-linux-x86_64.tar.gz`
(`fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf`) — do
not skip re-verifying this checksum after download if the tarball, URL, or
release tag ever changes, since that would mean re-establishing trust in a
new artifact.

```bash
curl -sL https://github.com/luciano-drozda/julia-tar/releases/download/julia-1.10.11/julia-1.10.11-linux-x86_64.tar.gz -o /home/claude/julia.tar.gz
echo "fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf  /home/claude/julia.tar.gz" | sha256sum -c -
tar -xzf /home/claude/julia.tar.gz -C /home/claude/
export PATH="/home/claude/julia-1.10.11/bin:$PATH"
julia --version
```

If the checksum check fails, stop and do not use the tarball — treat it as
untrusted and flag this to the user rather than installing it anyway.

For any subsequent `bash_tool` call in this conversation that needs Julia,
prepend `export PATH="/home/claude/julia-1.10.11/bin:$PATH"`.

## The rules, and why they matter

1. **Names are `snake_case`: lowercase letters, digits, and underscores
   only.** `x`, `dx1`, `flux_x`, `n_iter`, `tmp2` are all fine; anything with
   an uppercase letter is not. This applies equally to variable names *and*
   function names. The one structured exception is the reserved prefix used
   for sequential loop variables — see rule 2.

2. **Prefer iteration-independent loops over sequential ones, and name the
   loop variable to say which kind it is.** Write each loop body so that
   iteration `i` only reads inputs and writes outputs at index (or indices)
   derived from `i` — never reads a value that a *previous* iteration of the
   same loop wrote, unless the math genuinely requires a running/carried
   value (e.g. an accumulating sum, a genuine prefix scan). Don't introduce
   an accumulator or running variable that couples iterations together when
   the underlying computation is really elementwise — that coupling is
   exactly what makes a loop unsafe to reorder, chunk, or vectorize later.
   When a loop is unavoidably sequential (a true recurrence, scan, or
   reduction), that's fine — but make the dependency visible in the name:
   an ordinary iteration-independent loop uses a plain variable (`i`, `j`,
   `k`, ...), while a genuinely sequential loop's iteration variable must
   start with the reserved prefix `i_seq_` (e.g. `i_seq_k`, `i_seq_idx`).
   That way the loop-carried dependency is visible at a glance, without
   having to read the loop body.

   **Fuse a nest of iteration-independent loops into one when the nest is
   rectangular.** If an outer independent loop's body is *just* an inner
   independent loop (plus, optionally, more independent loops — no
   sequential loop or branch sitting between them), and the inner loop's
   trip count doesn't depend on the outer loop's variable (a rectangular
   nest, e.g. `for i = 1:n` wrapping `for j = 1:m` with `m` fixed), collapse
   the whole nest into one loop over the flattened index range, then recover
   each original index inside the body with `div` and `mod` (both already
   fair game under rule 11) instead of keeping the nest:
```julia
   # not fused: two independent loops nested
   for i = 1:n
       for j = 1:m
           y[i, j] = x[i, j] * 2.0
       end
   end
   # fused: one independent loop over the flattened range
   for idx = 1:n * m
       i = div(idx - 1, m) + 1
       j = mod(idx - 1, m) + 1
       y[i, j] = x[i, j] * 2.0
   end
```
   Only fuse when every loop in the nest is independent and rectangular in
   this sense. Don't fuse when: any loop in the nest is sequential
   (`i_seq_...`) — fusing would change what "one iteration" means for the
   carried state; the inner bound is computed from the outer variable
   (e.g. a triangular loop where `j`'s upper bound depends on `i`) — there
   is no fixed `m` to flatten against; or a boundary-vs-interior split
   (rule 4) still needs to sit between two of the loops — do the split
   first, then fuse each resulting rectangular interior nest on its own.
   A loop that started as `i_seq_` never gets folded into a fused loop's
   flattened range, even if it sits alongside independent loops in the
   same original nest.

3. **Loop headers use `=`, not `in`, and always name the iteration
   variable explicitly.** Write `for i = 1:n`, never `for i in 1:n`. This
   holds even when a loop's body never actually references its counter —
   give it a real name anyway rather than reaching for Julia's throwaway
   `_`: write `for j = 1:n` (with `j` genuinely unused inside) instead of
   `for _ = 1:n`. Every loop header keeps the same shape this way, with no
   special case to remember for the unused-counter case.

4. **Prefer loops over conditionals.** Keep `if`/`elseif`/`else` to the
   strict minimum the math requires, and never use `ifelse`. Concretely:
   - Replace `if`-based clipping/selection with the matching arithmetic
     function where one exists (`max`, `min`, `abs`, `sign` — see rule 11
     for which functions are fair game).
   - Handle boundary points by splitting the iteration range into separate
     loops (or single statements) for interior vs. boundary, instead of
     branching on the index inside one loop.
   - Don't use `if` to defensively re-implement what a `for i = a:b` range
     already guarantees.

5. **Don't type the function's input arguments.** Write
   `function grad(u, du, dx, n)`, never
   `function grad(u::Vector{Float64}, du::Vector{Float64}, dx::Float64, n::Int64)`.
   The caller's concrete types decide what runs; annotating inputs only adds
   friction without changing behavior for these kernels.

6. **Don't use keyword arguments.** Every input is positional — no `;`
   section in the signature, no `name=default` arguments. Call sites should
   never need `f(x, y; a=1.0)`; use `f(x, y, a)`.

7. **Only four variable "shapes" exist: `Float64`, `Int64`, `Array{Float64}`,
   `Array{Int64}`.** Every local variable and argument should be one of
   these (the function itself always returns nothing — see rule 16). In
   particular:
   - Don't store a comparison/condition in a `Bool` variable — write the
     comparison directly inside the `if`/`while` that needs it instead of
     assigning it to a name first.
   - Loop counters and sizes are `Int64`; physical/field quantities are
     `Float64`.
   - Don't introduce tuples, ranges-as-values, strings, or structs as local
     variables.
   - A kernel that "computes a scalar" writes it into a length-1
     `Array{Float64}` or `Array{Int64}` output argument rather than
     returning it — see rule 16.

8. **Never allocate an array inside the function.** No `zeros(n)`,
   `similar(x)`, `Array{Float64}(undef, n)`, `Vector{Int64}(undef, n)`, array
   comprehensions, `collect(...)`, etc. All arrays a kernel touches — inputs,
   outputs, and any scalar "results" boxed as length-1 arrays (rule 7) — are
   allocated by the caller and passed in; the kernel only reads and writes
   into them in place. Plain scalar locals (`Float64`/`Int64`) may still be
   declared and initialized freely — this rule is about arrays only.

9. **Never index an array with another array's element inline
   (`a[b[i]]`).** Indirect/gather-scatter indexing like `y[idx[i]] = x[i]` or
   `x[perm[i]]` is not allowed. If an index genuinely needs to be looked up
   from an array, read it into a plain `Int64` scalar on its own line first,
   then index with that scalar:
```julia
   # not allowed:
   y[idx[i]] = x[i]
   # do this instead:
   j = idx[i]
   y[j] = x[i]
```

10. **Never use a broadcasted/dot operator (`.+`, `.-`, `.*`, `./`, `.=`,
    `sin.(x)`, etc.).** Write the explicit `for` loop over the array instead.
    This applies to assignment too: no `y .= x`.

11. **Only call Julia functions that have a direct native-Fortran-intrinsic
    counterpart — plus any helper functions the user has defined and
    supplied.** Fair game: things like `abs`, `sqrt`, `exp`, `log`, `log10`,
    `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`,
    `mod`, `div`, `max`, `min`, `sign`, `floor`, `ceil`, `trunc` — these all
    mirror a standard Fortran intrinsic. Avoid Julia-only conveniences that
    don't have a Fortran counterpart, even if they'd be idiomatic
    elsewhere: `sum`, `prod`, `map`, `filter`, `reduce`, `foldl`/`foldr`,
    `enumerate`, `zip`, `findfirst`/`findall`, `clamp`, `ifelse`,
    `any`/`all`, `cumsum`, and `Float64(...)`/`Int64(...)` conversion
    calls — and (as already covered by other rules) comprehensions,
    broadcasting, and array-allocating functions. Before calling any Base
    function beyond ordinary arithmetic and indexing, ask "does Fortran
    have an intrinsic that does exactly this?" — if not and the user hasn't
    supplied it as their own function, compute it with an explicit
    loop/arithmetic instead. If the user provides their own Julia functions
    during the conversation, those are always fine to call.

12. **Integer division always goes through `div`.** Write `div(a, b)`, never
    `a ÷ b`, `fld(a, b)`, or `Int(floor(a / b))`.

13. **No compound assignment operators.** Write accumulations and products
    out in full: `x = x + 3.0`, `n = n * 2`, `s = s - dt`. Never `x += 3.0`,
    `n *= 2`, `s -= dt`, etc.

14. **Every function gets a `#`-comment header describing each input —
    never a `"""..."""` docstring.** Directly above `function ...`, write
    plain `#` comment lines: one line for what the kernel computes, then
    one short `#` line per input argument explaining what it represents
    (not its type — just its meaning, e.g. "grid spacing", "number of
    interior points", "length-1 output array holding the result").

15. **Comment the algorithm as you go.** Short, concise `#` comments along
    the implementation explaining *why* a step happens (e.g. "interior
    stencil", "boundary copied through unchanged", "accumulate
    sequentially"), not comments that just restate the Julia syntax.

16. **Every function ends with an explicit `return nothing` statement.**
    Because a kernel's real output is whatever it wrote into its array
    arguments (including any length-1 arrays boxing a scalar result, per
    rule 7), there's nothing meaningful to hand back to the caller — but say
    so explicitly. The last line of every function body must be
    `return nothing`, spelled out in full; don't leave it implicit, and
    don't shorten it to a bare `nothing`.

## Worked example

A 1-D three-point stencil (`y[i] = x[i-1] + x[i] + x[i+1]`, with the two
boundary points just copied through), given `x`, `y`, and `n` where `y` is
caller-allocated and `n = length(x)`. The interior loop is
iteration-independent (each `y[i]` only depends on `x` at fixed offsets
from `i`), so its variable stays a plain `i`:

```julia
# stencil(x, y, n)
#
# Three-point average stencil, boundary points copied through unchanged.
#
# x: input array of length n
# y: output array of length n, filled in place
# n: number of points in x and y
function stencil(x, y, n)
    # boundary points pass through unchanged
    y[1] = x[1]
    y[n] = x[n]
    # interior points: sum of the point and its two neighbors
    for i = 2:n - 1
        y[i] = x[i - 1] + x[i] + x[i + 1]
    end
    return nothing
end
```

Why this complies: `snake_case` names throughout; no argument types; no
keyword arguments; `y` is allocated by the caller, not inside `stencil`;
the interior loop's iterations don't depend on each other (order could be
scrambled and the result is identical), so it uses a plain `i` and a
`for i = ...` range instead of a branch on the index; the boundary cases
are handled by separate statements rather than an `if i == 1 ... elseif
i == n ...` inside the loop; there's no broadcasting, indirect indexing,
compound assignment, or Fortran-less Base function; every variable is
`Int64` (`i`, `n`) or a `Float64` array element (`x`, `y`); it has a
`#`-comment header and inline comments instead of a docstring; and it ends
with an explicit `return nothing`.

A second example with a genuinely sequential loop — a dot product, which
must accumulate a running sum, so its loop variable is prefixed `i_seq_`
and the scalar result is written into a length-1 output array rather than
returned:

```julia
# dotprod(x, y, n, out)
#
# Dot product of x and y, written into out.
#
# x: first input array of length n
# y: second input array of length n
# n: number of elements to sum
# out: length-1 output array; out[1] receives the result
function dotprod(x, y, n, out)
    s = 0.0
    # accumulate sequentially: each partial sum depends on the previous one
    for i_seq_k = 1:n
        s = s + x[i_seq_k] * y[i_seq_k]
    end
    out[1] = s
    return nothing
end
```

A third example showing fusion: scaling every entry of an `n`-by-`m`
matrix stored as a flat `Array{Float64}` (row `i`, column `j` at linear
offset `(i - 1) * m + j`). Iterating `i` and `j` independently over a
fixed `n`-by-`m` rectangle is exactly the case rule 2 says to fuse, so it
becomes one loop over `1:n * m` with `i` and `j` recovered via `div`/`mod`
rather than two nested loops:

```julia
# scale_matrix(x, y, n, m, factor)
#
# Multiplies every entry of the n-by-m matrix x by factor, writing into y.
#
# x: input array of length n * m, row-major (row i, col j at (i-1)*m+j)
# y: output array of length n * m, filled in place, same layout as x
# n: number of rows
# m: number of columns
# factor: scalar multiplier
function scale_matrix(x, y, n, m, factor)
    # rows and columns are both independent, and m is fixed across every
    # row, so the two loops fuse into one over the flattened range
    for idx = 1:n * m
        i = div(idx - 1, m) + 1
        j = mod(idx - 1, m) + 1
        k = (i - 1) * m + j
        y[k] = x[k] * factor
    end
    return nothing
end
```

Contrast with a non-compliant version a naive first draft might produce —
annotated/keyword args, an allocated output, `ifelse`, a branch inside the
loop instead of split statements, broadcasting, an unnecessary sequential
accumulator on what should be an independent loop, compound assignment, a
non-Fortran function, a loop header written with `in` instead of `=`, no
comment header, and returning the array instead of `return nothing`:

```julia
function stencil(x::Vector{Float64}, n::Int64; pad=0.0)
    y = zeros(n)
    total = 0.0
    for i in 1:n
        total += 1
        y[i] = ifelse(i == 1 || i == n, x[i], x[i - 1] .+ x[i] .+ x[i + 1])
    end
    return y
end
```

## Self-check before returning code

Read the draft back and confirm all of the following, fixing anything that
fails before sharing it:

- [ ] Every variable and function name is lowercase letters, digits, and
      underscores only
- [ ] The function signature has no `::Type` annotations and no `;` keyword
      arguments
- [ ] No array is created inside the function body (`zeros`, `similar`,
      `Array{...}(undef, ...)`, comprehensions, `collect`, etc.) — all
      arrays came in as arguments, including any length-1 array boxing a
      scalar result
- [ ] No `Bool`, `String`, tuple, struct, or range is stored in a variable;
      every variable is `Float64`, `Int64`, `Array{Float64}`, or
      `Array{Int64}`
- [ ] No `a[b[i]]`-style indirect indexing anywhere
- [ ] No broadcasted dot-operator or dot-call (`.+ .- .* ./ .= .sin. ...`)
      appears anywhere
- [ ] Every loop header uses `=` rather than `in`, and every loop names its
      iteration variable explicitly — never a throwaway `_`, even when the
      body doesn't use it
- [ ] `if`/`else` only appears where the math truly has no loop/range-based
      alternative, and `ifelse` is never used
- [ ] No loop carries state between iterations unless that dependency is
      mathematically required by the kernel being computed, and every such
      genuinely sequential loop's variable starts with `i_seq_` (ordinary
      independent loops do not use that prefix)
- [ ] No rectangular nest of purely iteration-independent loops was left
      un-fused — if the inner trip count doesn't depend on the outer
      variable and no sequential loop or boundary split needs to sit
      between them, it's collapsed into one loop with `div`/`mod`-recovered
      indices
- [ ] Every called function is either ordinary arithmetic/indexing, has a
      direct Fortran-intrinsic counterpart, or was supplied by the user
- [ ] Integer division uses `div`, never `÷`/`fld`/floor-based tricks
- [ ] No compound assignment operator (`+=`, `-=`, `*=`, `/=`, ...) appears
      anywhere — accumulations are written in expanded form
- [ ] The function has a `#`-comment header describing each input — no
      `"""..."""` docstring
- [ ] The implementation has short `#` comments explaining the algorithm
- [ ] The function ends with an explicit `return nothing` as its last
      line — not left implicit, and not shortened to a bare `nothing`