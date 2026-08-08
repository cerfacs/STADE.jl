---
name: skill-stade-kernels
description: >
  Use whenever writing, reviewing, or refactoring a Julia numerical kernel
  meant to be fed into STADE (the automatic differentiation engine) for
  tangent/adjoint/HVP generation -- stencils, linear algebra, integrators,
  elementwise/reduction kernels, and similar monoprocessor scientific-
  computing code. Trigger any time the user asks for a Julia kernel that
  STADE will differentiate, mentions the validation corpus, or asks to
  add/modify a kernel alongside STADE.jl itself, even without naming this
  skill. This is a *lighter* house style than the original one STADE grew
  out of: several syntax restrictions that STADE's parser and shape
  inference never actually needed have been dropped, so don't reapply them
  from habit -- check the rules below rather than assuming the old, more
  restrictive style.
---

# skill-stade-kernels: house style for kernels STADE differentiates

STADE parses each kernel's `Expr` tree, infers every variable's shape
(`Float64` / `Int64` / `Array{Float64}` / `Array{Int64}`) from how it's
used, traces which variables are reachable from an independent input, and
then generates tangent/adjoint/HVP code from that analysis. Every rule
below is either something STADE's pipeline actually depends on to get
that analysis right, or a genuine mathematical constraint of the kernel
style (allocation-free, single-processor, Fortran-portable). Rules that
turned out to be pure syntax preferences STADE never consumed have been
dropped -- see "What's no longer required" if you're used to a stricter
style and want to know what changed.

Apply every rule below to any kernel code you write in this conversation,
even if the user's prompt doesn't repeat the rules. When revising existing
kernel code, bring it into compliance rather than leaving violations in
place.

## Installing Julia in this environment

Julia isn't available via apt, and JuliaLang's official binary host
(`julialang-s3.julialang.org`) isn't reachable from this network. The user
has mirrored the official Julia 1.10.11 (LTS) linux-x86_64 tarball as a
GitHub release asset on their own account, on hosts that are reachable
(`github.com`, `release-assets.githubusercontent.com`). Its sha256 has been
verified to match the official JuliaLang checksum for
`julia-1.10.11-linux-x86_64.tar.gz`
(`fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf`) -- do
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

If the checksum check fails, stop and do not use the tarball -- treat it
as untrusted and flag this to the user rather than installing it anyway.

For any subsequent `bash_tool` call in this conversation that needs Julia,
prepend `export PATH="/home/claude/julia-1.10.11/bin:$PATH"`.

## The rules, and why they matter

1. **A sequential loop's iteration variable must start with `i_seq_`; an
   ordinary (iteration-independent) loop's variable must not.** This is
   the one naming convention that still matters, and it's load-bearing,
   not decorative: it's the only signal that tells STADE whether a loop
   carries a cross-iteration dependency, which in turn decides whether
   the adjoint sweep reverses that loop's iteration order. Get this wrong
   -- give a genuinely sequential loop (a running sum, a scan, a
   recurrence) a plain name, or vice versa -- and STADE won't error, it
   will silently generate a wrong derivative. Write each loop body so
   that iteration `i` only reads inputs and writes outputs at index (or
   indices) derived from `i`, never a value a *previous* iteration of the
   same loop wrote, unless the math genuinely requires a running/carried
   value. When it's unavoidably sequential, prefix the loop variable with
   `i_seq_` (e.g. `i_seq_k`, `i_seq_idx`); otherwise use any other valid
   identifier. Outside of this one prefix, variable and function names
   can be anything a valid Julia identifier allows -- case, underscores,
   camelCase, whatever reads best to you.

2. **Loop headers always name the iteration variable explicitly** --
   `for i = 1:n` and `for i in 1:n` are equivalent and both fine (Julia's
   parser produces the same `Expr` either way), and an unused counter can
   now be written `for _ = 1:n` if you'd rather not invent a name for it.

3. **Prefer loops over conditionals.** Keep `if`/`elseif`/`else` to the
   strict minimum the math requires, and never use `ifelse`. Concretely:
   - Replace `if`-based clipping/selection with the matching arithmetic
     function where one exists (`max`, `min`, `abs`, `sign` -- see rule 9
     for which functions are fair game).
   - Handle boundary points by splitting the iteration range into separate
     loops (or single statements) for interior vs. boundary, instead of
     branching on the index inside one loop.
   - Don't use `if` to defensively re-implement what a `for i = a:b` range
     already guarantees.
   - Genuine data-dependent branching (a condition on an independent
     input, not just an index) is fine when the math needs it -- STADE's
     activity analysis handles it -- but don't reach for it as a
     substitute for the loop-based patterns above.

4. **Don't use keyword arguments.** Every input is positional -- no `;`
   section in the signature, no `name=default` arguments. Call sites
   should never need `f(x, y; a=1.0)`; use `f(x, y, a)`.

5. **Only four variable "shapes" exist: `Float64`, `Int64`,
   `Array{Float64}`, `Array{Int64}`.** Every local variable and argument
   should be one of these (the function itself always returns nothing --
   see rule 14). In particular:
   - Don't store a comparison/condition in a variable -- write the
     comparison directly inside the `if`/`while` that needs it instead of
     assigning it to a name first. STADE's shape inference has no notion
     of a boolean value, so a stored comparison won't be classified
     correctly and could end up mistakenly tracked as differentiable.
   - Loop counters and sizes are `Int64`; physical/field quantities are
     `Float64`. A type annotation on an argument (`x::Float64`) is
     harmless if you want to write one for your own documentation -- STADE
     infers shapes from how a variable is actually used, not from any
     annotation -- but it doesn't replace this rule; the inferred shape
     still has to be one of the four above.
   - Don't introduce tuples, ranges-as-values, strings, or structs as
     local variables.
   - A kernel that "computes a scalar" writes it into a length-1
     `Array{Float64}` or `Array{Int64}` output argument rather than
     returning it -- see rule 14.

6. **Never allocate an array inside the function.** No `zeros(n)`,
   `similar(x)`, `Array{Float64}(undef, n)`, `Vector{Int64}(undef, n)`,
   array comprehensions, `collect(...)`, etc. All arrays a kernel touches
   -- inputs, outputs, and any scalar "results" boxed as length-1 arrays
   (rule 5) -- are allocated by the caller and passed in; the kernel only
   reads and writes into them in place. Plain scalar locals
   (`Float64`/`Int64`) may still be declared and initialized freely --
   this rule is about arrays only.

7. **Never index an array with another array's element inline
   (`a[b[i]]`).** Indirect/gather-scatter indexing like `y[idx[i]] = x[i]`
   or `x[perm[i]]` is not allowed -- STADE's snapshot analysis assumes
   every write's index is derivable without depending on another array's
   runtime contents, and a data-dependent index reopens aliasing
   questions it has no way to check. If an index genuinely needs to be
   looked up from an array, read it into a plain `Int64` scalar on its
   own line first, then index with that scalar:
```julia
   # not allowed:
   y[idx[i]] = x[i]
   # do this instead:
   j = idx[i]
   y[j] = x[i]
```

8. **Never use a broadcasted/dot operator (`.+`, `.-`, `.*`, `./`, `.=`,
   `sin.(x)`, etc.).** Write the explicit `for` loop over the array
   instead. This applies to assignment too: no `y .= x`.

9. **Only call Julia functions that have a direct native-Fortran-intrinsic
   counterpart -- plus any helper functions the user has defined and
   supplied.** Fair game: things like `abs`, `sqrt`, `exp`, `log`, `log10`,
   `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`,
   `mod`, `div`, `max`, `min`, `sign`, `floor`, `ceil`, `trunc` -- these all
   mirror a standard Fortran intrinsic, and each one has a matching
   derivative rule already registered in STADE. Avoid Julia-only
   conveniences that don't have a Fortran counterpart, even if they'd be
   idiomatic elsewhere: `sum`, `prod`, `map`, `filter`, `reduce`,
   `foldl`/`foldr`, `enumerate`, `zip`, `findfirst`/`findall`, `clamp`,
   `ifelse`, `any`/`all`, `cumsum`, and `Float64(...)`/`Int64(...)`
   conversion calls -- and (as already covered by other rules)
   comprehensions, broadcasting, and array-allocating functions. Calling
   something outside this list means STADE has no derivative rule for it
   and will hard-error rather than guess. If the user provides their own
   Julia functions during the conversation, those are always fine to call.

10. **Integer division always goes through `div`.** Write `div(a, b)`,
    never `a ÷ b`, `fld(a, b)`, or `Int(floor(a / b))`. STADE's shape
    inference specifically looks for a literal `div(...)` call as one of
    its two structural signals for proving a variable is `Int64`; any
    other spelling of integer division won't be recognized, and the
    variable could silently default to `Float64` instead.

11. **Every function gets a `#`-comment header describing each input --
    never a `"""..."""` docstring.** Directly above `function ...`, write
    plain `#` comment lines: one line for what the kernel computes, then
    one short `#` line per input argument explaining what it represents
    (not its type -- just its meaning, e.g. "grid spacing", "number of
    interior points", "length-1 output array holding the result"). A
    docstring isn't just a style mismatch here: Julia's parser wraps a
    docstring-preceded definition in a macro-call `Expr` rather than a
    plain `function ... end` one, which STADE's file reader doesn't
    recognize as a kernel at all -- the kernel would silently fail to
    load.

12. **Comment the algorithm as you go.** Short, concise `#` comments along
    the implementation explaining *why* a step happens (e.g. "interior
    stencil", "boundary copied through unchanged", "accumulate
    sequentially"), not comments that just restate the Julia syntax.

13. **Every function ends with an explicit `return nothing` statement.**
    Because a kernel's real output is whatever it wrote into its array
    arguments (including any length-1 arrays boxing a scalar result, per
    rule 5), there's nothing meaningful to hand back to the caller -- but
    say so explicitly. The last line of every function body must be
    `return nothing`, spelled out in full; don't leave it implicit, and
    don't shorten it to a bare `nothing`.

## What's no longer required

If you've written kernels under a stricter version of this style before,
here's what changed and why it's safe:

- **General `snake_case` on ordinary names is gone.** STADE identifies
  every variable and function purely by symbol identity -- nothing in its
  parsing, shape inference, activity analysis, or code generation reads
  case at all. The one exception is the `i_seq_` prefix in rule 1, which
  survived because it's genuinely load-bearing, not because of casing
  habits.
- **`for i = ...` vs. `for i in ...`** were never actually different to
  STADE -- Julia's own parser produces an identical `Expr` for both, so
  there was nothing to relax here in the first place.
- **A loop's counter no longer has to be a real name when it's unused.**
  `for _ = 1:n` is fine now.
- **Type annotations on arguments are allowed.** They're inert -- STADE
  infers shapes from usage, never from a declared type -- so annotate if
  it helps you or your editor, or don't.
- **Compound assignment is allowed.** `x += 3.0`, `n *= 2`, and friends
  desugar to the same thing as writing them out in full, so there's no
  reason to forbid them anymore.

## Worked example

A 1-D three-point stencil (`y[i] = x[i-1] + x[i] + x[i+1]`, with the two
boundary points just copied through), given `x`, `y`, and `n` where `y` is
caller-allocated and `n = length(x)`. The interior loop is
iteration-independent (each `y[i]` only depends on `x` at fixed offsets
from `i`), so its variable is a plain, ordinary name -- here shown with a
type annotation and mixed case to make clear both are fine:

```julia
# stencil(x, y, n)
#
# Three-point average stencil, boundary points copied through unchanged.
#
# x: input array of length n
# y: output array of length n, filled in place
# n: number of points in x and y
function stencil(x::Vector{Float64}, y, n::Int64)
    # boundary points pass through unchanged
    y[1] = x[1]
    y[n] = x[n]
    # interior points: sum of the point and its two neighbors
    nInterior = n - 2
    for idx = 2:n - 1
        y[idx] = x[idx - 1] + x[idx] + x[idx + 1]
    end
    return nothing
end
```

Why this complies: no argument is a keyword argument; `y` is allocated by
the caller, not inside `stencil`; the interior loop's iterations don't
depend on each other, so its variable (`idx`) carries no `i_seq_` prefix;
the boundary cases are handled by separate statements rather than a branch
on the index inside the loop; there's no broadcasting, indirect indexing,
or Fortran-less Base function; every variable is one of the four allowed
shapes; it has a `#`-comment header and inline comments instead of a
docstring; and it ends with an explicit `return nothing`. The type
annotations and the unused `nInterior` local (camelCase, not `i_seq_`-
prefixed since it never touches a loop) are both fine under this style.

A second example with a genuinely sequential loop -- a dot product, which
must accumulate a running sum, so its loop variable is prefixed `i_seq_`
and the scalar result is written into a length-1 output array rather than
returned. Note the compound-assignment form is now allowed:

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
        s += x[i_seq_k] * y[i_seq_k]
    end
    out[1] = s
    return nothing
end
```

Contrast with a non-compliant version -- a keyword argument, an allocated
output, `ifelse`, a branch inside the loop instead of split statements,
broadcasting, a sequential-looking accumulator loop that isn't marked with
the `i_seq_` prefix (this is the one that would silently break rather than
error), a non-Fortran function, no comment header, and returning the array
instead of `return nothing`:

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

- [ ] Every genuinely sequential loop's variable starts with `i_seq_`, and
      no non-sequential loop's variable does
- [ ] Every loop header names its iteration variable explicitly (`_` is
      fine when genuinely unused) -- `=` and `in` are both fine
- [ ] `if`/`else` only appears where the math truly has no loop/range-based
      alternative, and `ifelse` is never used
- [ ] The function signature has no `;` keyword arguments (type
      annotations are fine if present)
- [ ] No `Bool`, `String`, tuple, struct, or range is stored in a
      variable; every variable's inferred shape is `Float64`, `Int64`,
      `Array{Float64}`, or `Array{Int64}`
- [ ] No array is created inside the function body (`zeros`, `similar`,
      `Array{...}(undef, ...)`, comprehensions, `collect`, etc.) -- all
      arrays came in as arguments, including any length-1 array boxing a
      scalar result
- [ ] No `a[b[i]]`-style indirect indexing anywhere
- [ ] No broadcasted dot-operator or dot-call (`.+ .- .* ./ .= .sin. ...`)
      appears anywhere
- [ ] Every called function is either ordinary arithmetic/indexing, has a
      direct Fortran-intrinsic counterpart, or was supplied by the user
- [ ] Integer division uses `div`, never `÷`/`fld`/floor-based tricks
- [ ] The function has a `#`-comment header describing each input -- no
      `"""..."""` docstring
- [ ] The implementation has short `#` comments explaining the algorithm
- [ ] The function ends with an explicit `return nothing` as its last
      line -- not left implicit, and not shortened to a bare `nothing`
