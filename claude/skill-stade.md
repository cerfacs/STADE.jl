---
name: skill-stade
description: >
  Use whenever writing, reviewing, or refactoring a Julia numerical kernel
  meant to be fed into STADE (the automatic differentiation engine) for
  tangent/adjoint/HVP generation -- stencils, linear algebra, integrators,
  elementwise/reduction kernels, and similar monoprocessor scientific-
  computing code. Covers fusing rectangular nests of independent loops
  into one flattened loop, and the other STADE-parseability constraints
  below.
---

# skill-stade: house style for kernels STADE differentiates

STADE parses a kernel's `Expr` tree, infers every variable's shape
(`Float64` / `Int64` / `Array{Float64}` / `Array{Int64}`) from how the
kernel uses it, traces which variables reach an independent input, then
generates tangent/adjoint/HVP code from that analysis. Each rule below is
something STADE's pipeline depends on to get that analysis right, or a
real constraint of the allocation-free, single-processor kernel style.

Apply every rule below to any kernel code you write in this conversation,
even if the request does not repeat the rules. When you revise existing
kernel code, bring it into compliance instead of leaving violations in
place.

## Installing Julia in this environment

Julia is not available through apt, and JuliaLang's official binary host
(`julialang-s3.julialang.org`) is not reachable from this network. The
user mirrors the official Julia 1.10.11 (LTS) linux-x86_64 tarball as a
GitHub release asset on their own account, on hosts this network can
reach (`github.com`, `release-assets.githubusercontent.com`). Its sha256
matches the official JuliaLang checksum for
`julia-1.10.11-linux-x86_64.tar.gz`
(`fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf`).
Re-verify this checksum after download if the tarball, URL, or release
tag ever changes, because that means trusting a new artifact.

```bash
curl -sL https://github.com/luciano-drozda/julia-tar/releases/download/julia-1.10.11/julia-1.10.11-linux-x86_64.tar.gz -o /home/claude/julia.tar.gz
echo "fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf  /home/claude/julia.tar.gz" | sha256sum -c -
tar -xzf /home/claude/julia.tar.gz -C /home/claude/
export PATH="/home/claude/julia-1.10.11/bin:$PATH"
julia --version
```

If the checksum check fails, stop. Do not use the tarball. Treat it as
untrusted and tell the user, instead of installing it anyway.

For any later `bash_tool` call in this conversation that needs Julia,
add `export PATH="/home/claude/julia-1.10.11/bin:$PATH"` first.

## The rules, and why they matter

1. **Prefer loops where iteration `i` reads inputs and writes outputs only at indices derived from `i`.**
   Such loops fuse (rule 2) and run as one GPU thread per index.

   A loop stops qualifying when it **carries a value between its own
   iterations**: iteration `i` reads something an earlier iteration wrote
   at a *different* location, so the iterations only produce the right
   answer in order. A scan, a prefix sum, a recurrence like `u[i] = c *
   u[i - 1]`, an in-place Gauss-Seidel relaxation, and an outer loop
   counting sweeps of one are all of this kind. Write one only where the
   math needs it, and say so in a comment (rule 13). This is the property
   rule 2 means when it refuses to fuse such a nest.

   A **commutative accumulation is not that**, even though it too writes
   a location every iteration touches. Both `loss[1] = loss[1] + f(i)`,
   accumulating at a fixed index, and a scatter-accumulate `v[j] = v[j] +
   f(i)` whose target `j` is derived from `i`, give the same result in
   any iteration order. STADE compiles each to an atomic add, so a loop
   whose only cross-iteration coupling is one of these fuses under rule 2
   and runs one thread per index like any other. The test that matters is
   order-dependence, not whether the word "accumulator" fits.

   Reassociating a sum does change floating-point rounding, so a fused
   reduction agrees with its sequential form to within rounding rather
   than bit for bit. Where a kernel needs bitwise reproducibility instead,
   keep the loop sequential and say so in a comment (rule 13).

   `fuse_ii_loops` and the GPU codegen
   stages (`cgen_*`/`jgen_*`) each reprove, from the loop's own
   structure, which loops are safe to fuse or run in parallel.

2. **Fuse a rectangular nest of independent loops into one loop over the
   flattened range.** If an outer independent loop's body is only
   another independent loop, optionally several, nested deeper, with no
   value-carrying loop or branch between them, and the inner loop's trip
   count does not depend on the outer variable, collapse the nest into one
   loop over `1:n * m` and recover each original index with `div`/`mod`
   (rule 10).

   This step is not cosmetic. STADE can turn an outer independent loop
   into one GPU thread per outer index. Each thread then runs whatever
   is left inside that loop by itself, in sequence. Fuse first, and the
   full `n * m` range becomes independent GPU threads. Leave the nest
   unflattened, and only the outer `n` iterations become threads, each
   one running the whole inner `m`-iteration loop alone.

   Do not fuse when a loop in the nest carries a value between its own
   iterations (rule 1). Do not fuse when the inner bound depends on the
   outer variable, as
   in a triangular loop with no fixed trip count to flatten against. Do
   not fuse when a boundary/interior split (rule 4) needs to separate two
   loops in the nest. Split first, then fuse each resulting rectangular
   interior nest on its own.
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

3. **Name the loop variable explicitly.** `for i = 1:n` and `for i in
   1:n` parse to the same expression, so both are fine. An unused
   counter can be written `for _ = 1:n`.

4. **Prefer loops over conditionals.** Keep `if`/`elseif`/`else` to the
   minimum the math needs, and never use `ifelse`.
   - Replace `if`-based clipping or selection with the matching
     arithmetic function where one exists (`max`, `min`, `abs`, `sign`
     -- rule 10 lists which functions are allowed).
   - Handle boundary points with a separate loop or statement for
     interior versus boundary, instead of a branch on the index inside
     one loop.
   - Do not use `if` to check a condition a `for i = a:b` range already
     guarantees.
   - Genuine data-dependent branching, a condition on an independent
     input rather than just an index, is fine when the math needs it.
     STADE's activity analysis handles it. Do not use it as a substitute
     for the loop-based patterns above.

5. **Do not use keyword arguments.** Every input is positional. The
   signature takes no `;` section and no `name=default` argument. Write
   `f(x, y, a)`, not `f(x, y; a=1.0)`.

6. **Every variable has one of four shapes: `Float64`, `Int64`,
   `Array{Float64}`, `Array{Int64}`.**
   - Do not store a comparison in a variable. Write the comparison
     directly inside the `if`/`while` that needs it. STADE rejects a
     stored comparison at parse time, because Bool is not one of the
     four shapes, so this fails loudly rather than silently.
   - Loop counters and sizes are `Int64`. Physical or field quantities
     are `Float64`.
   - A type annotation on an argument, such as `x::Float64`, does not
     change how STADE reads the kernel: STADE infers each variable's
     shape from how the kernel uses it, then drops the annotation. But
     the annotation is still a real Julia type. Suppose it disagrees
     with how the kernel actually uses the argument, for example
     annotating a loop bound as `Float64` when the kernel uses it as an
     `Int64` index. The kernel then fails to run as ordinary Julia, even
     though STADE differentiates it without complaint. Only annotate an
     argument with the type it is actually used as.
   - Do not use a tuple, a range, a string, or a struct as a local
     variable.
   - A kernel that computes a scalar writes it into a length-1
     `Array{Float64}` or `Array{Int64}` output argument instead of
     returning it (rule 14).

7. **Never allocate an array inside the function.** No `zeros(n)`,
   `similar(x)`, `Array{Float64}(undef, n)`, `Vector{Int64}(undef, n)`,
   array comprehensions, `collect(...)`, and so on. Every array a kernel
   touches, input, output, or a length-1 array boxing a scalar result
   (rule 6), comes from the caller. A plain scalar local (`Float64` or
   `Int64`) can still be declared and set freely. This rule covers arrays
   only.

8. **Never index an array with another array's element inline
   (`a[b[i]]`).** STADE rejects this at parse time, so a gather or
   scatter index always needs the scalar-read-first form below.
```julia
   # not allowed:
   y[idx[i]] = x[i]
   # do this instead:
   j = idx[i]
   y[j] = x[i]
```

9. **Never use a broadcast operator (`.+`, `.-`, `.*`, `./`, `.=`,
   `sin.(x)`, and so on).** Write the explicit `for` loop instead. This
   includes assignment: no `y .= x`.

10. **Call only a function STADE has a derivative rule for, or one the
    user supplied.** The allowed set: `abs`, `sqrt`, `exp`, `log`,
    `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`,
    `tanh`, `mod`, `div`, `max`, `min`, `sign`, `floor`, `ceil`, `trunc`.
    Each one already has a matching derivative rule registered in STADE.

    Avoid convenience functions with no rule of their own: `sum`, `prod`,
    `map`, `filter`, `reduce`, `foldl`/`foldr`, `enumerate`, `zip`,
    `findfirst`/`findall`, `clamp`, `ifelse`, `any`/`all`, `cumsum`, and
    `Float64(...)`/`Int64(...)` conversion calls. Comprehensions,
    broadcasting, and array-allocating functions are already covered by
    other rules.

    Call anything outside this list and STADE raises a parse error
    instead of guessing a derivative. A function the user supplied
    during the conversation is always fine to call.

11. **Integer division always goes through `div`.** Write `div(a, b)`,
    never `a ÷ b`, `fld(a, b)`, or `Int(floor(a / b))`. STADE's shape
    inference treats a literal `div(...)` call as one of the signals
    that proves a variable is `Int64`. The other spellings never reach
    shape inference at all: STADE rejects `÷`, `fld`, and `Int(...)` at
    parse time, each with its own error.

12. **Give every function a `#`-comment header describing each input,
    never a `"""..."""` docstring.** Above `function ...`, write plain
    `#` lines: one line for what the kernel computes, then one short
    line per argument describing what it means, not its type.

    A docstring is not only a style mismatch here. Julia's parser wraps
    a docstring-preceded definition in a macro-call expression, not a
    plain `function ... end` one. STADE's file reader looks only for
    `function ... end` definitions, so it will not find this kernel. It
    raises a parse error such as "expected exactly one function
    definition ... found 0".

13. **Comment the algorithm as you write it.** Short `#` comments should
    explain why a step happens, for example "interior stencil" or
    "boundary copied through unchanged", not restate the Julia syntax.

14. **End every function with an explicit `return nothing`.** A
    kernel's real output is whatever it wrote into its array arguments,
    including any length-1 array boxing a scalar result (rule 6). There
    is nothing meaningful to return to the caller, but write it out in
    full. Do not leave it implicit, and do not shorten it to a bare
    `nothing`.

    A kernel can in fact omit any return statement and still parse.
    Write `return nothing` anyway: it documents, in the code itself,
    that the kernel's output lives in its arguments.

## Worked examples

**Naming and type-annotation freedom.** The interior loop below is
iteration-independent. The signature mixes a type annotation with a plain argument, and the unused
local uses camelCase, to show both are fine:

```julia
# stencil(x, y, n)
#
# Three-point average stencil, boundary points copied through unchanged.
#
# x: input array of length n
# y: output array of length n, filled in place
# n: number of points in x and y
function stencil(x::Vector{Float64}, y, n::Int64)
    y[1] = x[1]
    y[n] = x[n]
    nInterior = n - 2
    for idx = 2:n - 1
        y[idx] = x[idx - 1] + x[idx] + x[idx + 1]
    end
    return nothing
end
```

**A value-carrying loop with compound assignment.** A dot product must
accumulate a running sum, so it cannot be fused, and its result goes
into a length-1 output array.
The compound-assignment form (`+=`, `-=`, `*=`, `/=`, `^=`) desugars to a
plain assignment before STADE ever sees it, so it is allowed:

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
    # running sum -- carried from one iteration to the next
    for k = 1:n
        s += x[k] * y[k]
    end
    out[1] = s
    return nothing
end
```

`÷=`, `%=`, `\=`, and `.=` are not covered by this desugaring and remain
parse errors.

**Contrast.** A non-compliant draft: a keyword argument, an allocated
output, `ifelse`, a branch inside the loop instead of split statements,
broadcasting, a non-whitelisted function, no comment header, and
returning the array instead of `return nothing`:

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

Read the draft back and confirm all of the following, fixing anything
that fails before you share it:

- [ ] No rectangular nest of purely independent loops was left unfused:
      if the inner trip count does not depend on the outer variable, and
      no value-carrying loop or boundary split needs to separate them, it
      is one loop with `div`/`mod`-recovered indices
- [ ] Every loop header names its iteration variable explicitly (`_` is
      fine when genuinely unused). `=` and `in` are both fine
- [ ] `if`/`else` appears only where the math has no loop- or
      range-based alternative, and `ifelse` never appears
- [ ] The signature has no `;` keyword arguments (type annotations are
      fine, and must match how the argument is actually used)
- [ ] No `Bool`, `String`, tuple, struct, or range is stored in a
      variable. Every variable's inferred shape is `Float64`, `Int64`,
      `Array{Float64}`, or `Array{Int64}`
- [ ] No array is created inside the function body (`zeros`, `similar`,
      `Array{...}(undef, ...)`, comprehensions, `collect`, and so on).
      Every array came in as an argument, including any length-1 array
      boxing a scalar result
- [ ] No `a[b[i]]`-style indirect indexing appears anywhere
- [ ] No broadcast operator or dot-call (`.+ .- .* ./ .= .sin. ...`)
      appears anywhere
- [ ] Every called function is either ordinary arithmetic/indexing, one
      of the whitelisted intrinsics from rule 10, or supplied by the
      user
- [ ] Integer division uses `div`, never `÷`/`fld`/floor-based tricks
- [ ] Any compound assignment is `+=`, `-=`, `*=`, `/=`, or `^=` only.
      `÷=`, `%=`, `\=`, and `.=` never appear
- [ ] The function has a `#`-comment header describing each input, and
      no `"""..."""` docstring
- [ ] The implementation has short `#` comments explaining the
      algorithm
- [ ] The function ends with an explicit `return nothing` as its last
      line, not left implicit and not shortened to a bare `nothing`