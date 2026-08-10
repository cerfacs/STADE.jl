# Implementation plan: `keep_push_pop=false`

Array-indexed snapshot storage for GPU-portable adjoint/HVP code in STADE.jl

## 0. Who this is for, and what you have

This plan is meant to be actionable without re-deriving anything from
scratch. It assumes you have:

- The current `STADE.jl` (the file with `parse_`/`shape_`/`der_`/`emit_`/
  `act_`/`snap_`/`lin_`/`inl_`/`tgen_`/`agen_`/`hvp_`/`cgen_`/`jgen_`/
  `val_`/`io_`/`stade_` prefixed functions, and its own `skill-stade.md`
  governing the coding conventions below — read that file's rules before
  touching anything, they are not optional style preferences).
- `validate_corpus.jl` and the `val-corpus/` golden corpus (21 kernels).
- Three reference files produced by hand during the design of this
  feature, which should travel with this plan: `advection_b_arr.jl`,
  `branchsel_b_arr.jl` (hand-written array-indexed rewrites of two real
  corpus adjoints, proven bit-for-bit numerically identical to the
  originals), and `counter_vs_index.jl` (a minimal executable proof that
  a running-counter-based index is *not* sufficient — see §2).
- GPU hardware and job-launching access that the author of this plan did
  not have. Use it — §7 is written specifically around what only you can
  check.

Everything below was derived by reading STADE.jl's actual current code
(not guessed), and by generating and hand-testing real output from it.
Line numbers will have drifted by the time you read this; function names
and the section banners they live under should not have.

## 1. What problem this solves, in one paragraph

`agen_*` (adjoint) and `hvp_*` (Hessian-vector product) generate
checkpointing code: whenever a value needs restoring in the backward
sweep, the forward sweep `push!`s it onto a growable `Vector`, and the
backward sweep `pop!`s it back off in LIFO order (the backward sweep's
own loops are reversed specifically to make this line up). This is
correct and already validated against finite differences for the whole
corpus. It is also fundamentally inexpressible inside a GPU kernel: no
device backend can call `push!`/`pop!` on a `Vector`, and even if they
could, LIFO order across thousands of concurrently-scheduled,
unordered-execution GPU threads isn't a coherent concept. `cgen_*`
currently handles this by refusing to split any loop that contains a
`push!`/`pop!` anywhere in its body (`cgen_contains_stackop`), which is
correct but leaves real parallelism on the table — confirmed directly:
`advection_b` (the real corpus adjoint) currently splits into 2 GPU
kernels; a hand-rewritten version with the stack replaced by direct
array indexing splits into 4, using `cgen_*` completely unmodified.

The goal of this work is to make `agen_*`/`hvp_*` able to generate that
array-indexed form directly, gated behind a `keep_push_pop::Bool` kwarg
(default `true`, preserving all current behavior) on the public
generator entry points.

## 2. The alternative that looks simpler and isn't: read this before starting

It's tempting to replace `push!(stack, v)` / `pop!(stack)` with a
manually-incremented counter (`stack[idx+=1] = v` / `v = stack[idx];
idx -= 1`) instead of deriving a real position from the enclosing loop
structure. This *is* simpler to generate — it needs no trip-count
bookkeeping, no per-site offsets, nothing structural. **Do not do this.**
It doesn't solve the problem: the counter's value at any point is still
exactly as history-order-dependent as `push!`/`pop!` was, so it's just
as unsafe to parallelize, and `cgen_device_assign`'s current
atomic-detection logic (see §6) would actually fail to catch this and
silently emit a race condition rather than correctly refusing to split
the loop. `counter_vs_index.jl`, included with this plan, demonstrates
this concretely and executably: given a save pass and a restore pass
that visit the same 6 indices in different (both entirely legitimate)
orders — which is exactly what an unordered GPU scheduler is free to
do — the counter-indexed version restores `[6.0, 4.0, 5.0, 2.0, 1.0,
3.0]` instead of the correct `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`; the
position-derived index restores correctly regardless of order. Run it
yourself before starting if you want to see it firsthand.

The index must be a pure function of the enclosing loop variables (or a
constant, for a non-loop site) — never anything that accumulates across
execution order.

## 3. Scope of the `keep_push_pop` kwarg

Add `keep_push_pop::Bool = true` to:

- `stade_adjoint`, `stade_adjoint_file`, and whatever `stade_adjoint`
  multi-kernel/corpus sibling exists (check current `stade_*` section —
  it may be named `stade_adjoint_corpus` or similar).
- `stade_hvp`, `stade_hvp_file`, and its corpus sibling.

Add it, accepted but ignored, to:

- `stade_tangent`, `stade_tangent_file`, and its corpus sibling — tangent
  mode never emits `push!`/`pop!` at all (confirmed: `tgen_*` gives every
  active statement a shadow line directly, no stacks), so this is a pure
  interface-consistency no-op. Document it as such in the docstring so a
  caller iterating uniformly over all three modes doesn't need to
  special-case tangent.

Do **not** add it to `stade_cuda(_file)` / `stade_amdgpu(_file)` /
`stade_metal(_file)` / `stade_jacc(_file)`. Those consume whatever `.jl`
file already exists on disk; they have no opinion on how it was
produced. A file generated with `keep_push_pop=false` simply has no
`push!`/`pop!` left in it, and `cgen_*`'s existing, *unmodified*
`cgen_contains_stackop` check will correctly find nothing to exclude —
verified directly (§6). No cgen_/jgen_ code needs to change for this
feature at all.

## 4. Architecture: parameterize the emission primitive, don't rewrite the traversal

`agen_forward_body`/`agen_backward_body` already do all the hard,
already-validated work: deciding which statements need a snapshot (via
`snap_*`, upstream and unaffected by this change), walking the
statement tree in the correct nesting order, and reversing backward
loops correctly (`agen_body_has_snapshot`/`agen_negate_step`). The only
places that literally construct `Expr(:call, :push!, ...)` / `Expr(:(=),
lhs, Expr(:call, :pop!, ...))` are two call sites inside
`agen_forward_body` and `agen_backward_body`. The recommended design —
by direct analogy with how `cgen_*` parameterizes vendor differences
through a `gpu_backend` value while keeping `cgen_body`'s traversal
completely backend-agnostic — is to extract those two call sites into a
small pair of helpers and parameterize *only those* by a strategy, e.g.:

```
agen_emit_push(stack_name::Symbol, value, ctx) -> Expr
agen_emit_pop(stack_name::Symbol, ctx) -> Expr   # returns the rhs expr; caller wraps the lhs assign
```

where `ctx` carries whatever the `:indexed` strategy needs to compute a
position (the currently-enclosing loop variables/bounds, and the site's
static base offset within its stack — see §5). `ctx` is exactly the
kind of small, purely-local, thread-through-the-recursion value
`cgen_body`'s own `owner`/`kernels` arguments already model — build it
the same way, don't invent a different pattern.

This keeps the diff small and isolated to the two emission sites, and
means every already-validated piece of `agen_*`'s planning and
traversal logic is untouched and doesn't need re-validating from
scratch — only the new strategy's own output needs new validation.

**Naming/prefix decision, left to you once you're looking at the real
current code**: whether these helpers should live under the `agen_`
prefix (recommended default — this is a mode of `agen_`'s own emission,
not a separate pipeline stage) or deserve a new prefix. If you add a new
prefix, it must be added to `skill-stade.md`'s prefix list (see the
existing entries for `cgen_`/`jgen_` for the documentation pattern to
follow), same as every other stage in this file's history.

## 5. Stack-size derivation at generation time

This is the part that actually requires new logic, and it splits into
two tiers of difficulty. **Implement and fully validate Tier A first, as
a complete, shippable unit, before starting Tier B.**

### Tier A: closed-form sizes (the common case)

For each snapshot site, define its **local multiplicity**: the product
of `cgen_trip_count(lo, step, hi)`-style trip-count expressions of every
loop enclosing the statement that triggers the site, from outermost to
innermost, or the literal `1` if the site isn't inside any loop. This is
computable by threading one more accumulator through the *same*
recursive walk `snap_*`'s site-selection (or a mirror of it) already
does — multiply into a running product on descent into a `:for`, divide
back out on the way out (or just push/pop it on a small stack, cheaper
and clearer).

Multiple sites can share one stack — confirmed directly from
`agen_site_stack_name`: one stack per variable name for `:value`/`:array`
sites, and a *single* kernel-wide `branch_stack` (all `:if`s) and
`tripcount_stack` (all loop-bound snapshots) pooled across the entire
kernel. A stack's total size is the **sum** of the local multiplicities
of every site that maps to it — group the `(stack_name,
local_multiplicity)` records the walk produces, by name, and sum. Fold
to a single term when only one site maps to a stack (mirror the pattern
`cgen_sum_excluding` already uses for "0/1/n terms" — don't emit a
degenerate `+()` call).

Each site also needs a **static base offset** within its (possibly
shared) stack: process sites destined for the same stack in a fixed,
deterministic order (declaration order is simplest and matches how
`agen_stack_names`/`agen_stack_map` already enumerate them) and assign
each one an offset equal to the running sum of the multiplicities of the
sites processed before it. This offset is a compile-time-*constructed*,
runtime-*evaluated* expression (since trip counts are runtime values
like `n`), computed once — it is not a mutable counter (see §2); it's a
fixed formula baked into the generated code at both the push site and
the matching pop site for that one syntactic location.

The actual index at a given site = `base_offset + local_position`, where
`local_position` is the same kind of row-major flattening
`cgen_loopvar_from_tid` already computes elsewhere: for enclosing loops
L1 (outermost) ⊃ L2 ⊃ ... ⊃ Lk (innermost, directly containing the
site), with position-within-loop `pos(Li) = (var_i - lo_i) / step_i + 1`
for each,
```
local_position = pos(L1) * (tripcount(L2) * ... * tripcount(Lk))
                + pos(L2) * (tripcount(L3) * ... * tripcount(Lk))
                + ...
                + pos(Lk)
```
i.e. exactly standard row-major multi-dimensional indexing, one term per
nesting level. For the single-loop case (the common one — confirmed for
`advection`, `branchsel`, `ttgc`) this degenerates to
`local_position = i_x - lo + 1` (or just `i_x`, when `lo=1, step=1`),
matching `advection_b_arr.jl`'s hand-derived
`(i_seq_ - 1) * n_inner + (i_x - 1)` exactly (`n_inner` there is
`tripcount(inner)`, `(i_seq_ - 1)` is `pos(outer)`).

**Critical correctness invariant to test explicitly**: the *same* base
offset and the *same* index formula must be used at both the forward
(push→store) site and the matching backward (pop→load) site for one
syntactic location. Compute each site's offset once, store it (in
whatever plan/context structure carries site information from the
sizing pass through to both `agen_forward_body` and
`agen_backward_body`), and reuse it — never recompute independently in
the two places, or a drift between them is a silent wrong-answer bug
that finite-difference validation may or may not catch depending on
problem size.

`initstacks_*`'s signature must grow to accept whatever kernel arguments
the size expressions reference (today it takes zero arguments for
adjoint mode). Recommend computing the *minimal* referenced set (a
free-variable collection over the size expressions, structurally like
`cgen_free_vars` but simpler — no loop-variable or locally-assigned-
scalar exclusion needed, since skill-jade loop bounds can only reference
arguments/constants/other loop-invariant expressions, enforced already
by `parse_check_expr`) rather than the full original argument list —
consistent with how the rest of this codebase treats "pass only what's
read." This is a genuine judgment call with a defensible alternative
(just reuse the full primal argument list, simpler but passes unused
array arguments into a function that never touches them); pick one and
document the choice in `skill-stade.md`.

Each stack's element type is unchanged from today (`Float64` for
`:value`/`:array`, `Int64` for `:branch`/`:tripcount`) — only the
allocation changes, from `Vector{T}()` to `Vector{T}(undef, size_expr)`.

**Test kernels for Tier A** (chosen because they're already hand-solved
and numerically verified — use them as the primary correctness oracle
before trusting anything else): `advection.jl` (nested loop, single
site) and `branchsel.jl` (non-loop, single site, no signature change
needed at all). Your auto-generated `advection_b.jl`/`branchsel_b.jl`
under `keep_push_pop=false` should be semantically equivalent to
`advection_b_arr.jl`/`branchsel_b_arr.jl` (same index formulas, up to
naming) — diff them, and additionally re-run the bit-for-bit numerical
comparison methodology those hand-written files were checked with
(fresh copies of every mutated array per run, compare outputs across
many random seeds).

### Tier B: ragged/data-dependent trip counts (confirmed real, `mg_vcycle`)

Do not assume this doesn't occur in practice — it does, confirmed by
generating `mg_vcycle`'s actual adjoint and inspecting it directly. In
`mg_vcycle.jl`:
```julia
for i_seq_level = 1:num_levels - 1
    n = nl - 1
    ...
    ncg = div(nl, 2)
    ...
    nl = ncg
end
```
`n` — which several inner loops use as a bound — is reassigned every
outer sequential iteration, following a halving sequence seeded from
`nfine`/`num_levels`. It is not expressible as a single closed-form
product of top-level kernel arguments the way Tier A assumes. The
generated adjoint has 11 separate stacks with sites inside this nest
(`u_stack`, `f_stack`, `left_stack`, `right_stack`, `branch_stack`,
`tripcount_stack`, and others — regenerate `mg_vcycle_b.jl` yourself to
see the full list).

The sequence of `n`/`nl` values is still fully deterministic and
knowable before the main forward sweep runs — it depends only on
`nfine`/`num_levels`, never on the numerical work itself — so this is
not fundamentally unsolvable, just harder: `initstacks_*` needs a small
loop of its own that **replays only the scalar bookkeeping that
determines the ragged bound** (a slice of the original code — just
`n = nl - 1; ...; nl = div(nl, 2)`, none of the actual array arithmetic)
and accumulates a running total as it goes, rather than allocating from
one precomputed expression.

Detecting when Tier B applies: a loop's bound-determining symbol is ever
an assignment target inside an *ancestor* sequential loop. This is a
checkable static condition over the parsed kernel body, not a
heuristic.

**Recommended sequencing**: ship Tier A first, with Tier B kernels
(`mg_vcycle.jl`, `mg_vcycle_multi.jl` — the two known instances) failing
**loudly and clearly** under `keep_push_pop=false` (e.g. `error("...
ragged/data-dependent loop bound not yet supported by
keep_push_pop=false, see mg_vcycle in skill-stade.md")`), never silently
producing a wrong or under-sized buffer. This matches the fail-loud
posture used everywhere else in this codebase (`cgen_ingest`'s combined
error message is the model to follow) and lets Tier A ship and validate
independently, with Tier B as a distinct, separately-scoped follow-up.

## 6. What `cgen_*` needs — nothing, but verify this claim, don't just trust it

Confirmed directly, not assumed: run `stade_cuda` (completely
unmodified) against both the real `advection_b` (push!/pop!-based) and
the hand-written `advection_b_arr` (indexed) — the former splits into 2
kernels, the latter into 4. `cgen_contains_stackop` correctly finds
nothing once there are no more `push!`/`pop!` calls anywhere (with
`keep_push_pop=false` used throughout, this check becomes vacuously
true everywhere in the generated file — the stack-safety exclusion rule
has nothing left to exclude), and `cgen_device_assign`'s existing
occurs-check correctly recognizes that an indexed stack write/read's
position *does* depend on the loop's own thread-mapped variable (by
construction — that's exactly what §5 builds), so it's emitted as a
plain, non-atomic, race-free write. Re-run this exact comparison
yourself for every Tier A kernel as part of validation, not just
`advection`.

There is one latent gap in `cgen_device_assign` worth hardening as cheap
insurance before relying on this more heavily (found while designing
this feature, independent of it, but this work increases the chance of
tripping it if Tier A/B sizing ever has a bug):

```julia
if stmt.lhs isa Expr && stmt.lhs.head == :ref && !cgen_expr_contains(stmt.lhs.args[2:end], thread_var)
    terms = cgen_flatten_sum(stmt.rhs)
    self_idx = findfirst(t -> t == stmt.lhs, terms)
    if self_idx !== nothing
        # ... atomic ...
    end
end
return Expr(:(=), stmt.lhs, stmt.rhs)   # plain write — reached whenever self_idx is nothing
```

A write whose index does *not* depend on the parallelized loop's thread
variable, and which is *not* an accumulation pattern (`x[k]=x[k]+v`), is
currently treated as an ordinary safe plain write. It isn't safe — any
thread-invariant-indexed write from a parallel loop is a race regardless
of whether it's additive. This has never been hit in practice (every
corpus case tested so far that had a thread-invariant index was
additive), but a bug in Tier A/B's offset arithmetic — an offset that
should depend on the outer loop variable but doesn't, say — would
produce exactly this pattern, and today it would be silently
parallelized wrong rather than caught. Recommend changing the condition
to treat *any* thread-invariant-indexed write as needing protection
(atomic, or an explicit refusal to split), not only additive ones, as a
defensive measure before `keep_push_pop=false` sees real use. This is a
small, independent, low-risk change to `cgen_*` alone — do it separately
from, and can be sequenced either before or after, the main Tier A/B
work.

## 7. `hvp_*`: likely far less work than it looks, but verify before assuming

`hvp_*` does not call `agen_forward_body`/`agen_backward_body`'s
internal Julia logic again — it works by syntactically transforming the
**already-generated adjoint `Expr`** that `agen_*` produced
(`hvp_double_body`/`hvp_double_stmt`, "differentiate a statement list
agen_ already produced"). This matters a lot for scoping this task:

- `hvp_double_stmt` has an explicit special case matching
  `Expr(:call, :push!, stack, val)` syntactically, and produces a
  doubled push onto `shadow_of[stack]` alongside the original.
- `hvp_tangent_expr` has an explicit special case matching
  `Expr(:call, :pop!, stack)`, producing `pop!(shadow_of[stack])`.

Once `agen_*` (§4–§5) emits array-indexed stores/loads instead, a push
becomes an ordinary `Expr(:(=), Expr(:ref, stack, idx), val)` and a pop
becomes an ordinary `Expr(:ref, stack, idx)` read. **Both of these
already have fully generic handling in the current code**, independent
of the two push!/pop!-specific special cases:

- `hvp_double_stmt`'s generic `e.head == :(=)` branch already calls
  `hvp_shadow_lvalue(lhs, shadow_of)`, which (per its own docstring)
  already shadows "a bare Symbol to a bare Symbol; an array-ref to the
  same indices on the shadow array" — exactly what an indexed push
  needs, with no new code.
- `hvp_tangent_expr`'s generic `expr.head == :ref` branch already
  produces `Expr(:ref, shadow_of[expr.args[1]], expr.args[2:end]...)` —
  exactly what an indexed pop needs, with no new code.

So the hypothesis to verify first, before writing anything new for
`hvp_*`: **`:indexed` mode may already work correctly for `hvp_*` with
zero code changes**, and the two existing push!/pop!-specific branches
simply become dead code (never matched, since there are no more literal
`push!`/`pop!` calls to find) rather than something that needs updating.
Test this directly on `advection.jl` (via `stade_hvp`/`stade_hvp_file`
with `keep_push_pop=false`) before assuming otherwise.

What almost certainly *does* need updating, independent of the above:
`hvp_shadow_stack_inits` allocates each shadow stack (`du_stack_d`
alongside `du_stack`) as its own `Vector{Float64}()` — this needs the
same Tier A/B sizing treatment as `agen_init_emit`'s primal-stack
allocation, using the same size expressions (a shadow stack is exactly
as large as its primal counterpart). Confirm where `hvp_shadow_stack_inits`'s
output actually lands in the generated file (folded into the main `_hv`
function body, or its own companion `initstacks_*_hv` function — this
determines whether it needs its own signature change or can reuse
`agen_init_emit`'s) before implementing.

## 8. A real interaction to resolve, not defer silently

Once a stack becomes a genuine indexed array, and the loop using it
becomes GPU-splittable, that array becomes a candidate device-kernel
argument via `cgen_free_vars` — exactly as intended, since that's the
entire point of this work. For the kernels tested so far (`advection`)
this is clean: `du_stack` is used only inside the two loops that *both*
become splittable, so it only ever needs to be a device array, never
also indexed from surviving host-only code in the same function.

This won't hold in general. If a stack is shared across multiple sites
(§5's per-variable/kernel-wide sharing), and some of those sites end up
inside loops that remain host-only for unrelated reasons (a genuinely
sequential `i_seq_` loop, for instance — those never become
GPU-splittable, sizing or no sizing) while others end up in loops that
do get split, the *same* buffer needs to be both a CuArray-style
argument to a device kernel and directly, scalar-indexed on the host
within the same function. That collides with `CUDA.allowscalar(false)`/
`AMDGPU.allowscalar(false)`/`Metal.allowscalar(false)` in exactly the
way already flagged as an open, unrelated, pre-existing gap in
`skill-stade.md` (search it for "allowscalar" — a `cgen_`-generated host
function mixing a device-kernel-argument use and a host-scalar-indexed
use of the same array argument in one function). This work makes that
gap concretely reachable for the first time, rather than only
theoretical. Recommend resolving that gap as a prerequisite or
co-requisite of shipping `keep_push_pop=false` broadly, rather than
discovering it via a confusing runtime failure on whichever corpus
kernel first happens to trigger it. Check which corpus kernels (beyond
`advection`) actually hit this before deciding how urgent it is — it may
turn out to be rare in practice, or it may not.

## 9. Validation plan

### Phase 1 — no GPU required (do this first, it's the primary correctness oracle)

1. Run `validate_corpus.jl`'s existing finite-difference checks with
   `keep_push_pop=false` for adjoint and hvp generation, across the
   whole corpus except the confirmed Tier B kernels
   (`mg_vcycle.jl`/`mg_vcycle_multi.jl`, expected to error loudly per
   §5 until Tier B lands). All of these should still pass — this proves
   the transform preserves numerical correctness, generalizing the
   bit-for-bit check already done by hand for `advection`/`branchsel` to
   the entire corpus, automatically.
2. For every kernel where `keep_push_pop=false` succeeds, run the
   resulting `_b.jl`/`_hv.jl` through `stade_cuda_file` (or another
   backend) unmodified, and confirm it produces **strictly more or
   equal** split kernels than the `keep_push_pop=true` version of the
   same file did. `advection`: 2 → 4 is the known baseline; every other
   kernel should show an equal-or-larger jump, never smaller.
3. Diff the auto-generated `advection_b.jl`/`branchsel_b.jl` (under
   `keep_push_pop=false`) against `advection_b_arr.jl`/
   `branchsel_b_arr.jl` — should match up to naming/cosmetic
   differences, since those were hand-derived using exactly the
   algorithm in §5.
4. Confirm `keep_push_pop=true` output is byte-identical to what it was
   before this work started, for every kernel — this is a pure
   backward-compatibility check and should never fail.

### Phase 2 — requires real GPU hardware (this is the part only you can do)

5. Take at least one Tier A kernel's `keep_push_pop=false` adjoint
   (start with `advection` — cleanest, already hand-verified), run it
   through `stade_cuda_file` (or whichever backend your hardware
   actually is — most likely CUDA on a cloud GPU box), and **actually
   compile and execute** the resulting kernels with concrete input data.
6. Compare the GPU-computed gradient against the CPU-computed one
   (either `keep_push_pop=true` or `=false` on CPU — Phase 1 already
   proved these agree) — should match within the tolerance appropriate
   to whatever precision was used (see `stade_gpu`'s `precision` kwarg
   from the existing `cgen_*` work if you want to test `Float32`
   specifically).
7. Time it: compare wall-clock between the GPU-split version and the
   pure-CPU version at a large problem size. This is the first real
   signal on whether the whole effort delivers actual speedup, which
   nothing prior to this point in the project has been able to measure
   at all (no GPU hardware was available when `cgen_*`/`jgen_*` were
   built or when this feature was designed — every check up to Phase 1
   here has necessarily been structural, "does it parse," never
   numerical-on-real-hardware).
8. Repeat 5–7 for a Tier B kernel once §5's Tier B work lands.
9. Repeat 5–7 across whichever of CUDA/AMDGPU/Metal/JACC your hardware
   can actually run — note explicitly in your results which backends
   got real execution testing versus which remain structural-only
   (same distinction the project has maintained throughout; don't blur
   it).

## 10. Suggested phased task breakdown

1. Tier A sizing + `:indexed` emission strategy for `agen_*` (adjoint)
   only; `hvp_*` deferred.
2. Add `keep_push_pop` kwarg to `stade_adjoint`/`stade_adjoint_file`
   (+ corpus sibling), default `true`, wired to the new strategy.
3. Phase 1 validation (§9.1–9.4) for adjoint mode, corpus minus Tier B
   kernels, which should fail loudly per §5.
4. Investigate and most likely confirm the §7 hypothesis for `hvp_*`
   (may require zero new code beyond `hvp_shadow_stack_inits`'s sizing);
   add the `keep_push_pop` kwarg to `stade_hvp`/`stade_hvp_file`.
5. Phase 1 validation for hvp mode, same corpus subset.
6. Phase 2 GPU execution validation (§9.5–9.7), at minimum for
   `advection`.
7. *(separate follow-up, not blocking the above)* Resolve the §8
   host/device array-crossing interaction.
8. *(separate follow-up)* Tier B ragged-bound sizing for
   `mg_vcycle`/`mg_vcycle_multi`.
9. *(separate follow-up)* Harden `cgen_device_assign`'s atomic-detection
   heuristic per §6.
10. Update `skill-stade.md` throughout — new frozen shapes if any,
    the `keep_push_pop` kwarg's scope (§3), the Tier A/B distinction,
    and any newly-adopted prefix, following the documentation pattern
    every other addition in this file's history has used. Do this as
    you go, not as a single pass at the end — every prior stage of this
    project was documented at the same time it was built.

## 11. Definition of done

- `keep_push_pop::Bool = true` exists on `stade_adjoint`/`_file`/
  (corpus sibling) and `stade_hvp`/`_file`/(corpus sibling); accepted
  and documented as a no-op on the `stade_tangent` family.
- `keep_push_pop=false` output contains zero `push!`/`pop!` calls for
  every Tier A kernel, and fails with a clear, specific error (not a
  wrong answer, not a crash with an unrelated message) for Tier B
  kernels until §5's Tier B work lands.
- Every corpus kernel's existing finite-difference validation still
  passes under `keep_push_pop=false` (Tier A kernels) or is explicitly,
  loudly excluded with a documented reason (Tier B kernels, pending).
- `keep_push_pop=true` behavior is provably unchanged (byte-identical
  output) for every kernel.
- `stade_cuda_file` (and friends), run unmodified against
  `keep_push_pop=false` output, splits strictly more (or equal) loops
  into GPU kernels than against `keep_push_pop=true` output, for every
  kernel tested.
- At least one kernel's GPU-split kernels have been compiled and
  executed on real hardware, with results matching the CPU reference
  numerically.
- `skill-stade.md` reflects all of the above, in the same style and
  location conventions as every prior addition to this project.