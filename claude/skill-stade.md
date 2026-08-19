---
name: skill-stade
description: >
  Use whenever writing, reviewing, or refactoring Julia code that is part
  of STADE itself (the automatic differentiation engine for skill-jade
  kernels) — as opposed to the numerical kernels STADE differentiates,
  which are governed by skill-jade instead. Trigger any time the user
  asks for code implementing or modifying a STADE pipeline stage (parse,
  shape inference, activity analysis, snapshot/TBR analysis, derivative
  rules, linearization, tangent or adjoint codegen, file I/O, validation),
  even without naming "skill-stade" or STADE explicitly.
---

# skill-stade: house style and architecture contract for STADE

STADE is a single-file (`STADE.jl`), function-only Julia AD engine for
skill-jade-compliant kernels. This skill encodes the architecture
contract that keeps independently-developed pieces mergeable.

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

## Hard rules

1. **No `module` blocks.** Everything lives at top level in one file,
   organized by `# ==== prefix_* ====` comment banners, not namespaces.
2. **No `struct` or `@enum` definitions.** Records are plain
   `NamedTuple`s with a documented, fixed key set (see "Frozen data
   shapes" below). Open-ended key sets (argument names, a rule table)
   use `Dict`, not `NamedTuple`.
3. **No global `const` (or any top-level mutable/shared state).** Any
   lookup table a function needs (derivative rules, emit templates)
   is built fresh *inside* the function that uses it and returned —
   never registered into module-level state.
4. **Every function name starts with its stage's prefix**:
   `parse_`, `shape_`, `der_`, `emit_`, `act_`, `snap_`, `lin_`,
   `tgen_`, `agen_`, `hvp_`, `val_`, `io_`, `stade_`. `io_` is the only
   prefix permitted to touch the filesystem (open/read/write) —
   every other stage, including `stade_` itself, operates purely on
   in-memory `Expr`/`NamedTuple` values. This prefix rule is the
   entire substitute for module-based namespacing — treat a name
   collision across prefixes as a bug to fix before merging.
5. **Frozen data shapes** (do not add/rename/remove keys without
   flagging it as a breaking-change event across every stage that
   touches the shape):
   - kernel: `(sig=..., body=...)`
   - kernel sig: `(name, args, kinds::Dict{Symbol,Symbol}, independents, dependents)`
     — `kinds` values are one of `:scalar_float :scalar_int :array_float :array_int`
     — `independents`/`dependents` are auto-derived (every float-kinded
       arg) by `parse_kernel`; never required from the caller
   - statement: `(kind=:assign|:for|:if, ...)` — see spec doc for per-kind keys.
     `:while` is intentionally not supported yet — do not add it
     speculatively; wait for a concrete kernel that needs it.
   - snapshot site: `(kind=:value|:array|:branch|:tripcount, array, at)`
   - linearization node: `(op, args, darg_exprs)`
6. **Expressions are raw Julia `Expr`/`Symbol`/`Number` — never a
   custom expression type.** Validation/normalization happens as
   functions over `Expr`, not translation into a parallel AST.
7. **Every stage function is pure** given its documented inputs — no
   hidden reliance on another stage's internals, only on the frozen
   shape its input/output is documented to have (except `io_*`, whose
   entire job is the filesystem side effect). This is what lets a
   contributor build `agen_*` against a hand-written fake `snap_plan`
   output before `snap_*` itself exists.
8. **Comments are short and sparse.** One line, only where the code
   isn't self-explanatory (a non-obvious invariant, a deliberate
   deviation from the obvious approach). Don't restate what the code
   already says. Longer design rationale belongs in the conversation
   / spec doc, not inline in STADE.jl.
9. **STADE.jl contains no corpus-specific code.** Nothing in it may
   reference any particular kernel inside `val-corpus-*` by name.
10. **A loop nest is whatever `for` structure the kernel source actually
   contains — never "un-fuse" it back into synthetic sub-loops.**
   skill-jade fuses a rectangular nest of iteration-independent loops (see
   skill-jade rule 2) into one `for idx = 1:n * m` loop with the original
   indices recovered inside the body via `div`/`mod`, rather than writing
   literal nested `for`s. Every stage that walks loop nests for indexing or
   tripcount purposes (`snap_*`'s `pos0`/`stride` math in the `:indexed`
   strategy, `agen_tier_b_walk`'s ancestor-loop check, GPU split counting
   in `stade_cuda`) must treat a fused single loop as a depth-1 nest with
   tripcount `n * m` — its `pos0`/`stride` formula already reduces to the
   depth-1 case (`stride(L1) = 1`) with no special-casing needed. Do not
   add logic that pattern-matches `div(idx - 1, m)`/`mod(idx - 1, m)`
   inside a loop body to reconstruct a synthetic multi-level nest; the
   closed-form Tier A formula is defined over the *actual* enclosing `for`
   statements a snapshot site sits inside, and a fused loop's body-level
   `div`/`mod` expressions are ordinary assignment statements to that
   formula, not loop structure.

## `keep_push_pop`: the `:indexed` snapshot-storage strategy

`stade_adjoint`/`stade_hvp` (and their `_file`/`_corpus` wrappers) take a
`keep_push_pop::Bool = true` keyword. `true` is the original behaviour
(unchanged, byte-identical output — verified against the pre-feature
codebase across the whole corpus). `false` replaces every snapshot
stack's `push!`/`pop!` pair with a direct write/read into a pre-sized
`Vector`, indexed by a compile-time-derived, runtime-evaluated position —
this is what makes GPU code generation for these push/pop sites
*possible at all*: a shared mutable stack pointer has no meaning once a
loop is split across independent GPU threads (`counter_vs_index.jl`
demonstrates concretely why a running counter isn't a valid substitute
either — a GPU launch doesn't guarantee threads visit indices in any
particular order, so only a closed-form, thread-local index works).

**Tier A (implemented):** a snapshot site's index is computable in
closed form from kernel arguments and its own enclosing loop nest alone.
For a site inside loop nest `L1..Lk` (outermost to innermost), with
`pos0(Li) = (Li.var - Li.lo) / Li.step` (0-based) and
`stride(Li) = product of tripcount(Lj) for j > i`:

```
local_position = 1 + sum_i pos0(Li) * stride(Li)
index = base_offset + local_position
```

`base_offset` for one occurrence = the sum of the *local multiplicities*
(product of enclosing-loop tripcounts, or 1 if none) of every earlier
occurrence sharing its stack, in kernel-source order — i.e. stacks are
still shared across sites the same way they are today (`agen_site_stack_name`
unchanged), just laid out as consecutive fixed-size blocks instead of a
LIFO stream. A stack's total size is the sum of all its occurrences'
multiplicities (folded to a single term, no degenerate `+()`, when only
one site maps to it). `initstacks_*`'s signature grows to accept the
*minimal* free-variable set the size expressions actually reference
(built via the same free-variable collection every other sizing helper
in this file uses) — this is the one unavoidable calling-convention
change, verified against a hand-derived reference
(`advection_b_arr.jl`, multi-site/looped case; `branchsel_b_arr.jl`,
single unlooped site, no signature change needed at all).

Matching a specific push site to its corresponding pop site is done by
a **structural key**, not a running counter: `agen_site_key(body, idx[, bv])`
= `(objectid(body), idx, bv_or_nothing)`, where `body` is the *kernel.body-side*
`Vector` containing the statement and `idx` its position in it.
`agen_backward_body` threads a `primal_body` argument (kernel.body's own
matching sub-`Vector`, passed down structurally parallel to `lin_plan`)
purely so it can reconstruct this same key at the corresponding backward
site — deliberately *not* relying on any assumption about generation-time
call ordering between forward and backward (branch-scalar hoisting
already reorders pops relative to a naive full reversal; a counter-based
scheme breaks silently under that reordering, a running-counter approach
was tried and rejected during design for exactly this reason).
`agen_local_position` is recomputed fresh from the current `loop_ctx` at
both the push and pop call sites rather than cached, since `loop_ctx`
is guaranteed structurally identical at both (same `(var,lo,hi,step)`
values from the same statement, regardless of which direction the
generated loop actually iterates at runtime).

Branch-scalar hoisting's discard-pop (the `__snap_discard` lines) is
*dropped entirely* under `:indexed` mode — it exists only to keep a
shared LIFO pointer in sync, which is meaningless once storage is
randomly addressable. The corresponding forward-written slot is simply
never read back — a harmless over-snapshot, the same philosophy `snap_*`
already applies to its own iteration-independent elision.

**Tier B (implemented — closed-form, GPU-eligible ragged-block tables,
with a narrow `push!`/`pop!` fallback for what still can't resolve):**
a loop whose own bound-determining symbol is ever an assignment target
inside an *ancestor sequential loop* has a trip count that isn't a
closed-form function of kernel arguments (`mg_vcycle`/`mg_vcycle_multi`,
whose inner loop bounds derive from `nl`, reassigned once per outer
sequential `i_seq_level` iteration, are two corpus instances; the
dedicated Tier B stress corpus below adds three more shapes).
`agen_tier_b_offender`/`agen_tier_b_walk` still detect this exactly as
before, but `agen_emit`/`stade_hvp` no longer refuse on it, and no
longer fall every ragged occurrence back to `push!`/`pop!` either:
`agen_layout` resolves each ragged loop into a **ragged block** — a
per-ancestor-iteration table pair that keeps the occurrence indexed and
mutation-free, the same as a closed-form Tier A site, just addressed
through one extra table lookup instead of a static formula. This
happened in five sub-stages, kept distinct below since each one's own
design rationale matters for anyone touching this code:

**Ragged-block layout (identifying ancestor loops as layout elements).**
`agen_ragged_block(stmt, ...)` recognizes a genuine ancestor loop AL — a
sequential loop whose body contains a descendant loop it alone governs
(`agen_tier_b_walk(stmt.body, own_reassigned)` finds an offender) — and
lays out AL's *own* body via the **existing, unmodified**
`agen_indexed_layout` (the boolean in_ragged/tainted_stacks machinery
from before), reused verbatim as an "AL-scoped, single-owner"
sub-engine: `loop_ctx` resets to `[]` at AL's own frame, so AL's body
is laid out exactly as if it were a fresh top-level kernel. Critically,
the sub-call's own `seq_reassigned` seed is the *incoming* set from
*outside* AL only, **not** unioned with AL's own newly-introduced
reassignments: a var reassigned as a top-level statement of `stmt.body`
itself (e.g. mg_vcycle's `n = nl - 1`, sitting directly in
`i_seq_level`'s own body) is fixed for the whole of one AL iteration,
so from the sub-call's perspective it must not read as "reassigned
inside an ancestor sequential loop" — getting this backwards was the
first bug found while building this (see `agen_ragged_block`'s own
comment): it tainted every stack touched anywhere inside AL, including
ones with no raggedness of their own.

`agen_layout` (the new top-level entry, replacing `agen_indexed_layout`
for `keep_push_pop=false`) walks `kernel.body` once, maintaining a
per-stack running offset exactly like `agen_indexed_layout`'s own
`running` — except this one is allowed to become a runtime expression
(referencing a block's own computed total) rather than staying
closed-form throughout. That's what lets a stack have **multiple**
ragged blocks in sequence — e.g. mg_vcycle's `branch_stack`, written by
both the descend and ascend passes — chain correctly: block 2's own
`base` is block 1's `base + total_sym`, read back out of the same
`current` dict a plain Tier A occurrence's base already uses. Single-
level raggedness only (deliberate scope restriction, matching what the
whole corpus needs): a genuine AL-within-AL surfaces in the sub-call's
*own* `tainted_stacks` rather than a second, recursive table-building
attempt, and `agen_layout` routes exactly those stacks — nowhere else —
to the old whole-kernel `push!`/`pop!` fallback (`layout.tainted_stacks`,
same per-*stack*-not-per-occurrence granularity as before). For the
whole current corpus this fallback is empty: every stack in all five
Tier B stress kernels resolves into a table.

**Table-construction codegen.** `agen_tier_b_block_stmts` emits, for one
block and every stack it touches, a `prefix_<stack>_<block_id>` table
(one entry per AL iteration, the cumulative offset *before* that
iteration) and a `__tot_<stack>_<block_id>` running total, built inside
a real loop over AL's own header — `agen_pos0(header)+1` as the write
index is deliberately the *same formula* the read side
(`agen_site_index`) will use, so a write at iteration k and a later
read at iteration k can't independently drift (the same guarantee
Tier A's own `pos0`/`stride` formulas already rely on). `agen_tier_b_block_skeleton` supplies the scalar state (`n`,
`nc`, ...) this needs by replicating AL's own top-level scalar
reassignments (dropping every array-touching statement, same
`agen_expr_reads_array` filter step 2's sizing pass already used) —
but unlike step 2's whole-kernel replica, it **stops at any nested
`:for` entirely**, not even an empty loop: everything inside a ragged
loop's own body is already captured by the closed-form `local_size`
formula the layout step above computed, so re-executing it here would
be redundant O(n)-per-block-iteration work for zero new information.
That's the actual payoff of the closed-form design over step 2's
sizing pass: sizing no longer touches the ragged loops themselves at
all, only the handful of scalar statements that govern their bounds.
`agen_tier_b_kernel_skeleton` is the real integration point — splices
each block's own table-construction loop into the right point of a
full walk of `kernel.body`, so a kernel's static prelude
before/between/after blocks (mg_vcycle's own `n = n * 2; nl = nfine;
hl = h1`, needed before block 1's own skeleton first reads `nl`/`hl`)
still runs in the right order.

**`agen_site_index`'s `:ragged` offset kind.** A resolved occurrence's
key maps to `(:ragged, stack, block_id, local_offset)` rather than the
Tier A `(stack, offset)` shape; the final index is
`base[stack] + prefix_table[pos0(header)+1] + local_offset +
local_position` — the *whole-stack* base (0, or a prior block's/static
prefix's own total), a runtime table lookup keyed by the current AL
iteration, the block-local static offset (from the AL-scoped sub-call,
unchanged Tier A math), and the ordinary within-iteration position.
Missing `base[stack]` here was the first real bug: every ragged
block's own table is relative to "within this block alone" (it always
starts its own `__tot_*` at 0), so a second block touching the same
stack silently overlapped the first block's index range rather than
continuing after it — found by tracing a `BoundsError` down to a
forward-sweep-only content comparison against the proven `:stack`-mode
pipeline (a corrupted 131-vs-248 vote split on `branch_stack`'s 0/1
flags, restored to an exact match once `base` was added).

A `local_offset`/`local_position` formula can legitimately reference a
block-local scalar directly (e.g. `nc`, or even a kernel *argument*
that's immediately reassigned inside the kernel body — mg_vcycle's `n`
is exactly this, `n = n * 2` being its own first statement, so it's
really just a pre-declared local by the time any ragged block reads
it). Reading such a variable as a bare in-scope reference is **not**
safe at an arbitrary push/pop site: `agen_forward_body` always
preserves program order, so this was never a problem for it, but
`agen_backward_body` reverses per-*statement* order too — a
block-boundary occurrence whose forward statement came last in the
block's body becomes the *first* thing the reversed loop executes,
potentially before the scalar-recompute/tripcount-pop machinery that
preceded it in forward order. This dependency on program order never
existed under plain `push!`/`pop!` (no index formula at all) — it's
new, and specific to closed-form Tier B indexing. The fix: every
block-local scalar a formula depends on (`blk.value_vars`, computed in
`agen_layout`) gets its *own* per-iteration value table too
(`val_<var>_<block_id>`, built alongside the offset/total tables in
Phase B), and `agen_site_index` substitutes a bare reference to it
with a table lookup (`agen_substitute_vars`, a pure AST substitution)
— removing the dependency on program order entirely, the same way the
prefix/total tables already removed it for offsets. The safety filter
for *which* variables need this treatment must also catch a reassigned
kernel argument like `n` above, not just genuine locals — trusting
"it's a kernel argument, always safe" was the second bug found here.

Both fixes are deliberately mutation-free: every index is a pure
expression (a table lookup plus ordinary position formulas), never an
embedded mutation like `stack[cursor += 1]` — the scheme originally
considered and set aside (see the *former* "Known blocking issue" this
entry replaces): `hvp_double_stmt`'s doubling transform reuses the same
index sub-`Expr` for both the shadow and primal writes as two separate
statements, so any embedded mutation gets evaluated twice.

**Wiring (`agen_init_emit`/`agen_adjoint_emit`/`hvp_emit`/`agen_emit`/
`stade_hvp`).** `agen_init_emit` now builds Tier B's table-construction
code (via `agen_tier_b_kernel_skeleton`) alongside the existing Tier A
and `push!`/`pop!`-fallback logic, and returns `(expr, table_names,
tot_names, val_names)` instead of a bare `Expr` — the three extra name
lists get appended, in that exact order, to both `initstacks_*`'s own
return tuple and `_b`/`_hv`'s argument list (`tier_b_extra_args`), so
`val_init_stacks`' generic splat-the-whole-tuple convention keeps
working unmodified; Phase D adds no special-cased calling convention,
only more return values and more parameters kept in lockstep by
construction. `agen_emit`/`stade_hvp` call `agen_layout` instead of
`agen_indexed_layout` at the top level (`agen_indexed_layout` remains,
unchanged, as the sub-engine `agen_ragged_block` calls internally).
`hvp_shadow_stack_inits` had its own bug in the same family as the
`value_vars` one above: it re-evaluated `layout.block_totals`
formulas (which can legitimately reference a block-local scalar) at
the very top of `_hv`, before anything establishes that scalar's
value. Fixed by allocating the shadow stack via `length()` of the
already-built primal stack (itself passed in as `_hv`'s own argument)
instead of re-deriving any size formula — simpler and more robust for
Tier A too, not just Tier B, since it needs nothing but the primal
stack that's already there.

**HVP shadow reuse needed no new code.** A shadow stack is exactly the
same shape as its primal counterpart, and `hvp_shadow_lvalue`/
`hvp_tangent_expr` were already general enough (predating this work
entirely) to copy an index sub-`Expr` verbatim into the shadow
statement, whatever it looks like — Tier B's `:ragged` index is just
another opaque expression from their point of view. Once the index is
pure (the fix two paragraphs up) and the tables are real function
arguments (the wiring paragraph above), reuse follows for free: no
second table, no shadow-specific codegen. Verified by inspecting a
generated `_hv` signature directly (one `prefix_*`/`__tot_*`/`val_*`
per block, no `_d`-suffixed duplicates) and by a dedicated HVP stress
pass (below).

**A pre-existing validation-infrastructure gap, found and fixed along
the way.** `stade_validate_adjoint_file`/`stade_validate_hvp_file`/
`stade_validate_tangent_file` never threaded `keep_push_pop`/
`site_level_tbr` through to the `stade_adjoint`/`stade_hvp`/
`stade_tangent` calls inside `stade_validate_from_baseline` — they
always validated the `keep_push_pop=true` math regardless of what flags
generated the file being tested moments earlier. This means finite-
difference correctness of `keep_push_pop=false` specifically was never
actually exercised by this convenience path in *any* earlier phase of
this feature, Tier A included — the missing-`base` bug above could not
possibly have been caught by it. Fixed by threading both flags through
`stade_validate_from_baseline` and its three `_file` wrappers, and
deriving `stack_arg_names` directly from the generated `initstacks`
Expr's own argument list (`adjoint_out.initstacks.args[1].args[2:end]`)
rather than recomputing it separately — so it can never drift from
whatever `agen_init_emit` actually built. This is a general fix, not
Tier-B-specific: it's what makes `keep_push_pop=false` validation
trustworthy for Tier A kernels too, going forward.

**Validated:** all 19 Tier A corpus kernels pass central
finite-difference validation under `keep_push_pop=false` for both
adjoint and HVP generation. The 5-kernel Tier B stress corpus
(`mg_vcycle`, `mg_vcycle_multi`, `cascadic_mg_prolong` — a growing-only
bound — `windowed_relax_retire` — an `if`-gated reassignment — and
`richardson_substep` — a non-spatial, substep-count domain) passes
tangent/adjoint/HVP validation across the full `keep_push_pop` ×
`site_level_tbr` 2×2 matrix, now via the *fixed* validation
infrastructure with real random seeding — 30/30 combinations, genuine
nonzero max relative errors in the 1e-8–1e-10 range (not the trivial
all-zero results a seed-less test would trivially "pass" with). A
dedicated HVP stress pass — 5 kernels × 5 random baselines × 10 trials
— adds 250 more finite-difference checks, all passing, specifically
targeting the multi-block stacks (`branch_stack`, `left_stack`, etc.)
most likely to reveal a table-chaining or cross-derivative ordering
bug if one existed. `keep_push_pop=true` output is provably
byte-identical to the pre-feature codebase across the full corpus, Tier
A and Tier B alike, at every checkpoint of this work. GPU-split-count
regression checked via `stade_cuda`: `:indexed` mode never reduces the
number of loops split versus `:stack` mode, and often increases it for
Tier A kernels (e.g. `advection` 2→4, `matvec_loss`/`relu_field` 0→2);
Tier B's own ragged loops are **still** not GPU-split-eligible under
either mode (see the follow-up below) — this closed-form design gets
Tier B off `push!`/`pop!` and onto real indexing, which is the
prerequisite, but doesn't by itself change `cgen_contains_stackop`'s
verdict on the loop.

One test-setup wrinkle worth recording, not a code bug: mg_vcycle's
random baseline generator can pick `nfine`/`num_levels` combinations
that are mathematically incompatible with the multigrid hierarchy
(e.g. `nfine=3, num_levels=5`, forcing a coarse-grid point count
negative partway through). `:stack` mode tolerated this silently (a
negative-length Julia range is just empty, not an error); closed-form
ahead-of-time sizing correctly surfaces it as an immediate allocation
error instead. Worked around by pinning a compatible baseline for
testing; the baseline generator itself has no per-argument constraint
mechanism to fix this properly, which would be a reasonable follow-up
independent of Tier B.

**Known follow-up (not blocking, not yet done):** GPU-launch plumbing
for a resolved ragged block's own loop body. The loop is index-based
now, same shape as any Tier A GPU-split candidate, but its trip count
is a *runtime* value (the current AL iteration's `n`, not a kernel
argument), and the launch itself needs to happen *inside* the
still-host-side ancestor sequential loop, once per iteration, with a
freshly-read trip count each time — `cgen_launch_expr`'s range argument
already accepts an arbitrary expression, but whether `cgen_device_body`
and friends handle a runtime-scalar range correctly needs to be
verified, not assumed, before `cgen_contains_stackop`'s refusal is
lifted for these loops.

`cgen_device_assign`'s atomic-write detection has already been hardened
(independent of, but recommended alongside, this feature): it now does
a real occurs-check for the enclosing device loop's thread variable
anywhere in a write's index expression, and refuses loudly (rather than
silently emitting an unprotected write) for any thread-invariant-indexed
write that isn't a plain additive accumulation, since a non-additive
race has no atomic-wrapper fix.

### Testing convention

Every change is checked against the shared golden corpus (19 kernels), 
using `validate_corpus.jl` (central finite differences)
as the primary correctness oracle, checked against random seed vectors.
A structural diff against existing Tapenade-generated `_b.jl` files 
inside `val-corpus-tapenade-adjoint` is a secondary style check, not a correctness requirement.

## `site_level_tbr`: per-statement (not per-variable) TBR pruning

`stade_adjoint`/`stade_hvp` (and their `_file`/`_corpus` wrappers) take a
`site_level_tbr::Bool = false` keyword. `snap_value_needed_vars`/
`agen_value_needed_vars` decide "does var's value matter *anywhere*" at
whole-*variable* granularity — every write to a value-needed var gets a
snapshot, even when the specific nonlinear read that made it value-needed
happens *after* that write, reading the write's own new value rather than
what it destroyed (`affine_loss`'s `v_stack` was the original motivating
case: `v` is value-needed because of a later `v^2`, but that read happens
after `v`'s only write, so the write's snapshot is provably dead weight).

**Direction rule:** a write needs a snapshot iff (a) it's self-referencing
with its own occurrence appearing under a non-`+`/`-` op in its own rhs, or
(b) some nonlinear read of the same variable occurred *strictly before* it
in true forward execution order — never because of a read that comes
*after*. (Confirmed against `snap_read_before`'s own existing forward
convention; an earlier backward-liveness formulation of this got the
direction wrong and was corrected before landing — see the design
discussion in the conversation history that produced this feature.) Loops
need a **forward** fixed point: a read near the top of a loop body is, for
every iteration but the first, actually preceded by the *previous*
iteration's reads from later in the body — `snap_fwd_walk_loop!`/
`agen_fwd_walk_loop!` iterate to convergence (monotone, since `seen` only
grows) before recording final per-site decisions.

**Deliberately duplicated, not shared** (per Hard Rule 7): `snap_*`
(`snap_fwd_walk!`/`snap_fwd_walk_loop!`/`snap_value_needed_sites`) and
`agen_*` (`agen_fwd_walk!`/`agen_fwd_walk_loop!`/`agen_value_needed_sites`)
are independent, hand-written-identical implementations, exactly like
every other `snap_`/`agen_` pair in this file. `stade_site_level_tbr_check`
is the *one* place allowed to compare them: it computes both, asserts
exact `Dict` equality (not just subset), and only then hands each stage
its own copy — `snap_plan` gets `snap_sites`, `agen_emit`/`hvp_emit` get
`agen_sites`. This assertion runs automatically, every time the flag is
on — not just in dev testing — so a future silent divergence between the
two hand-maintained copies fails loudly instead of desyncing push/pop
counts.

**New frozen shape:** site-level TBR decision = `Dict{Any,Bool}` keyed by
`agen_site_key(body, idx)`. `ectx` gained a `push_pop` field
(`Union{Nothing,Dict{Any,Bool}}`); `agen_needs_snapshot(lhs, rhs, var,
value_needed, key=nothing)` is polymorphic on `value_needed`'s type — a
`Set{Symbol}` (default) behaves exactly as before and ignores `key`, a
`Dict` looks up `key` instead. `snap_plan`/`snap_check_assign!` take a
parallel `site_needed` keyword with the same fallback behavior.

**Deliberately out of scope** (still whole-variable, unaffected by the
flag regardless of its setting): `agen_block_boundary_vars`,
`agen_exempt_vars`, `agen_if_branch_scalar_vars` — these model block-scope
restoration and branch-hoist fallback, different questions from "does
this one write need a snapshot" that this analysis doesn't cover.

**Validated (CPU):** full corpus (22 kernels) plus three stress kernels
added specifically to exercise the fixed point and conservative `if`
merge — `stress_cross_iter` (single-loop write-then-read, cross-iteration
coupling), `stress_nested` (same coupling through a nested loop), and
`stress_if_asym` (asymmetric nonlinear-only-in-one-branch read) — pass
central-difference validation for adjoint and HVP, in *both*
`keep_push_pop=true` and `keep_push_pop=false` (`agen_layout_walk!`/
`agen_indexed_layout` also take a `push_pop` keyword, wired the same way).
`keep_push_pop=false` validation for `unet`/`transformer` hits a
pre-existing gap in `val_validate_adjoint`'s test harness (can't supply
values for derived/internal sizing free-variables like `hw2`, `n_d`) —
confirmed pre-existing by reproducing identically with `site_level_tbr=
false`, not a regression from this feature.

**Known blocking issue — resolved.** Porting `site_level_tbr=true`-generated
adjoint code to CUDA/AMDGPU via `stade_cuda`/`stade_amdgpu` used to fail for
2 of 25 corpus+stress kernels (`unet`, `transformer`), both tripping
`cgen_device_assign`'s data-race occurs-check on a scatter write
(`xpad0[yi] = x[idx]`-shaped: the write's index is a *bare symbol* computed
one or more assignment-hops earlier in the same device-loop body from the
thread variable, e.g. `yi` derived from `idx` via `div`/`mod` unravel/
reflatten arithmetic — not literally containing `idx` itself). Root cause
confirmed by diffing generated code: `cgen_body` only splits a loop onto
the GPU when it contains no `push!`/`pop!` (`cgen_contains_stackop`); with
the flag off, these loops kept their now-pruned snapshot push and were
never split at all, so `cgen_device_assign` never ran on them. Pruning the
push made the loop GPU-eligible for the first time, exposing that
`cgen_expr_contains`'s occurs-check was purely syntactic (top-level index
expression only) with no transitive tracing through same-body scalar
let-bindings.

**Fix:** `cgen_device_body` now threads a `thread_dep::Set{Symbol}`
(starting as `{thread_var}`) through the device loop body, adding any
scalar (`lhs isa Symbol`) whose rhs references something already in
`thread_dep`, and removing a symbol from it if a later reassignment
breaks the chain (dataflow-correct, not a monotonic once-true-always-true
flag). `cgen_device_assign` takes `thread_dep` instead of a bare
`thread_var` for its check, via a new `cgen_expr_contains_any` (same
occurs-check, generalized to a symbol set). This is strictly *more
precise*, not more permissive in an unsound way: it still only accepts a
write whose index provably, syntactically depends on the thread variable
somewhere in its derivation — exactly the same soundness bar the original
check used (neither version proves *injectivity*; a genuinely
non-injective thread-dependent index, e.g. `arr[idx % 5]`, was accepted
before this fix and still is — that's an inherent, documented limitation
of "depends on thread_var" as a race proxy, unrelated to this fix, and
still the kernel author's responsibility). Both flag states, all 25
corpus+stress kernels, both CUDA and AMDGPU: 0 failures. `jgen_device_assign`
(the JACC target) reuses `cgen_expr_contains` directly rather than
duplicating it, and was **not** touched by this fix — it doesn't currently
reproduce `cgen_device_assign`'s error-refusal at all for the
non-additive/non-dependent case (a separate, pre-existing gap, out of
scope here — see `jgen_` v1.x note below for what *was* fixed on the
JACC side).

## `jgen_`: JACC.jl v1.x migration

`jgen_launch_expr` used to emit `JACC.parallel_for(N, f, args...)` — a
plain positional-argument function call, which was JACC.jl's pre-1.0
(`v0.0.x`-era) API. JACC.jl's stable v1.0 line (released ~March 2026)
replaced this with a macro form: `JACC.@parallel_for range=N f(args...)`
(confirmed against `juliagpu.github.io/JACC.jl/stable`'s current
example, and structurally verified via `Meta.parse` to get the exact
`Expr` shape — `Expr(:macrocall, Expr(:., :JACC,
QuoteNode(Symbol("@parallel_for"))), nothing, Expr(:(=), :range, n_iter),
Expr(:call, fname, fargs...))`). The underlying kernel function's own
signature is unchanged (loop index still its own first parameter,
supplied internally by the macro) — only the launch call's shape moved,
so `jgen_kernel_def` itself needed no change, only `jgen_launch_expr`.

`jgen_preamble()`'s `Pkg.add("JACC")` is now pinned to
`Pkg.add(name = "JACC", version = "1")` — the previous unpinned call is
exactly what let this mismatch happen silently in the first place (any
fresh install pulls whatever's newest); pinning to the major version
this code actually targets means a future v2 breaking change surfaces
as a resolvable Pkg version conflict instead of a silent runtime API
mismatch.

**Validated:** the built-in self-test (`jgen_* round-tripped...`) checks
for `JACC.@parallel_for` + `range = ` in the generated source. Every
corpus + stress kernel's `stade_jacc` output was regenerated and
confirmed to use the v1.x macro form with zero errors; `validate_corpus.jl`
(unaffected by this change — a JACC-only code path) stayed at 75/75.
Not independently confirmed against a running JACC v1.x install (no GPU
hardware, of any vendor or through any target, has been available to
actually execute anything `cgen_`/`jgen_` produce, at any point in this
codebase's history) — this rests on documentation/example evidence, same
epistemic status as the rest of `jgen_`'s design notes above.

## Self-check before returning STADE code

- [ ] No `module`, `struct`, or `@enum` anywhere
- [ ] No top-level `const`, and no other shared mutable state
- [ ] Every function name carries its correct stage prefix
- [ ] Any `NamedTuple` produced/consumed matches the frozen shape spec
      exactly — no ad hoc extra or renamed keys, and no `:while` support
- [ ] Expressions are plain `Expr`/`Symbol`/`Number`, never a custom type
- [ ] The function's only inputs are the documented upstream shapes —
      no reaching into another stage's internal helpers
- [ ] No public entry point requires the caller to specify
      independents/dependents — those are auto-derived
- [ ] Comments are one line or less, and present only where non-obvious
- [ ] No filesystem access outside `io_*`
- [ ] No reference to any specific corpus kernel by name
- [ ] Loop-nest analysis (`pos0`/`stride`, Tier B ancestor-loop detection,
      GPU split counting) walks the kernel source's actual `for` structure
      only — no logic reconstructs a synthetic nest by pattern-matching a
      fused loop's `div`/`mod` body statements
- [ ] If touching `site_level_tbr`'s `snap_*`/`agen_*` pair, both copies
      were changed identically and `stade_site_level_tbr_check` still
      passes across the full corpus + stress kernels
- [ ] If touching Tier B (`agen_ragged_block`/`agen_layout`'s block
      resolution, `agen_tier_b_block_stmts`/`agen_tier_b_kernel_skeleton`'s
      table codegen, or `agen_site_index`'s `:ragged` branch), every
      index built stays a *pure* expression — never one with an
      embedded mutation (see the Tier B entry's `hvp_double_stmt`
      double-evaluation note for why that specifically breaks HVP);
      any block-local scalar (or reassigned kernel argument) a formula
      depends on is in `value_vars` and gets substituted via its own
      value table, not read as a bare in-scope reference; the
      remaining `push!`/`pop!` fallback (`layout.tainted_stacks`)
      stays a per-*stack*, not per-occurrence, decision; and a
      `keep_push_pop=false` validation claim is only trustworthy if it
      went through `stade_validate_*_file` with `keep_push_pop`/
      `site_level_tbr` explicitly passed (the defaults validate
      `:stack` mode's math regardless of what was generated)