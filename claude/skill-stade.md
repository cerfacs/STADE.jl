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

**Tier B (detected, not yet implemented):** a loop whose own
bound-determining symbol is ever an assignment target inside an
*ancestor sequential loop* has a trip count that isn't a closed-form
function of kernel arguments (`mg_vcycle`/`mg_vcycle_multi`, whose inner
loop bounds derive from `nl`, reassigned once per outer sequential
`i_seq_level` iteration, are the two confirmed corpus instances).
`agen_tier_b_offender`/`agen_tier_b_walk` detect this up front and
`agen_emit`/`stade_hvp` refuse loudly (`error(...)`, naming the
offending symbol) rather than emit a wrong or under-sized buffer.
Actual Tier B support (ragged/data-dependent sizing) is a separate,
not-yet-implemented follow-up.

**Validated:** all 19 Tier A corpus kernels pass central
finite-difference validation under `keep_push_pop=false` for both
adjoint and HVP generation (`stade_validate_adjoint_against_file`,
`val_validate_hvp`); the 2 Tier B kernels refuse loudly as designed.
`keep_push_pop=true` output is provably byte-identical to the
pre-feature codebase across the full corpus. GPU-split-count regression
checked via `stade_cuda`: `:indexed` mode never reduces the number of
loops split into parallel GPU kernels versus `:stack` mode, and often
increases it (removing the push/pop sequential dependency unlocks
splits `:stack` mode couldn't attempt at all — e.g. `advection` 2→4,
`matvec_loss`/`relu_field` 0→2).

**Known follow-up (not blocking, not yet done):** Tier B ragged sizing.

`cgen_device_assign`'s atomic-write detection has already been hardened
(independent of, but recommended alongside, this feature): it now does
a real occurs-check for the enclosing device loop's thread variable
anywhere in a write's index expression, and refuses loudly (rather than
silently emitting an unprotected write) for any thread-invariant-indexed
write that isn't a plain additive accumulation, since a non-additive
race has no atomic-wrapper fix.

## Testing convention

### Indirect-indexing shape-inference fix (2026-08-12)

`shape_force_int_expr!` (the backward pass of `shape_propagate_int!`,
which pushes int-ness discovered on an assignment's LHS back onto
whatever RHS operands it depends on) had no case for `expr.head ==
:ref`. This meant a gather/permutation-table array used only to
produce indices for other arrays -- e.g. `i_node = i_cell_to_node[i_loc,
i_cell]`, with `i_node` itself later confirmed int by its own use as a
subscript elsewhere -- never had that int-ness propagated back onto
`i_cell_to_node`, the array actually being indexed. It defaulted to
`array_float`, so downstream adjoint generation treated a pure
integer connectivity/index table as a differentiable field, producing
spurious shadow arguments (`i_cell_to_nodeb`, etc.) that don't
correspond to any real gradient. Fixed by adding an `expr.head ==
:ref` case: the base array symbol is marked int whenever the
indexing result is (or becomes) known int -- sound, since indexing a
true `Vector{Float64}` field can never itself be forced int by any of
this file's existing int-evidence rules. Verified against the full
20-kernel corpus (`phase1_adjoint.jl`/`phase1_hvp.jl`/
`phase1_splitcount.jl`/byte-compat diff against the pre-feature
baseline): zero regressions, all previously-passing kernels
unaffected. Confirmed fixed on a real unstructured-mesh kernel (mesh
connectivity + periodicity permutation tables): those arrays are now
correctly classified `array_int`, and the spurious shadow arguments
are gone from the generated adjoint.

**Separately found and fixed (2026-08-13):** that same real-world
kernel's adjoint was numerically wrong under finite-difference
validation. Initially misdiagnosed as an array-aliasing issue (two
mesh cells sharing a node, both accumulating into `res[i_k_node]`
where `i_k_node` comes from a connectivity table) -- but a
matched-settings comparison showed a *disjoint* (non-aliased) mesh
failing just as badly, ruling that out. The actual cause: a scalar
(`cavgx`/`cavgy`/`cavgz`) written by one nested sub-loop and read
unchanged by a *later sibling* sub-loop within the same repeating
outer iteration had no restore mechanism reaching that sibling read --
see the write-up immediately below, which is the fix. Confirmed via
machine-precision complex-step differentiation against a from-scratch
reimplementation (ruling out finite-difference truncation noise as an
alternative explanation) and via instrumented traces of the generated
code showing the scalar frozen at one outer iteration's value
throughout the entire backward sweep.

### Cross-sibling-loop scalar restoration fix (2026-08-13)

A scalar written by one nested sub-loop and read unchanged by a
*later sibling* sub-loop within the same repeating outer iteration
(e.g. `cavgx`/`cavgy`/`cavgz`: built by one `for i_loc` loop, read by
a second, separate `for i_loc` loop, both inside a repeating `for
i_cell`) had no restore mechanism reaching that sibling read. The
per-write-statement push/pop machinery only restores a var to what it
held immediately *before its own next write* -- correct for a var read
again within the *same* loop that writes it (`vere` elsewhere in this
file), but wrong here, since nothing re-established the var's
post-loop value before the sibling's backward code ran. Only the
last-processed outer iteration (in backward order) ever saw the
correct value; every earlier one silently inherited the previous
iteration's corrupted leftover state -- with no aliasing or indirect
indexing involved at all.

`agen_nested_write_vars`/`agen_block_boundary_vars` find the affected
variables for a given block `body`: value-needed scalars written only
inside nested sub-`:for`/`:if`s of `body` (never as a top-level
statement of `body` itself). `agen_forward_body` now pushes each such
var once at the end of `body` (its value as this iteration leaves it);
`agen_backward_body` pops/restores it once at the very start of
processing `body`'s own backward code -- using the SAME `(:value,
var)` stack and the same `agen_emit_push`/`agen_emit_pop`/
`agen_site_key` machinery every other site already uses (key =
`agen_site_key(body, 0, var)`, the `idx=0` sentinel never colliding
with a real per-statement index), so it works identically under
`keep_push_pop=true` and `=false` with no separate code path, and HVP
inherits it for free since `hvp_emit` calls the same
`agen_forward_body`/`agen_backward_body`. `agen_body_has_snapshot`
also checks `agen_block_boundary_vars` now, so a loop needing only
this treatment still gets correctly reversed.

Verified: the original failing case matches a complex-step reference
to `8.9e-16` (was 1-3% off); a disjoint/aliased x `i_njac`=0/1/2 test
matrix is ~1e-9 everywhere (was 0.8%-5.5%); the full 19-kernel Tier A
corpus is unaffected (still passes adjoint + HVP validation, byte-
identical generated output for 19 of 21 kernels); `mg_vcycle`/
`mg_vcycle_multi` are the only corpus kernels whose generated
`keep_push_pop=true` output changed (they have this pattern too) and
both still pass their own validation correctly afterward.

### val_generate_baseline fixes for indirect-indexing kernels (2026-08-13)

Four bugs surfaced getting the website's "Validate" button working for
up.jl (mesh connectivity args); all generic, none up.jl-specific:

1. `val_zeros_like`/`val_random_values_like` only populated float-arg
   keys, but `val_unflatten` unconditionally reads `template[a]` for
   every `:array_int` arg too (to pass real, non-perturbed data
   through unchanged) -- caused `KeyError(:i_cell_to_node)`. Fixed:
   both now also copy the array_int entries through.
2. `val_random_values` filled array_int args from a hardcoded `1:3`,
   unrelated to what they actually index into. Fixed: uses `1:N`
   where N is the shared array size `val_grow_shapes` already grows
   every array dimension to -- safe against whatever it indexes,
   with no per-argument "what does this index into" analysis needed.
3. `io_read_baseline_yaml` has no per-kernel type info, so array_int
   values always come back Float64 -- `ArgumentError: invalid index:
   3.0`. Fixed: new `val_coerce_int_arrays!(kernel, values)`, called
   at both call sites right after reading a baseline back.
4. A float arg used as a `/` divisor (e.g. cell/node "volume") has no
   guarantee of staying positive or of a comparable scale to whatever
   it's divided against under plain `randn()` -- for a kernel with an
   iterative relaxation, that can make the per-step gain exceed 1 and
   blow up over repeated iterations to astronomical output magnitudes,
   at which point `epsilon=1e-6` finite differences become numerically
   meaningless (the step's own contribution falls below float64
   precision at that scale) -- not an adjoint bug, but it looks like
   one. Fixed: `val_divisor_args` statically finds every arg ever used
   as a `/` divisor (kernel-agnostic, same style as `val_arg_ndims`)
   and draws it narrowly around 1.0 instead of wide `randn()`;
   `val_generate_baseline` also rejects a candidate whose own output
   is non-finite or too large, and adaptively narrows the scalar_int
   range (an iteration-count arg is a common divergence amplifier)
   toward 1 on repeated divergence, down to below the caller's own
   `int_lo` if needed as a last resort -- a working baseline with a
   smaller iteration count beats none.

With these, up.jl's tangent and HVP validate reliably (~1e-8). Adjoint
validation sometimes still reports elevated error on a random
baseline -- confirmed via independent complex-step differentiation
this is not an adjoint-correctness bug (HVP, which calls the identical
adjoint function internally, passes consistently) but FD-precision
sensitivity in `val_validate_adjoint`'s dot-product check methodology
for this still numerically stiff kernel; not yet resolved.

### Testing convention

Every change is checked against the shared golden corpus (19 kernels), 
using `validate_corpus.jl` (central finite differences)
as the primary correctness oracle, checked against random seed vectors.
A structural diff against existing Tapenade-generated `_b.jl` files 
inside `val-corpus-tapenade-adjoint` is a secondary style check, not a correctness requirement.

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