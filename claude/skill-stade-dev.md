---
name: skill-stade-dev
description: >
  Use whenever writing, reviewing, or refactoring Julia code that is part
  of STADE itself (the automatic differentiation engine for skill-stade
  kernels) — as opposed to the numerical kernels STADE differentiates,
  which are governed by skill-stade instead. Trigger any time the user
  asks for code implementing or modifying a STADE pipeline stage (parse,
  shape inference, activity analysis, snapshot/TBR analysis, derivative
  rules, linearization, tangent or adjoint codegen, GPU codegen, file
  I/O, validation).
---

# skill-stade-dev: house style and architecture contract for STADE

STADE is a single-file (`STADE.jl`), function-only Julia AD engine for
kernels compliant with skill-stade. This skill is the contract
that keeps independently-developed pieces mergeable.

## Installing Julia

Julia isn't on apt, and JuliaLang's binary host isn't reachable. The
user mirrors the official 1.10.11 LTS tarball as a GitHub release asset.

```bash
curl -sL https://github.com/luciano-drozda/julia-tar/releases/download/julia-1.10.11/julia-1.10.11-linux-x86_64.tar.gz -o /home/claude/julia.tar.gz
echo "fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf  /home/claude/julia.tar.gz" | sha256sum -c -
tar -xzf /home/claude/julia.tar.gz -C /home/claude/
export PATH="/home/claude/julia-1.10.11/bin:$PATH"
```

The sha256 matches the official JuliaLang checksum. **If it fails, stop**
— treat the tarball as untrusted and tell the user. Re-verify if the
URL or tag ever changes. Prepend the `export PATH` line to every
subsequent `bash_tool` call needing Julia.

## Hard rules

1. **No `module` blocks.** One file, top level, organized by
   `# ==== prefix_* ====` banners.
2. **No `struct` or `@enum`.** Records are plain `NamedTuple`s with a
   fixed key set. Open-ended key sets use `Dict`.
3. **No global `const`** or any top-level mutable/shared state. Lookup
   tables are built fresh inside the function that uses them.
4. **Every function name carries its stage prefix**: `parse_`, `shape_`,
   `der_`, `emit_`, `act_`, `snap_`, `lin_`, `tgen_`, `agen_`, `hvp_`,
   `cgen_`, `jgen_`, `ii_`, `val_`, `io_`, `stade_`. `io_` is the only
   prefix permitted to touch the filesystem. This replaces namespacing —
   a cross-prefix name collision is a bug to fix before merging.
5. **Frozen data shapes.** Changing a key is a breaking-change event
   across every stage touching it.
   - kernel `(sig, body)`
   - sig `(name, args, kinds::Dict{Symbol,Symbol}, independents, dependents)`;
     kinds ∈ `:scalar_float :scalar_int :array_float :array_int`;
     independents/dependents auto-derived by `parse_kernel`, never
     required from a caller
   - statement `(kind=:assign|:for|:if, ...)`. **`:while` is
     deliberately unsupported** — wait for a kernel that needs it
   - snapshot site `(kind=:value|:array|:branch|:tripcount, array, at)`
   - linearization node `(op, args, darg_exprs)`
   - `ii_plan` `Dict{site_key => :independent|:reduction|:mixed|:recompute}`
6. **Expressions are raw `Expr`/`Symbol`/`Number`** — never a custom AST.
7. **Every stage function is pure** given its documented inputs (except
   `io_*`). This is what lets a contributor build `agen_*` against a
   hand-written fake `snap_plan` output.
8. **Comments are short and sparse** — one line, only where the code
   isn't self-explanatory. Design rationale belongs here, not inline.
9. **No corpus-specific code in `STADE.jl`.** Nothing may name a kernel
   in `val-corpus/`.
10. **A loop nest is whatever `for` structure the source contains.**
    skill-stade fuses a rectangular nest into one `for idx = 1:n*m` with
    indices recovered by `div`/`mod`. Treat that as a depth-1 nest with
    tripcount `n*m`. **Never pattern-match body-level `div`/`mod` to
    reconstruct a synthetic nest** — those are ordinary assignments, not
    loop structure.
11. **`snap_*`/`agen_*` duplicated pairs must be changed identically.**
    `stade_ii_plan_check` and `stade_site_level_tbr_check` assert exact
    equality between the two copies; treat a disagreement as a
    one-sided edit, not as a reason to relax the assertion.

## Public flag surface

`stade_tangent`/`stade_adjoint`/`stade_hvp` (and `_file`/`_corpus`
wrappers) take exactly two: `keep_push_pop::Bool = true` and
`fuse_ii_loops::Bool = false`. Site-level TBR is unconditional.

Any new snapshot-exclusion mechanism must go through
`agen_ii_override_ectx`. `agen_push_pop_source` reads `ectx.push_pop`,
so modifying `value_needed` alone is silently ignored.

### `keep_push_pop`

`true` must stay byte-identical to `ii_plan = nothing` output — both
flags off is an exact rollback, and that is what makes a two-flag
surface defensible for this much machinery.

`false` replaces every `push!`/`pop!` with an indexed write/read into a
pre-sized `Vector`, addressed by a closed-form, thread-local position.
This is what makes GPU codegen possible at all: a shared stack pointer
has no meaning once a loop splits across threads, and a running counter
is no substitute either, since a launch guarantees no visit order.

**Tier A**: index computable in closed form from kernel arguments and
the site's own loop nest. Push/pop sites are matched by a *structural*
key, `agen_site_key(body, idx[, bv])`, never a running counter —
branch-scalar hoisting already reorders pops, which silently breaks
counters.

**Tier B**: a loop whose bound is reassigned inside *any* ancestor loop
has no closed-form trip count. The ancestor does not have to carry a
value between its own iterations. `agen_ragged_block`
resolves it into per-ancestor-iteration prefix/value tables, keeping the
site indexed and mutation-free; a narrow per-*stack* (never
per-occurrence) `push!`/`pop!` fallback remains for what can't resolve.

**Rematerialized restores** (`agen_remat_plan`, `keep_push_pop=false` only): a
pre-write save whose old value is an earlier same-block write of a pure
expression (read-only-after-load arrays, ints defined once per block, literals)
is replaced by recomputing that expression in the backward sweep, and its slot
disappears. Without it a tiled convolution's per-tap saves force 255 registers
and spills. A new snapshot-exclusion mechanism still has to leave the site-level
push map consistent for layout and emission: `agen_remat_apply` does that. Never
let a float local into the recomputed expression, and never skip the array-write
rule: the sabotage of purity, nested-write clearing and block-local int rules
all fail `remat_shapes`.

**Cross-iteration restores** extend the same pass to a loop-local scalar's first
write (`agen_remat_cross_*`): its save is dropped and each reverse iteration
rebuilds the end-of-iteration value from the last write's pure rhs. The scalar
must be written only at the top level of loop bodies, never rewritten in a
statement nested in a body that writes it, read only after a write in its own
body, and the decision is all-or-nothing per scalar. Never replace the per-body
coverage by a kernel-wide read count (it never fires on a tiled kernel whose
register names are reused across loops) or by an outer body's write (the first
iteration would read that write and later ones the previous iteration's value).

**Array snapshot** (`agen_asnap_plan`, `keep_push_pop=false` only): a loop that
scatters into an array `A` and reads `A` after every write pushes the old value
at each write, and the push is a read-modify-write that keeps the loop on the
host. One copy of the whole `A` at the head of each iteration of the enclosing
loop, restored at the tail of its reverse, replaces those pushes and lets the
scatter offload (`ttgc`: 18 to 6 host loops). It is right only when every
derivative read of `A` follows every write to `A` in the body (G4). The
mechanism leaves the per-write pushes alone for every other array. Wiring: a
pseudo-site `<A>_snap` (stack `<A>_snap_stack`) joins the snapshot plan, the
push map gets `agen_site_key(B, 0, A) => true` and loses the write sites, and
`initstacks_*` takes the array, returns `__len_A` after the stacks and the
adjoint takes it as a trailing argument (`length` is not a whitelisted
intrinsic, so no generated kernel calls it; `cgen_is_length_call` admits it in
`initstacks_*` only). Correctness guards G4, G7, G9, G12 must each fail an
oracle or the array-state comparison when removed. G6, G10, G13, G14 are cost
guards: removed, the plan changes and the result stays right, and the stack
grows (a whole-array copy per iteration costs `length(A)` where a per-slice
write costs a few slots). Do not widen G14 past scatters: it is what keeps the
copy off `mpnn`, `transformer` and `advection`. An array-state comparison
against `keep_push_pop=true` is mandatory, because the gradient can stay right
while the restore is missing.

**Live-out rule of the ii classifier.** An `:independent` loop is fused at the
forward position and must not consume an adjoint that code after the loop seeds.
An active accumulator read after its loop defers to the backward position
(`:mixed`); any other live-out local refuses the loop. Change both classifier
copies identically.

**A site collides with itself.** A plain accumulation `a[t + k]` under an inner
serial loop is a cross-thread race even with a single write site; the pairwise
overlap analysis never sees it. `cgen_write_self_overlaps` counts only positive
evidence (integer coefficient on the thread variable, an offset that moves with
the loop through integer-scaled sums); a symbolic stride stays trusted.

Tier B index expressions must stay **pure** — an embedded mutation
double-evaluates under `hvp_double_stmt`. Any block-local scalar a
formula depends on must be in `value_vars` and substituted through its
own value table, never read as a bare in-scope reference.

The sizing loop writes a block's value tables and totals **before the first
retiring assignment**: a write to a size-relevant scalar that some loop, at any
depth, has already read as a bound. Look at nested loop headers too, not only
top-level ones (`retire_nested`, `retire_if`).

`initstacks_*`'s signature grows to accept the free variables its size
expressions reference — including scalars **derived in the kernel body**
(`n_d = n * d`), whose defining assignments are hoisted into its body in
dependency order. Its body must have **no free variable outside its
parameter list**; that invariant is guarded.

### `fuse_ii_loops`

Eliminates snapshots for provably-safe loops. Named after Tapenade's
`$AD II-LOOP`, but *proven* rather than asserted: `snap_ii_plan` and
`agen_ii_plan` are independent implementations, cross-checked. That caps
coverage at what can be proven, and the trade is deliberate.

| kind | forward position | backward position |
|---|---|---|
| `:independent` | *replaced* by one fused un-reversed loop | already emitted |
| `:reduction` | ordinary loop, `vn_red` excluded from push | `agen_emit_ii_loop(recompute=true)` |
| `:mixed` | ordinary loop, `vn_local` excluded | same; **never split** across positions |
| `:recompute` | ordinary loop, `vn_local` excluded | same as `:reduction` |

`:recompute` exists for loops containing an **escaping array write**:
the others fuse differentiation at the forward position, moving that
array's adjoint too early; `:recompute` moves nothing and replaces
snapshots with re-execution.

**The walker is post-order** — nested loops are classified before the
loop containing them, so the enclosing loop can see they're covered.
Reversing this silently disables nested classification.

Constraints, each of which is load-bearing for correctness:

- `ii_body_has_surviving_snapshot` must key on **site-level** decisions,
  not whole-variable `value_needed`. A pure accumulation
  (`res[i] = res[i] + auxres`) is value-needed as a variable yet emits no
  push; treating it as one refuses every scatter-accumulate loop.
- `agen_emit_ii_loop`'s `fwd` half is the real primal at the forward
  position but only a **recompute** at the backward one, where it must
  be filtered to scalar assignments. Re-executing array writes applies
  every accumulation twice — **gradients stay bit-identical while this
  happens**, so only an array-*state* comparison catches it.
- `ii_recomputable`'s scalar rule is **live-in**, not "assigned
  elsewhere". A scalar reassigned in a later unrelated region is fine.
- `ii_array_intact` is the exact dual of
  `ii_body_has_escaping_array_write` — one asks whether a *read* is
  reachable after the loop, the other a *write*. The wraparound pass is
  mandatory.
- Classification must check the candidate loop's **own header**, not
  only its body: `for i = 1:cur` with `cur` retired each pass gets its
  header re-run at the backward position against the wrong value.
- `agen_ii_covered_write_check`'s covered-kind tuple must list **every**
  kind. Missing one is the difference between "classified" and "the
  stack actually disappears".
- A local assigned two or more times with a read between the first and
  last assignment (`ii_versions_read`) has several live VERSIONS per
  iteration. Fusing or recomputing such a loop rebuilds the scalar from its
  last assignment only, so every read sees the final version. Both
  classifiers refuse it, except a self-accumulation `v = v + t`, which has
  one version. A register-tiled convolution reloads its weight scalar per
  tap; with the flag on, the gradient of its input came out wrong (relative
  error about 2) until this check. `ii_multiversion` is the witness.

Measured effect (peak stack slots, `false` → `true`): `ttgc` 504 → 240,
`transformer` 2406 → 1095, `unet` 432 → 324. Every other kernel is
unaffected — dominated by loops with no value-needed local scalar.
**Don't justify extensions on breadth this won't deliver.**

## Validation

The corpus is in `val-corpus/`. `validate_corpus.jl` runs
four oracles per kernel — tangent, adjoint, HVP (all central differences,
~1e-8 floor) and the **exact tangent-vs-adjoint dot-product identity**
`<Yb, J·Xd> == <J'·Yb, Xd>` (~1e-15, no epsilon). The exact oracle
catches systematic adjoint errors two orders below what FD can see; it
does *not* validate the tangent, since a bug shared by both codes
cancels, so it complements rather than replaces.

- **Pass the flags you generated with** through to the *validator*, not
  only to generation. The validator's defaults will otherwise check a
  different mode's math than the one under test, and the flagged path
  goes unexercised.
- Run in halves — `validate_corpus.jl` take kernel names.
  Backgrounded runs get reaped.
- The baseline generator draws random integer arguments and retries.
  It gates candidates on the `keep_push_pop=false` `initstacks_*`, so a
  dimensionally incoherent draw is redrawn automatically — generic, with
  no per-kernel annotation. It catches only the *loud* class; a
  valid-but-wrong shape (`d != h*dk`) needs the relation **derived in
  the kernel body** instead, making the bad state unrepresentable.

## GPU (`cgen_`/`jgen_`)

`stade_gpu` ingests **already-generated** code; its only keywords are
`precision` and `keep_all_atomic`. **Only `keep_push_pop=false` output is
a GPU target** — `:stack` mode's growable `Vector`s are inherently
host-only. The caller pre-offloads array arguments; stacks must come
from the converted `initstacks_*_cuda`, not the host `initstacks_*`.

Measure **loops offloaded**, never device-kernel count — adjacent
splittable loops merge, so fewer kernels can mean more offloaded.

**An HVP must offload exactly as well as its adjoint.** It is the adjoint
with every statement doubled by `hvp_double_stmt` — same loops, same
bounds, same snapshot sites. Any gap is a defect, not a property of the
mode. `validate_offload.jl` measures both against one ceiling.

Nothing downstream of `cgen_snapshot_save_dead` may assume a snapshot
save and its covering overwrite are **adjacent**. That holds for an
adjoint and is two statements early for an HVP, whose interleaved shadow
twin sits between them. mpnn is the witness: adjoint fully offloaded,
HVP launching one kernel per graph edge and per graph node. `cgen_`/
`jgen_` are the only stages that ingest already-doubled code, so they are
the only place this class can appear.

**Split-gate hazards.** Plain writes through unknown indices are trusted;
only provable overlap is refused. `cgen_split_loop_gate` therefore ends
with `cgen_overlap_proven_arrays` (`:overlap`) and
`cgen_rmw_hazard_arrays` (`:rmw`, gathered indices only, a wider rule lost
cellscatter, mpnn and entry_dead). It starts by folding a CSE'd
accumulation back to `A[e] += x` (`cgen_fold_accumulates`), without which
`idm_cse_acc` is refused. `validate_ownership.jl` is the oracle, and its
known-flag table stays empty; `idm_dup_write` is the designated flag.

**Save-only scalars.** `cgen_saveonly_synth` lets a register-tiled loop
offload although its stack saves read a scalar before assignment. It needs
the four conditions in the manual (end-of-body push, no later read, no read
outside loops that write it, every other read covered). Never relax
`cgen_use_before_def` for a read that computes something: only the bare
`X_stack[i] = v` save is exempt. Sabotage each guard: the ownership check
fails with relative error 0.9 to 1.5 when one is removed.

`jgen_` gets no free zero-trip guard. `cgen_launch_expr` survives a
zero-trip loop by launching one block whose threads all fail the device
kernel's own `__tid > trip_count` check; JACC's callee is a plain indexed
function with no such guard, and `range = 0` raises `DivideError` on a
live device. `jgen_launch_expr` must skip the launch instead.

**The `ii` callers use the strict gate only.** `cgen_reduction_only_loop`
(`snap_ii_classify`, `agen_ii_classify`) must stay byte-identical in result.
The GPU split uses `cgen_split_loop_gate`, which adds the accumulate-only and
consume-only idioms. Never route an `ii` caller through it, and never let the
strict gate call `cgen_array_ok_idiom`: AD output changes with no oracle
failing. `validate_ii_coverage.jl` pins `idm_src_consume` at zero for this.

**Plain writes are trusted, and the trust is tested.** Iterations of an
offloaded loop own disjoint elements for plain writes. After any change to a
gate or to `cgen_device_assign`, run `validate_ownership.jl` (CPU, no GPU),
`validate_idiom_oracle.jl` and `validate_atomic_guarantee.jl`. A new flag in
the ownership check is a finding: reproduce it on the previous source before
calling it a regression, and list it in `own_known()` with its reason.

A GPU parity pass means nothing unless the kernel **wrote** something.
Elements *compared* is the wrong count: an all-zero integer draw leaves
arrays at full length while every loop bound collapses to zero trips.
Count elements changed. GPU baselines live in their own `.gpu.yaml`
namespace, because `.yaml` is gitignored and `validate_zerotrip.jl`
deliberately draws integers from `[0, 2]`.

## Self-check before returning STADE code

- [ ] No `module`, `struct`, `@enum`, top-level `const`, or shared state
- [ ] Correct stage prefix on every function; no filesystem outside `io_*`
- [ ] `NamedTuple`s match the frozen shapes; no `:while`; no ad hoc keys
- [ ] Plain `Expr`/`Symbol`/`Number`, never a custom type
- [ ] Only documented upstream shapes as inputs
- [ ] No caller-supplied independents/dependents
- [ ] Comments one line, only where non-obvious
- [ ] No corpus kernel named in `STADE.jl`
- [ ] Loop-nest analysis walks actual `for` structure; no synthetic nests
- [ ] Duplicated `snap_*`/`agen_*` pairs edited identically; the
      `stade_*_check` assertions pass
- [ ] Tier B index expressions pure; block-local scalars via value tables
- [ ] `initstacks_*` has no free variable outside its parameter list
- [ ] A `keep_push_pop=false` claim went through `stade_validate_*_file`
      with the flag explicitly passed
- [ ] A GPU claim was measured for **both** `:adjoint` and `:hvp`, and the
      run changed a nonzero number of elements

## Discipline

1. Adversarial test kernels **before** implementing.
2. A minimal synthetic passing is not sufficient — validate on the corpus.
3. Check generated code directly, not just `ok=true`.
4. Run the corpus after every change, with matching flags.
5. Permanent regression test using the shape that caught the bug.
6. When a guard test breaks, the gate is wrong, not the test.
7. **Sabotage-test every new guard.** One never seen to fail is one you
   don't know you have.
8. **Measure the property, not a proxy.** Device-kernel count is not
   offload coverage; a liveness walker that cannot kill is not liveness;
   a single-kernel fingerprint is not default-path drift. Print the
   per-item detail beside any aggregate and check they agree.
9. **Verify a mechanism before writing it down in a comment.** A
   confidently wrong comment is worse than none.
10. **Keep a corpus kernel for every syntactic shape the analyses can
    encounter.** A shape absent from the corpus is a path that is never
    executed, and an unexecuted path is where defects survive. When
    adding a mechanism, enumerate the shapes it can meet and check one
    of each is represented.
11. **A loop that may not run has not run.** Any analysis concluding
    "this write establishes X" must prove the write executes: at the
    body's own top level, in BOTH arms of an `:if`, or inside a loop
    whose LITERAL bounds prove an iteration. A runtime bound proves
    nothing. The same applies in reverse -- an unprovable write also
    destroys anything established earlier, so it must clear the fact
    rather than be skipped.

    This exact bug has been written six separate times, in
    `norm_first_touch`, `cgen_last_assign_is_zero`,
    `ii_kill_and_collect!`, `cgen_loop_convergent_constant`, and the
    post-loop `known_consts` update in both `cgen_body` and `jgen_body`.
    Every instance produced silently wrong gradients; none was caught by
    the corpus unaided, and two needed a GPU run. The sixth was found by
    the fix for the fifth, not by the audit that followed it -- so audit
    by construction, not by recall.

    The audit that works: grep for a recursion into `stmt.body` whose
    result is ASSIGNED to accumulated state (`state = f(stmt.body, ...)`).
    That is the shape every instance took. Pure collectors that union,
    existence searches that return early, and codegen walkers that emit
    statements are all immune -- an unexecuted loop can only remove
    things from a union, never add a false fact. What is left after that
    filter is small enough to read line by line.

    Two ways to reach it. `test/validate_zerotrip.jl` runs the whole
    corpus with `int_lo = 0`, so any loop whose bound IS an integer
    argument executes zero times -- 19 kernels draw one, deterministically
    (validate_corpus seeds per kernel name). Run it alongside the ordinary
    corpus when touching any of these analyses.

    That does not reach a bound computed INSIDE the kernel, which is
    where all six were actually found. For those, retire a width past
    zero (`w = w - 3`, or `i_w0 - 5`) so every draw in the generator's
    range empties the loop. `retire_empty`, `entry_empty` and `ii_kill`
    are the three that exist; copy their shape rather than inventing one.

    The generator used to floor its draws at 1, on the stated grounds
    that "a zero-iteration loop would trivially pass". That is backwards:
    a zero-trip loop is where these bugs live. What passes trivially is a
    draw carrying NO SIGNAL -- every derivative check comparing 0 to 0 --
    and `val_generate_baseline` now rejects those explicitly instead,
    which is what makes `int_lo = 0` usable.