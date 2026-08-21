---
name: skill-stade
description: >
  Use whenever writing, reviewing, or refactoring Julia code that is part
  of STADE itself (the automatic differentiation engine for skill-jade
  kernels) — as opposed to the numerical kernels STADE differentiates,
  which are governed by skill-jade instead. Trigger any time the user
  asks for code implementing or modifying a STADE pipeline stage (parse,
  shape inference, activity analysis, snapshot/TBR analysis, derivative
  rules, linearization, tangent or adjoint codegen, GPU codegen, file
  I/O, validation), even without naming "skill-stade" or STADE
  explicitly.
---

# skill-stade: house style and architecture contract for STADE

STADE is a single-file (`STADE.jl`), function-only Julia AD engine for
skill-jade-compliant kernels. This skill is the contract that keeps
independently-developed pieces mergeable.

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
    skill-jade fuses a rectangular nest into one `for idx = 1:n*m` with
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

**Tier B**: a loop whose bound is reassigned inside an ancestor
sequential loop has no closed-form trip count. `agen_ragged_block`
resolves it into per-ancestor-iteration prefix/value tables, keeping the
site indexed and mutation-free; a narrow per-*stack* (never
per-occurrence) `push!`/`pop!` fallback remains for what can't resolve.

Tier B index expressions must stay **pure** — an embedded mutation
double-evaluates under `hvp_double_stmt`. Any block-local scalar a
formula depends on must be in `value_vars` and substituted through its
own value table, never read as a bare in-scope reference.

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

Measured effect (peak stack slots, `false` → `true`): `ttgc` 504 → 240,
`transformer` 2406 → 1095, `unet` 432 → 324. Every other kernel is
unaffected — dominated by loops with no value-needed local scalar.
**Don't justify extensions on breadth this won't deliver.**

## Validation

The corpus is **29 kernels** in `val-corpus/`. `validate_corpus.jl` runs
four oracles per kernel — tangent, adjoint, HVP (all central differences,
~1e-8 floor) and the **exact tangent-vs-adjoint dot-product identity**
`<Yb, J·Xd> == <J'·Yb, Xd>` (~1e-15, no epsilon). The exact oracle
catches systematic adjoint errors two orders below what FD can see; it
does *not* validate the tangent, since a bug shared by both codes
cancels, so it complements rather than replaces.

Current state: **116/116 in both stack modes**, 30 in-file self-test
suites.

- **Pass the flags you generated with** through to the *validator*, not
  only to generation. The validator's defaults will otherwise check a
  different mode's math than the one under test, and the flagged path
  goes unexercised.
- Run in halves — `validate_corpus_sel.jl` / `validate_corpus_stk_sel.jl`
  take kernel names. Backgrounded runs get reaped.
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

Status: 16/16 tested kernels validated exactly on real hardware. Four
(`mpnn`, `ttgc`, `transformer`, `unet`) blocked on shadow scalars a
kernel reads without receiving as parameters; Tier B blocked because
`cgen_ingest` rejects its own emitted indirect index. See
`stade_gpu_plan.md`.

Measure **loops offloaded**, never device-kernel count — adjacent
splittable loops merge, so fewer kernels can mean more offloaded.

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