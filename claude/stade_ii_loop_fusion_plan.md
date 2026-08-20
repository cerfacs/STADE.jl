# STADE: next step to minimize stack-variable footprint

**Context for whoever picks this up:** `STADE.jl` (the attached file) has a
working, validated feature (`fuse_ii_loops::Bool=false` on `stade_tangent`/
`stade_adjoint`/`stade_hvp`, matching `keep_push_pop`'s existing presence
on all three) that eliminates snapshot stacks for provably-safe loops. It
works, but on real kernels it currently eliminates far fewer stacks than
it could. This document says precisely why, and what to do about it. It
does not recount how the existing feature was built — that history isn't
needed to continue the work; only the current behavior and its one real
blocker are.

## What exists today (validated, don't re-derive)

- `ii_plan` (`snap_ii_plan`/`agen_ii_plan`, cross-checked by
  `stade_ii_plan_check`) classifies each `:for` loop as `:independent`
  (fully self-contained, fuse forward+backward into one pass),
  `:reduction` (a pure `+`/`-` accumulator, fuse into a backward-only
  distribution pass), `:mixed` (both present, treated like `:reduction`
  — see below for why), or unclassified.
- `agen_emit_ii_loop` does the actual codegen for all three kinds, reused
  as-is by `stade_hvp` (`hvp_emit` shares the exact same underlying
  `agen_forward_body`/`agen_backward_body` calls the adjoint path uses).
- Stack cleanup (`agen_used_stack_names`, `agen_block_boundary_vars`'s
  `ii_plan`-awareness) removes a stack from the generated signature when
  it's genuinely unused — not just when its var is `ii_plan`-covered
  (see below, this distinction matters). Applied independently to both
  `stade_adjoint`'s and `stade_hvp`'s own output, since code that shares
  one kernel's `initstacks_*` across both calls needs their signatures to
  stay in sync with each other, not just each individually correct.
- All of this is validated: central-difference comparison against the
  unfused baseline, plus direct diffing of generated code on real corpus
  kernels (`val-corpus/ttgc.jl`, `val-corpus/transformer.jl`), not just
  synthetic tests.

## The one thing actually blocking further stack reduction

**Eligibility is currently loop-scoped, not statement-scoped.**
`ii_body_has_escaping_array_write` refuses classifying a candidate loop
*at all* if *any* array write anywhere inside it (recursively) has its
array read elsewhere in the kernel — even when that escaping write has
nothing to do with the scalar variables you actually want to fuse.

**Concrete, verified example (`ttgc.jl`):** `res`/`res2` are written
inside the same `i_seq_k` loop that also computes `aeresk`/`factor`
(value-needed scalars). `res`/`res2` genuinely escape (read later via
`up[i] = res[i] / node_vol[i]`). Because that write sits in the same loop
nest as the scalar computation, at every level of nesting, `vere`/`re`/
`aerex`/`aerey`/`aerez`/`aeresk`/`factor` never get classified — their
stacks (`re_stack`, `aerex_stack`, etc.) still scale with `i_ncell` even
with `fuse_ii_loops=true`. Only `cavgx`/`cavgy`/`cavgz` (a separate loop
that never touches `res`/`res2`) currently benefits.

**Verify this directly before trusting it** — call
`ii_body_has_escaping_array_write(kernel.body, loop.body, kernel.sig.kinds,
active_map)` on the relevant loops in `val-corpus/ttgc.jl` and confirm it
returns `true` at every level containing that chain. Don't take this
document's word for it — re-verify against whatever revision of the file
is actually present, since the exact loop-variable names may have changed
again by the time this is read.

## What would fix it: two options, with their real risks

**Option A — statement-level fusion (the real fix).** Eligibility should
stop asking "can this whole loop be classified" and start asking "which
individual write-sites in this loop are safe to fuse," independent of
whatever else the loop contains. The scalar-level analysis
(`ii_escapes_nested`, `vn_ind`/`vn_red`) already computes this correctly
per-variable — it's the array-escape check that's coarsened it back down
to a whole-loop yes/no. Codegen (`agen_emit_ii_loop`) would need to fuse
only the proven-safe statements and leave everything else (the escaping
array write, e.g.) at its ordinary, unfused position within the same loop
body.

**The real risk, already proven to bite once:** the `:mixed` codegen's
first design tried exactly this kind of split (differentiate some
statements at one position, defer others to another) and produced
silently wrong gradients — a variable fused at the forward position can
be read by a statement deferred to the backward position, and that
variable's own "collect every contribution, then distribute" step needs
*all* contributions before it can safely run. The fix there was to give up
the split entirely and defer everything together. **Before attempting
Option A, explicitly check whether the same cross-dependency can occur
between a fused scalar and a deferred array write** (does any fused
statement's value feed into the escaping array write, or vice versa?). If
it can, the same "don't split, defer more together" logic likely applies,
and the benefit may be smaller than hoped.

**Option B — widen the fusion unit.** Instead of proving a smaller
statement subset safe, widen the *candidate* so the escaping consumer is
included in the same classified unit (the way `ttgc`'s own `cavgx` case
already works — its consumer sits inside the same outer `i_cell` loop, so
nothing escapes at that granularity). For `res`/`res2` specifically this
would mean including the `up[i] = res[i] / node_vol[i]` loop in the
same fusion unit as the assembly loop — a larger, structurally different
unit than anything currently classified, though a genuinely smaller one
to reason about than before, now that there's only one consumer loop
instead of three. Untried, uncertain payoff, and likely still blocked by
the same kind of cross-dependency risk as Option A once the unit contains
genuinely different computations.

**Recommendation:** attempt Option A first, scoped narrowly (one specific
escaping-array-write shape at a time, starting with `ttgc`'s `res`/`res2`
case specifically, matching a hand-built minimal reproduction before
touching the real kernel). Don't attempt both at once.

## Hard-won constraints — respect these, don't rediscover them by breaking them

- **A linear read is still an escape, not just a nonlinear one.** The
  existing `agen_var_value_needed!`/`snap_var_value_needed!` machinery
  (used elsewhere in this file for TBR) only flags *nonlinear* reads —
  wrong for escape-checking, because *any* downstream read (even a plain
  `y = y + t` copy) still accumulates into a shadow that must be
  distributed. Escape-checking uses `ii_expr_reads` (any occurrence,
  linear or not) for exactly this reason — don't swap it back to the
  nonlinear-only detector when extending this code.
- **`site_level_tbr` is always on now** (no longer an opt-in flag) —
  `agen_push_pop_source`'s reliance on `ectx.push_pop` means any new
  exclusion mechanism must go through `agen_ii_override_ectx`, not just
  modify `value_needed` (modifying `value_needed` alone is silently
  ignored once `ectx.push_pop` is set — this exact gap caused a real
  wrong-gradient bug once already).
- **Array writes still need the order-aware, ancestor-scoped treatment**
  `ii_escapes_nested`/`ii_find_ancestor_path` already give scalars — a
  blanket "read anywhere in the kernel" check is over-conservative (a
  read positioned before the write, at a non-repeating level, can never
  observe it) but a naive check is *unsound* for the reasons above. Reuse
  the existing path-walking machinery; don't rebuild it.
- **`agen_block_boundary_vars` needs `ii_plan` awareness for the stack to
  actually disappear from the signature**, on top of the loop's own
  push/pop being suppressed. A var only loses its stack when *every*
  write-site is `ii_plan`-covered, including sibling resets outside the
  classified loop (`ttgc`'s own `cavgx = 0.0` reset is a genuine example
  of a write-site that stays uncovered on purpose).

## Discipline that has consistently caught real bugs — keep doing this

Every extension to this feature has followed the same pattern, and every
time it was skipped or rushed, a real bug slipped through:

1. **Build adversarial test kernels before implementing**, targeting the
   specific new mechanism, not just a happy-path case.
2. **A minimal synthetic test passing is not sufficient evidence.** Two
   separate real bugs (an array-escape correctness bug, and the `:mixed`
   cross-dependency bug) passed a minimal synthetic test and only failed
   on `val-corpus/ttgc.jl` or `val-corpus/transformer.jl`. Always validate
   against the real corpus before trusting a new mechanism.
3. **Check generated code directly, not just central-difference error
   numbers.** A byte-identical or structurally-sane diff against the
   unfused baseline is what actually confirms a fix, not just `ok=true`.
4. **Run the validator after every change**: `validate_corpus.jl`
5. **Lock in a permanent regression test for whatever you fix**, using the
   real corpus shape that caught the bug, not a simplified version of it.