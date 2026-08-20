# STADE implementation plan: fused-sweep adjoint generation for iteration-independent loops

**Working name:** `fuse_ii_loops` (mirrors `keep_push_pop`/`site_level_tbr`: a new
`Bool = false` keyword on `stade_adjoint`/`stade_hvp` and their `_file`/`_corpus`
wrappers).

**Source baseline for this revision:** `src-work.zip`'s `STADE.jl` (8029 lines,
up from 7354). This supersedes the plan written against the earlier `src-ttgc`
snapshot — the `cgen_*`/`jgen_*` stages changed substantially in the interim,
and the `val-corpus` `ttgc` kernel itself grew into something much closer to
the real R&R Hydra-scale kernel the Hascoët paper's own benchmark describes.
Every change below reflects a direct read of the new source, not a guess.

**Origin:** Tapenade's `II_LOOP` differentiation mode and Hascoët/Fidanova/
Held, *Adjoining Independent Computations* (2001) — background unchanged from
the prior revision of this plan, not repeated here.

## Implementation status (updated after building and testing Phase 1+2)

Phases 1 and 2 (the eligibility oracle — `ii_plan`, no codegen) are
**implemented and tested**, not just designed. This section is the accurate,
current record; §§1–10 below are kept as the original design writeup and are
now historical in places where the real implementation made a different
(and better-justified) choice — noted inline where that happened.

**What exists:** `snap_ii_plan`/`agen_ii_plan` (duplicated per Hard Rule 7,
cross-checked by `stade_ii_plan_check`), classifying each eligible `:for`
loop as `:independent` or `:reduction`, keyed by `agen_site_key`. No `ectx`
change, no `agen_forward_body`/`agen_backward_body` change — purely
additive, appended after the file's existing self-test block.

**Classification granularity ended up outermost-first**, matching Tapenade's
own observed `II_LOOP` behavior more closely than the original plan assumed:
a loop nest (e.g. `ttgc`'s `for i_cell` containing `for i_loc`/`for i_k`) is
judged as ONE fusion unit at the outermost level that succeeds; only a
*failing* outer loop causes the walker to recurse into its own `:for`
children looking for a smaller eligible unit. This resolved the
`cavgx`-escapes-to-a-sibling-inner-loop question the original plan raised in
§7/§9 without needing any special-casing: `cavgx` never escapes the *outer*
`i_cell` loop, only an inner one, and escape is checked at whatever
granularity is being tested.

**Reused exactly what §3 predicted, plus more**: `cgen_reduction_only_loop`
(now returning a `synth` map, not a bare `Bool` — a `src-work` change, see
§0) is the primal-independence oracle, called on every `:for` loop
regardless of `i_seq_` naming. `cgen_locally_assigned_scalars` and
`cgen_scalar_reduction_vars` — found while reading the newer `cgen_*` code,
not anticipated in the original plan — turned out to be the exact existing
primitives needed to split a loop's locally-written value-needed scalars
into the `:independent` (fresh every iteration) and `:reduction`
(self-referencing accumulator, no fresh reset) buckets, replacing the
hand-rolled classification §4 originally sketched.

**Three rounds of testing found four real bugs**, all fixed, all now locked
in as permanent regression tests appended to the file:

1. *Corpus testing (`ttgc.jl`)* — a flat, order-unaware "does this name get
   read nonlinearly anywhere else" escape check produced a false positive:
   `ttgc` genuinely reuses `vere`/`i_loc`/`i_node` across two unrelated
   `i_cell` loops, and a later loop's *fresh* overwrite isn't a read of the
   earlier loop's value. Fixed by tracking program order and killing a var
   out of the "alive" set on a fresh (non-self-referencing) reassignment.
2. *Corpus testing (`ttgc.jl`)* — `i_loc`/`i_node` (integer loop-index
   vars) were being treated as escaping data, because the value-needed
   analysis was never gated on `active_map` (int-kinded vars can't be
   active) the way `snap_check_assign!` already gates it elsewhere. Fixed
   by adding the same gate.
3. *Corpus testing (`mg_vcycle.jl`)* — the recursive "check a failed
   loop's children" fallback was unsound: it applied a top-level-only
   escape scope (`kernel_body[idx+1:end]`) to a target nested inside a
   *repeating* ancestor (`i_seq_level`), missing the wraparound case where
   a read positioned earlier in the ancestor's own body is reachable on the
   next ancestor iteration. Fixed by restricting to top-level-only targets
   first (retracting `mg_vcycle`'s and `transformer.jl`'s previously-unsound
   positives — 13 of 15 original results were retracted), then building a
   proper `ii_find_for_path`/`ii_escapes_nested` extension with adversarial
   tests written *before* the implementation, confirmed against four
   deliberately corpus-independent stress kernels, then re-verified sound
   against the real `mg_vcycle`/`transformer` cases.
4. *Adversarial testing (not corpus-derived)* — six further corpus-
   independent kernels (genuine-escape-still-caught, buried array
   recurrence, unstable trip count, multi-assignment-site independent var,
   array-write exclusion genericity) found no further bugs, which is itself
   a meaningful (if weaker) result after three rounds that did find bugs.

**Validated** against all four of `src-work`'s validation harnesses after
every change: `validate_corpus.jl` (84/84 `ok`), `validate_corpus_keep_push_
pop_false.jl` (84/84 `ok`), `validate_corpus_keep_push_pop_false_site_level_
tbr_true.jl` (84/84 `ok`), `validate_corpus_gpu.jl` (232 `gen_ok`, 20
`gen_error` — an unrelated, pre-existing `initstacks*_b`-ingestion issue, not
caused by this work). Since Phase 1+2 never touches codegen, every one of
these runs is a check that nothing regressed, not a check that fusion itself
is correct — that can only be verified once Phase 3 exists.

**Real corpus result, current and verified (not the original plan's
estimate):** `ttgc.jl` — 2 sites (`:independent`, both top-level `i_cell`
loops); `unet.jl` — 2 sites; `mg_vcycle.jl` — 1 site (nested, via the
extension); `transformer.jl` — 11 sites (10× `:independent`, 1×
`:reduction`, nested inside its one sequential layer loop; structurally
spot-checked, not individually hand-audited beyond what the general
adversarial suite covers). Every other corpus kernel: 0 sites — either no
independent-loop structure at all, or (honestly) not yet provably so under
this analysis's current scope.

**`known_consts` threading: implemented, and turned out to matter more than
the original gap description suggested.** It's not a narrow edge case —
`cgen_reduction_only_loop`'s Phase 1 proof can fail *outright* (refusing the
entire loop) because of an *unrelated* self-referencing-with-reset scalar
(a `wb`-style accumulator) sitting anywhere in the loop body, even when that
variable has nothing to do with whatever this analysis actually cares about.
Confirmed directly before fixing anything: `cgen_reduction_only_loop(body,
var, Dict())` returned `nothing` for a loop containing both a genuinely
independent chain and an unrelated `wb`-style var, while `cgen_reduction_
only_loop(body, var, Dict(:wb=>0.0))` accepted the same loop. Fixed by
building `known_consts` locally in `snap_ii_plan_walk!`/`agen_ii_plan_walk!`,
fresh per body-list, populated by preceding literal scalar assigns and
invalidated by non-literal assigns or `:if`-touched vars — mirroring
`cgen_body`'s own convention exactly (confirmed by reading it directly: its
own recursive calls into a nested `:if` branch pass no `known_consts`
argument at all, so each body-list starts fresh, never inherited from a
caller). Three tests built before the fix: the positive case, and two
negative controls (pre-loop value mismatches the in-loop convergent
constant; pre-loop value isn't a literal at all) — all three matched
expectations. Real-corpus effect: none of the four positive sites changed
(2/2/1/11, same as before) — this fix's actual payoff is for Phase 3, once
generated adjoint accumulators (which have exactly this reset-to-known-
constant shape, e.g. `ttgc_b.jl`'s `resib`/`aerexb`) become inputs to a
similar proof; no primal kernel in the current corpus happens to combine an
unrelated `wb`-shaped var with an otherwise-eligible independent loop.

**Mixed `:reduction`+`:independent` splitting: implemented.**
`agen_ii_classify`/`snap_ii_classify` now check each half (`vn_red`,
`vn_ind`) against its own condition independently rather than requiring the
whole loop to be homogeneous, returning a new `:mixed` tag when both halves
are present and both pass. This is a genuine relaxation, not a new failure
mode: a pure reduction accumulator and a fully-contained independent chain
in the same loop body don't interact, so refusing the whole loop for not
being homogeneous was never buying any safety. Two negative controls
confirm the independence of the two checks is real, not accidental: the
independent half genuinely escaping still refuses the whole loop even
though the reduction half alone would pass; the reduction half being read
nonlinearly inside the loop (violating its own purity condition) still
refuses the whole loop even though the independent half alone would pass.
The existing `ii_stress_mixed` test from an earlier round was already
non-vacuous (both `t*t` and `acc*acc` are genuine nonlinear uses) and was
reused rather than rebuilt, with its expected outcome updated from `:none`
to `:mixed`. Real-corpus effect: none of the four positive kernels contain
a loop mixing both shapes, so this is validated only by the adversarial
suite, same honest caveat as the `:if`-nesting extension.

**Both gaps from the prior round are now closed.** The eligibility oracle
(`ii_plan`) has no more known, stated soundness or completeness gaps beyond
what's inherent to Phase 1+2's own scope (it classifies loop *shapes*, not
yet generating any code). Every extension in this file, across five rounds,
was built adversarial-tests-first.

## Phase 3: started, and two real correctness bugs found and fixed via
## actual codegen + numerical validation

Phase 3 (`agen_emit_ii_loop`, `ectx.ii_plan`, `agen_forward_body`/
`agen_backward_body` dispatch) is now implemented for `:independent` sites
only. `:reduction`/`:mixed` sites deliberately fall through to ordinary,
unfused codegen — reasoned through carefully before implementation: a
standalone reduction's adjoint total isn't fully accumulated until *after*
its downstream consumer's own (logically later, hence backward-sweep-
earlier) code has run, so fusing it the same way as `:independent` would
read the shadow too early, silently producing a wrong (understated) result.
(`ttgc`'s own `cavgx` does *not* need this separate treatment — at the
classification granularity this analysis actually uses, outermost loop
first, `cavgx`'s accumulation and its consumer are both inside the same
classified `i_cell` loop, so ordinary reverse-statement-order differentiation
already gets the ordering right. The problem case is a reduction whose
consumer sits outside the candidate loop entirely.)

**What's plumbed:** `fuse_ii_loops::Bool=false` threads through
`stade_adjoint`/`stade_adjoint_file`/`stade_adjoint_corpus`/
`stade_validate_adjoint_file`/`stade_validate_from_baseline`, matching the
existing `keep_push_pop`/`site_level_tbr` convention. `ectx` gained
`ii_plan`. `agen_forward_body` gained `lin_body`/`unsafe` keyword arguments,
threaded through its entire recursion (`:for`, `:if.then`, `:if.els`) so
`agen_emit_ii_loop` can be invoked at any nesting depth `ii_plan` finds a
site. Stated scope limitation: `stacks`/`agen_stack_map`/`agen_init_emit`
are untouched, so a fused var's stack is still *declared* (unused) in
`initstacks_*` — harmless for correctness, not yet cleaned up. Also
untested in combination with `keep_push_pop=false` (Tier A/B `:indexed`
layout is not `ii_plan`-aware).

**First result looked good, and was wrong.** A minimal hand-written test
kernel passed central-difference validation both fused and unfused, with
comparable error magnitudes. But running the same validation against the
real motivating kernels (`ttgc`, `mg_vcycle`, `unet`) immediately found a
serious bug: `agen_emit_ii_loop` fused a loop's *entire* backward
differentiation, not just the scalar `vn_ind`/`vn_red` subset Phase 1+2
actually proved safe. A value-needed **array** write in the same loop body
(`mg_vcycle`'s `r[j,i_seq_level]`, later divided by nothing directly but
read nonlinearly by the prolong statement; `ttgc`'s `res`, later divided by
`node_vol`) got its backward code moved to run before its true downstream
consumer had executed at all — `max_rel_err` ~1–2 on all three kernels, not
noise. Traced to the exact `push!`/`pop!`/`push!`/`pop!` pair in generated
code, confirmed the mechanism, reproduced on a minimal adversarial kernel,
and fixed by refusing `:independent`/`:reduction`/`:mixed` classification
whenever the loop body writes an array that's value-needed anywhere else in
the kernel (`ii_body_has_value_needed_array_write`, later generalized — see
below). `ttgc`/`mg_vcycle` immediately passed with byte-identical results
to unfused (since their remaining sites became `:reduction`-only or empty,
never dispatching to fusion) — but `unet` *still* failed.

**The real bug was more general than the first fix caught.** `unet.jl`
still had 2 `:independent` sites, unaffected by the array-write-refusal fix,
and still failed validation with `max_rel_err` ~1.9. Investigation traced it
to `p1[idx] = max(m1, m2)` (inside the classified loop) later read by
`p1pad[yi] = p1[idx]` — a **purely linear** copy, not a nonlinear read at
all. Every existing "is this var value-needed" check in this file
(`agen_var_value_needed!`/`snap_var_value_needed!`) deliberately only marks
a *nonlinear* read as "needed" — correct for its own purpose (does the OLD
value need protecting for someone else's partial derivative) but the wrong
tool here: a write's *shadow* accumulates from **any** downstream read,
linear or not (the adjoint of a straight copy is itself an identity
pass-through, not zero), and that accumulation must land before a fused
loop reads the shadow. This meant the escape-detection gap applied to
**both** the array check and the original scalar `ii_escapes_nested`
mechanism (which also only used the nonlinear-only detector) — a latent bug
in the scalar path too, not yet triggered by any existing test. Fixed by
adding `ii_expr_reads` (a true "does this expression mention this name at
all" detector, linear or nonlinear) and using it in both
`ii_kill_and_collect!` (scalar escape detection) and a rebuilt
`ii_body_has_escaping_array_write` (array escape detection: is this array
read *anywhere* outside the loop, not just nonlinearly). After this fix,
all four real kernels passed with byte-identical results to unfused — not
because the bug was masked, but because the corrected, sound eligibility
check found zero genuine `:independent` sites left in any of them (`unet`
dropped to 0 sites; `transformer` recomposed into mostly `:reduction`/
`:mixed`; `ttgc`/`mg_vcycle` unchanged). Fixing several existing regression
tests that turned out to have accidental array escapes of their own (a
shared test array both accumulated inside the classified loop and read
elsewhere) was part of restoring a clean run — each was fixed by removing
the accidental escape, preserving what the test was actually meant to
exercise, not by weakening the check.

**What's now actually validated, honestly stated:** the `:independent`
fusion *mechanism* itself (`agen_emit_ii_loop`) is validated correct via
central-difference comparison on a hand-built minimal kernel (both a pure-
scalar-chain case and a genuinely-non-escaping-array-write case), locked in
as a permanent end-to-end numerical regression test — the one test in this
file that exercises generated code, not just classification. But the
current, now-sound eligibility oracle finds **zero** genuine `:independent`
sites anywhere in the real corpus — every kernel that used to classify
`:independent` did so because of an escape-detection gap that's now closed.
This is a real, if deflating, result: proving `agen_emit_ii_loop` correct
took priority over preserving corpus coverage, and coverage should be
expected to stay near zero until the *array*-side of Phase 1+2 gets its own
proper escape analysis (mirroring `ii_escapes_nested`'s order-aware,
ancestor-scoped design, rather than the current blanket "read anywhere in
the kernel" conservative refusal) — real, well-scoped future work, not
attempted here given the priority on establishing correctness first.

**Full validation sweep after every change in this round**, all clean:
`validate_corpus.jl`/`validate_corpus_keep_push_pop_false.jl`/
`validate_corpus_keep_push_pop_false_site_level_tbr_true.jl` all 84/84 `ok`;
`validate_corpus_gpu.jl` 232 `gen_ok`/20 known `gen_error`, unchanged
throughout.

## Array-side order-aware escape analysis: built, and an honest result

Built `ii_body_has_escaping_array_write`'s order-aware replacement,
mirroring `ii_escapes_nested`'s design exactly: reuses `ii_find_
ancestor_path` to walk the same repeating/non-repeating levels, checking
each level's "after" (and "before", wraparound, for a repeating level) for
a read of the array. Deliberately does **not** model any "kill" for
arrays, unlike the scalar side — proving a later write safely overwrites
(kills) this write's contribution would need index-equality reasoning this
analysis doesn't attempt, so an array stays "checked" all the way out to
the kernel top level once flagged. Five adversarial tests built before the
implementation (unrelated read before the loop at a non-repeating level —
the direct analog of the earlier `vere` false positive; genuine read
after; wraparound through a repeating ancestor; same-repeating-level
after; fully contained) all matched expectations on the first run.
Confirmed numerically correct on the newly-accepted case via central-
difference validation.

**The honest result:** this did *not* restore coverage on `ttgc`/
`mg_vcycle`/`unet` — checked directly, not assumed. Their escapes are
genuine: `mg_vcycle`'s `r[j,i_seq_level]` really is read later by the
prolong statement within the same `i_seq_level` iteration; `unet`'s `p1`
really is copied by a later statement; `ttgc`'s `res`/`res2` really are
divided by `node_vol` after the assembly loop closes. Order-awareness
fixes a real imprecision (a name-reuse false positive, the array-side
analog of the `vere` bug found early in Phase 1+2), but none of these
three kernels' refusals were caused by that imprecision — they were always
correct refusals, just for a blunter reason than necessary. `transformer`/
`ttgc`'s site counts are unchanged. This is a case where an investment
in analytical precision was worth making (it's real, tested infrastructure
now, not a dead end) but didn't pay off for the specific motivating
kernels the way the plan's phrasing had hoped. Restoring real coverage on
those three would need something structurally different — e.g. widening
the fusion *unit* to include the escaping consumer too (the way `ttgc`'s
own `cavgx` case already works by construction, since its consumer sits
inside the same classified outer loop) — not a more precise version of
"does this array escape a single candidate loop," which is what both the
old and new checks answer.

**Full validation sweep after this round**, all clean: same 84/84 × 3,
232/20 GPU as every round before it.

## `:reduction` codegen: implemented, and this is the round with a real win

`agen_emit_reduction_adjoint` turned out not to need a new function at
all — `agen_emit_ii_loop` (built for `:independent`) already does exactly
the right thing (un-reversed loop, recompute-then-differentiate per
iteration, reduction var excluded from push/pop) when invoked from
`agen_backward_body`'s `:for` dispatch instead of `agen_forward_body`'s.
The only real design work was figuring out *where* it needs to run:
`agen_forward_body`'s handling of a `:reduction` site is barely touched —
the primal loop keeps its exact position and structure (the primal value is
often still needed elsewhere, e.g. `ttgc`'s own `aerex = cavgx * re`), just
with the reduction var excluded from `value_needed` so nothing pushes its
old value. `agen_backward_body`, when it reaches this same site at its
*normal*, unfused backward-sweep position, calls `agen_emit_ii_loop` there
instead of the ordinary reversed-pop treatment. This is what makes it safe,
unlike fusing at the forward position: by the time the reverse sweep
reaches this loop's own unchanged position, everything logically after it
in the primal has already run its own backward code (processed earlier in
the reverse walk), so the shadow being distributed is already fully
accumulated. `:mixed` sites still fall through to ordinary treatment — they
would need per-*statement* splitting within one body's own differentiation,
which `agen_backward_body` isn't built for today.

Validated on the real motivating kernel: `ttgc.jl`'s `cavgx`/`cavgy`/`cavgz`
`:reduction` site now generates genuinely different code (confirmed by
diffing fused vs. unfused output — no more `push!(cavgx_stack, cavgx)` in
the forward loop, an un-reversed `for i_loc = 1:4` distribution loop at the
backward position instead of the reversed pop-based one) and produces
**byte-identical** gradients to the unfused baseline — expected here since
each `i_loc` iteration writes a distinct `i_node`, making the summation
order-independent even at the floating-point bit level. `transformer.jl`
(11 `:reduction`/`:mixed` sites, mostly `:reduction`) also byte-identical.
A harder adversarial case built specifically to stress a different property
— every iteration accumulating into the *same* shared downstream slot,
where floating-point summation order genuinely differs between the
un-reversed distribution loop and the reversed unfused traversal — still
passed central-difference validation with appropriately small (not
bit-identical, correctly) error, confirming the mechanism is right and not
just coincidentally matching on the easy disjoint-write case. Both locked
in as permanent end-to-end numerical regressions.

**Full validation sweep after this round**, all clean: same 84/84 × 3,
232/20 GPU as every round before it.

**Real, validated coverage as of this round:** `ttgc.jl` and
`transformer.jl` now generate genuinely fused/distributed adjoint code via
`fuse_ii_loops=true`, both confirmed numerically correct — this is the
first round where `fuse_ii_loops=true` actually changes generated code for
real corpus kernels *and* that change is validated right. `:independent`
remains at zero real-corpus coverage (its mechanism is proven correct on
synthetic kernels, but every real site currently classifying `:independent`
turned out to have a genuine array or linear-read escape once analysis was
made sound). `mg_vcycle.jl`/`unet.jl` remain fully unfused (zero eligible
sites of any kind).

## Stack cleanup: found to be more nuanced than expected, fixed correctly

Started from the stated scope limitation ("a fused var's stack is still
declared, unused, in `initstacks_*`"), expecting this to be a small,
low-risk signature-trimming task. Checking the actual generated code first
(rather than assuming) found this framing was wrong in an important way:
`ttgc`'s `cavgx_stack` etc. are **not** simply unused — they're still
genuinely load-bearing, just for a completely different reason than
before. `agen_block_boundary_vars` (an existing, separate mechanism that
threads a var's value correctly across a *repeating ancestor's* own
iterations, unrelated to fusion) still pushes/pops `cavgx` at the `i_cell`
loop boundary. A blanket "drop every `ii_plan`-covered var's stack" would
have been a real bug — an undefined-variable error, not a cosmetic fix.

**Two-part fix, built in the right order.** First, a safe, provably-correct
post-hoc mechanism (`agen_used_stack_names`/`agen_drop_unused_stack_args`/
`agen_drop_unused_stack_allocs`): generate the adjoint body first, exactly
as before, then scan the *actual generated code* for which stack names
still have any `push!`/`pop!` call at all, and only drop what's provably
unreferenced — correct by construction, since it reads the already-correct
output rather than predicting it. This alone changed nothing on the
minimal test kernels, which exposed the real blocker: `agen_block_
boundary_vars` unconditionally treats *any* value-needed var written only
inside a nested sub-loop as needing restoration, with no awareness of
`ii_plan`'s own, strictly more precise escape proof. Second fix:
`agen_block_boundary_vars` gained an `ii_plan` parameter (default
`nothing`, every existing caller unaffected) and now excludes a var only
when *every* write-site of it — checked via a new `agen_ii_covered_
write_check`, not just the one inside the classified loop — is
`ii_plan`-covered. This is why `ttgc`'s `cavgx` correctly keeps its stack
(its `= 0.0` reset is a sibling statement outside the classified loop, not
covered) while a fully-contained synthetic kernel's var correctly loses
its stack entirely.

**Result, confirmed by checking, not assumed:** the minimal `stub_indep`
test kernel's adjoint signature shrank from `stub_indep_b(x, xb, y, yb, n,
t_stack, s_stack)` to `stub_indep_b(x, xb, y, yb, n)` — `initstacks_*`
correspondingly takes and allocates nothing. This is the first case where
`fuse_ii_loops=true` produces a strictly smaller, cleaner signature than
the unfused baseline, not just different internal code. `ttgc`'s own
stacks are confirmed unchanged (still declared, still genuinely used) via
a permanent regression test — guarded to skip gracefully rather than fail
when `val-corpus` isn't bundled alongside a standalone copy of this file,
since it's the only test in this suite with that dependency. Three new
adversarial kernels (fully-covered → removed, partially-covered → kept,
and the real-corpus retention check) all built and validated via central-
difference comparison, all locked in as permanent regressions.

**Full validation sweep after this round**, all clean: same 84/84 × 3,
232/20 GPU as every round before it.

**`:if`-nested targets: extended and tested, not just `:for`.** The
original gap ("a target reachable only through an `:if` is never even
considered") is closed. `ii_find_ancestor_path` generalizes the `:for`-only
path-finder to also descend into `:if.then`/`:if.els`, tagging each level
with whether its own containing body repeats (a `:for` body: yes, unless
it's the literal kernel top; an `:if` branch taken in isolation: never, since
one evaluation runs at most one branch exactly once — but the `:if`
*statement's own position* one level further out follows the ordinary rule,
so a whole conditional sitting inside a repeating ancestor still gets the
wraparound treatment it needs). The sibling branch is never examined at all,
by construction — only the one branch actually containing the target is ever
descended into, so a name reused independently in the other branch can never
register as a false escape. Six adversarial kernels were built before the
implementation (fully-contained, same-branch-after escape, post-if escape,
sibling-branch-reuse-is-safe, wraparound-through-a-repeating-`:if`-ancestor,
and the full combination), all matched expectations on the first run against
the real implementation. One pre-existing regression test from an earlier
round (`ii_stress_ifnest`) turned out to be vacuous — its own local variable
was never actually value-needed (used only under `+`), so it "passed" for
the wrong reason regardless of whether `:if`-nesting was handled at all;
fixed to use a genuinely nonlinear consumption and its assertion corrected
to match now-verified behavior, rather than left silently misleading.
Real-corpus effect: none of the four positive corpus kernels' site counts
changed (still `ttgc`: 2, `unet`: 2, `mg_vcycle`: 1, `transformer`: 11) —
two other corpus kernels do contain `:for`-under-`:if` structure but still
classify with zero sites, for reasons not individually traced (either Phase
1 recurrence or a genuine escape); not a red flag, just an honest note that
this extension's real-world yield in the current corpus was zero, validated
only by the adversarial suite, not by a new corpus win.

None of these gaps have caused an unsound *accept* so far (all err
toward under-classifying, confirmed by the adversarial suite's positive
controls still passing) — but "no bug found yet" is exactly the phrase that
preceded each of the four bugs found earlier, so treat this as tested-and-
currently-clean, not proven.

## 0. What changed in `src-work` that affects this plan

Diffing function signatures against the prior snapshot turns up a cluster of
changes, all inside `cgen_*`/`jgen_*` (the GPU-target stages), none yet
touching `agen_*`/`snap_*` (the adjoint-generation stages this plan modifies).
Confirmed by direct inspection, not assumed:

- **`cgen_reduction_only_loop`'s signature changed**: it now takes a third
  argument, `known_consts::Dict{Symbol,Any} = Dict{Symbol,Any}()`, and returns
  `Union{Nothing,Dict{Symbol,Any}}` (a `synth` map of provably-convergent
  locally-reset scalars) instead of a plain `Bool`. It no longer just answers
  "is this loop safe" — it also answers "and here's what to inject at the top
  of the split body to make it correct."
- **New convergent-constant proof machinery**: `cgen_loop_convergent_constant`/
  `cgen_terminal_value_walk`/`cgen_var_assigned_anywhere`/
  `cgen_all_assigned_scalars` prove that a locally-assigned scalar which
  *fails* the plain "already defined before self-reference" test can still be
  safe if it provably resets to the **same literal** on every control-flow
  path through the loop body, *and* that literal matches what the variable is
  known to hold on entry (`known_consts`, populated by `cgen_body` from a
  literal top-level assignment immediately preceding the loop). The
  motivating case is `clamped_sumsq_b`'s `wb`: additively self-referencing
  (`wb = wb + lossb[1]`) with a fresh `wb = 0.0` reset in *both* arms of an
  `:if`, later in the same body.
- **New idiomatic-reduction codegen** (`cgen_idiomatic_scalar_reduction` +
  `cgen_idiomatic_reduction_value`/`jgen_idiomatic_reduction_value`, gated by
  a new `keep_all_atomic::Bool = true` keyword threaded through
  `stade_gpu`/`stade_cuda`/`stade_amdgpu`/`stade_metal`/`stade_jacc` and their
  `_file` wrappers): a loop already proven safe by `cgen_reduction_only_loop`
  and matching the narrow single-statement shape
  `target = target ± f(arr[loopvar], ...)` gets compiled to a real `dot`/
  `sum(abs2,·)`/`mapreduce` (or `JACC.@parallel_reduce`) call instead of an
  atomic-accumulate kernel, when `keep_all_atomic=false`.
- **The previously-documented injectivity gap is closed**: the earlier
  skill-file note ("a genuinely non-injective thread-dependent index...was
  accepted before this fix and still is") no longer describes this codebase.
  `cgen_expr_injective_ok`, threaded through `cgen_device_body`/
  `cgen_device_assign` as `injective_dep` alongside the existing
  `thread_dep`, now refuses (falls back to atomic, doesn't silently accept)
  a write whose index is thread-dependent only through a chain that passes
  through an array read anywhere along it.
- **The `val-corpus` `ttgc` kernel and its reference adjoint grew
  substantially.** The version this plan previously analyzed (13 args, 6
  scalar stacks) is superseded by a version with 23 args that adds: 3D terms
  (`cavgy`/`cavgz`, `aerey`/`aerez`, `sky`/`skz`), a second residual
  (`res2`/`gamma`-weighted), periodic-boundary node-pairing fixups
  (`for k = 1:npernode_half`, gather into `resperio` then scatter back), and
  a genuinely sequential Jacobi mass-matrix relaxation
  (`for i_seq_ = 1:i_njac`, with `mup`/`up` coupled across sweeps through a
  *different* node index each time). `initstacks_ttgc_b` now declares
  **15** stacks, not 6: `cavgx/y/z`, `vere`, `re`, `aerex/y/z`, `aeresk`,
  `factor` (10 per-iteration scalars — the direct analogs of what the prior
  plan targeted) plus `res`, `res2`, `up`, `mup`, `auxu` (5 whole-array or
  cross-loop-nest values — a materially different problem, see §1).
- **No work toward this plan's feature exists yet.** `grep -i
  "fuse\|ii_loop\|fused"` across the new `STADE.jl` turns up nothing beyond
  unrelated uses of the English word "fuses" in comments. `agen_*`/`snap_*`
  are otherwise unchanged from the prior snapshot (confirmed via the same
  function-signature diff) — the `site_level_tbr`/`keep_push_pop`
  architecture this plan builds on is exactly as previously analyzed.

Net effect on this plan: the reusable oracle from §3 got *better* (it now
proves more than my original design assumed), the reduction-adjoint shape in
§6 has a second, more directly relevant precedent to draw on (§0's `wb`
convergent-constant proof, which is architecturally closer to an *adjoint*
accumulator pattern than the primal-side dot-product cases it was written
for), and the corpus now contains real examples of exactly the two things
this plan must correctly refuse (the Jacobi sweep) and a new good candidate
for what it should accept (the periodic-pairing loops) — see §7.

## 1. Scope (updated)

**In scope (Phase 1 target), unchanged in kind, bigger in surface area:**
loops with no genuine primal cross-iteration data dependency — both plain
iteration-independent loops and pure-reduction loops — get reverse-mode code
generated as a single un-reversed loop with recompute and adjoint
accumulation interleaved per iteration, eliminating storage for every
snapshot site whose sole enclosing dynamic scope is that loop. In the grown
`ttgc` kernel this now concretely means the `cavgx/y/z`, `vere`, `re`,
`aerex/y/z`, `aeresk`, `factor` stacks (10 of the 15) are the realistic
target, not a toy 6-stack example.

**New, explicit non-goal, visible now that the corpus grew:** the `res`,
`res2`, `up`, `mup`, `auxu` stacks are a *different* problem and Phase 1 must
not claim to touch them. These are kernel-argument arrays (or, for `auxu`, a
scalar reused *across* an entire Jacobi sweep, not within one loop's
iterations) written by one loop nest and read — not just accumulated into,
but read directly, e.g. `resperio[k,1] = res[i1]` — by a *different*,
later loop nest entirely. The II-loop fusion technique only removes storage
for a value that crosses forward-sweep-to-backward-sweep *within a single
loop's own iterations*; it says nothing about a value that has to survive
from one loop nest to a structurally different, later one. That's a cross-
loop-nest liveness problem, closer in spirit to what `site_level_tbr` already
targets (proving a snapshot dead) than to what this plan's fusion targets
(proving a snapshot *unnecessary in kind*, not just prunable) — worth a
dedicated follow-up plan of its own, not folded into this one.

**Also explicitly out of scope, confirmed correct-to-exclude by the new
corpus itself:** `for i_seq_ = 1:i_njac`'s Jacobi relaxation is exactly the
"sweep-style recurrence" shape `cgen_reduction_only_loop`'s own test suite
(`sweep_plan` test, §3 below) already exists to refuse — `mup[i_k]` written
during sweep *k*, `up` read at `nbr[i_k]` (a different index, via the mesh
connectivity) during sweep *k+1*. Confirms the primal-independence oracle
this plan reuses gives the right answer on the harder, real kernel, for
free, with no new logic.

## 2. Why the eligibility test cannot be "loop var isn't `i_seq_`"

Unchanged from the prior revision: `cavgx`'s accumulation loop is a genuine
reduction that must be fusable even though it carries state across
iterations (confirmed by the user this loop was meant to be `i_seq_`-tagged
and wasn't, a corpus typo, not a rule change). Eligibility is a dataflow
property, not a naming-convention lookup — see the prior plan's §2 for the
full argument, which the new source doesn't change.

## 3. Phase 1: primal-independence oracle (updated for the new signature)

Reuse `cgen_reduction_only_loop(body, loopvar, known_consts)` as the
primal-independence proof, but account for its richer new contract:

- It now returns `nothing` (refused) or a `synth::Dict{Symbol,Any}` — the
  scalars it proved converge to a known literal despite not being defined
  before their own self-reference. An eligibility wrapper,
  `snap_ii_primal_eligible`/duplicated `agen_ii_primal_eligible` (Hard Rule
  7), should treat `synth` as a genuine result to carry forward into Phase 2
  and §6, not discard it — a loop var flagged in `synth` needs the same
  literal injected at the top of the *fused adjoint* loop body too, if that
  same variable turns out to also be an adjoint accumulator inside the
  eligible loop (plausible: `resib`/`aerexb`/`vereb`-style adjoint
  accumulators in `ttgc_b.jl`'s own reverse sweep already have exactly this
  reset-to-0.0-every-iteration shape — see §6).
- It must be called on **every** `:for` loop, not only `i_seq_`-prefixed
  ones. Confirmed by re-reading its only current caller
  (`cgen_body`/`jgen_body`, STADE.jl ~5680): today it's gated by
  `!stmt.sequential ? Dict{Symbol,Any}() : cgen_reduction_only_loop(...)` —
  i.e. a plain-named loop is *already* unconditionally trusted (empty
  `synth`, no proof needed) and only an `i_seq_`-tagged loop is run through
  the prover. This is precisely the eligibility split this plan needs
  (§2's plain-independent vs. reduction distinction) and should be copied
  verbatim into the new `agen_ii_primal_eligible` dispatch rather than
  reinvented — see §5.
- Reject any loop `agen_tier_b_offender` already flags (Tier B raggedness
  stays out of scope per §1, unchanged from the prior revision).

`known_consts` itself needs an adjoint-side source. `cgen_body` builds it by
scanning literal top-level scalar assignments immediately preceding a loop in
the *primal*. The adjoint-side equivalent is `agen_local_shadow_inits`/
`agen_local_primal_inits` (STADE.jl ~2421-2422), which already initialize
every shadow/primal local to a literal at the top of `agen_adjoint_emit`'s
generated function — so, unlike `cgen_body`, `agen_*`'s `known_consts` is
almost always trivially available (every local adjoint accumulator starts at
a known `0.0` by construction, already emitted before `agen_forward_body`
even runs). This is a genuine simplification versus the primal-side case,
worth confirming explicitly during implementation rather than assuming.

## 4. Phase 2: adjoint-side eligibility

Unchanged in substance from the prior revision (extend the existing
`site_level_tbr` direction-rule walk — `snap_fwd_walk!`/`snap_fwd_walk_loop!`
and the `agen_` duplicates — to also track whether a nonlinear read stays
inside the write's own iteration). One addition given §0/§3's findings: the
new `synth` map from Phase 1 should be threaded into this analysis so that a
write covered by `synth` (provably reset every iteration) is treated as
"freshly defined at iteration start" for the *adjoint* accumulator sharing
its name, exactly the way `cgen_reduction_only_scalar_walk`'s own `synth`
parameter already treats it on the primal side (STADE.jl ~5178: `defined =
union(defined_in, Set{Symbol}(keys(synth)))`) — reuse that exact convention
rather than inventing a parallel one.

**New frozen shape**, unchanged from the prior revision:
`ii_plan = Dict{Any,Symbol}` keyed by `agen_site_key(body, idx)` of the loop
statement, valued `:none | :independent | :reduction`. Computed by
`snap_ii_plan`/duplicated `agen_ii_plan`, cross-checked via a new
`stade_ii_plan_check` following the exact `stade_site_level_tbr_check`
pattern (Hard Rule 7).

## 5. Phase 3: codegen restructuring — now with a concrete precedent to mirror

This is still the largest piece of work, but the new source gives it a
direct, already-shipped architectural precedent that the prior revision of
this plan didn't have: `cgen_body` (STADE.jl ~5680-5701) *already* does
exactly this shape of per-statement dispatch on `:for` loops:

```julia
elseif stmt.kind == :for
    flush_pending!()
    synth = !stmt.sequential ? Dict{Symbol,Any}() : cgen_reduction_only_loop(stmt.body, stmt.var, known_consts)
    if synth !== nothing && !cgen_contains_stackop(stmt.body)
        # special-cased emission (idiomatic reduction, or a split kernel)
        ...
    else
        push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, cgen_body(stmt.body, ...)))
    end
    delete!(known_consts, stmt.var)
```

`agen_forward_body`/`agen_backward_body` gaining an analogous branch —
"is this `:for` statement `ii_plan`-eligible? if so, hand it to
`agen_emit_ii_loop`/`agen_emit_reduction_adjoint` instead of the default
per-statement walk" — is consistent with, not foreign to, this codebase's
existing style for exactly this kind of loop-level special-casing. It
doesn't reduce the actual amount of new code needed (§5's core claim from
the prior revision stands: this changes *whether the kernel body is walked
once or twice* for eligible loops, which `cgen_body`'s dispatch never had to
do, since `cgen_body`/`cgen_backward_body` don't exist as a forward/backward
pair the way `agen_forward_body`/`agen_backward_body` do), but it de-risks
the *shape* of the change considerably: the `ectx.ii_plan` field, the
per-statement `if`/`else` on loop eligibility, and the "fall through to
today's behavior otherwise" discipline should all be built to match
`cgen_body`'s own dispatch idiom, including its detail of clearing tracked
state after the loop (`delete!(known_consts, stmt.var)` ↔ whatever
`ectx.ii_plan`-adjacent state Phase 2 accumulates should be scoped the same
way).

Everything else from the prior revision's §5 is unchanged: `agen_emit_ii_loop`
builds one un-reversed loop (primal statements, then that same iteration's
adjoint statements via `agen_backward_body`'s existing per-statement logic
called at loop-body granularity); `agen_backward_body` skips a statement
already claimed via `agen_site_key`; `ectx` gains `ii_plan::Union{Nothing,
Dict{Any,Symbol}}`, `nothing` when `fuse_ii_loops=false` for byte-identical
"off" output.

## 6. The two code shapes — `:reduction` case now has a second, better precedent

`:independent` (Tapenade's literal recipe) is unchanged from the prior
revision.

`:reduction` gets a materially better precedent from §0's discovery. The
prior revision's example was `cavgx`'s primal accumulation (adjoint = a
simple broadcast, no recompute needed). The new source's own test suite
(`wb_plan`, STADE.jl ~7955) proves the *same convergent-constant machinery*
correctly handles a shape that is structurally an **adjoint** accumulator,
not a primal one:

```julia
wb = 0.0
for i_seq_x = n:-1:1
    wb = wb + loss[1]
    if u[i_seq_x] > 0.0
        ub[i_seq_x] = ub[i_seq_x] + wb
        wb = 0.0
    else
        wb = 0.0
    end
end
```

— additive self-reference, reset via *both* arms of an `:if`, matching the
prior-value in `known_consts` from a literal pre-loop assignment. This is a
strictly harder case (branch-conditional reset) than anything `ttgc_b.jl`'s
own reverse sweep currently needs (its `resib = 0.0`/`aerexb = 0.0`/
`vereb = 0.0`/`factorb = 0.0`/`aereskb = 0.0` resets are all unconditional,
one per loop body, no `:if`), so the machinery §3 recommends reusing is
already proven on a harder version of the exact problem Phase 2's `:reduction`
code shape needs solved for adjoint accumulators living *inside* a fused
`:independent` loop (not just accumulators feeding a separate post-loop
reduction). Concretely: when `agen_emit_ii_loop` builds the fused loop for
`ttgc`'s `vere`/`re`/`aerex`/`aeresk`/`factor` chain, its own adjoint
accumulators (`resib`, `aerexb`, `vereb`, `factorb`, `aereskb`) need the same
"provably resets every iteration, safe to leave un-pushed" proof — reuse
`cgen_loop_convergent_constant`'s *algorithm* (walk the fused loop body,
same `:unchanged`/`:unknown`/literal state machine) for this internal-to-
`agen_emit_ii_loop` bookkeeping rather than writing a new one.

`cgen_idiomatic_scalar_reduction` (the new BLAS-call-shaped reduction
detector) is **not** directly reusable here — it targets a different output
(a `dot`/`sum`/`mapreduce` call replacing an entire primal loop for GPU
performance), not an interleaved recompute-then-adjoint loop. Noting this
explicitly so a future implementer doesn't reach for it by pattern-matching
the name; the relevant precedent is the *convergent-constant proof*
underneath it, not the reduction-call codegen built on top of it.

## 7. Interaction with `keep_push_pop`/`site_level_tbr`/Tier A/B, and with the
new `keep_all_atomic` GPU flag

Unchanged from the prior revision for `keep_push_pop`/`site_level_tbr`/Tier
A/B: fusion takes priority for loops it applies to, `agen_stack_map`/
`agen_layout`/`agen_init_emit` must filter `ii_plan`-covered sites out before
sizing anything (mirroring the existing `agen_exempt_vars` filtering
precedent).

New consideration from §0: `keep_all_atomic` is a `cgen_*`/`jgen_*`-only flag
and has no direct interaction with `fuse_ii_loops` (an `agen_*`-only flag) —
they operate on different generated functions (`ttgc_b` vs. `ttgc`/its GPU
variants) and neither reads the other's output. The only place they could
plausibly interact is if `fuse_ii_loops=true`-generated adjoint code is later
run back *through* `cgen_*`/`jgen_*` for its own GPU port — see §8, this
stays a follow-up, not a Phase 1 dependency.

## 8. GPU interaction — still a follow-up, but more concrete now

The prior revision's caution stands (don't claim GPU behavior without tracing
the actual generated `Expr` through `cgen_device_body`), but §0's findings
make the shape of that follow-up clearer:

- `cgen_contains_stackop`'s gate is unchanged — a fused loop (no
  `push!`/`pop!` at all) would, untested, read as trivially clean to it,
  same prediction as the prior revision.
- Once fused, `agen_emit_ii_loop`'s output, if ever run back through
  `cgen_body`, would hit the **same** `stmt.kind == :for` dispatch branch
  described in §5 — meaning a fused adjoint loop could, in principle, become
  a *second* candidate for `cgen_reduction_only_loop`'s own proof, on top of
  already being proven independent once for the primal. Whether the fused
  loop's adjoint statements (which reference `..b` shadow variables, `beta`,
  `gamma`, `dt`, etc., not present in the primal) still satisfy
  `cgen_reduction_only_loop`'s array-index-mismatch and self-reference checks
  is an open question that needs its own stress kernel, not an assumption —
  flagging this explicitly rather than folding an optimistic claim into
  Phase 1's definition of done.
- The closed injectivity gap (§0) removes one specific *known* risk (a fused
  loop's scatter-style writes, e.g. `res[i_k_node] += auxres`-shaped
  accumulation now inside one fused loop instead of split across two sweeps,
  won't silently accept a non-injective index) but doesn't establish that GPU
  codegen on fused adjoint loops works — it only means that if it's ever
  tried, this particular failure mode is already guarded against.

## 9. Testing plan (updated for the grown corpus)

Same staged structure as the prior revision, with the target kernel now
concretely identified:

1. Add `fuse_ii_loops::Bool = false`, thread it through `stade_adjoint`/
   `stade_hvp`, their `_file`/`_corpus` wrappers, and
   `stade_validate_from_baseline` from the start (don't repeat the bug the
   `keep_push_pop`/`site_level_tbr` history already had to fix).
2. Central finite-difference validation across the full corpus,
   `fuse_ii_loops=true` vs `false`, both `keep_push_pop` states — unchanged
   goal, bigger corpus (this `val-corpus` snapshot has more kernels than the
   one the prior revision worked from; re-enumerate at implementation time
   rather than assuming the old count).
3. Dedicated stress kernels — same four shapes as the prior revision
   (elementwise-independent chain, pure reduction, primal-independent-but-
   adjoint-ineligible, mixed eligible/ineligible nest), **plus** two new ones
   the grown corpus motivates directly:
   - an adjoint accumulator reset via `:if` branches inside an otherwise-
     eligible loop (the `wb_plan` shape from §6, ported to the adjoint side —
     this is the one case not already covered by `ttgc`'s own reverse sweep,
     since `ttgc_b.jl`'s resets are all unconditional today),
   - a periodic-pairing-style loop (`for k = 1:npernode_half`, gather-then-
     scatter through two arbitrary index arrays) as a positive `:independent`
     candidate distinct from `ttgc`'s mesh-assembly shape — confirm
     `cgen_reduction_only_loop`/the new eligibility wrapper actually accepts
     it before assuming it does.
4. `ttgc` itself, using the **current** (grown) reference: after fusing the
   10 scalar per-iteration stacks (§1), `initstacks_ttgc_b` should shrink
   from 15 declared stacks to 5 (`res`/`res2`/`up`/`mup`/`auxu` remaining,
   per §1's explicit non-goal) — document this as the actual target, not the
   "collapse to Tapenade's empty `initstacks`" framing the prior revision
   used against the smaller kernel, which no longer matches what's in the
   corpus. The Jacobi loop and periodic-fixup loops are real,
   already-present positive/negative test cases (§1) — use them rather than
   only synthetic stress kernels.
5. `stade_ii_plan_check` cross-check (§4), run automatically whenever
   `fuse_ii_loops=true`, unchanged from the prior revision.

## 10. Definition of done for Phase 1 (updated — reflects actual completion)

- [x] `agen_ii_classify`/`snap_ii_classify` reuse `cgen_reduction_only_loop`'s
      3-argument signature correctly, including its `synth` return value,
      called on every `:for` loop regardless of `i_seq_` naming (the
      original checklist item worried about inheriting `cgen_body`'s own
      `!stmt.sequential` gate by accident — the actual implementation makes
      the plain-vs-`i_seq_` split irrelevant by construction: eligibility is
      driven by `cgen_locally_assigned_scalars`/`cgen_scalar_reduction_vars`,
      not by the loop's own naming)
- [x] `snap_ii_plan`/`agen_ii_plan` pair implemented, duplicated per Hard
      Rule 7, cross-checked via `stade_ii_plan_check`
- [ ] `agen_emit_ii_loop` (§6 `:independent`) — **not started**; Phase 1+2
      deliberately stopped short of codegen
- [ ] `agen_emit_reduction_adjoint` (§6 `:reduction`) — **not started**
- [ ] `agen_forward_body`/`agen_backward_body` dispatch on `ectx.ii_plan` —
      **not started**; `ectx` has no new field yet
- [ ] `agen_stack_map`/`agen_layout`/`agen_init_emit` filtering — **not
      started**, depends on the above
- [x] (Partial, Phase 1+2 scope only) Full corpus + regression stress
      kernels pass under the eligibility oracle: three real bugs found via
      corpus testing and fixed (order-unaware escape check, missing
      `active_map` gate, unsound nested-target scope), all locked in as
      permanent regression tests; six further corpus-independent
      adversarial kernels found no additional bugs; four more adversarial
      kernels built *before* the nested-`:for` extension existed, all
      matched expectations once it did. **Not yet done**: central-
      difference validation of `fuse_ii_loops=true`-generated code, since
      no code is generated yet — that's a Phase 3 item, listed above.
- [x] All four of `src-work`'s validation harnesses re-run after every
      change (additive and refactoring alike): `validate_corpus.jl`,
      `validate_corpus_keep_push_pop_false.jl`,
      `validate_corpus_keep_push_pop_false_site_level_tbr_true.jl` all
      84/84 `ok`; `validate_corpus_gpu.jl` 232/252 `gen_ok`, with the
      remaining 20 confirmed to be the pre-existing, unrelated
      `initstacks*_b`-ingestion class, not a regression from this work
- [ ] `fuse_ii_loops=false` byte-identical output — **not yet meaningful**,
      no such flag exists yet (no codegen)
- [x] (Revised target) `ttgc`'s eligible sites measured against the
      **current** (grown, 15-stack) reference: 2 top-level `:independent`
      sites found and verified, covering the `cavgx/y/z`, `vere`, `re`,
      `aerex/y/z`, `aeresk`, `factor` scalars across both `i_cell` loop
      nests — the actual stack-count reduction this implies can only be
      confirmed once Phase 3 generates real code
- [x] The Jacobi loop(s) in `ttgc` confirmed (directly, via `agen_ii_
      classify` called on them explicitly, not inferred) to classify
      `:none`; the periodic-pairing loops confirmed to classify `:none` too
      (correctly — no active scalar content, only int index bookkeeping and
      array writes, both out of scope by design) — note this is a
      **correction** to the original plan's expectation that the periodic
      loops would be a good `:independent` positive test candidate; they
      aren't, and that's fine
- [ ] GPU interaction (§8) — still an explicit follow-up, still untested,
      unchanged from the original plan

**New checklist items the original plan didn't anticipate, now done:**
- [x] Dead-code audit after three rounds of incremental editing — removed
      `ii_escapes`/`ii_top_level_index` once confirmed fully superseded by
      `ii_escapes_nested` (a strict generalization, verified equivalent on
      top-level targets), re-validated all four harnesses afterward
- [x] Adversarial-before-implementation discipline established and
      followed for the `:for`-nesting extension specifically (four stress
      kernels written and run against the pre-extension code to confirm
      expected pre-extension behavior, *then* the extension was built and
      the same four kernels re-run to confirm the expected post-extension
      behavior) — worth continuing for the `:if`-nesting extension and the
      nested-`known_consts` gap when either is next attempted

**Not yet attempted, explicitly out of this session's scope:**
- [x] ~~`:if`-nested target scope extension~~ — **done**: `ii_find_
      ancestor_path` handles `:for` and `:if` ancestors uniformly, six
      adversarial tests built before the implementation, all passed on
      first run, full validation sweep unchanged afterward
- [x] ~~`known_consts` threading for nested targets~~ — **done**: built
      locally per body-list in the plan walkers, mirroring `cgen_body`'s
      exact convention; found (before fixing) that this was blocking whole-
      loop classification outright in the presence of an unrelated
      `wb`-style var, not just missing a narrow edge case; three tests
      (one positive, two negative controls) built before the fix
- [x] ~~Mixed `:reduction`+`:independent`-in-one-loop splitting~~ — **done**:
      each half now checked independently, new `:mixed` tag; two negative
      controls confirm the independence of the checks is real

**Remaining, honestly: nothing currently known within Phase 1+2's own
scope.** Five rounds of adversarial-tests-first extension there have not
turned up a further gap. Phase 3, however, has real, well-understood open
work — see the "Phase 3: started..." section above for the full account:

- [x] ~~`agen_emit_reduction_adjoint`~~ — **done**: turned out to need no
      new function, `agen_emit_ii_loop` (built for `:independent`) already
      does the right thing when invoked from `agen_backward_body`'s `:for`
      dispatch instead of `agen_forward_body`'s. Validated on `ttgc.jl`
      itself (structural diff confirmed, byte-identical gradients) and a
      harder shared-accumulation-target adversarial case (correctly
      non-identical but within tolerance). This is the first round with
      real, validated `fuse_ii_loops=true` coverage on real corpus kernels.
- [x] ~~Array-side escape analysis for Phase 1+2~~ — **done, order-aware,
      mirrors `ii_escapes_nested`'s design** — but did **not** restore
      coverage on `ttgc`/`mg_vcycle`/`unet`, checked directly: their
      escapes are genuine, not artifacts of the old blanket check. See
      the "Array-side order-aware escape analysis" section above for the
      honest account and what restoring real coverage there would
      actually require (widening the fusion unit, not a more precise
      single-loop escape check)
- [x] ~~`stacks`/`agen_stack_map`/`agen_init_emit` are not `ii_plan`-aware~~
      — **done, and more nuanced than the original framing suggested**: a
      var's `ii_plan` coverage does not by itself mean its stack is unused
      (see the "Stack cleanup" section above) — fixed via a safe post-hoc
      scan of generated code plus making `agen_block_boundary_vars` itself
      `ii_plan`-aware. Confirmed: a fully-contained synthetic kernel's
      signature shrinks correctly; `ttgc`'s own stacks correctly stay
      (block-boundary restoration genuinely still needs them)
- [x] ~~`:mixed` codegen~~ — **done**: first design (split across forward/
      backward positions) had a real, non-obvious bug, caught by real-
      corpus validation (`transformer.jl`), fixed by simplifying to defer
      everything to the backward position rather than patching the split.
      See the "`:mixed` codegen" section above.
- [ ] `keep_push_pop=false` interaction with `fuse_ii_loops=true` is
      untested
- [ ] HVP (`stade_hvp`) does not have `fuse_ii_loops` threaded through at
      all — this round scoped to the adjoint path only

## `:mixed` codegen: real design flaw found, fixed by simplifying rather
## than patching

First attempt split the differentiation across two positions, mirroring
how `:independent` and `:reduction` each work alone: `vn_ind` fused at the
forward position, `vn_red` deferred to the backward position. A minimal
synthetic test passed. `transformer.jl`'s own `:mixed` sites — a
LayerNorm-style variance computation — failed central-difference
validation with `max_rel_err` ~1.4, not noise.

**The bug was a genuine design flaw, not an implementation slip.**
Debugging required building an isolated reproduction matching
`transformer`'s exact nesting shape (abstract reasoning about the generated
code wasn't converging fast enough) before the mechanism became clear: a
`vn_ind` variable can be read by *both* an ordinary statement and a
`vn_red` statement's own accumulation term (`diff` feeding both `y[...] +=
diff*diff` and `s2 = s2 + diff*diff`). `diff`'s own "collect every
contribution, then distribute to its inputs" step needs *both*
contributions available before it runs — but the `vn_red`-side contribution
isn't available until the backward position. Splitting `diff`'s
differentiation across both positions silently dropped that contribution:
the shadow accumulated correctly but was never distributed or reset,
because the one statement that would have done so (`diff`'s own write) was
the one skipped at the position where the drop mattered.

**Fixed by simplifying, not patching around it.** Rather than detecting the
cross-dependency case and treating it specially (real complexity, more
surface for another version of the same bug), `:mixed` now gets the exact
same treatment as `:reduction`: both `vn_ind` and `vn_red` excluded from
push at the forward position (still correct, still the actual benefit),
but *all* differentiation deferred to the backward position rather than
splitting any of it forward. This is provably safe because `vn_ind`'s own
"no escape" proof only ever guaranteed nothing *outside* the loop reads
it — it never required forward-position fusion specifically, that was
just an optimization opportunity that turned out to be unsafe in this
case. The now-unused `skip_stmts` machinery (added for the split design)
was removed rather than left as dead code.

Verified by diffing generated code (not just numerically): the fix
produces a real structural change — `diff_stack` fully eliminated, the
backward-position distribution loop un-reversed and matching primal
direction — not just a numerically-lucky match. Both `ttgc.jl` (already
`:reduction`-only, unaffected) and `transformer.jl` (now genuinely fixed)
validated byte-identical to unfused. Two regression tests locked in: the
simple case (no cross-dependency, confirms the basic mechanism still
works) and the cross-dependency case itself, built to specifically catch
a return of the split-based bug.

**Full validation sweep after this round**, all clean: same 84/84 × 3,
232/20 GPU as every round before it.

**State of Phase 3 as of this round:** all three `ii_plan` classification
kinds (`:independent`, `:reduction`, `:mixed`) now have real codegen.
`ttgc.jl` and `transformer.jl` both generate genuinely fused/distributed
adjoint code, both confirmed numerically correct via central-difference
validation and, where feasible, direct diffing of generated code.
`:independent`'s own mechanism remains proven-correct-but-zero-real-
coverage (every real site has a genuine array or linear-read escape, as
found two rounds ago) — `:reduction` and `:mixed` are what's actually
carrying real coverage on the corpus right now.

## User-reported: `re`/`aerex`/`aerey`/`aerez` still scale with `i_ncell`,
## and `site_level_tbr=true` made `cavgx/y/z_stack` "worse" — investigated

Both parts checked directly, not assumed.

**Part 1 (`re_stack`/`aerex_stack`/etc. still growing with `i_ncell`) is
real, and expected — a known, already-documented limitation, now traced
to its exact mechanism.** `ii_plan` classifies only `ttgc`'s `cavgx`/
`cavgy`/`cavgz` accumulation loop; the sibling loop computing `vere`/`re`/
`aerex`/`aerey`/`aerez`/`aeresk`/`factor` is never classified at all,
traced precisely: `res`/`res2` get written *inside* the same `i_k` loop
that computes `aeresk`/`factor`, and `res`/`res2` genuinely escape (read
later via `up[i] = res[i] / node_vol[i]`). The current eligibility check
refuses the *whole* containing loop wherever *any* escaping array write
sits inside it, at every level of nesting — since `res`'s write is
co-located with `aeresk`/`factor`'s own computation in the same `i_k`
loop, there is no smaller loop-shaped boundary that separates them at the
current granularity. This is exactly the "fuse only the proven-safe
statements, not the whole loop" gap already stated as future work, not a
regression — confirmed by direct classification tracing (`ii_body_has_
escaping_array_write` returns `true` at every candidate level containing
that chain).

**Part 2 (`site_level_tbr=true` keeping `cavgx/y/z_stack`) was not "worse
optimization" — it was a genuine, serious wrong-gradient bug**, confirmed
by central-difference validation (`max_rel_err ~0.17` on `ttgc.jl` before
the fix, `ok=false`). Root cause: `agen_push_pop_source` ignores
`value_needed` *entirely* once `ectx.push_pop` is set (`site_level_tbr`'s
own, `ii_plan`-unaware per-site Dict takes over the push/pop decision
completely) — so every place in this file that suppresses a push/pop pair
by excluding a var from `value_needed` had **zero effect** whenever
`site_level_tbr=true`. For `:reduction` specifically this was worse than
a missed optimization: the forward-position recompute still pushed
`cavgx` (because the stale, fusion-unaware site decision said to), but
fusion's whole design assumes nothing at that position ever needs
popping — leaving a genuinely unmatched push every iteration, permanently
growing the stack and corrupting every later, unrelated pop.

This combination — `site_level_tbr=true` *and* `fuse_ii_loops=true`
together — had never actually been tested end to end before; every prior
`fuse_ii_loops` regression in this file only ever exercised the default
`site_level_tbr=false` path, which is exactly why this went unnoticed
through several otherwise-careful rounds.

**Fix**: `agen_ii_override_ectx`/`agen_ii_force_no_snapshot!` — builds a
new `ectx` whose `push_pop` Dict (a copy, starting from whatever
`site_level_tbr` already decided) additionally forces `false` for every
site writing a fusion-covered var, applied everywhere `value_needed` is
locally modified for `ii_plan` purposes (`agen_emit_ii_loop`, and the
plain-recursion `:reduction`/`:mixed` branches in `agen_forward_body`
that don't go through it). Verified via generated-code diff (the
forward-position push disappeared, matching the `site_level_tbr=false`
shape exactly) and via a full 24-combination sweep (4 real corpus kernels
× the existing synthetic test kernels × both `site_level_tbr` settings),
all passing. Locked in as a permanent regression test — the one test in
this file combining both flags end to end.

**Full validation sweep after this fix**, all clean: same 84/84 × 3,
232/20 GPU as every round before it.

## `site_level_tbr` removed as an opt-in flag — now always on

Per direct request: `site_level_tbr::Bool` is gone from every function
signature in `STADE.jl` (`stade_tangent`, `stade_adjoint`, `stade_hvp`,
their `_corpus`/`_file` wrappers, `stade_validate_from_baseline`, and the
three `stade_validate_*_file` functions). Site-level TBR is no longer a
choice — `stade_site_level_tbr_check(kernel)` (the cross-checked snap_*/
agen_* computation, unchanged) now runs unconditionally inside
`stade_adjoint`/`stade_hvp`, the same way it previously only ran when the
flag was `true`.

**One regression test needed updating for more than cosmetic reasons.**
The `agen_ii_override_ectx` interaction test (added last round, after the
`site_level_tbr=true` + `fuse_ii_loops=true` stack-imbalance bug) used to
compare `site_level_tbr=false` against `site_level_tbr=true` explicitly.
With the flag gone there's only one mode to test — updated to a single
validation call, keeping the same kernel shape that originally caught the
bug, so a regression of `agen_ii_override_ectx` itself would still be
caught.

**External validation scripts weren't touched** (`validate_corpus_keep_
push_pop_false_site_level_tbr_true.jl`, `validate_corpus_gpu.jl`), since
they're outside `STADE.jl` and weren't part of the request — but both
explicitly pass `site_level_tbr` as a kwarg and would now error unmodified.
Verified the underlying change is sound by running locally-adjusted copies
(the kwarg simply removed from each call, since it's the only behavior now)
rather than skipping this check: 84/84 `ok` and 232 `gen_ok`/20 `gen_error`
(the same pre-existing, unrelated class flagged from the start), both
unchanged from before this round. Worth noting for whoever maintains those
scripts going forward: `validate_corpus_keep_push_pop_false_site_level_tbr_
true.jl` and `validate_corpus_keep_push_pop_false.jl` now test the exact
same thing (site-level TBR is no longer a variable between them) and could
reasonably be merged or one retired.

**Full validation sweep**, all clean: `validate_corpus.jl` and
`validate_corpus_keep_push_pop_false.jl` unmodified and still 84/84 `ok`
(neither ever referenced the flag); the two scripts that did, run via
locally-adjusted copies, matched their pre-change results exactly.