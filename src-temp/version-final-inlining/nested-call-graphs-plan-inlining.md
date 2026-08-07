# Plan: nested call graphs in STADE, via source-level inlining

## 1. Goal and why inlining is the right approach

skill-jade kernels are Fortran-subroutine-shaped: no structs/tuples,
no return values, every function ends `return nothing`, outputs
happen by mutating array arguments in place. So "kernel `A` calls
kernel `B`" means a **bare subroutine-call statement** —
`callee_name(a, b, c)` on its own line — not a call embedded in an
expression.

An earlier design made `:call` a first-class statement kind carried
all the way through every stage. That turned out to be expensive for
one specific reason: once `A` calls `B`, their forward/backward AD
sweeps interleave, so the snapshot stacks `agen_*` builds can no
longer be locals private to one generated function — they'd have to
be threaded as explicit arguments through the whole call tree, plus a
new "interface activity summary" abstraction to know which of `B`'s
dependents are reachable from which independents without re-running
`B`'s full analysis at every call site. That's a lot of new surface
area across `parse_/shape_/act_/snap_/lin_/agen_` for something that
has a much smaller solution.

**Inlining sidesteps this by construction.** Splice `B`'s body into
`A` (renaming `B`'s locals to avoid collisions) *before* `parse_kernel`
ever runs, and the result is just one flat function again. There is no
call boundary left for any downstream stage to reason about — activity
analysis, snapshot planning, linearization, and adjoint codegen all
already do the right thing, because they're working over exactly the
kind of single-function body they already understand. Every frozen
data shape and every existing stage function is untouched.

The cost that doesn't go away: **code size**. Each call site to a
helper kernel gets its own fully-expanded copy. For a small corpus
this is very likely fine (and it's the tradeoff Tapenade makes for
small callees too) — worth a one-line note in the spec doc so nobody's
surprised by generated-file size later, but not worth solving up
front.

## 2. Scope

- A DAG of kernels, resolved from a corpus supplied up front as a
  `Dict{Symbol,Expr}` (one `function...end` `Expr` per kernel name) —
  not auto-discovered at runtime.
- **Recursion is out of scope.** Detect a cycle in the call graph and
  hard-error with a clear cycle trace, rather than relying on an
  iteration cap to eventually give up.
- A nested call is always a **bare statement** (`callee(args...)` on
  its own line), never embedded inside an expression — matches how
  these kernels actually return results (mutated array args, not
  values).
- Call arguments are **bare symbols only** — no expressions, no
  indexing — matching the existing "compute it on its own line first"
  discipline STADE already applies to indirect indexing. This keeps
  substitution a pure symbol→symbol rename, nothing more.

## 3. New stage: `inl_*`

A new pipeline stage, `inl_*`, runs on raw `Expr` **before**
`parse_kernel`. It is the one stage in STADE that operates outside the
`parse_kernel`-produced kernel/body shape — flag this explicitly in
skill-stade's prefix list (rule 4) so it's not mistaken for a
namespace collision later. Nothing below it changes: `parse_*` through
`agen_*` receive a single already-flat `Expr(:function, ...)` and
never see a `:call`-to-a-user-kernel statement at all.

### 3.1 Entry point

```
inl_inline_calls(kernels::Dict{Symbol,Expr}) -> Dict{Symbol,Expr}
```

Pure function: same input always produces the same output (see §3.4
on determinism). Returns one fully-inlined `Expr` per kernel name —
every user-kernel call anywhere in that kernel's call graph has been
expanded away, leaving only intrinsic calls (`parse_intrinsic_whitelist`)
and arithmetic/comparison ops for `parse_kernel` to see.

### 3.2 Call graph and cycle check

Before inlining anything, walk each kernel body (recursively through
`for`/`if`, same traversal shape `snap_collect_reassigned` already
uses) collecting which other keys of `kernels` it calls. Build a
`Dict{Symbol,Set{Symbol}}` adjacency map and topologically sort it.

- **Unknown callee** (a bare-statement call to a name not in
  `kernels` and not an intrinsic): hard error at this stage, naming
  the caller and the unresolved callee — don't let it fall through to
  a confusing `parse_kernel` error later about an unwhitelisted call.
- **Cycle**: hard error naming the full cycle (`A -> B -> A`), not
  "maximum iterations reached."

Process kernels in topological order (callees inlined into, and
finalized, before any kernel that calls them is processed) — so a
three-level call chain `A -> B -> C` only ever does one substitution
pass per level, using `B`'s *already-fully-inlined* body when
expanding it into `A`.

### 3.3 Per-call-site substitution

For one call statement `callee_name(arg1, arg2, ...)` found in a
caller's (already partially inlined) body:

1. Look up `callee_name`'s parsed signature (`params = callee.sig.args`,
   in the same declared order) and its already-inlined body.
2. **Kind check the call site** against the callee's own declared
   signature *before* substituting — `arg_i`'s kind in the caller
   (`scalar_float`/`scalar_int`/`array_float`/`array_int`, from
   `shape_infer` run on the caller so far) must exactly match
   `params[i]`'s declared kind. This is the one thing pure inlining
   would otherwise silently lose: without this check, a caller passing
   a `scalar_int` where the callee's signature says `scalar_float`
   just gets quietly merged in and reinterpreted by whatever
   `shape_infer` derives from the combined body, instead of failing
   loudly at the call site. Implement as `inl_check_call_kinds`,
   run once per call site.
3. Rename every variable assigned inside the callee's body that is
   **not** one of `params` (i.e. every genuine local) with a
   deterministic, collision-free suffix (§3.4). Parameters are *not*
   renamed — they get substituted directly with the caller's actual
   argument expressions in the next step.
4. Substitute: build `subst = Dict(zip(params, call_args))` and walk
   the (now locally-renamed) callee body replacing every parameter
   symbol with the caller's corresponding argument symbol.
5. **Splice at the statement-list level**, not by nesting `Expr(:block)`
   wrappers: replace the one call-statement entry in the caller's
   `Vector{Expr}` body with the N statements of the substituted callee
   body, in place. Do this on the already-`parse_strip_lines`-flattened
   statement vector, so there is no begin-block flattening,
   `clean_expr`, or `unwrap_begin` step needed at all — that machinery
   only exists to clean up nesting artifacts from splicing at the raw
   nested-`Expr` level, and doesn't arise if you splice a flat list
   into a flat list.

### 3.4 Deterministic naming (no `rand`)

Suffix each inlined local with `_<callee_name>_c<call_site_counter>`
(a per-*caller* counter that increments once per call site processed,
not per callee), e.g. inlining `dotprod` at the first call site inside
`mg_vcycle` renames its locals `x_dotprod_c1`, `acc_dotprod_c1`, etc.

This matters for two reasons beyond style:
- **Purity.** `inl_inline_calls` must be a pure function of its input
  per rule 7 (no stage may have hidden nondeterminism) — a `rand()`-based
  suffix makes the same source produce a different `Expr` on every
  run, which breaks reproducing a golden-corpus failure and breaks
  any test that compares inlined output structurally.
- **Legibility.** Generated names stay meaningful in error messages
  and in the final differentiated source, instead of opaque `x_a1b2`.

Still run `parse_check_snake_case` over the final suffixed name — the
callee name and counter are both already snake_case-safe by
construction, but check anyway rather than assuming.

### 3.5 Staying in `Expr`-space throughout

Do not round-trip through `String`/`Meta.parse` at any point in this
stage. Build and splice `Expr`/`Symbol` values directly, and hand the
final assembled `Expr(:function, sig, body)` straight to
`parse_kernel`. `io_expr_to_source` already exists for the one place
text output is actually needed — writing the final file in `io_*` —
there's no reason for `inl_*` to serialize and re-parse in the middle.

### 3.6 No `@capture`/MacroTools dependency

Everything above is expressible with the same plain
`expr.head`/`expr.args` pattern-matching every other stage already
uses — `expr.head == :function` for a definition,
`expr.head == :call && expr.args[1] isa Symbol` for a call site. Adding
MacroTools would be the only external dependency in an otherwise
single-file, dependency-light engine; avoid it for consistency with
the rest of STADE's style, not just to save an import.

## 4. Frozen shapes and existing stages

**No changes.** No new statement kind, no new key on any frozen shape
in skill-stade.md §5. `parse_*` through `val_*` behave exactly as
today, because by the time they see a kernel, it has no user-kernel
calls left in it — only intrinsics and arithmetic, which they already
handle.

## 5. `stade_*` public API

Add a thin multi-kernel entry point that does inlining, then defers to
the existing single-kernel pipeline unchanged:

```
stade_adjoint_corpus(kernels::Dict{Symbol,Expr}; independents=..., dependents=...) 
    -> Dict{Symbol,<same shape stade_adjoint returns today>}
```

Implementation is essentially:

```
inlined = inl_inline_calls(kernels)
Dict(name => stade_adjoint(expr) for (name, expr) in inlined)
```

(independents/dependents overrides, if used, apply per top-level
kernel exactly as `stade_adjoint` already supports — a call site's
arguments are already substituted away by the time `parse_kernel` ever
sees them, so there's no separate override mechanism needed for
inlined locals.)

`stade_adjoint(expr::Expr; ...)` (single kernel, no calls) stays
exactly as it is today — `stade_adjoint_corpus` is additive, not a
replacement.

## 6. Rollout order

1. `inl_*` call graph construction + cycle/unknown-callee detection,
   tested in isolation against a couple of hand-built `Dict{Symbol,Expr}`
   fixtures (including a deliberate cycle, to check the error message).
2. `inl_check_call_kinds` + the rename/substitute/splice pipeline,
   tested by inlining a small real pair of corpus kernels and manually
   inspecting the resulting `Expr` (or its `io_expr_to_source` text)
   for correctness — no gradient involved yet, this step is purely
   about getting the right *primal* code out.
3. Feed an inlined result through the existing, unmodified
   `parse_kernel` → `act_analyze` → `snap_plan` → `lin_build` →
   `agen_emit` pipeline and confirm it produces a working adjoint —
   this should require zero changes to any of those stages, which is
   itself the check that the design achieved its goal.
4. Add `stade_adjoint_corpus` as the public multi-kernel entry point.
5. Add a real multi-kernel fixture (a caller + at least one two-deep
   call chain) to `val_fixtures.jl` and validate with
   `val_finite_diff_check` the same way every existing tier is — don't
   add this speculatively ahead of a concrete need, consistent with
   the skill's own norm for `:while`.

## 7. Open items to settle while implementing

- **Multiple call sites to the same callee.** Each call site gets its
  own independent expansion + its own suffix counter value — nothing
  is shared or cached across call sites, unlike the registry-based
  design's per-kernel-once analysis. This is intentional (simpler,
  and correctness doesn't depend on it) but costs more generated code
  the more a small kernel is reused; revisit only if a real corpus
  kernel makes this cost concrete.
- **Array aliasing at a call site** (e.g. passing the same array as
  two different arguments to the same callee): inlining makes this a
  non-issue for correctness — the substituted body just reads/writes
  the same caller-side array symbol twice, exactly as if it had been
  written inline by hand, so `act_/snap_` see the true aliasing
  automatically. Worth a short fixture confirming this rather than
  assuming it.
- **Where `inl_check_call_kinds` gets the caller's kind info from,
  mid-inlining.** Since inlining happens before `parse_kernel`, the
  caller doesn't have a `shape_infer`-derived kind map yet at the
  point a given call site is processed. Simplest option: run a
  lightweight local shape pass (or reuse `shape_infer`'s existing
  logic against the caller's Expr-so-far) purely for this check,
  discarding the result afterward — `parse_kernel` will derive the
  real one from the final fully-inlined body regardless.
