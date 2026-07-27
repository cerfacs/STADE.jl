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
   `tgen_`, `agen_`, `val_`, `io_`, `stade_`. `io_` is the only
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
   reference `all_b.jl` or any particular kernel by name. Corpus
   fixtures live only in `val_fixtures.jl`.

## Testing convention

Every change is checked against the shared golden corpus (19 kernels,
tiered M1–M4 by complexity: straight-line → conditionals → loop-carried
recurrence → `mg_vcycle`'s full nested-stack complexity), using
`val_finite_diff_check` (central finite differences against the
Tapenade-generated ground truth in `all_b.jl`, via `val_fixtures.jl`)
as the primary correctness oracle, checked against random seed vectors.
All four tiers currently pass against the hand-written ground-truth
fixtures (worst observed max_rel_err ~4.7e-8 across M1–M4). Note: avoid
"dotproduct"/"dotprod" in any STADE-internal name — the corpus already
has a kernel called `dotprod`, and reusing that word for the test
infrastructure itself invites exactly the kind of confusable naming
this skill's prefix rule is meant to prevent. A structural diff against
the corpus's existing Tapenade-generated `_b.jl` files is a secondary
style check only, never a correctness requirement.

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