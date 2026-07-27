# ============================================================
# STADE.jl -- source-to-source AD for skill-jade-compliant Julia
# kernels. No modules, structs, @enum, or top-level const --
# see skill-stade.md for the full house-style contract.
#
# Note: this file contains no corpus-specific code. all_b.jl (the
# Tapenade ground-truth corpus) is never referenced here -- it's
# loaded only by the separate val_fixtures.jl, which builds test
# closures around specific corpus functions to validate this engine.
#
# Pipeline stages, in the order data flows through them:
#
#   parse_  Parse        raw Expr -> validated (sig, body) kernel;
#                         rejects anything violating skill-jade.
#   shape_  Shape infer   each variable's kind: scalar_float,
#                         scalar_int, array_float, array_int.
#   der_    Derivatives   rule table -- how to differentiate each
#                         whitelisted operator/intrinsic.
#   emit_   Emit          shared Expr-building helpers (loops, ifs,
#                         return nothing) used by both codegen modes.
#   act_    Activity      which variables carry a derivative at all,
#                         i.e. are reachable from the independents.
#   snap_   Snapshot      which writes need a push/pop for the
#                         reverse sweep (the TBR-equivalent check).
#   lin_    Linearize     builds each statement's derivative-tree,
#                         shared input to both codegen directions.
#   tgen_   Tangent gen   forward-mode codegen.
#   agen_   Adjoint gen   reverse-mode codegen: forward sweep with
#                         pushes, reversed backward sweep with pops,
#                         plus the companion initstacks_* function.
#   val_    Validate      correctness checking against ground truth
#                         (finite differences; later the adjoint
#                         identity once tangent codegen is real).
#   io_     File I/O      the ONLY stage touching the filesystem --
#                         reads a kernel .jl file, writes a
#                         generated .jl file.
#   stade_  Public API    stade_tangent / stade_adjoint (Expr in,
#                         Expr out) and stade_*_file wrappers
#                         (path in, path out), wiring every stage
#                         above together for an end user.
#
# Phase 0: frozen contracts + stub bodies. Real implementations land
# stage by stage, each checked against the val_* corpus fixtures.
# ============================================================

# kernel        :: (sig=kernel_sig, body=statement_list)
# kernel_sig    :: (name::Symbol, args::Vector{Symbol},
#                    kinds::Dict{Symbol,Symbol},   # :scalar_float :scalar_int :array_float :array_int
#                    independents::Vector{Symbol}, dependents::Vector{Symbol})
#     independents/dependents auto-derived by parse_kernel = every
#     float-kinded arg (matches the corpus convention: every _b.jl
#     differentiates all floats, both directions). Never required
#     from the caller -- see parse_override_indep_dep for the
#     opt-in exclusion escape hatch.
# statement     :: one of
#     (kind=:assign, lhs, rhs)                                  # lhs/rhs :: Expr|Symbol|Number
#     (kind=:for, var::Symbol, lo, hi, sequential::Bool, body)  # body :: statement_list
#     (kind=:if, cond, then::statement_list, els::statement_list)
#     -- :while intentionally unsupported for now, see skill-stade.md
# statement_list :: Vector{NamedTuple}
# active_map    :: Dict{Symbol,Bool}                # var/array name -> is-active
# snapshot_site :: (kind=:value|:array|:branch|:tripcount, array::Symbol, at::Int)
# snapshot_plan :: Vector{snapshot_site}
# lin_node      :: (op::Symbol, args::Vector, darg_exprs::Vector)
# lin_plan      :: Vector{lin_node}
# der_rule_pair :: (tangent::Function, adjoint::Function)


# ==================== parse_* ================================
# Raw Expr -> validated kernel. Enforces every skill-jade rule as a
# hard error (snake_case, no annotations, no kwargs, only the four
# variable shapes, no indirect indexing, no broadcasting, i_seq_
# prefix discipline, whitelisted calls only, div-not-÷, no compound
# assignment, comment header present, terminal `return nothing`).

function parse_kernel(expr::Expr)
    sig0 = parse_stub_sig()
    kinds = shape_infer(sig0, Vector{NamedTuple}())
    indep, dep = parse_infer_indep_dep(sig0.args, kinds)
    sig = (name = sig0.name, args = sig0.args, kinds = kinds,
           independents = indep, dependents = dep)
    return (sig = sig, body = Vector{NamedTuple}())
end

function parse_infer_indep_dep(args::Vector{Symbol}, kinds::Dict{Symbol,Symbol})
    floats = [a for a in args if kinds[a] in (:scalar_float, :array_float)]
    return (independents = floats, dependents = floats)
end

# escape hatch -- never required by the public API
function parse_override_indep_dep(kernel, independents, dependents)
    independents === nothing && dependents === nothing && return kernel
    sig = kernel.sig
    new_sig = (name = sig.name, args = sig.args, kinds = sig.kinds,
               independents = independents === nothing ? sig.independents : independents,
               dependents = dependents === nothing ? sig.dependents : dependents)
    return (sig = new_sig, body = kernel.body)
end

function parse_stub_sig()
    return (name = :stub, args = Symbol[], kinds = Dict{Symbol,Symbol}(),
            independents = Symbol[], dependents = Symbol[])
end

function parse_check_snake_case(name::Symbol)
    return true
end

function parse_check_loop_prefix(var::Symbol, sequential::Bool)
    return true
end


# ==================== shape_* =================================
# Infer each argument/local's kind syntactically; cross-check
# against an optional explicit manifest passed by the caller.

function shape_infer(sig, body)
    return Dict{Symbol,Symbol}()
end


# ==================== der_* ====================================
# Derivative rule table -- built fresh per call, never stored.

function der_rule(op::Symbol)
    table = der_table()
    haskey(table, op) || error("no derivative rule registered for $op")
    return table[op]
end

function der_table()
    return Dict{Symbol,NamedTuple}(
        :+ => (tangent = der_tangent_stub, adjoint = der_adjoint_stub),
    )
end

function der_tangent_stub(args, dargs)
    return :(0.0)
end

function der_adjoint_stub(args, out_adjoint)
    return Expr[]
end


# ==================== emit_* ===================================
# Shared Expr-building helpers used by both codegen directions.

function emit_forloop(var::Symbol, lo, hi, body_exprs::Vector)
    return Expr(:for, Expr(:(=), var, Expr(:call, :(:), lo, hi)), Expr(:block, body_exprs...))
end

function emit_if(cond, then_exprs::Vector, else_exprs::Vector)
    return Expr(:if, cond, Expr(:block, then_exprs...), Expr(:block, else_exprs...))
end

function emit_return_nothing()
    return :(return nothing)
end

# returns a String (Expr can't hold comments) -- prepend at file-write time
function emit_comment_header(name::Symbol, arg_docs::Vector{String})
    return "# $(name)(...)\n"
end


# ==================== act_* ====================================
# Forward taint analysis from independents through assignments.

function act_analyze(kernel)
    return Dict{Symbol,Bool}()
end


# ==================== snap_* ===================================
# Push/pop site analysis (the TBR-equivalent). For each write,
# decide whether a later agen_ sweep needs a recorded snapshot.

function snap_plan(kernel, active_map)
    return NamedTuple[]
end


# ==================== lin_* =====================================
# Shared derivative-tree representation, swept differently by each
# codegen direction.

function lin_build(kernel, active_map)
    return NamedTuple[]
end


# ==================== tgen_* =====================================
# Forward-mode codegen: single sweep, original statement/loop order.

function tgen_emit(kernel, lin_plan)
    return Expr(:block, emit_return_nothing())
end


# ==================== agen_* =====================================
# Reverse-mode codegen: forward sweep w/ pushes, reversed backward
# sweep w/ pops, plus the companion initstacks_* generator.

function agen_emit(kernel, lin_plan, snapshot_plan)
    return (adjoint = Expr(:block, emit_return_nothing()),
            initstacks = Expr(:block, :(return nothing)))
end


# ==================== val_* =======================================
# Correctness oracle: <y, J*x> == <J'*y, x>, checked against random
# seed vectors. Deliberately not named "dotproduct" -- see skill.

function val_finite_diff_check(f_eval::Function, f_grad::Function, x0::Vector{Float64};
                                epsilon::Float64 = 1e-6, trials::Int = 10, rtol::Float64 = 1e-3)
    n = length(x0)
    g = f_grad(x0)   # gradient at x0 is independent of direction -- compute once
    results = NamedTuple[]
    worst_rel_err = 0.0
    for t in 1:trials
        d = randn(n)
        d = d ./ sqrt(sum(d .^ 2))
        fd = (f_eval(x0 .+ epsilon .* d) - f_eval(x0 .- epsilon .* d)) / (2 * epsilon)
        ad = sum(g .* d)
        denom = max(abs(fd), abs(ad), 1e-12)
        rel_err = abs(fd - ad) / denom
        worst_rel_err = max(worst_rel_err, rel_err)
        push!(results, (direction = d, finite_diff = fd, adjoint_derivative = ad, rel_err = rel_err))
    end
    return (ok = worst_rel_err <= rtol, max_rel_err = worst_rel_err, trials = results)
end

function val_check_fixture(fixture; epsilon::Float64 = 1e-6, trials::Int = 10, rtol::Float64 = 1e-3)
    return val_finite_diff_check(fixture.f_eval, fixture.f_grad, fixture.x0;
                                  epsilon = epsilon, trials = trials, rtol = rtol)
end

function val_run_corpus(tier::Symbol)
    return NamedTuple[]
end


# ==================== io_* ====================================
# File-level entry points. The only stage permitted to touch the
# filesystem -- everything above operates purely on Expr in memory.

function io_read_kernel(path::String)
    src = read(path, String)
    parsed = Meta.parseall(src)
    defs = [e for e in parsed.args if e isa Expr && e.head == :function]
    length(defs) == 1 || error("expected exactly one function definition in $path, found $(length(defs))")
    return defs[1]
end

function io_expr_to_source(expr::Expr)
    return string(expr) * "\n"
end

# matches the corpus convention: foo_b.jl bundles initstacks_foo_b,
# foo_b, and a copy of foo itself, in that order
function io_write_kernel_file(path::String, primal_expr::Expr, generated::Vector{Expr})
    parts = [io_expr_to_source(e) for e in vcat(generated, [primal_expr])]
    open(path, "w") do f
        write(f, join(parts, "\n"))
    end
    return nothing
end


# ==================== stade_* (public API) ========================
# independents/dependents auto-derived -- see parse_infer_indep_dep.
# Override kwargs exist only for the rare exclusion case.

function stade_tangent(expr::Expr; independents::Union{Vector{Symbol},Nothing}=nothing,
                        dependents::Union{Vector{Symbol},Nothing}=nothing)
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    active_map = act_analyze(kernel)
    lin_plan = lin_build(kernel, active_map)
    return tgen_emit(kernel, lin_plan)
end

function stade_adjoint(expr::Expr; independents::Union{Vector{Symbol},Nothing}=nothing,
                        dependents::Union{Vector{Symbol},Nothing}=nothing)
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    active_map = act_analyze(kernel)
    snapshot_plan = snap_plan(kernel, active_map)
    lin_plan = lin_build(kernel, active_map)
    return agen_emit(kernel, lin_plan, snapshot_plan)
end

function stade_tangent_file(in_path::String, out_path::String)
    primal_expr = io_read_kernel(in_path)
    tangent_expr = stade_tangent(primal_expr)
    io_write_kernel_file(out_path, primal_expr, [tangent_expr])
    return out_path
end

function stade_adjoint_file(in_path::String, out_path::String)
    primal_expr = io_read_kernel(in_path)
    adjoint_out = stade_adjoint(primal_expr)
    io_write_kernel_file(out_path, primal_expr, [adjoint_out.initstacks, adjoint_out.adjoint])
    return out_path
end


# ==================== smoke test ===================================
# Confirms the file loads and the call chain executes end-to-end.
# Not a correctness test -- that's val_* 's job once stage bodies
# above are filled in.

let
    trivial = :(function stub(x, n, y)
        return nothing
    end)
    tangent_out = stade_tangent(trivial)
    adjoint_out = stade_adjoint(trivial)
    @assert tangent_out isa Expr
    @assert adjoint_out.adjoint isa Expr && adjoint_out.initstacks isa Expr
    println("STADE.jl Phase 0 skeleton loaded and round-tripped a stub kernel OK")
end