# ============================================================
# check_der.jl -- standalone correctness check for STADE's der_*
# derivative rule table.
#
# Tests ONLY der_rule / der_table and the (tangent, adjoint) functions
# they return -- no dependency on parse_/shape_/act_/lin_/tgen_/agen_,
# none of which are wired up yet. Run with:
#
#   julia check_der.jl
#
# Two independent checks per whitelisted operator, both driven off a
# small standalone Expr evaluator (chk_eval) that is deliberately NOT
# part of STADE.jl, so a bug in der_*'s own simplification helpers
# can't quietly make its own test pass:
#
#   1. tangent/adjoint self-consistency (exact, no numerics involved):
#      sum_i adjoint_i(seed) * dargs_i  ==  tangent(dargs) * seed
#      This must hold exactly because both sides are built from the
#      same der_partials(op, args) -- it's a regression guard against
#      the tangent and adjoint rules for one operator drifting apart,
#      independent of whether either is calculus-correct.
#
#   2. each partial vs. a central finite difference of the *real*
#      Base.<op> function -- this is the actual correctness check
#      against calculus. Piecewise-constant ops (floor/ceil/trunc/
#      sign/div) are included too: away from their jump points a
#      small enough step keeps the finite difference at exactly 0,
#      matching der_partials' 0.0.
# ============================================================

isdefined(Main, :der_table) || include(joinpath(@__DIR__, "STADE.jl"))

using Test

# ---- standalone Expr evaluator (independent of STADE.jl) ----------

function chk_eval(e, env::Dict{Symbol,Float64})
    e isa Number && return Float64(e)
    if e isa Symbol
        haskey(env, e) || error("chk_eval: unbound symbol $e")
        return env[e]
    end
    if e isa Expr && e.head === :call
        op = e.args[1]
        argvals = [chk_eval(a, env) for a in e.args[2:end]]
        f = getfield(Base, op)
        return f(argvals...)
    end
    error("chk_eval: cannot evaluate $e")
end

# central finite difference of Base.<op>(values...) w.r.t. values[i]
function chk_finite_diff(op::Symbol, values::Vector{Float64}, i::Int, h::Float64)
    f = getfield(Base, op)
    plus = copy(values);  plus[i]  += h
    minus = copy(values); minus[i] -= h
    return (f(plus...) - f(minus...)) / (2h)
end

# ---- test cases: (op, arg symbols, arg values, fd step) -----------
# arg values are chosen to stay well inside every op's safe domain
# (sqrt/log > 0, asin/acos in (-1,1), no ties for max/min, no exact
# integers for floor/ceil/trunc/sign/div) so the finite difference is
# never taken across a genuine singularity or jump point.

const CASES = [
    (:+,     [:x, :y, :z], [1.3, -2.1, 0.7],  1e-6),
    (:-,     [:x, :y],     [3.4, 1.1],        1e-6),
    (:-,     [:x],         [2.2],             1e-6),  # unary negation
    (:*,     [:x, :y, :z], [1.7, -0.6, 2.2],  1e-6),
    (:/,     [:x, :y],     [3.1, 1.4],        1e-6),
    (:^,     [:x, :y],     [1.7, 2.3],        1e-6),
    (:abs,   [:x],         [-2.4],            1e-6),
    (:sqrt,  [:x],         [2.6],             1e-6),
    (:exp,   [:x],         [0.8],             1e-6),
    (:log,   [:x],         [2.1],             1e-6),
    (:log10, [:x],         [5.7],             1e-6),
    (:sin,   [:x],         [0.9],             1e-6),
    (:cos,   [:x],         [0.9],             1e-6),
    (:tan,   [:x],         [0.6],             1e-6),
    (:asin,  [:x],         [0.4],             1e-6),
    (:acos,  [:x],         [0.4],             1e-6),
    (:atan,  [:x],         [1.2],             1e-6),
    (:sinh,  [:x],         [0.7],             1e-6),
    (:cosh,  [:x],         [0.7],             1e-6),
    (:tanh,  [:x],         [0.7],             1e-6),
    (:mod,   [:x, :y],     [7.4, 2.1],        1e-6),
    (:div,   [:x, :y],     [7.4, 2.1],        1e-4),  # piecewise-constant
    (:max,   [:x, :y, :z], [1.0, 2.7, 0.3],   1e-6),
    (:min,   [:x, :y, :z], [1.0, 2.7, 0.3],   1e-6),
    (:sign,  [:x],         [1.6],             1e-4),  # piecewise-constant
    (:floor, [:x],         [3.2],             1e-4),  # piecewise-constant
    (:ceil,  [:x],         [3.2],             1e-4),  # piecewise-constant
    (:trunc, [:x],         [3.2],             1e-4),  # piecewise-constant
]

const WHITELIST = [:+, :-, :*, :/, :^, :abs, :sqrt, :exp, :log, :log10,
                    :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh,
                    :tanh, :mod, :div, :max, :min, :sign, :floor, :ceil, :trunc]

# ---- table completeness --------------------------------------------

@testset "der_table completeness" begin
    @test length(WHITELIST) == 27
    table = der_table()
    @test length(table) == length(WHITELIST)
    for op in WHITELIST
        @test haskey(table, op)
        rule = der_rule(op)
        @test rule.tangent isa Function
        @test rule.adjoint isa Function
    end
    @test all(op -> op in WHITELIST, keys(der_table()))
    @test_throws Exception der_rule(:not_a_real_op)
end

# every CASES op must be on the whitelist and vice versa, so this file
# can't silently drift out of sync with skill-jade's rule 11 list
@testset "CASES matches the whitelist" begin
    tested_ops = unique([c[1] for c in CASES])
    @test sort(tested_ops) == sort(unique(WHITELIST))
end

# ---- per-operator tangent/adjoint checks ---------------------------

@testset "der_rule($op) arity=$(length(vals))" for (op, syms, vals, h) in CASES
    env = Dict(zip(syms, vals))
    rule = der_rule(op)
    n = length(syms)

    # (1) tangent/adjoint self-consistency, exact to floating point
    dargs = [0.3, -1.7, 0.9, 2.4][1:n]
    out_seed = 0.6
    tangent_val = chk_eval(rule.tangent(syms, dargs), env)
    adjoint_contribs = rule.adjoint(syms, out_seed)
    @test length(adjoint_contribs) == n
    adjoint_dot = sum(chk_eval(adjoint_contribs[i], env) * dargs[i] for i in 1:n)
    @test isapprox(adjoint_dot, tangent_val * out_seed; atol = 1e-9, rtol = 1e-9)

    # (2) each partial vs. central finite difference of the real Base op
    for i in 1:n
        unit = [j == i ? 1.0 : 0.0 for j in 1:n]
        tangent_partial = chk_eval(rule.tangent(syms, unit), env)
        adjoint_partial = chk_eval(rule.adjoint(syms, 1.0)[i], env)
        fd_partial = chk_finite_diff(op, vals, i, h)
        @test isapprox(tangent_partial, fd_partial; atol = 1e-4, rtol = 1e-4)
        @test isapprox(adjoint_partial, fd_partial; atol = 1e-4, rtol = 1e-4)
    end
end

println("check_der.jl: all der_* checks passed")