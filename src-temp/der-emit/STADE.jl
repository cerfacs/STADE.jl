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
#
# Every rule pair is built from a per-operator "local partials" list:
# partials[i] is the Expr|Symbol|Number for d(op(args...))/d(args[i]),
# symbolic in terms of the primal args. tangent/adjoint are then just
# the two standard contractions of that list against a seed:
#   tangent: sum_i partials[i] * dargs[i]          (forward accumulation)
#   adjoint: [partials[i] * out_adjoint for each i] (one contribution per
#            arg -- turning each entry into an `argib = argib + ...`
#            accumulation statement is agen_'s job, not ours)
# A few algebraic simplifications (dropping *1, *0, +0 terms, folding
# literal-Number arithmetic) keep the generated Exprs close to the
# corpus's hand-differentiated style.

function der_rule(op::Symbol)
    table = der_table()
    haskey(table, op) || error("no derivative rule registered for $op")
    return table[op]
end

function der_table()
    return Dict{Symbol,NamedTuple}(
        :+     => (tangent = der_tangent_add,     adjoint = der_adjoint_add),
        :-     => (tangent = der_tangent_sub,     adjoint = der_adjoint_sub),
        :*     => (tangent = der_tangent_mul,     adjoint = der_adjoint_mul),
        :/     => (tangent = der_tangent_divide,  adjoint = der_adjoint_divide),
        :^     => (tangent = der_tangent_pow,     adjoint = der_adjoint_pow),
        :abs   => (tangent = der_tangent_abs,     adjoint = der_adjoint_abs),
        :sqrt  => (tangent = der_tangent_sqrt,    adjoint = der_adjoint_sqrt),
        :exp   => (tangent = der_tangent_exp,     adjoint = der_adjoint_exp),
        :log   => (tangent = der_tangent_log,     adjoint = der_adjoint_log),
        :log10 => (tangent = der_tangent_log10,   adjoint = der_adjoint_log10),
        :sin   => (tangent = der_tangent_sin,     adjoint = der_adjoint_sin),
        :cos   => (tangent = der_tangent_cos,     adjoint = der_adjoint_cos),
        :tan   => (tangent = der_tangent_tan,     adjoint = der_adjoint_tan),
        :asin  => (tangent = der_tangent_asin,    adjoint = der_adjoint_asin),
        :acos  => (tangent = der_tangent_acos,    adjoint = der_adjoint_acos),
        :atan  => (tangent = der_tangent_atan,    adjoint = der_adjoint_atan),
        :sinh  => (tangent = der_tangent_sinh,    adjoint = der_adjoint_sinh),
        :cosh  => (tangent = der_tangent_cosh,    adjoint = der_adjoint_cosh),
        :tanh  => (tangent = der_tangent_tanh,    adjoint = der_adjoint_tanh),
        :mod   => (tangent = der_tangent_mod,     adjoint = der_adjoint_mod),
        :div   => (tangent = der_tangent_intdiv,  adjoint = der_adjoint_intdiv),
        :max   => (tangent = der_tangent_max,     adjoint = der_adjoint_max),
        :min   => (tangent = der_tangent_min,     adjoint = der_adjoint_min),
        :sign  => (tangent = der_tangent_sign,    adjoint = der_adjoint_sign),
        :floor => (tangent = der_tangent_floor,   adjoint = der_adjoint_floor),
        :ceil  => (tangent = der_tangent_ceil,    adjoint = der_adjoint_ceil),
        :trunc => (tangent = der_tangent_trunc,   adjoint = der_adjoint_trunc),
    )
end

# ---- generic contraction: local partials -> tangent / adjoint --------

# sum_i partials[i] * dargs[i], dropping zero terms
function der_tangent_generic(op::Symbol, args, dargs)
    partials = der_partials(op, args)
    length(partials) == length(dargs) || error("der_tangent_generic: arity mismatch for $op")
    terms = [der_mul(partials[i], dargs[i]) for i in 1:length(partials)]
    return der_sum_terms(terms)
end

# [partials[i] * out_adjoint for each arg] -- per-arg contribution, not
# yet folded into an `argib = argib + ...` accumulation (agen_'s job)
function der_adjoint_generic(op::Symbol, args, out_adjoint)
    partials = der_partials(op, args)
    return [der_mul(p, out_adjoint) for p in partials]
end

# ---- local partials, one function per skill-jade-whitelisted op ------

function der_partials(op::Symbol, args)
    op === :+     && return der_partials_add(args)
    op === :-     && return der_partials_sub(args)
    op === :*     && return der_partials_mul(args)
    op === :/     && return der_partials_divide(args)
    op === :^     && return der_partials_pow(args)
    op === :abs   && return der_partials_abs(args)
    op === :sqrt  && return der_partials_sqrt(args)
    op === :exp   && return der_partials_exp(args)
    op === :log   && return der_partials_log(args)
    op === :log10 && return der_partials_log10(args)
    op === :sin   && return der_partials_sin(args)
    op === :cos   && return der_partials_cos(args)
    op === :tan   && return der_partials_tan(args)
    op === :asin  && return der_partials_asin(args)
    op === :acos  && return der_partials_acos(args)
    op === :atan  && return der_partials_atan(args)
    op === :sinh  && return der_partials_sinh(args)
    op === :cosh  && return der_partials_cosh(args)
    op === :tanh  && return der_partials_tanh(args)
    op === :mod   && return der_partials_mod(args)
    op === :div   && return der_partials_intdiv(args)
    op === :max   && return der_partials_maxmin(:max, args)
    op === :min   && return der_partials_maxmin(:min, args)
    op === :sign  && return der_partials_sign(args)
    op === :floor && return der_partials_floor(args)
    op === :ceil  && return der_partials_ceil(args)
    op === :trunc && return der_partials_trunc(args)
    error("no derivative partials defined for $op")
end

# +(x1,...,xn) (unary + included): every partial is 1
function der_partials_add(args)
    return [1.0 for _ in args]
end

# unary negation -x: partial is -1.
# binary/n-ary subtraction x1-x2-...-xn: first arg +1, every other -1
function der_partials_sub(args)
    length(args) == 1 && return [-1.0]
    return [i == 1 ? 1.0 : -1.0 for i in 1:length(args)]
end

# *(x1,...,xn): d/dxi = product of every other argument
function der_partials_mul(args)
    return [der_product_excluding(args, i) for i in 1:length(args)]
end

# x / y: d/dx = 1/y, d/dy = -x/y^2
function der_partials_divide(args)
    length(args) == 2 || error("der_partials_divide: / takes exactly 2 args")
    x, y = args[1], args[2]
    px = der_div_expr(1.0, y)
    py = der_neg(der_div_expr(x, der_pow_expr(y, 2)))
    return [px, py]
end

# x^y: d/dx = y*x^(y-1), d/dy = x^y*log(x)  -- the y-branch is only
# meaningful for x>0, same caveat any AD tool has for a non-integer
# or negative base; y is almost always an inactive Int64 literal in
# skill-jade kernels, so that branch is usually dropped upstream
function der_partials_pow(args)
    length(args) == 2 || error("der_partials_pow: ^ takes exactly 2 args")
    x, y = args[1], args[2]
    px = der_mul(y, der_pow_expr(x, der_sub(y, 1)))
    py = der_mul(der_pow_expr(x, y), Expr(:call, :log, x))
    return [px, py]
end

# abs(x): d/dx = sign(x)  (subgradient 0 at x==0, matching Julia's sign)
function der_partials_abs(args)
    length(args) == 1 || error("der_partials_abs: abs takes exactly 1 arg")
    return [Expr(:call, :sign, args[1])]
end

# sqrt(x): d/dx = 1/(2*sqrt(x))
function der_partials_sqrt(args)
    length(args) == 1 || error("der_partials_sqrt: sqrt takes exactly 1 arg")
    x = args[1]
    return [der_div_expr(1.0, der_mul(2.0, Expr(:call, :sqrt, x)))]
end

# exp(x): d/dx = exp(x)
function der_partials_exp(args)
    length(args) == 1 || error("der_partials_exp: exp takes exactly 1 arg")
    return [Expr(:call, :exp, args[1])]
end

# log(x): d/dx = 1/x
function der_partials_log(args)
    length(args) == 1 || error("der_partials_log: log takes exactly 1 arg")
    return [der_div_expr(1.0, args[1])]
end

# log10(x): d/dx = 1/(x*log(10))
function der_partials_log10(args)
    length(args) == 1 || error("der_partials_log10: log10 takes exactly 1 arg")
    x = args[1]
    return [der_div_expr(1.0, der_mul(x, Expr(:call, :log, 10.0)))]
end

# sin(x): d/dx = cos(x)
function der_partials_sin(args)
    length(args) == 1 || error("der_partials_sin: sin takes exactly 1 arg")
    return [Expr(:call, :cos, args[1])]
end

# cos(x): d/dx = -sin(x)
function der_partials_cos(args)
    length(args) == 1 || error("der_partials_cos: cos takes exactly 1 arg")
    return [der_neg(Expr(:call, :sin, args[1]))]
end

# tan(x): d/dx = 1/cos(x)^2
function der_partials_tan(args)
    length(args) == 1 || error("der_partials_tan: tan takes exactly 1 arg")
    x = args[1]
    return [der_div_expr(1.0, der_pow_expr(Expr(:call, :cos, x), 2))]
end

# asin(x): d/dx = 1/sqrt(1 - x^2)
function der_partials_asin(args)
    length(args) == 1 || error("der_partials_asin: asin takes exactly 1 arg")
    x = args[1]
    return [der_div_expr(1.0, Expr(:call, :sqrt, der_sub(1.0, der_pow_expr(x, 2))))]
end

# acos(x): d/dx = -1/sqrt(1 - x^2)
function der_partials_acos(args)
    length(args) == 1 || error("der_partials_acos: acos takes exactly 1 arg")
    x = args[1]
    return [der_neg(der_div_expr(1.0, Expr(:call, :sqrt, der_sub(1.0, der_pow_expr(x, 2)))))]
end

# atan(x): d/dx = 1/(1 + x^2)
function der_partials_atan(args)
    length(args) == 1 || error("der_partials_atan: atan takes exactly 1 arg")
    x = args[1]
    return [der_div_expr(1.0, der_add(1.0, der_pow_expr(x, 2)))]
end

# sinh(x): d/dx = cosh(x)
function der_partials_sinh(args)
    length(args) == 1 || error("der_partials_sinh: sinh takes exactly 1 arg")
    return [Expr(:call, :cosh, args[1])]
end

# cosh(x): d/dx = sinh(x)
function der_partials_cosh(args)
    length(args) == 1 || error("der_partials_cosh: cosh takes exactly 1 arg")
    return [Expr(:call, :sinh, args[1])]
end

# tanh(x): d/dx = 1 - tanh(x)^2
function der_partials_tanh(args)
    length(args) == 1 || error("der_partials_tanh: tanh takes exactly 1 arg")
    x = args[1]
    return [der_sub(1.0, der_pow_expr(Expr(:call, :tanh, x), 2))]
end

# mod(x,y) = x - y*floor(x/y): d/dx = 1, d/dy = -floor(x/y)  (a.e. --
# floor's own derivative is 0 a.e., same caveat as floor/ceil/trunc below)
function der_partials_mod(args)
    length(args) == 2 || error("der_partials_mod: mod takes exactly 2 args")
    x, y = args[1], args[2]
    return [1.0, der_neg(Expr(:call, :floor, der_div_expr(x, y)))]
end

# div(x,y): integer floor division -- piecewise constant, derivative 0
# a.e. w.r.t. both args (skill-jade rule 12: div is for Int64
# loop-counter/size arithmetic, never applied to active floats)
function der_partials_intdiv(args)
    length(args) == 2 || error("der_partials_intdiv: div takes exactly 2 args")
    return [0.0, 0.0]
end

# max/min(x1,...,xn): subgradient split via sign(), so no branching is
# needed in the generated Expr. For arg i, compare xi against M_i, the
# max/min of every *other* argument. At n==2 this is the usual
# 0.5*(1+sign(x-y)) / 0.5*(1-sign(x-y)) split; it generalizes cleanly to
# n-ary max/min via a nested max/min call over "everyone else".
function der_partials_maxmin(op::Symbol, args)
    n = length(args)
    partials = Vector{Any}(undef, n)
    for i in 1:n
        others = der_others(args, i)
        if isempty(others)
            partials[i] = 1.0
            continue
        end
        rest = der_reduce_call(op, others)
        indicator = der_mul(0.5, der_add(1.0, Expr(:call, :sign, der_sub(args[i], rest))))
        partials[i] = op === :max ? indicator : der_sub(1.0, indicator)
    end
    return partials
end

# sign(x): piecewise constant (jump at 0), derivative 0 a.e.
function der_partials_sign(args)
    length(args) == 1 || error("der_partials_sign: sign takes exactly 1 arg")
    return [0.0]
end

# floor(x): piecewise constant, derivative 0 a.e.
function der_partials_floor(args)
    length(args) == 1 || error("der_partials_floor: floor takes exactly 1 arg")
    return [0.0]
end

# ceil(x): piecewise constant, derivative 0 a.e.
function der_partials_ceil(args)
    length(args) == 1 || error("der_partials_ceil: ceil takes exactly 1 arg")
    return [0.0]
end

# trunc(x): piecewise constant, derivative 0 a.e.
function der_partials_trunc(args)
    length(args) == 1 || error("der_partials_trunc: trunc takes exactly 1 arg")
    return [0.0]
end

# ---- named tangent/adjoint entry points registered in der_table ------

function der_tangent_add(args, dargs); return der_tangent_generic(:+, args, dargs); end
function der_adjoint_add(args, out_adjoint); return der_adjoint_generic(:+, args, out_adjoint); end

function der_tangent_sub(args, dargs); return der_tangent_generic(:-, args, dargs); end
function der_adjoint_sub(args, out_adjoint); return der_adjoint_generic(:-, args, out_adjoint); end

function der_tangent_mul(args, dargs); return der_tangent_generic(:*, args, dargs); end
function der_adjoint_mul(args, out_adjoint); return der_adjoint_generic(:*, args, out_adjoint); end

function der_tangent_divide(args, dargs); return der_tangent_generic(:/, args, dargs); end
function der_adjoint_divide(args, out_adjoint); return der_adjoint_generic(:/, args, out_adjoint); end

function der_tangent_pow(args, dargs); return der_tangent_generic(:^, args, dargs); end
function der_adjoint_pow(args, out_adjoint); return der_adjoint_generic(:^, args, out_adjoint); end

function der_tangent_abs(args, dargs); return der_tangent_generic(:abs, args, dargs); end
function der_adjoint_abs(args, out_adjoint); return der_adjoint_generic(:abs, args, out_adjoint); end

function der_tangent_sqrt(args, dargs); return der_tangent_generic(:sqrt, args, dargs); end
function der_adjoint_sqrt(args, out_adjoint); return der_adjoint_generic(:sqrt, args, out_adjoint); end

function der_tangent_exp(args, dargs); return der_tangent_generic(:exp, args, dargs); end
function der_adjoint_exp(args, out_adjoint); return der_adjoint_generic(:exp, args, out_adjoint); end

function der_tangent_log(args, dargs); return der_tangent_generic(:log, args, dargs); end
function der_adjoint_log(args, out_adjoint); return der_adjoint_generic(:log, args, out_adjoint); end

function der_tangent_log10(args, dargs); return der_tangent_generic(:log10, args, dargs); end
function der_adjoint_log10(args, out_adjoint); return der_adjoint_generic(:log10, args, out_adjoint); end

function der_tangent_sin(args, dargs); return der_tangent_generic(:sin, args, dargs); end
function der_adjoint_sin(args, out_adjoint); return der_adjoint_generic(:sin, args, out_adjoint); end

function der_tangent_cos(args, dargs); return der_tangent_generic(:cos, args, dargs); end
function der_adjoint_cos(args, out_adjoint); return der_adjoint_generic(:cos, args, out_adjoint); end

function der_tangent_tan(args, dargs); return der_tangent_generic(:tan, args, dargs); end
function der_adjoint_tan(args, out_adjoint); return der_adjoint_generic(:tan, args, out_adjoint); end

function der_tangent_asin(args, dargs); return der_tangent_generic(:asin, args, dargs); end
function der_adjoint_asin(args, out_adjoint); return der_adjoint_generic(:asin, args, out_adjoint); end

function der_tangent_acos(args, dargs); return der_tangent_generic(:acos, args, dargs); end
function der_adjoint_acos(args, out_adjoint); return der_adjoint_generic(:acos, args, out_adjoint); end

function der_tangent_atan(args, dargs); return der_tangent_generic(:atan, args, dargs); end
function der_adjoint_atan(args, out_adjoint); return der_adjoint_generic(:atan, args, out_adjoint); end

function der_tangent_sinh(args, dargs); return der_tangent_generic(:sinh, args, dargs); end
function der_adjoint_sinh(args, out_adjoint); return der_adjoint_generic(:sinh, args, out_adjoint); end

function der_tangent_cosh(args, dargs); return der_tangent_generic(:cosh, args, dargs); end
function der_adjoint_cosh(args, out_adjoint); return der_adjoint_generic(:cosh, args, out_adjoint); end

function der_tangent_tanh(args, dargs); return der_tangent_generic(:tanh, args, dargs); end
function der_adjoint_tanh(args, out_adjoint); return der_adjoint_generic(:tanh, args, out_adjoint); end

function der_tangent_mod(args, dargs); return der_tangent_generic(:mod, args, dargs); end
function der_adjoint_mod(args, out_adjoint); return der_adjoint_generic(:mod, args, out_adjoint); end

function der_tangent_intdiv(args, dargs); return der_tangent_generic(:div, args, dargs); end
function der_adjoint_intdiv(args, out_adjoint); return der_adjoint_generic(:div, args, out_adjoint); end

function der_tangent_max(args, dargs); return der_tangent_generic(:max, args, dargs); end
function der_adjoint_max(args, out_adjoint); return der_adjoint_generic(:max, args, out_adjoint); end

function der_tangent_min(args, dargs); return der_tangent_generic(:min, args, dargs); end
function der_adjoint_min(args, out_adjoint); return der_adjoint_generic(:min, args, out_adjoint); end

function der_tangent_sign(args, dargs); return der_tangent_generic(:sign, args, dargs); end
function der_adjoint_sign(args, out_adjoint); return der_adjoint_generic(:sign, args, out_adjoint); end

function der_tangent_floor(args, dargs); return der_tangent_generic(:floor, args, dargs); end
function der_adjoint_floor(args, out_adjoint); return der_adjoint_generic(:floor, args, out_adjoint); end

function der_tangent_ceil(args, dargs); return der_tangent_generic(:ceil, args, dargs); end
function der_adjoint_ceil(args, out_adjoint); return der_adjoint_generic(:ceil, args, out_adjoint); end

function der_tangent_trunc(args, dargs); return der_tangent_generic(:trunc, args, dargs); end
function der_adjoint_trunc(args, out_adjoint); return der_adjoint_generic(:trunc, args, out_adjoint); end

# ---- small Expr-building helpers shared by every rule above -----------
# (constant folding for literal-Number operands, elision of trivial
# *1/*0/+0 terms -- keeps generated Exprs close to hand-written style)

function der_is_zero_num(e)
    return e isa Number && e == 0
end

function der_is_one_num(e)
    return e isa Number && e == 1
end

function der_is_negone_num(e)
    return e isa Number && e == -1
end

function der_neg(a)
    a isa Number && return -a
    return Expr(:call, :-, a)
end

function der_add(a, b)
    a isa Number && b isa Number && return a + b
    der_is_zero_num(a) && return b
    der_is_zero_num(b) && return a
    return Expr(:call, :+, a, b)
end

function der_sub(a, b)
    a isa Number && b isa Number && return a - b
    der_is_zero_num(b) && return a
    der_is_zero_num(a) && return der_neg(b)
    return Expr(:call, :-, a, b)
end

function der_mul(a, b)
    a isa Number && b isa Number && return a * b
    der_is_zero_num(a) && return 0.0
    der_is_zero_num(b) && return 0.0
    der_is_one_num(a) && return b
    der_is_one_num(b) && return a
    der_is_negone_num(a) && return der_neg(b)
    der_is_negone_num(b) && return der_neg(a)
    return Expr(:call, :*, a, b)
end

function der_div_expr(a, b)
    a isa Number && b isa Number && return a / b
    der_is_zero_num(a) && return 0.0
    der_is_one_num(b) && return a
    return Expr(:call, :/, a, b)
end

function der_pow_expr(base, expo)
    base isa Number && expo isa Number && return base ^ expo
    der_is_one_num(expo) && return base
    der_is_zero_num(expo) && return 1.0
    return Expr(:call, :^, base, expo)
end

# sum of a list of terms, dropping literal-zero terms; 0.0 if every
# term was zero (or the list was empty)
function der_sum_terms(terms)
    acc = 0.0
    seen = false
    for t in terms
        der_is_zero_num(t) && continue
        acc = seen ? der_add(acc, t) : t
        seen = true
    end
    return acc
end

# product of args[j] for every j != i (used by the n-ary * rule)
function der_product_excluding(args, i)
    factors = der_others(args, i)
    isempty(factors) && return 1.0
    acc = factors[1]
    for k in 2:length(factors)
        acc = der_mul(acc, factors[k])
    end
    return acc
end

# every element of args except the i-th
function der_others(args, i)
    return [args[j] for j in 1:length(args) if j != i]
end

# left-fold exprs into nested binary calls of op: op(op(op(a1,a2),a3),...)
function der_reduce_call(op::Symbol, exprs)
    length(exprs) == 1 && return exprs[1]
    acc = exprs[1]
    for k in 2:length(exprs)
        acc = Expr(:call, op, acc, exprs[k])
    end
    return acc
end


# ==================== emit_* ===================================
# Shared Expr-building helpers used by both codegen directions.

# for var = lo:hi ... end  -- or  for var = lo:step:hi ... end  when
# step isn't the literal 1 (e.g. the adjoint reverse sweep's
# descending `for i_seq_x = i_n:-1:1`). step is a required argument
# rather than an optional/nothing-defaulted one, because it mirrors
# a field the frozen :for statement shape *always* carries --
# parse_kernel fills in the Int64 literal 1 itself for a plain
# `lo:hi` header, so callers pulling straight from a parsed
# statement (stmt.step) never have to decide whether to pass it.
# The 2-arg-vs-3-arg choice is made right here instead, so a plain
# forward loop still emits the cleaner `lo:hi` form.
function emit_forloop(var::Symbol, lo, hi, step, body_exprs::Vector)
    range_expr = step == 1 ? Expr(:call, :(:), lo, hi) :
                              Expr(:call, :(:), lo, step, hi)
    return Expr(:for, Expr(:(=), var, range_expr), Expr(:block, body_exprs...))
end

# if cond ... else ... end -- omits the else clause entirely when
# else_exprs is empty (rather than emitting an empty `else end`
# block), matching skill-jade's "keep if/else to the strict minimum"
# spirit, and round-tripping parse_if's els = NamedTuple[] for a
# source `if` with no `else`.
function emit_if(cond, then_exprs::Vector, else_exprs::Vector)
    then_block = Expr(:block, then_exprs...)
    isempty(else_exprs) && return Expr(:if, cond, then_block)
    return Expr(:if, cond, then_block, Expr(:block, else_exprs...))
end

function emit_return_nothing()
    return :(return nothing)
end

# Builds the skill-jade rule-14 #-comment header:
#   # name(arg1, arg2, ...)
#   #
#   # <summary, possibly multi-line>
#   #
#   # arg1: <doc>
#   # arg2: <doc>
#   ...
# args and arg_docs must be the same length and in matching order.
# returns a String (Expr can't hold comments) -- prepend at file-write time
function emit_comment_header(name::Symbol, args::Vector{Symbol}, summary::String, arg_docs::Vector{String})
    length(args) == length(arg_docs) ||
        error("emit_comment_header: args and arg_docs must be the same length")
    lines = String["# $(name)($(join(args, ", ")))", "#"]
    for summary_line in split(summary, "\n")
        push!(lines, "# $(summary_line)")
    end
    push!(lines, "#")
    for k in 1:length(args)
        push!(lines, "# $(args[k]): $(arg_docs[k])")
    end
    return join(lines, "\n") * "\n"
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