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
#     float-kinded arg. Override via parse_override_indep_dep if needed.
# statement     :: one of
#     (kind=:assign, lhs, rhs)                                  # lhs/rhs :: Expr|Symbol|Number
#     (kind=:for, var::Symbol, lo, hi, step, sequential::Bool, body)  # body :: statement_list
#     (kind=:if, cond, then::statement_list, els::statement_list)
#     -- :while intentionally unsupported for now, see skill-stade.md
#     -- lo/hi/step are all Expr|Symbol|Number; a plain `lo:hi` header
#        parses with `step` set to the Int64 literal `1`.
# statement_list :: Vector{NamedTuple}
# active_map    :: Dict{Symbol,Bool}                # var/array name -> is-active
# snapshot_site :: (kind=:value|:array|:branch|:tripcount, array::Symbol, at::Int)
# snapshot_plan :: Vector{snapshot_site}
# lin_node/lin_plan :: internal-only shape, documented at the lin_*
#     section header below.
# der_rule_pair :: (tangent::Function, adjoint::Function)


# ==================== parse_* ================================
# Raw Expr -> validated kernel. Enforces every skill-jade rule that
# is actually visible at the Expr level as a hard error: snake_case,
# no argument annotations/kwargs (at the def or at any call site),
# only the four variable shapes (no Bool/String/tuple/range stored
# in a variable), no indirect indexing, no broadcasting, i_seq_
# prefix discipline, the intrinsic whitelist, div-not-÷, no compound
# assignment. Two skill-jade rules are pure *source-text* concerns
# -- the `#`-comment header, and `for i = ...` vs `for i in ...`
# (Julia's parser produces an identical Expr for both) -- and can't
# be checked from an Expr at all, so they're left to source review.

function parse_kernel(expr::Expr)
    expr.head == :function ||
        error("parse_kernel: expected a `function ... end` definition, got Expr(:$(expr.head), ...)")
    length(expr.args) == 2 ||
        error("parse_kernel: malformed function Expr (expected signature + body)")
    name, args = parse_signature(expr.args[1])
    body = parse_statements(parse_strip_lines(expr.args[2]))
    parse_check_local_names(body)
    kinds = shape_infer((name = name, args = args), body)
    indep, dep = parse_infer_indep_dep(args, kinds)
    sig = (name = name, args = args, kinds = kinds,
           independents = indep, dependents = dep)
    return (sig = sig, body = body)
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

# lowercase letters, digits, underscores only; can't be empty or
# start with a digit
function parse_check_snake_case(name::Symbol)
    s = String(name)
    isempty(s) && return false
    isdigit(s[1]) && return false
    return all(c -> islowercase(c) || isdigit(c) || c == '_', s)
end

# the reserved i_seq_ prefix marks a genuinely sequential loop --
# the prefix and the sequential flag must agree, and the whole name
# must still be snake_case
function parse_check_loop_prefix(var::Symbol, sequential::Bool)
    has_prefix = startswith(String(var), "i_seq_")
    return has_prefix == sequential && parse_check_snake_case(var)
end

# ---- function signature: name + positional, untyped arguments ----

function parse_signature(sig_expr)
    sig_expr isa Expr && sig_expr.head == :call ||
        error("parse_kernel: function signature must be a plain `name(args...)` call -- no where-clauses or return-type annotations")
    name_expr = sig_expr.args[1]
    name_expr isa Symbol ||
        error("parse_kernel: function name must be a plain identifier")
    parse_check_snake_case(name_expr) ||
        error("parse_kernel: function name :$(name_expr) is not snake_case")

    args = Symbol[]
    for a in sig_expr.args[2:end]
        if a isa Symbol
            parse_check_snake_case(a) ||
                error("parse_kernel: argument name :$a is not snake_case")
            push!(args, a)
        elseif a isa Expr && a.head == :parameters
            error("parse_kernel: keyword arguments aren't allowed (found a `;` section in the signature of :$(name_expr))")
        elseif a isa Expr && a.head == :(::)
            error("parse_kernel: argument `$(a)` has a type annotation, which isn't allowed")
        elseif a isa Expr && a.head == :kw
            error("parse_kernel: argument `$(a)` has a default value -- every argument must be positional")
        else
            error("parse_kernel: unsupported argument form `$(a)` in the signature of :$(name_expr)")
        end
    end
    return name_expr, args
end

# ---- statement-list plumbing ----

# drop LineNumberNodes; if the last statement is a bare `return
# nothing` (the only body-level return skill-jade allows), drop it
# too -- any other `return`, anywhere, is a hard error
#
# NB: `nothing` written in source parses to the Symbol :nothing, not
# the Nothing singleton -- unlike `true`/`false`, which the parser
# does special-case as real Bool literals, `nothing` is just an
# ordinary identifier that only resolves to the singleton value at
# evaluation time. Both forms are checked for, since a caller could
# in principle hand parse_kernel an Expr built programmatically
# rather than parsed from source text.
function parse_is_nothing_literal(x)
    return x === nothing || x === :nothing
end

function parse_strip_lines(block_expr)
    block_expr isa Expr && block_expr.head == :block ||
        error("parse_kernel: expected a `begin ... end`-style function body")
    stmts = [s for s in block_expr.args if !(s isa LineNumberNode)]
    if !isempty(stmts) && parse_is_nothing_literal(stmts[end])
        error("parse_kernel: don't shorten `return nothing` to a bare `nothing`")
    end
    if !isempty(stmts) && stmts[end] isa Expr && stmts[end].head == :return
        last_stmt = stmts[end]
        length(last_stmt.args) == 1 && parse_is_nothing_literal(last_stmt.args[1]) ||
            error("parse_kernel: the only body-level `return` skill-jade allows is exactly `return nothing`")
        stmts = stmts[1:end-1]
    end
    for s in stmts
        s isa Expr && s.head == :return &&
            error("parse_kernel: `return` may only appear once, as the final `return nothing`")
    end
    return stmts
end

function parse_statements(stmts::Vector)
    return NamedTuple[parse_statement(s) for s in stmts]
end

function parse_statement(stmt)
    stmt isa Expr || error("parse_kernel: unsupported statement `$(stmt)`")
    if stmt.head == :(=)
        return parse_assign(stmt)
    elseif stmt.head == :for
        return parse_for(stmt)
    elseif stmt.head == :if
        return parse_if(stmt)
    elseif stmt.head == :while
        error("parse_kernel: `while` loops aren't supported yet (see skill-stade.md)")
    elseif stmt.head in (:+=, :-=, :*=, :/=, :÷=, :^=, :%=, Symbol("\\="), :.=)
        error("parse_kernel: compound/broadcast assignment `$(stmt)` isn't allowed -- write it out in full, e.g. `x = x + ...`")
    else
        error("parse_kernel: unsupported statement form `Expr(:$(stmt.head), ...)`")
    end
end

# ---- assignment ----

function parse_assign(stmt::Expr)
    lhs = parse_lvalue(stmt.args[1])
    rhs = stmt.args[2]
    parse_check_expr(rhs, false)
    return (kind = :assign, lhs = lhs, rhs = rhs)
end

function parse_lvalue(lhs)
    if lhs isa Symbol
        return lhs
    elseif lhs isa Expr && lhs.head == :ref
        parse_check_no_indirect_indexing(lhs)
        for idx in lhs.args[2:end]
            parse_check_expr(idx, false)
        end
        return lhs
    else
        error("parse_kernel: unsupported assignment target `$(lhs)` -- only a plain variable or an array element is allowed")
    end
end

# ---- for loop ----

function parse_for(stmt::Expr)
    header = stmt.args[1]
    header isa Expr && header.head == :(=) ||
        error("parse_kernel: unsupported `for` header `$(header)`")
    var = header.args[1]
    var isa Symbol ||
        error("parse_kernel: `for` loop variable must be a plain identifier")
    var == Symbol("_") &&
        error("parse_kernel: `for` loops must name their iteration variable -- never a throwaway `_`")

    range_expr = header.args[2]
    range_expr isa Expr && range_expr.head == :call && range_expr.args[1] == :(:) ||
        error("parse_kernel: `for` loop range must be written `lo:hi` or `lo:step:hi`")
    range_args = range_expr.args[2:end]
    if length(range_args) == 2
        lo, step, hi = range_args[1], 1, range_args[2]
    elseif length(range_args) == 3
        lo, step, hi = range_args[1], range_args[2], range_args[3]
    else
        error("parse_kernel: unsupported range arity in `for` loop header `$(range_expr)`")
    end
    parse_check_expr(lo, false)
    parse_check_expr(step, false)
    parse_check_expr(hi, false)

    sequential = startswith(String(var), "i_seq_")
    parse_check_loop_prefix(var, sequential) ||
        error("parse_kernel: loop variable :$var doesn't follow the i_seq_ prefix convention")

    body = parse_statements(parse_strip_lines(stmt.args[2]))
    return (kind = :for, var = var, lo = lo, hi = hi, step = step, sequential = sequential, body = body)
end

# ---- if statement (plain if/else only -- no elseif chains yet) ----

function parse_if(stmt::Expr)
    length(stmt.args) in (2, 3) || error("parse_kernel: unsupported `if` form")
    cond = stmt.args[1]
    parse_check_expr(cond, true)

    then_block = stmt.args[2]
    then_block isa Expr && then_block.head == :block ||
        error("parse_kernel: malformed `if` body")
    then_stmts = parse_statements(parse_strip_lines(then_block))

    els_stmts = NamedTuple[]
    if length(stmt.args) == 3
        els_block = stmt.args[3]
        els_block isa Expr && els_block.head == :elseif &&
            error("parse_kernel: `elseif` chains aren't representable in the current statement shape yet -- use nested `if`/`else`")
        els_block isa Expr && els_block.head == :block ||
            error("parse_kernel: malformed `else` body")
        els_stmts = parse_statements(parse_strip_lines(els_block))
    end
    return (kind = :if, cond = cond, then = then_stmts, els = els_stmts)
end

# ---- expression validity ----
# built fresh per call rather than as top-level consts -- see
# skill-stade.md rule 3.

function parse_arith_ops()
    return Set{Symbol}([:+, :-, :*, :/, :^])
end

function parse_compare_ops()
    return Set{Symbol}([:>, :<, :>=, :<=, :(==), :(!=)])
end

function parse_intrinsic_whitelist()
    return Set{Symbol}([
        :abs, :sqrt, :exp, :log, :log10, :sin, :cos, :tan,
        :asin, :acos, :atan, :sinh, :cosh, :tanh,
        :mod, :div, :max, :min, :sign, :floor, :ceil, :trunc,
    ])
end

# walk an expression (assignment rhs, index, loop bound, or if
# condition) and hard-error on anything skill-jade forbids.
# in_condition allows the comparison operators -- they may only
# appear directly in an if/while header, never stored in a variable.
function parse_check_expr(expr, in_condition::Bool)
    if expr isa Bool
        error("parse_kernel: literal `$(expr)` -- Bool isn't one of the four allowed variable shapes")
    elseif expr isa Symbol || expr isa Number
        return nothing
    elseif !(expr isa Expr)
        error("parse_kernel: literal `$(repr(expr))` of type $(typeof(expr)) isn't one of the four allowed variable shapes (Float64, Int64, Array{Float64}, Array{Int64})")
    end

    if expr.head == :call
        op = expr.args[1]
        op isa Symbol ||
            error("parse_kernel: only calls to a plain function name are allowed, got `$(expr)`")
        call_args = expr.args[2:end]
        any(a -> a isa Expr && a.head == :parameters, call_args) &&
            error("parse_kernel: keyword arguments aren't allowed at call sites (`$(expr)`)")
        if op in parse_arith_ops()
            # arithmetic, arity-agnostic (covers unary minus too)
        elseif in_condition && op in parse_compare_ops()
            # comparison, only legal directly inside an if/while condition
        elseif op in parse_intrinsic_whitelist()
            # has a direct Fortran-intrinsic counterpart
        elseif op == :÷
            error("parse_kernel: use `div(a, b)` instead of `a ÷ b`")
        elseif op == :(:)
            error("parse_kernel: ranges aren't one of the four allowed variable shapes")
        elseif op in parse_compare_ops()
            error("parse_kernel: comparison `$(expr)` can't be stored in a variable -- only allowed directly in an if/while condition")
        else
            error("parse_kernel: call to `$(op)` isn't a whitelisted intrinsic (no direct Fortran counterpart) and wasn't user-supplied")
        end
        for a in call_args
            parse_check_expr(a, in_condition)
        end
    elseif expr.head == :ref
        parse_check_no_indirect_indexing(expr)
        for idx in expr.args[2:end]
            parse_check_expr(idx, false)
        end
    elseif expr.head == :comparison
        in_condition ||
            error("parse_kernel: comparison `$(expr)` can't be stored in a variable -- only allowed directly in an if/while condition")
        for a in expr.args
            parse_check_expr(a, false)
        end
    elseif in_condition && expr.head in (:&&, :||)
        for a in expr.args
            parse_check_expr(a, true)
        end
    elseif expr.head == :. || expr.head == :.=
        error("parse_kernel: broadcasting (`$(expr)`) isn't allowed -- write the explicit loop instead")
    elseif expr.head == :tuple
        error("parse_kernel: tuples aren't one of the four allowed variable shapes")
    else
        error("parse_kernel: unsupported expression form `Expr(:$(expr.head), ...)` in `$(expr)`")
    end
    return nothing
end

# rejects a[b[i]]-style indirect indexing: no index expression may
# itself read another array
function parse_check_no_indirect_indexing(ref_expr::Expr)
    for idx in ref_expr.args[2:end]
        parse_contains_ref(idx) &&
            error("parse_kernel: indirect indexing `$(ref_expr)` isn't allowed -- read the index into a scalar on its own line first")
    end
    return nothing
end

function parse_contains_ref(expr)
    expr isa Expr || return false
    expr.head == :ref && return true
    return any(parse_contains_ref, expr.args)
end

# ---- naming discipline over the parsed body (args + function name
# are already checked in parse_signature; this covers locals) ----

function parse_check_local_names(body::Vector{NamedTuple})
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            parse_check_snake_case(var) ||
                error("parse_kernel: variable name :$var is not snake_case")
        elseif stmt.kind == :for
            parse_check_local_names(stmt.body)   # loop var already checked in parse_for
        elseif stmt.kind == :if
            parse_check_local_names(stmt.then)
            parse_check_local_names(stmt.els)
        end
    end
    return nothing
end


# ==================== shape_* =================================
# Infer each argument/local's kind syntactically -- no explicit
# manifest to cross-check against, since skill-jade kernels never
# carry type annotations. Two syntactic signals decide a variable's
# kind:
#   array vs scalar: is it ever indexed (`v[...]`) anywhere?
#   int vs float:    is there any evidence it must be an index/size
#                     (a for-loop variable, a range bound, a `div`
#                     argument, or an array-subscript expression)?
# Anything not provably Int64 defaults to Float64, matching rule 7's
# "physical/field quantities are Float64" -- loop/size variables are
# the exception that has to be proven, not assumed.

function shape_infer(sig, body)
    vars = Set{Symbol}(sig.args)
    shape_collect_vars!(body, vars)

    is_array = Dict{Symbol,Bool}(v => false for v in vars)
    shape_mark_arrays!(body, is_array)

    is_int = Dict{Symbol,Bool}(v => false for v in vars)
    shape_mark_int_direct!(body, is_int)
    for _ in 1:8
        shape_propagate_int!(body, is_int) || break
    end

    kinds = Dict{Symbol,Symbol}()
    for v in vars
        if is_array[v]
            kinds[v] = is_int[v] ? :array_int : :array_float
        else
            kinds[v] = is_int[v] ? :scalar_int : :scalar_float
        end
    end
    return kinds
end

# ---- variable discovery: every arg + every name ever assigned to
# or read, across the whole (possibly nested) statement_list ----

function shape_collect_vars!(body::Vector{NamedTuple}, vars::Set{Symbol})
    for stmt in body
        if stmt.kind == :assign
            push!(vars, stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1])
            shape_collect_expr_vars!(stmt.lhs, vars)
            shape_collect_expr_vars!(stmt.rhs, vars)
        elseif stmt.kind == :for
            push!(vars, stmt.var)
            shape_collect_expr_vars!(stmt.lo, vars)
            shape_collect_expr_vars!(stmt.hi, vars)
            shape_collect_expr_vars!(stmt.step, vars)
            shape_collect_vars!(stmt.body, vars)
        elseif stmt.kind == :if
            shape_collect_expr_vars!(stmt.cond, vars)
            shape_collect_vars!(stmt.then, vars)
            shape_collect_vars!(stmt.els, vars)
        end
    end
    return nothing
end

function shape_collect_expr_vars!(expr, vars::Set{Symbol})
    if expr isa Symbol
        push!(vars, expr)
    elseif expr isa Expr
        if expr.head == :call
            for a in expr.args[2:end]   # skip the operator/function name
                shape_collect_expr_vars!(a, vars)
            end
        elseif expr.head == :ref
            push!(vars, expr.args[1])
            for a in expr.args[2:end]
                shape_collect_expr_vars!(a, vars)
            end
        else
            for a in expr.args
                shape_collect_expr_vars!(a, vars)
            end
        end
    end
    return nothing
end

# ---- array vs scalar: does this name ever appear as v[...]? ----

function shape_mark_arrays!(body::Vector{NamedTuple}, is_array::Dict{Symbol,Bool})
    for stmt in body
        if stmt.kind == :assign
            shape_mark_arrays_expr!(stmt.lhs, is_array)
            shape_mark_arrays_expr!(stmt.rhs, is_array)
        elseif stmt.kind == :for
            shape_mark_arrays_expr!(stmt.lo, is_array)
            shape_mark_arrays_expr!(stmt.hi, is_array)
            shape_mark_arrays_expr!(stmt.step, is_array)
            shape_mark_arrays!(stmt.body, is_array)
        elseif stmt.kind == :if
            shape_mark_arrays_expr!(stmt.cond, is_array)
            shape_mark_arrays!(stmt.then, is_array)
            shape_mark_arrays!(stmt.els, is_array)
        end
    end
    return nothing
end

function shape_mark_arrays_expr!(expr, is_array::Dict{Symbol,Bool})
    expr isa Expr || return nothing
    if expr.head == :ref
        haskey(is_array, expr.args[1]) && (is_array[expr.args[1]] = true)
        for a in expr.args[2:end]
            shape_mark_arrays_expr!(a, is_array)
        end
    else
        for a in expr.args
            shape_mark_arrays_expr!(a, is_array)
        end
    end
    return nothing
end

# ---- int evidence, pass 1: direct syntactic roles ----
# a for-loop's own variable, anything feeding a range bound, a
# `div` argument, an array-subscript expression, or an operand
# compared against an Int64 literal (e.g. `i_branch == 1` -- a
# branch/mode selector, not a physical quantity) is Int64.

function shape_mark_int_direct!(body::Vector{NamedTuple}, is_int::Dict{Symbol,Bool})
    for stmt in body
        if stmt.kind == :assign
            shape_mark_int_from_div_and_index!(stmt.lhs, is_int)
            shape_mark_int_from_div_and_index!(stmt.rhs, is_int)
        elseif stmt.kind == :for
            is_int[stmt.var] = true
            shape_mark_int_leaves!(stmt.lo, is_int)
            shape_mark_int_leaves!(stmt.hi, is_int)
            shape_mark_int_leaves!(stmt.step, is_int)
            shape_mark_int_from_div_and_index!(stmt.lo, is_int)
            shape_mark_int_from_div_and_index!(stmt.hi, is_int)
            shape_mark_int_from_div_and_index!(stmt.step, is_int)
            shape_mark_int_direct!(stmt.body, is_int)
        elseif stmt.kind == :if
            shape_mark_int_from_div_and_index!(stmt.cond, is_int)
            shape_mark_int_from_comparisons!(stmt.cond, is_int)
            shape_mark_int_direct!(stmt.then, is_int)
            shape_mark_int_direct!(stmt.els, is_int)
        end
    end
    return nothing
end

# mark every bare-variable leaf inside expr as Int64
function shape_mark_int_leaves!(expr, is_int::Dict{Symbol,Bool})
    if expr isa Symbol
        haskey(is_int, expr) && (is_int[expr] = true)
    elseif expr isa Expr
        start = expr.head == :call ? 2 : 1   # skip the operator/function name
        for a in expr.args[start:end]
            shape_mark_int_leaves!(a, is_int)
        end
    end
    return nothing
end

# find `div(...)` arguments and array-subscript expressions inside
# expr and mark their leaves Int64
function shape_mark_int_from_div_and_index!(expr, is_int::Dict{Symbol,Bool})
    expr isa Expr || return nothing
    if expr.head == :call
        if expr.args[1] == :div
            for a in expr.args[2:end]
                shape_mark_int_leaves!(a, is_int)
            end
        end
        for a in expr.args[2:end]
            shape_mark_int_from_div_and_index!(a, is_int)
        end
    elseif expr.head == :ref
        for a in expr.args[2:end]
            shape_mark_int_leaves!(a, is_int)
            shape_mark_int_from_div_and_index!(a, is_int)
        end
    else
        for a in expr.args
            shape_mark_int_from_div_and_index!(a, is_int)
        end
    end
    return nothing
end

# comparison operators, kept local to shape_* rather than reusing
# parse_compare_ops's whitelist -- shape_* shouldn't reach into
# parse_*'s internals for something this small
function shape_compare_ops()
    return Set{Symbol}([:>, :<, :>=, :<=, :(==), :(!=)])
end

# a bare variable (or subscript-free expression) compared against an
# Int64 literal is itself Int64 -- e.g. `i_branch == 1`. Comparing
# against a Float64 literal (`u[i] > 0.0`) gives no evidence either
# way, so it's left alone; default-to-float already handles that
# case correctly.
function shape_mark_int_from_comparisons!(expr, is_int::Dict{Symbol,Bool})
    expr isa Expr || return nothing
    if expr.head == :call && length(expr.args) == 3 && expr.args[1] in shape_compare_ops()
        a, b = expr.args[2], expr.args[3]
        b isa Integer && !(b isa Bool) && shape_mark_int_leaves!(a, is_int)
        a isa Integer && !(a isa Bool) && shape_mark_int_leaves!(b, is_int)
    end
    for a in (expr.head == :call ? expr.args[2:end] : expr.args)
        shape_mark_int_from_comparisons!(a, is_int)
    end
    return nothing
end

# ---- int evidence, pass 2: propagate through plain assignments ----
# A bare copy `lhs = rhs` forces lhs and rhs to share one kind, since
# assignment can't change it -- evidence has to flow both ways.

function shape_propagate_int!(body::Vector{NamedTuple}, is_int::Dict{Symbol,Bool})
    changed = false
    for stmt in body
        if stmt.kind == :assign && stmt.lhs isa Symbol
            if !is_int[stmt.lhs] && shape_expr_int_status(stmt.rhs, is_int) == :int
                is_int[stmt.lhs] = true
                changed = true
            end
            # backward: a bare copy forces the rhs variable to match
            # a lhs kind discovered from evidence elsewhere (the
            # forward direction is already covered by the check above)
            if stmt.rhs isa Symbol && haskey(is_int, stmt.rhs) &&
               is_int[stmt.lhs] && !is_int[stmt.rhs]
                is_int[stmt.rhs] = true
                changed = true
            end
        elseif stmt.kind == :for
            changed = shape_propagate_int!(stmt.body, is_int) || changed
        elseif stmt.kind == :if
            changed = shape_propagate_int!(stmt.then, is_int) || changed
            changed = shape_propagate_int!(stmt.els, is_int) || changed
        end
    end
    return changed
end

# tri-state: :int (provably integer), :float (provably float), or
# :unknown (not yet decidable from current evidence)
function shape_expr_int_status(expr, is_int::Dict{Symbol,Bool})
    if expr isa Symbol
        return get(is_int, expr, false) ? :int : :unknown
    elseif expr isa Integer
        return :int
    elseif expr isa AbstractFloat
        return :float
    elseif expr isa Expr && expr.head == :call
        op = expr.args[1]
        arg_status = [shape_expr_int_status(a, is_int) for a in expr.args[2:end]]
        if op == :div
            return :int
        elseif op == :/
            return :float
        elseif op in (:sqrt, :exp, :log, :log10, :sin, :cos, :tan,
                      :asin, :acos, :atan, :sinh, :cosh, :tanh)
            return :float
        elseif op in (:+, :-, :*, :^, :abs, :sign, :max, :min, :mod, :floor, :ceil, :trunc)
            any(==(:float), arg_status) && return :float
            all(==(:int), arg_status) && return :int
            return :unknown
        else
            return :unknown
        end
    else
        return :unknown
    end
end


# ==================== der_* ====================================
# Derivative rule table -- built fresh per call, never stored.
#
# Every rule pair is built from a per-operator "local partials" list:
# partials[i] is d(op(args...))/d(args[i]), symbolic in the primal
# args. tangent/adjoint are the two standard contractions against a
# seed: tangent sums partials[i]*dargs[i]; adjoint returns one
# contribution per arg (partials[i]*out_adjoint), left for agen_ to
# turn into an accumulation statement. A few algebraic simplifications
# (dropping *1, *0, +0 terms, folding literal arithmetic) keep the
# generated Exprs simple.

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

# for var = lo:hi ... end -- or for var = lo:step:hi ... end when
# step isn't the literal 1 (e.g. a reversed, descending sweep). step
# is required rather than optional, since it mirrors a field the
# frozen :for statement shape always carries -- parse_kernel fills in
# the Int64 literal 1 for a plain `lo:hi` header, so callers pulling
# from a parsed statement never have to decide whether to pass it.
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

# `return nothing` when there's nothing to hand back; `return v` (one
# var) or `return v1,v2,...` (several) otherwise -- the only way a
# scalar function argument's new value escapes the call, since Julia
# passes scalars by value (unlike an array argument, mutated in place
# and needing no return at all). Shared by tgen_/agen_: forward mode
# uses it for a reassigned scalar arg's shadow, reverse mode for
# every scalar-float arg's accumulated adjoint, and agen_'s
# initstacks_ generator reuses the exact same bare/tuple/nothing
# shape to hand back its stack(s).
function emit_return_scalars(vars::Vector{Symbol})
    isempty(vars) && return emit_return_nothing()
    length(vars) == 1 && return Expr(:return, vars[1])
    return Expr(:return, Expr(:tuple, vars...))
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
# Forward taint analysis from independents through assignments,
# swept to a fixed point rather than once. Whole-variable
# granularity, monotonic: once a variable is shown reachable from an
# independent by some assignment, it stays marked active for the
# rest of the map, even past a later statement that overwrites it
# with something inactive -- activity records "does this variable
# ever carry a derivative", not "does it carry one right now".
#
# A single top-down pass over the body already catches taint that
# flows forward in textual order. It misses taint a loop carries
# from one (conceptual) iteration into an earlier statement of the
# *same* loop body -- e.g. `b[i] = a[i]` appearing before
# `a[i] = x[i]` inside one `for` block only sees `a` active on the
# lap after `a` was first activated. Re-running the whole-body pass
# until nothing changes handles that without unrolling anything: at
# most one variable can flip false->true per pass, so a bound of
# (#variables + 1) passes always reaches the fixed point.

function act_analyze(kernel)
    active_map = Dict{Symbol,Bool}(v => false for v in keys(kernel.sig.kinds))
    for v in kernel.sig.independents
        active_map[v] = true
    end
    bound = length(active_map) + 1
    for _ in 1:bound
        act_propagate!(kernel.body, active_map, kernel.sig.kinds) || break
    end
    return active_map
end

# one pass over a statement_list, recursing into nested for/if
# bodies; returns whether any active_map entry flipped during this
# pass
function act_propagate!(body::Vector{NamedTuple}, active_map::Dict{Symbol,Bool}, kinds::Dict{Symbol,Symbol})
    changed = false
    for stmt in body
        if stmt.kind == :assign
            changed = act_propagate_assign!(stmt, active_map, kinds) || changed
        elseif stmt.kind == :for
            # loop bounds are always Int64 (shape_*'s doing) -- only
            # the body can possibly touch activity
            changed = act_propagate!(stmt.body, active_map, kinds) || changed
        elseif stmt.kind == :if
            # both arms visited unconditionally: activity is a static
            # may-reach property, not tied to which branch runs
            changed = act_propagate!(stmt.then, active_map, kinds) || changed
            changed = act_propagate!(stmt.els, active_map, kinds) || changed
        end
    end
    return changed
end

# lhs becomes active the first time its rhs reads an already-active
# variable. Int64-kinded targets (loop counters, branch selectors,
# sizes) never become active -- only Float64/Array{Float64} can
# carry a derivative at all (skill-jade rule 7).
function act_propagate_assign!(stmt::NamedTuple, active_map::Dict{Symbol,Bool}, kinds::Dict{Symbol,Symbol})
    var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
    kinds[var] in (:scalar_int, :array_int) && return false
    active_map[var] && return false   # already active -- nothing new to learn
    act_expr_active(stmt.rhs, active_map) || return false
    active_map[var] = true
    return true
end

# does expr read any variable currently marked active? Array reads
# (`v[i]`) test the array's own name (args[1] of a :ref), matching
# the whole-variable/whole-array granularity the map is defined at.
function act_expr_active(expr, active_map::Dict{Symbol,Bool})
    if expr isa Symbol
        return get(active_map, expr, false)
    elseif expr isa Expr
        start = expr.head == :call ? 2 : 1   # skip the operator/function name
        for a in expr.args[start:end]
            act_expr_active(a, active_map) && return true
        end
    end
    return false
end


# ==================== snap_* ===================================
# Push/pop site analysis (the TBR-equivalent). For each write,
# decide whether the reverse sweep needs a recorded snapshot.
#
# Whole-variable granularity: a site names the variable/array as a
# whole, never a specific index. `at` is a shared counter assigned in
# forward-sweep order -- the reverse sweep pops in exact reverse.
#
#   :array / :value -- an active write whose old value is genuinely
#     needed later. Flagged per :assign by two rules:
#       1. self-reference: the written variable also appears on its
#          own rhs, except a pure accumulation (same exact slot, one
#          direct `+`/`-` operand -- see snap_is_pure_accumulation),
#          which never needs the old value since old and new are the
#          same quantity for adjoint purposes.
#       2. cross-statement: written inside a sequential loop and read
#          by some other statement anywhere in the kernel -- a write
#          in a non-sequential loop never triggers this, even if read
#          again later, since nothing overwrites it before that read.
#   :branch -- one per `if`, unconditionally: the reverse sweep must
#     replay whichever arm the forward sweep actually took.
#   :tripcount -- a loop's bounds reference a variable reassigned
#     elsewhere in the kernel, so its trip count could be gone by the
#     time the reverse sweep needs to replay it. Deliberately
#     conservative: doesn't try to prove the reassignment happens
#     strictly after or outside this loop, just flags it regardless.

function snap_plan(kernel, active_map)
    reassigned = snap_collect_reassigned(kernel.body)
    read_anywhere = snap_collect_reads(kernel.body)
    sites = NamedTuple[]
    counter = Ref(0)
    snap_walk!(kernel.body, active_map, kernel.sig.kinds, false, reassigned, read_anywhere, sites, counter)
    return sites
end

# every variable ever reassigned via a scalar (non-array) :assign,
# anywhere in the kernel, at any nesting depth
function snap_collect_reassigned(body)
    reassigned = Set{Symbol}()
    for stmt in body
        if stmt.kind == :assign
            stmt.lhs isa Symbol && push!(reassigned, stmt.lhs)
        elseif stmt.kind == :for
            union!(reassigned, snap_collect_reassigned(stmt.body))
        elseif stmt.kind == :if
            union!(reassigned, snap_collect_reassigned(stmt.then))
            union!(reassigned, snap_collect_reassigned(stmt.els))
        end
    end
    return reassigned
end

# every variable read anywhere in the kernel: rhs of every :assign,
# plus every :if's condition (for-bounds are handled separately by
# the :tripcount rule, since a bound isn't a normal "value read")
function snap_collect_reads(body)
    reads = Set{Symbol}()
    for stmt in body
        if stmt.kind == :assign
            snap_collect_expr_vars!(stmt.rhs, reads)
        elseif stmt.kind == :for
            union!(reads, snap_collect_reads(stmt.body))
        elseif stmt.kind == :if
            snap_collect_expr_vars!(stmt.cond, reads)
            union!(reads, snap_collect_reads(stmt.then))
            union!(reads, snap_collect_reads(stmt.els))
        end
    end
    return reads
end

# collect every Symbol leaf appearing in expr (bare read or array
# name inside a :ref) into vars
function snap_collect_expr_vars!(expr, vars)
    if expr isa Symbol
        push!(vars, expr)
    elseif expr isa Expr
        start = expr.head == :call ? 2 : 1
        for a in expr.args[start:end]
            snap_collect_expr_vars!(a, vars)
        end
    end
    return nothing
end

# one forward-order pass, emitting sites as they're found; `seq` is
# whether the walk is currently nested inside at least one
# sequential loop
function snap_walk!(body, active_map, kinds, seq, reassigned, read_anywhere, sites, counter)
    for stmt in body
        if stmt.kind == :assign
            snap_check_assign!(stmt, active_map, kinds, seq, read_anywhere, sites, counter)
        elseif stmt.kind == :for
            snap_check_tripcount!(stmt, reassigned, sites, counter)
            snap_walk!(stmt.body, active_map, kinds, seq || stmt.sequential, reassigned, read_anywhere, sites, counter)
        elseif stmt.kind == :if
            counter[] = counter[] + 1
            push!(sites, (kind = :branch, array = :cond, at = counter[]))
            snap_walk!(stmt.then, active_map, kinds, seq, reassigned, read_anywhere, sites, counter)
            snap_walk!(stmt.els, active_map, kinds, seq, reassigned, read_anywhere, sites, counter)
        end
    end
    return nothing
end

function snap_check_assign!(stmt, active_map, kinds, seq, read_anywhere, sites, counter)
    var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
    active_map[var] || return nothing
    snap_is_pure_accumulation(stmt.lhs, stmt.rhs, var) && return nothing
    needs_site = snap_count_var_refs(stmt.rhs, var) > 0 || (seq && var in read_anywhere)
    needs_site || return nothing
    kind = kinds[var] == :array_float ? :array : :value
    counter[] = counter[] + 1
    push!(sites, (kind = kind, array = var, at = counter[]))
    return nothing
end

function snap_check_tripcount!(stmt, reassigned, sites, counter)
    bound_vars = Set{Symbol}()
    snap_collect_expr_vars!(stmt.lo, bound_vars)
    snap_collect_expr_vars!(stmt.hi, bound_vars)
    snap_collect_expr_vars!(stmt.step, bound_vars)
    for bv in bound_vars
        if bv in reassigned
            counter[] = counter[] + 1
            push!(sites, (kind = :tripcount, array = bv, at = counter[]))
        end
    end
    return nothing
end

# is `lhs = rhs` an identity-preserving self-update -- one where
# d(new)/d(old) = 1, so old and new are the same quantity for adjoint
# purposes and nothing needs recording? Two shapes qualify: lhs as
# any direct top-level `+` operand, or lhs as specifically the
# left/minuend operand of a binary `-` (the right operand does not
# qualify -- d(a-b)/db = -1, not an identity). Either way, lhs's
# exact slot must not appear anywhere else in rhs (a different index
# of the same array is a different slot and doesn't disqualify it).
function snap_is_pure_accumulation(lhs, rhs, var)
    (rhs isa Expr && rhs.head == :call) || return false
    op = rhs.args[1]
    if op == :+
        matches = 0
        for a in rhs.args[2:end]
            a == lhs && (matches = matches + 1)
        end
        matches == 1 || return false
    elseif op == :- && length(rhs.args) == 3
        rhs.args[2] == lhs || return false
    else
        return false
    end
    return snap_count_expr_occurrences(rhs, lhs) == 1
end

# how many times does `var`'s name appear anywhere in expr (as a
# bare read, or as an array name inside a :ref)?
function snap_count_var_refs(expr, var)
    if expr isa Symbol
        return expr == var ? 1 : 0
    elseif expr isa Expr
        start = expr.head == :call ? 2 : 1
        total = 0
        for a in expr.args[start:end]
            total = total + snap_count_var_refs(a, var)
        end
        return total
    end
    return 0
end

# how many times does the exact sub-expression `target` (a whole
# Symbol or a whole `:ref`) appear anywhere in expr? Unlike
# snap_count_var_refs (bare variable name), this distinguishes
# different indices of the same array.
function snap_count_expr_occurrences(expr, target)
    total = expr == target ? 1 : 0
    if expr isa Expr
        start = expr.head == :call ? 2 : 1
        for a in expr.args[start:end]
            total = total + snap_count_expr_occurrences(a, target)
        end
    end
    return total
end


# ==================== lin_* =====================================
# Shared derivative-tree representation, swept differently by each
# codegen direction. Builds structure only, using der_partials to
# attach each :op node's local partials so both directions can read
# them straight off the tree.
#
# lin_node -- one node of a statement's rhs, mirroring its primal
#   expression tree one-for-one:
#     kind = :leaf -- a variable/array-element read or a literal;
#       op = :leaf, args = [], children = [], partials = [].
#     kind = :op -- expr rebuilt from the processed children; op is
#       the operator; children is one lin_node per call argument;
#       args mirrors children's exprs; partials = der_partials(op,
#       args) -- what both directions contract against a seed.
#   Every node carries active::Bool (for :leaf, whether the
#   referenced variable is active; for :op, any child active) --
#   lets codegen skip subtrees that provably carry no derivative.
# lin_stmt -- one processed statement, parallel to the frozen
#   `statement` shape; only :assign gets a built tree, :for/:if just
#   thread their own fields through with a recursively-built body:
#     (kind=:assign, lhs, active::Bool, tree::lin_node)
#     (kind=:for, var, lo, hi, step, sequential, body::lin_plan)
#     (kind=:if, cond, then::lin_plan, els::lin_plan)
# lin_plan :: Vector{lin_stmt}

function lin_build(kernel, active_map)
    return lin_build_body(kernel.body, active_map)
end

function lin_build_body(body, active_map)
    plan = NamedTuple[]
    for stmt in body
        push!(plan, lin_build_stmt(stmt, active_map))
    end
    return plan
end

function lin_build_stmt(stmt, active_map)
    if stmt.kind == :assign
        tree = lin_build_expr(stmt.rhs, active_map)
        return (kind = :assign, lhs = stmt.lhs, active = tree.active, tree = tree)
    elseif stmt.kind == :for
        return (kind = :for, var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step,
                sequential = stmt.sequential, body = lin_build_body(stmt.body, active_map))
    elseif stmt.kind == :if
        return (kind = :if, cond = stmt.cond,
                then = lin_build_body(stmt.then, active_map),
                els = lin_build_body(stmt.els, active_map))
    end
    error("lin_build_stmt: unrecognized statement kind :$(stmt.kind)")
end

# recursively mirror one primal sub-expression into a lin_node
function lin_build_expr(expr, active_map)
    if expr isa Symbol
        return (kind = :leaf, expr = expr, op = :leaf, args = [], children = NamedTuple[],
                partials = [], active = get(active_map, expr, false))
    elseif expr isa Number
        return (kind = :leaf, expr = expr, op = :leaf, args = [], children = NamedTuple[],
                partials = [], active = false)
    elseif expr isa Expr && expr.head == :ref
        # array-element read: a leaf for differentiation purposes --
        # the array's own name carries activity; indices are always
        # Int64 and never differentiated
        arr = expr.args[1]
        return (kind = :leaf, expr = expr, op = :leaf, args = [], children = NamedTuple[],
                partials = [], active = get(active_map, arr, false))
    elseif expr isa Expr && expr.head == :call
        op = expr.args[1]
        children = [lin_build_expr(a, active_map) for a in expr.args[2:end]]
        args = [c.expr for c in children]
        rebuilt = Expr(:call, op, args...)
        partials = der_partials(op, args)
        active = any(c.active for c in children)
        return (kind = :op, expr = rebuilt, op = op, args = args, children = children,
                partials = partials, active = active)
    end
    error("lin_build_expr: unsupported primal sub-expression $expr")
end


# ==================== tgen_* =====================================
# Forward-mode codegen: single sweep, original statement/loop order,
# no snapshot stacks -- every active statement gets a shadow
# ("d"-suffixed) line emitted right before its own primal line,
# computed from current (pre-this-statement) values. Always safe: a
# statement's tangent never depends on its own lhs's new value. The
# tangent line is emitted even when it collapses to 0.0, so a later
# active read sees the reset rather than a stale value.

function tgen_emit(kernel, lin_plan)
    fname = tgen_fname(kernel.sig.name)
    fargs = tgen_signature_args(kernel.sig)
    body_exprs = tgen_body(lin_plan)
    reassigned = tgen_reassigned_scalar_args(kernel)
    push!(body_exprs, emit_return_scalars([tgen_shadow(v) for v in reassigned]))
    return Expr(:function, Expr(:call, fname, fargs...), Expr(:block, body_exprs...))
end

tgen_fname(name::Symbol) = Symbol(string(name) * "_d")

# a Symbol becomes `<name>d`; an array-ref `v[i,...]` becomes
# `vd[i,...]` -- same indices, shadowed array name
function tgen_shadow(expr)
    expr isa Symbol && return Symbol(string(expr) * "d")
    if expr isa Expr && expr.head == :ref
        return Expr(:ref, Symbol(string(expr.args[1]) * "d"), expr.args[2:end]...)
    end
    error("tgen_shadow: expected a Symbol or array-ref, got $expr")
end

# every float arg gets its shadow appended right after it;
# Int64 args appear once, exactly as in the primal
function tgen_signature_args(sig)
    fargs = Symbol[]
    for a in sig.args
        push!(fargs, a)
        if sig.kinds[a] in (:scalar_float, :array_float)
            push!(fargs, tgen_shadow(a))
        end
    end
    return fargs
end

function tgen_body(plan)
    exprs = Any[]
    for stmt in plan
        if stmt.kind == :assign
            push!(exprs, Expr(:(=), tgen_shadow(stmt.lhs), tgen_tangent_expr(stmt.tree)))
            push!(exprs, Expr(:(=), stmt.lhs, stmt.tree.expr))
        elseif stmt.kind == :for
            inner = tgen_body(stmt.body)
            push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
        elseif stmt.kind == :if
            then_exprs = tgen_body(stmt.then)
            els_exprs = tgen_body(stmt.els)
            push!(exprs, emit_if(stmt.cond, then_exprs, els_exprs))
        end
    end
    return exprs
end

# bottom-up: sum_i partials[i]*tangent(child_i), via der_tangent_generic
# (which itself re-derives partials from node.op/node.args -- cheap,
# and keeps this function from needing to know der_mul/der_sum_terms).
# An entirely-inactive subtree collapses straight to the literal 0.0
# rather than trusting the generic contraction to fold it down itself.
function tgen_tangent_expr(node)
    node.active || return 0.0
    node.kind == :leaf && return tgen_shadow(node.expr)
    dargs = [tgen_tangent_expr(c) for c in node.children]
    return der_tangent_generic(node.op, node.args, dargs)
end

# scalar-float function args reassigned somewhere in the primal body
# -- the only shadows that can't just be read straight off the
# argument binding, since nothing else in the tangent function would
# otherwise expose their new value to the caller
function tgen_reassigned_scalar_args(kernel)
    arg_set = Set(kernel.sig.args)
    out = Symbol[]
    tgen_collect_reassigned_scalar_args!(kernel.body, arg_set, kernel.sig.kinds, out)
    return out
end

function tgen_collect_reassigned_scalar_args!(body, arg_set, kinds, out)
    for stmt in body
        if stmt.kind == :assign
            if stmt.lhs isa Symbol && stmt.lhs in arg_set && kinds[stmt.lhs] == :scalar_float && !(stmt.lhs in out)
                push!(out, stmt.lhs)
            end
        elseif stmt.kind == :for
            tgen_collect_reassigned_scalar_args!(stmt.body, arg_set, kinds, out)
        elseif stmt.kind == :if
            tgen_collect_reassigned_scalar_args!(stmt.then, arg_set, kinds, out)
            tgen_collect_reassigned_scalar_args!(stmt.els, arg_set, kinds, out)
        end
    end
    return nothing
end


# ==================== agen_* =====================================
# Reverse-mode codegen: forward sweep w/ pushes, reversed backward
# sweep w/ pops, plus the companion initstacks_* generator.
#
# Forward sweep: replays the primal's own statements exactly, in
# original order, unconditionally (never trying to prove a statement
# is dead for gradient purposes -- simpler, and harmless: unused
# recomputation costs time, not correctness), inserting a push!
# right before any write that needs one. Every push targets the
# exact lhs reference being overwritten (a scalar), not a whole-array
# copy -- simpler and uniform.
#
# Backward sweep: statement order reversed within every block; a
# sequential loop's own iteration direction is also reversed, but a
# non-sequential loop is left forward -- it has no cross-iteration
# coupling, so reversing it would be pointless. Each statement's
# incoming adjoint seed is distributed down through its lin_node tree
# via der_rule(op).adjoint, accumulating into `argb = argb + ...` at
# every active leaf. Two cases per statement:
#   - pure accumulation: the lhs's own occurrence in the rhs is the
#     same quantity as its own seed, so it's skipped when
#     distributing, and the shadow is never reset.
#   - anything else: the full seed is distributed to every active
#     leaf normally, and the lhs's shadow is reset to 0.0 afterward.
# A statement whose write has a snapshot site pops the old value back
# into the exact lhs location as the very first thing its backward
# code does, before any contribution is computed.

function agen_emit(kernel, lin_plan, snapshot_plan)
    active_map = act_analyze(kernel)
    adjoint_expr = agen_adjoint_emit(kernel, active_map, lin_plan, snapshot_plan)
    initstacks_expr = agen_init_emit(kernel, snapshot_plan)
    return (adjoint = adjoint_expr, initstacks = initstacks_expr)
end

agen_fname(name::Symbol) = Symbol(string(name) * "_b")
agen_init_fname(name::Symbol) = Symbol("initstacks_" * string(name) * "_b")

function agen_shadow(expr)
    expr isa Symbol && return Symbol(string(expr) * "b")
    if expr isa Expr && expr.head == :ref
        return Expr(:ref, Symbol(string(expr.args[1]) * "b"), expr.args[2:end]...)
    end
    error("agen_shadow: expected a Symbol or array-ref, got $expr")
end

# every float arg gets its adjoint appended right after it; Int64
# args appear once, exactly as in the primal
function agen_signature_args(sig)
    fargs = Symbol[]
    for a in sig.args
        push!(fargs, a)
        if sig.kinds[a] in (:scalar_float, :array_float)
            push!(fargs, agen_shadow(a))
        end
    end
    return fargs
end

function agen_adjoint_emit(kernel, active_map, lin_plan, sites)
    fname = agen_fname(kernel.sig.name)
    stacks = agen_stack_map(sites)
    fargs = vcat(agen_signature_args(kernel.sig), agen_stack_names(sites))

    read_anywhere = agen_collect_reads(kernel.body)
    reassigned = agen_collect_reassigned(kernel.body)
    unsafe = agen_unsafe_int_vars(kernel)

    body = Any[]
    append!(body, agen_local_primal_inits(kernel, active_map))
    append!(body, agen_local_shadow_inits(kernel, active_map))
    append!(body, agen_forward_body(kernel.body, active_map, false, read_anywhere, reassigned, stacks))
    append!(body, agen_backward_body(lin_plan, kernel.sig.kinds, unsafe, false, read_anywhere, reassigned, stacks))

    scalar_args = [a for a in kernel.sig.args if kernel.sig.kinds[a] == :scalar_float]
    push!(body, emit_return_scalars([agen_shadow(a) for a in scalar_args]))

    return Expr(:function, Expr(:call, fname, fargs...), Expr(:block, body...))
end

# int-kinded variables reassigned at more than one distinct :assign
# site anywhere in the kernel are evolving state whose correct value
# at a given point depends on the full history of the primal's
# control flow -- including a sibling block's own internal
# reassignment persisting afterward, since in Julia a `for` loop
# assigning to an already-outer variable modifies that outer
# variable, so it leaks past the loop that set it. Recomputing such a
# variable from a fixed starting point at the top of every block that
# reads it is wrong -- it silently discards whatever an earlier
# sibling block left it as. Anything transitively depending on such a
# variable inherits the same problem and is excluded too. None of
# this needs fixing by tracking history harder, though: the cases
# that actually matter for correctness are for-loop bounds, and those
# are already correctly restored via :tripcount regardless of what
# these intermediate variables hold -- so the right fix is simply to
# never hoist/recompute a variable in this set at all, only ones that
# are genuinely self-contained.
#
# The set is seeded by genuine self-reference, not merely by having
# more than one assignment site -- a variable computed fresh from a
# loop's own iteration variable can legitimately appear at several
# independent sites (different loops), none depending on its own
# previous value, and must stay hoistable.
function agen_unsafe_int_vars(kernel)
    kinds = kernel.sig.kinds
    unsafe = Set{Symbol}()
    agen_seed_unsafe_self_ref!(kernel.body, kinds, unsafe)
    changed = true
    while changed
        changed = agen_propagate_unsafe!(kernel.body, kinds, unsafe)
    end
    return unsafe
end

function agen_seed_unsafe_self_ref!(body, kinds, unsafe)
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            if kinds[var] in (:scalar_int, :array_int) && agen_count_var_refs(stmt.rhs, var) > 0
                push!(unsafe, var)
            end
        elseif stmt.kind == :for
            agen_seed_unsafe_self_ref!(stmt.body, kinds, unsafe)
        elseif stmt.kind == :if
            agen_seed_unsafe_self_ref!(stmt.then, kinds, unsafe)
            agen_seed_unsafe_self_ref!(stmt.els, kinds, unsafe)
        end
    end
    return nothing
end

function agen_propagate_unsafe!(body, kinds, unsafe)
    changed = false
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            if kinds[var] in (:scalar_int, :array_int) && !(var in unsafe)
                refs = Set{Symbol}()
                agen_collect_expr_vars!(stmt.rhs, refs)
                if any(r in unsafe for r in refs)
                    push!(unsafe, var)
                    changed = true
                end
            end
        elseif stmt.kind == :for
            changed = agen_propagate_unsafe!(stmt.body, kinds, unsafe) || changed
        elseif stmt.kind == :if
            changed = agen_propagate_unsafe!(stmt.then, kinds, unsafe) || changed
            changed = agen_propagate_unsafe!(stmt.els, kinds, unsafe) || changed
        end
    end
    return changed
end

# ---- stack bookkeeping (shared by the forward sweep, backward
#      sweep, and initstacks_ generator) --------------------------

# one dedicated stack per distinct :array/:value variable; ALL
# :branch sites share one stack, and ALL :tripcount sites share
# another -- matches how snap_plan tags every :branch site's `array`
# field with the fixed sentinel :cond, so they naturally collapse to
# one name here
function agen_site_stack_name(site)
    site.kind in (:array, :value) && return Symbol(string(site.array) * "_stack")
    site.kind == :branch && return :branch_stack
    return :tripcount_stack
end

# distinct stack names, in order of first appearance in `sites`
function agen_stack_names(sites)
    names = Symbol[]
    for s in sites
        nm = agen_site_stack_name(s)
        nm in names || push!(names, nm)
    end
    return names
end

# (kind, array) -> stack name, for quick lookup during codegen
function agen_stack_map(sites)
    m = Dict{Tuple{Symbol,Symbol},Symbol}()
    for s in sites
        m[(s.kind, s.array)] = agen_site_stack_name(s)
    end
    return m
end

# ---- initstacks_ generator ---------------------------------------

function agen_init_emit(kernel, sites)
    fname = agen_init_fname(kernel.sig.name)
    names = agen_stack_names(sites)
    kind_of = Dict{Symbol,Symbol}()
    for s in sites
        kind_of[agen_site_stack_name(s)] = s.kind
    end
    body = Any[Expr(:(=), nm, agen_stack_alloc_expr(kind_of[nm])) for nm in names]
    push!(body, emit_return_scalars(names))
    return Expr(:function, Expr(:call, fname), Expr(:block, body...))
end

# :array/:value stacks hold the popped Float64 scalar itself (every
# push is of one exact lhs reference, never a whole-array copy);
# :branch/:tripcount stacks hold Int64 flags/bounds
function agen_stack_alloc_expr(kind)
    kind in (:array, :value) && return Expr(:call, Expr(:curly, :Vector, :Float64))
    return Expr(:call, Expr(:curly, :Vector, :Int64))
end

# ---- TBR predicate, duplicated (agen_-prefixed) from snap_*'s own
#      logic rather than calling it directly -- skill-stade's purity
#      rule only allows relying on another stage's *documented*
#      input/output shape (here: the sites list itself), not
#      reaching into its private helpers ------------------------------

function agen_collect_reassigned(body)
    reassigned = Set{Symbol}()
    for stmt in body
        if stmt.kind == :assign
            stmt.lhs isa Symbol && push!(reassigned, stmt.lhs)
        elseif stmt.kind == :for
            union!(reassigned, agen_collect_reassigned(stmt.body))
        elseif stmt.kind == :if
            union!(reassigned, agen_collect_reassigned(stmt.then))
            union!(reassigned, agen_collect_reassigned(stmt.els))
        end
    end
    return reassigned
end

function agen_collect_reads(body)
    reads = Set{Symbol}()
    for stmt in body
        if stmt.kind == :assign
            agen_collect_expr_vars!(stmt.rhs, reads)
        elseif stmt.kind == :for
            union!(reads, agen_collect_reads(stmt.body))
        elseif stmt.kind == :if
            agen_collect_expr_vars!(stmt.cond, reads)
            union!(reads, agen_collect_reads(stmt.then))
            union!(reads, agen_collect_reads(stmt.els))
        end
    end
    return reads
end

function agen_collect_expr_vars!(expr, vars)
    if expr isa Symbol
        push!(vars, expr)
    elseif expr isa Expr
        start = expr.head == :call ? 2 : 1
        for a in expr.args[start:end]
            agen_collect_expr_vars!(a, vars)
        end
    end
    return nothing
end

# see snap_is_pure_accumulation's comment -- identical logic,
# duplicated here for the same purity-rule reason as every other
# agen_-prefixed pair in this file.
function agen_is_pure_accumulation(lhs, rhs, var)
    (rhs isa Expr && rhs.head == :call) || return false
    op = rhs.args[1]
    if op == :+
        matches = 0
        for a in rhs.args[2:end]
            a == lhs && (matches = matches + 1)
        end
        matches == 1 || return false
    elseif op == :- && length(rhs.args) == 3
        rhs.args[2] == lhs || return false
    else
        return false
    end
    return agen_count_expr_occurrences(rhs, lhs) == 1
end

function agen_count_var_refs(expr, var)
    if expr isa Symbol
        return expr == var ? 1 : 0
    elseif expr isa Expr
        start = expr.head == :call ? 2 : 1
        total = 0
        for a in expr.args[start:end]
            total = total + agen_count_var_refs(a, var)
        end
        return total
    end
    return 0
end

# how many times does the exact sub-expression `target` appear
# anywhere in expr -- see snap_count_expr_occurrences's comment
function agen_count_expr_occurrences(expr, target)
    total = expr == target ? 1 : 0
    if expr isa Expr
        start = expr.head == :call ? 2 : 1
        for a in expr.args[start:end]
            total = total + agen_count_expr_occurrences(a, target)
        end
    end
    return total
end

function agen_needs_snapshot(lhs, rhs, var, seq, read_anywhere)
    agen_is_pure_accumulation(lhs, rhs, var) && return false
    agen_count_var_refs(rhs, var) > 0 && return true
    return seq && (var in read_anywhere)
end

# bound-variables of a :for statement that are reassigned somewhere
# else in the kernel -- the same set snap_plan's :tripcount sites are
# keyed on
function agen_tripcount_bound_vars(stmt, reassigned)
    bound_vars = Set{Symbol}()
    agen_collect_expr_vars!(stmt.lo, bound_vars)
    agen_collect_expr_vars!(stmt.hi, bound_vars)
    agen_collect_expr_vars!(stmt.step, bound_vars)
    return [bv for bv in bound_vars if bv in reassigned]
end

function agen_negate_step(step)
    step isa Number && return -step
    return Expr(:call, :-, step)
end

# ---- forward sweep (walks the raw primal `statement_list`) ---------

function agen_forward_body(body, active_map, seq, read_anywhere, reassigned, stacks)
    exprs = Any[]
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            # gate on THIS statement's own rhs activity, not the whole
            # variable's -- a variable written by several statements
            # can be active overall via one of them while another
            # write is a plain inactive literal; the backward sweep
            # only ever pops for a write whose own rhs is active, so
            # pushing here on every write regardless would push more
            # than gets popped
            if agen_expr_active(stmt.rhs, active_map) && agen_needs_snapshot(stmt.lhs, stmt.rhs, var, seq, read_anywhere)
                push!(exprs, Expr(:call, :push!, stacks[(agen_snapshot_kind(stmt.lhs), var)], stmt.lhs))
            end
            push!(exprs, Expr(:(=), stmt.lhs, stmt.rhs))
        elseif stmt.kind == :for
            for bv in agen_tripcount_bound_vars(stmt, reassigned)
                push!(exprs, Expr(:call, :push!, stacks[(:tripcount, bv)], bv))
            end
            inner = agen_forward_body(stmt.body, active_map, seq || stmt.sequential, read_anywhere, reassigned, stacks)
            push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
        elseif stmt.kind == :if
            nm = stacks[(:branch, :cond)]
            then_exprs = vcat(Any[Expr(:call, :push!, nm, 1)], agen_forward_body(stmt.then, active_map, seq, read_anywhere, reassigned, stacks))
            els_exprs = vcat(Any[Expr(:call, :push!, nm, 0)], agen_forward_body(stmt.els, active_map, seq, read_anywhere, reassigned, stacks))
            push!(exprs, emit_if(stmt.cond, then_exprs, els_exprs))
        end
    end
    return exprs
end

# does expr read any variable currently marked active? Duplicated
# (agen_-prefixed) from act_*'s own act_expr_active rather than
# calling it directly -- same purity rule as the snap_* duplicates
# above, applied to act_*'s private helper this time.
function agen_expr_active(expr, active_map)
    if expr isa Symbol
        return get(active_map, expr, false)
    elseif expr isa Expr
        start = expr.head == :call ? 2 : 1
        for a in expr.args[start:end]
            agen_expr_active(a, active_map) && return true
        end
    end
    return false
end

# an active lhs is always scalar_float or array_float -- :ref means
# the array kind (:array), a bare Symbol means the scalar kind
# (:value), matching how snap_plan classified the same write
agen_snapshot_kind(lhs) = lhs isa Symbol ? :value : :array

# ---- backward sweep (walks lin_plan, whose :for/:if fields mirror
#      the primal's own exactly -- only :assign carries a built tree) -

function agen_backward_body(plan, kinds, unsafe, seq, read_anywhere, reassigned, stacks)
    exprs = Any[]
    # int-kinded local assignments (index/bookkeeping helpers) never
    # carry gradients, so agen_backward_assign emits nothing at all
    # for them -- but the array indices they compute can still be
    # needed by OTHER statements in this same block once everything
    # else is reversed. A plain reversal would put the statement that
    # NEEDS such an index before the statement that computes it;
    # Julia's `for`/`if` bodies each have their own scope, so there's
    # no other point at which this could get recomputed. Recompute
    # them all up front instead, in their original (forward) relative
    # order -- but ONLY the ones not in `unsafe` (see
    # agen_unsafe_int_vars): a var reassigned at more than one site
    # elsewhere in the kernel can't be safely reconstructed this way,
    # and doesn't need to be -- whatever actually matters about it
    # downstream is a loop bound, already correctly restored via
    # :tripcount regardless.
    for stmt in plan
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            if kinds[var] in (:scalar_int, :array_int) && !(var in unsafe)
                push!(exprs, Expr(:(=), stmt.lhs, stmt.tree.expr))
            end
        end
    end
    for stmt in reverse(plan)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            kinds[var] in (:scalar_int, :array_int) && continue   # hoisted above, or unsafe (skipped entirely)
            append!(exprs, agen_backward_assign(stmt, kinds, seq, read_anywhere, reassigned, stacks))
        elseif stmt.kind == :for
            inner_seq = seq || stmt.sequential
            inner = agen_backward_body(stmt.body, kinds, unsafe, inner_seq, read_anywhere, reassigned, stacks)
            reverse_it = stmt.sequential || agen_body_has_snapshot(stmt.body, inner_seq, read_anywhere, reassigned)
            loop_expr = reverse_it ?
                emit_forloop(stmt.var, stmt.hi, stmt.lo, agen_negate_step(stmt.step), inner) :
                emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner)
            for bv in agen_tripcount_bound_vars(stmt, reassigned)
                push!(exprs, Expr(:(=), bv, Expr(:call, :pop!, stacks[(:tripcount, bv)])))
            end
            push!(exprs, loop_expr)
        elseif stmt.kind == :if
            nm = stacks[(:branch, :cond)]
            then_exprs = agen_backward_body(stmt.then, kinds, unsafe, seq, read_anywhere, reassigned, stacks)
            els_exprs = agen_backward_body(stmt.els, kinds, unsafe, seq, read_anywhere, reassigned, stacks)
            push!(exprs, Expr(:(=), :__branch, Expr(:call, :pop!, nm)))
            push!(exprs, emit_if(Expr(:call, :(==), :__branch, 1), then_exprs, els_exprs))
        end
    end
    return exprs
end

# a loop must be reversed in the backward sweep whenever ANY push
# happens inside it, at any nesting depth -- not just when THIS loop
# is itself sequential=true. LIFO stack discipline requires every
# loop enclosing a push to run in exact reverse, full stop: a loop
# with no value recurrence at all can still push a branch flag every
# iteration and must be walked backward to pop them correctly --
# reversal here is about stack order, not mathematical dependency.
function agen_body_has_snapshot(body, seq, read_anywhere, reassigned)
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            if stmt.active && agen_needs_snapshot(stmt.lhs, stmt.tree.expr, var, seq, read_anywhere)
                return true
            end
        elseif stmt.kind == :for
            !isempty(agen_tripcount_bound_vars(stmt, reassigned)) && return true
            agen_body_has_snapshot(stmt.body, seq || stmt.sequential, read_anywhere, reassigned) && return true
        elseif stmt.kind == :if
            return true   # every `if` pushes a branch flag, unconditionally
        end
    end
    return false
end

function agen_backward_assign(stmt, kinds, seq, read_anywhere, reassigned, stacks)
    var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
    is_accum = agen_is_pure_accumulation(stmt.lhs, stmt.tree.expr, var)
    exprs = Any[]
    if stmt.active
        if agen_needs_snapshot(stmt.lhs, stmt.tree.expr, var, seq, read_anywhere)
            nm = stacks[(agen_snapshot_kind(stmt.lhs), var)]
            push!(exprs, Expr(:(=), stmt.lhs, Expr(:call, :pop!, nm)))
        end
        lhsb = agen_shadow(stmt.lhs)
        if is_accum
            agen_distribute!(stmt.tree, lhsb, exprs; skip_expr = stmt.lhs)
        else
            # a leaf whose own slot exactly matches lhs is a GENUINE,
            # non-identity self-reference -- is_accum only catches the
            # identity case (+/-, coefficient exactly 1), so this is
            # different and needs different treatment. Such a leaf's
            # contribution can't be accumulated into lhsb the normal
            # way: lhsb is simultaneously the seed being read FROM and
            # a target being written TO, so `lhsb = lhsb + contribution`
            # reads its own not-yet-updated self mid-computation --
            # harmless in isolation, except the very next line
            # (unconditional reset) then throws that whole sum away,
            # silently dropping the contribution entirely. Collected
            # separately and applied as a REPLACEMENT of lhsb instead:
            # that replacement already reflects "lhsb now represents
            # the OLD slot's adjoint", making a further reset both
            # wrong (it would erase the value just computed) and
            # unnecessary.
            self_terms = Any[]
            agen_distribute!(stmt.tree, lhsb, exprs; self_expr = stmt.lhs, self_terms = self_terms)
            if isempty(self_terms)
                push!(exprs, Expr(:(=), lhsb, 0.0))
            else
                push!(exprs, Expr(:(=), lhsb, der_sum_terms(self_terms)))
            end
        end
    elseif !is_accum && kinds[var] in (:scalar_float, :array_float)
        # this specific write's rhs carries no active leaf at all --
        # there's nothing to distribute, but the shadow this write
        # "produced" still needs resetting here. Skipping the reset
        # because THIS write happens to be a constant would let
        # whatever an earlier (already-processed-in-reverse) statement
        # accumulated into the shadow leak into the next
        # (chronologically earlier) iteration's contribution instead
        # of starting fresh.
        push!(exprs, Expr(:(=), agen_shadow(stmt.lhs), 0.0))
    end
    return exprs
end

# recursively distribute `seed` down through a lin_node tree,
# accumulating `target = target + contribution` at every active leaf
# whose own slot differs from both `skip_expr` and `self_expr`.
# `skip_expr`, when set, is only honored against a *direct child of
# the node it's passed to* -- exactly the top-level operand a pure
# accumulation's own lhs occupies -- and is never forwarded to deeper
# recursion, since accumulation-skipping is a property of that one
# top-level slot, not of the variable in general. `self_expr`/
# `self_terms`, by contrast, DO propagate to any depth (a genuine
# self-reference can appear anywhere in the tree, not just at the
# top level) -- matching leaves push their contribution onto
# `self_terms` instead of emitting an accumulate statement, so the
# caller can combine them into a single replacement afterward.
function agen_distribute!(node, seed, exprs; skip_expr = nothing, self_expr = nothing, self_terms = nothing)
    node.active || return nothing
    if node.kind == :leaf
        skip_expr !== nothing && node.expr == skip_expr && return nothing
        if self_expr !== nothing && node.expr == self_expr
            push!(self_terms, seed)
            return nothing
        end
        target = agen_shadow(node.expr)
        push!(exprs, Expr(:(=), target, Expr(:call, :+, target, seed)))
        return nothing
    end
    contributions = der_adjoint_generic(node.op, node.args, seed)
    for (child, contrib) in zip(node.children, contributions)
        skip_expr !== nothing && child.expr == skip_expr && continue
        child.active || continue
        agen_distribute!(child, contrib, exprs; self_expr = self_expr, self_terms = self_terms)
    end
    return nothing
end

# ---- local shadow initialization -----------------------------------

# every local (non-argument) scalar variable needs to already exist
# before the forward sweep runs -- normally its first primal
# assignment would establish that, but a local flagged for
# snapshotting can get PUSHED (reading its pre-overwrite value) as
# early as the very first loop iteration, before any primal
# assignment to it has ever happened. Without this, that first push
# would be an UndefVarError. Harmless for locals that don't need it
# -- their real first assignment overwrites the 0.0 immediately.
function agen_local_primal_inits(kernel, active_map)
    arg_set = Set(kernel.sig.args)
    exprs = Any[]
    for v in sort(collect(keys(kernel.sig.kinds)))
        if !(v in arg_set) && kernel.sig.kinds[v] == :scalar_float && active_map[v]
            push!(exprs, Expr(:(=), v, 0.0))
        end
    end
    return exprs
end

# every ACTIVE local (non-argument) scalar variable needs its shadow
# declared and zeroed before the backward sweep can accumulate into
# it -- arrays can never be local under skill-jade (rule 8: no
# in-kernel allocation, so any array must be a caller-supplied arg),
# so this only ever has scalars to handle
function agen_local_shadow_inits(kernel, active_map)
    arg_set = Set(kernel.sig.args)
    exprs = Any[]
    for v in sort(collect(keys(kernel.sig.kinds)))
        if !(v in arg_set) && kernel.sig.kinds[v] == :scalar_float && active_map[v]
            push!(exprs, Expr(:(=), agen_shadow(v), 0.0))
        end
    end
    return exprs
end


# ==================== cgen_* =====================================
# CUDA codegen: turns a validated kernel -- or one of STADE's own
# tgen_/agen_-generated functions -- into a host launcher Expr plus a
# Vector of `@cuda`-callable device kernel Exprs, one per
# iteration-independent (sequential=false) loop that's actually safe
# to run data-parallel. Structurally independent of act_/snap_/lin_:
# this is a loop-nest transform, not a derivative, so it never touches
# activity/snapshot/linearization machinery.
#
# cuda_plan :: (host::Expr, kernels::Vector{Expr})   -- frozen shape,
# see skill-stade.md.
#
# Two ingestion paths feed the same cgen_kernel shape:
#   - a plain skill-jade kernel, via parse_kernel (unchanged) wrapped
#     by cgen_from_kernel
#   - one of STADE's own generated functions (tangent, adjoint, or
#     initstacks_), via cgen_parse_generated below. Needed because a
#     generated function's trailing `return` carries values (never
#     just `return nothing`), and an adjoint's forward/backward sweep
#     contains push!/pop! stack statements that skill-jade source can
#     never contain. cgen_parse_generated recognizes only the fixed,
#     small vocabulary tgen_body/agen_forward_body/agen_backward_body/
#     agen_init_emit themselves emit -- it is NOT a general Julia
#     parser and must never be pointed at arbitrary user source.
#
# cgen_kernel :: (name::Symbol, args::Vector{Symbol}, body::statement_list,
#                 ret::Vector{Symbol})   -- cgen-local, not the frozen
#     `kernel` shape (a generated function has no kinds/independents/
#     dependents to carry). empty ret => `return nothing`.
#
# cgen also recognizes two statement forms of its own, produced only
# by cgen_parse_generated and consumed only within this section --
# deliberately NOT added to the frozen `statement` shape in
# skill-stade.md, since no other stage ever needs to see them:
#   (kind=:stackpush, stack::Symbol, value)        -- a bare push! call
#   (kind=:assign, lhs, rhs)  where rhs is a bare `pop!(stack)` call
#     (kept under the ordinary :assign kind since structurally it is
#     one; cgen_is_pop_call recognizes it)
#
# Stack safety rule: a loop containing a push!/pop! anywhere in its
# body (at any depth) is NEVER split into a device kernel, regardless
# of its own sequential flag. A snapshot stack's LIFO order is a
# global, history-order-dependent invariant across every push/pop
# touching it; preserving that order across thousands of concurrently-
# scheduled GPU threads isn't possible without replacing the stack
# mechanism itself with pre-sized, positionally-indexed buffers (an
# agen_ change, not a cgen_ one -- see skill-stade.md's cgen_ note for
# the follow-up this opens). Such a loop is always emitted as ordinary
# host-side Julia via emit_forloop, exactly as it was parsed. This
# still parallelizes any part of a generated adjoint that doesn't
# touch a snapshot stack.
#
# Race-safety rule for a write inside a kernel that *is* split: a
# write to array[idx...] is atomic-free only if the enclosing device
# loop's own thread-mapped variable occurs, at any depth, in idx --
# i.e. a real occurs-check (cgen_expr_contains), not shallow top-level
# membership. Anything else assumed unsafe and wrapped in
# CUDA.@atomic, after flattening the rhs's +/- spine
# (cgen_flatten_sum) to find and drop the write's own prior value as a
# term, however many terms the sum has and wherever it appears in the
# sum (not just literally the second operand of a 2-ary +).
#
# skill-jade rule 8 (no in-kernel array allocation -- every array is a
# caller-supplied argument) means no kernel or generated function this
# section ever processes can contain a `zeros(...)`-style local
# allocation, so there is deliberately no CUDA.zeros conversion step
# here: the caller is responsible for passing CuArrays to any `_cuda`
# function.

function cgen_from_kernel(kernel)
    return (name = kernel.sig.name, args = kernel.sig.args, body = kernel.body, ret = Symbol[])
end

# ---- relaxed ingestion for STADE's own generated code -------------

function cgen_parse_generated(expr::Expr)
    expr.head == :function ||
        error("cgen_parse_generated: expected a `function ... end` definition")
    length(expr.args) == 2 ||
        error("cgen_parse_generated: malformed function Expr (expected signature + body)")
    name, args = cgen_parse_generated_signature(expr.args[1])
    stmts = [s for s in expr.args[2].args if !(s isa LineNumberNode)]
    isempty(stmts) && error("cgen_parse_generated: empty function body")
    last_stmt = stmts[end]
    last_stmt isa Expr && last_stmt.head == :return ||
        error("cgen_parse_generated: expected a trailing `return` statement")
    for s in stmts[1:end-1]
        s isa Expr && s.head == :return &&
            error("cgen_parse_generated: `return` may only appear once, at the end")
    end
    ret = cgen_parse_return_values(last_stmt)
    body = cgen_parse_generated_statements(stmts[1:end-1])
    return (name = name, args = args, body = body, ret = ret)
end

function cgen_parse_generated_signature(sig_expr)
    sig_expr isa Expr && sig_expr.head == :call ||
        error("cgen_parse_generated: unsupported signature form")
    name = sig_expr.args[1]
    name isa Symbol || error("cgen_parse_generated: function name must be a plain identifier")
    args = Symbol[a for a in sig_expr.args[2:end]]
    all(a -> a isa Symbol, args) ||
        error("cgen_parse_generated: only plain positional argument names are supported")
    return name, args
end

function cgen_parse_return_values(ret_stmt::Expr)
    length(ret_stmt.args) == 1 || error("cgen_parse_generated: malformed return `$(ret_stmt)`")
    v = ret_stmt.args[1]
    parse_is_nothing_literal(v) && return Symbol[]
    v isa Symbol && return Symbol[v]
    v isa Expr && v.head == :tuple && all(a -> a isa Symbol, v.args) && return Symbol[v.args...]
    error("cgen_parse_generated: unsupported return form `$(ret_stmt)` -- expected `return nothing`, a bare variable, or a tuple of variables")
end

function cgen_parse_generated_statements(stmts::Vector)
    return NamedTuple[cgen_parse_generated_statement(s) for s in stmts]
end

function cgen_parse_generated_statement(stmt)
    stmt isa Expr || error("cgen_parse_generated: unsupported statement `$(stmt)`")
    if stmt.head == :(=)
        return cgen_parse_generated_assign(stmt)
    elseif stmt.head == :for
        return cgen_parse_generated_for(stmt)
    elseif stmt.head == :if
        return cgen_parse_generated_if(stmt)
    elseif stmt.head == :call && stmt.args[1] == :push!
        length(stmt.args) == 3 || error("cgen_parse_generated: malformed push! `$(stmt)`")
        return (kind = :stackpush, stack = stmt.args[2], value = stmt.args[3])
    else
        error("cgen_parse_generated: unrecognized statement form `Expr(:$(stmt.head), ...)` -- this parser only understands STADE's own tgen_/agen_ output, never arbitrary user source")
    end
end

# lhs shape (plain var or array-ref) never differs from a plain
# kernel's, so parse_lvalue/parse_check_expr are reused verbatim --
# only the two relaxations below (pop!, stack allocation) are new
function cgen_parse_generated_assign(stmt::Expr)
    lhs = parse_lvalue(stmt.args[1])
    rhs = stmt.args[2]
    if cgen_is_pop_call(rhs) || cgen_is_stack_alloc(rhs)
        return (kind = :assign, lhs = lhs, rhs = rhs)
    end
    parse_check_expr(rhs, false)
    return (kind = :assign, lhs = lhs, rhs = rhs)
end

cgen_is_pop_call(rhs) = rhs isa Expr && rhs.head == :call && length(rhs.args) == 2 && rhs.args[1] == :pop!

# matches agen_stack_alloc_expr's own output: Vector{Float64}() / Vector{Int64}()
cgen_is_stack_alloc(rhs) = rhs isa Expr && rhs.head == :call && length(rhs.args) == 1 &&
    rhs.args[1] isa Expr && rhs.args[1].head == :curly && rhs.args[1].args[1] == :Vector

function cgen_parse_generated_for(stmt::Expr)
    header = stmt.args[1]
    header isa Expr && header.head == :(=) ||
        error("cgen_parse_generated: unsupported `for` header `$(header)`")
    var = header.args[1]
    var isa Symbol || error("cgen_parse_generated: `for` loop variable must be a plain identifier")
    range_expr = header.args[2]
    range_expr isa Expr && range_expr.head == :call && range_expr.args[1] == :(:) ||
        error("cgen_parse_generated: `for` loop range must be written `lo:hi` or `lo:step:hi`")
    range_args = range_expr.args[2:end]
    lo, step, hi = length(range_args) == 2 ? (range_args[1], 1, range_args[2]) :
                   length(range_args) == 3 ? (range_args[1], range_args[2], range_args[3]) :
                   error("cgen_parse_generated: unsupported range arity in `$(range_expr)`")
    sequential = startswith(String(var), "i_seq_")
    body_stmts = [s for s in stmt.args[2].args if !(s isa LineNumberNode)]
    body = cgen_parse_generated_statements(body_stmts)
    return (kind = :for, var = var, lo = lo, hi = hi, step = step, sequential = sequential, body = body)
end

function cgen_parse_generated_if(stmt::Expr)
    length(stmt.args) in (2, 3) || error("cgen_parse_generated: unsupported `if` form")
    cond = stmt.args[1]
    parse_check_expr(cond, true)
    then_block = stmt.args[2]
    then_stmts = cgen_parse_generated_statements([s for s in then_block.args if !(s isa LineNumberNode)])
    els_stmts = NamedTuple[]
    if length(stmt.args) == 3
        els_block = stmt.args[3]
        els_stmts = cgen_parse_generated_statements([s for s in els_block.args if !(s isa LineNumberNode)])
    end
    return (kind = :if, cond = cond, then = then_stmts, els = els_stmts)
end

# ---- single entry point: try the strict parser, fall back to the
#      relaxed one, and surface both failure reasons if neither fits
#      -- never a silent guess about which kind of input this is ----

function cgen_ingest(expr::Expr)
    kernel_err = nothing
    try
        return cgen_from_kernel(parse_kernel(expr))
    catch e
        kernel_err = e
    end
    try
        return cgen_parse_generated(expr)
    catch generated_err
        error("cgen_ingest: `$(expr.args[1])` is neither a valid skill-jade kernel ($(kernel_err)) nor recognizable STADE-generated code ($(generated_err))")
    end
end

# ---- stack-op detection (blocks kernel-splitting for a loop) -------

cgen_is_stackop_assign(stmt) = stmt.kind == :assign && cgen_is_pop_call(stmt.rhs)

function cgen_contains_stackop(body::Vector{NamedTuple})
    for stmt in body
        if stmt.kind == :stackpush || cgen_is_stackop_assign(stmt)
            return true
        elseif stmt.kind == :for
            cgen_contains_stackop(stmt.body) && return true
        elseif stmt.kind == :if
            (cgen_contains_stackop(stmt.then) || cgen_contains_stackop(stmt.els)) && return true
        end
    end
    return false
end

# ---- free-variable collection (duplicated from shape_/snap_/agen_'s
#      own copies rather than calling them -- see skill-stade.md rule
#      7 and agen_collect_expr_vars!'s own comment for precedent) ----

# collects from the loop's bounds as well as its body -- a device
# kernel's bounds check needs whatever variables lo/hi/step reference
# (e.g. an array-length argument), not just what the body touches
function cgen_free_vars(stmt, exclude::Symbol)
    vars = Set{Symbol}()
    cgen_collect_expr_vars!(stmt.lo, vars)
    cgen_collect_expr_vars!(stmt.hi, vars)
    cgen_collect_expr_vars!(stmt.step, vars)
    cgen_collect_vars!(stmt.body, vars)
    delete!(vars, exclude)
    return sort(collect(vars); by = string)
end

function cgen_collect_vars!(body::Vector{NamedTuple}, vars::Set{Symbol})
    for stmt in body
        if stmt.kind == :stackpush
            push!(vars, stmt.stack)
            cgen_collect_expr_vars!(stmt.value, vars)
        elseif stmt.kind == :assign
            cgen_collect_expr_vars!(stmt.lhs, vars)
            cgen_collect_expr_vars!(stmt.rhs, vars)
        elseif stmt.kind == :for
            cgen_collect_expr_vars!(stmt.lo, vars)
            cgen_collect_expr_vars!(stmt.hi, vars)
            cgen_collect_expr_vars!(stmt.step, vars)
            cgen_collect_vars!(stmt.body, vars)
            push!(vars, stmt.var)
        elseif stmt.kind == :if
            cgen_collect_expr_vars!(stmt.cond, vars)
            cgen_collect_vars!(stmt.then, vars)
            cgen_collect_vars!(stmt.els, vars)
        end
    end
    return nothing
end

function cgen_collect_expr_vars!(expr, vars::Set{Symbol})
    expr isa Symbol && (push!(vars, expr); return nothing)
    expr isa Expr || return nothing
    if expr.head == :ref
        push!(vars, expr.args[1])
        for idx in expr.args[2:end]
            cgen_collect_expr_vars!(idx, vars)
        end
    elseif expr.head == :call
        # args[1] is the operator/function name, not a variable
        for a in expr.args[2:end]
            cgen_collect_expr_vars!(a, vars)
        end
    else
        for a in expr.args
            cgen_collect_expr_vars!(a, vars)
        end
    end
    return nothing
end

# ---- host-side body walk: splits off one device kernel per eligible
#      iteration-independent loop, leaves everything else untouched --

# ---- GPU backend descriptor ----------------------------------------
# gpu_backend :: (suffix, kernel_tag, launch_macro::Symbol,
#                 threads_kw::Symbol, blocks_kw::Symbol, tid_rhs::Expr,
#                 atomic_macro::Expr, preamble::String)
#
# Everything above this point (parsing, free-var collection, stack-op
# detection, loop-splitting decision, +/- flattening for atomic
# detection) is genuinely backend-agnostic -- it never mentions CUDA.
# Only five things differ between CUDA.jl and AMDGPU.jl, and they're
# all syntactic, not semantic: the launch macro name, the two launch
# keyword names (`threads`/`blocks` vs `groupsize`/`gridsize` -- both
# resolve to the exact same `cld(n_iter, blocksize)` formula; AMDGPU's
# `gridsize` is a work-group count, not a total thread count, so nothing
# about the arithmetic below differs), the thread-index intrinsic
# names, the atomic macro's owning module, and the `using` preamble.
# Adding a third backend (oneAPI.jl, Metal.jl, ...) should only ever
# mean adding one more of these constructors -- if it turns out to
# need a change anywhere else in this section, that's a sign the new
# backend isn't actually the same programming model and deserves its
# own prefix instead of being forced in here.

function cgen_backend_cuda()
    return (
        suffix = "_cuda",
        kernel_tag = "cuda",
        launch_macro = Symbol("@cuda"),
        threads_kw = :threads,
        blocks_kw = :blocks,
        tid_rhs = :((blockIdx().x - 1) * blockDim().x + threadIdx().x),
        atomic_macro = Expr(:., :CUDA, QuoteNode(Symbol("@atomic"))),
        preamble = "import Pkg\nhaskey(Pkg.project().dependencies, \"CUDA\") || Pkg.add(\"CUDA\")\nusing CUDA\nCUDA.allowscalar(false)\n",
    )
end

function cgen_backend_amdgpu()
    return (
        suffix = "_amdgpu",
        kernel_tag = "amdgpu",
        launch_macro = Symbol("@roc"),
        threads_kw = :groupsize,
        blocks_kw = :gridsize,
        tid_rhs = :(workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x),
        atomic_macro = Expr(:., :AMDGPU, QuoteNode(Symbol("@atomic"))),
        preamble = "import Pkg\nhaskey(Pkg.project().dependencies, \"AMDGPU\") || Pkg.add(\"AMDGPU\")\nusing AMDGPU\nAMDGPU.allowscalar(false)\n",
    )
end

function cgen_body(body::Vector{NamedTuple}, kernels::Vector{Expr}, owner::Symbol, backend)
    exprs = Any[]
    for stmt in body
        if stmt.kind == :stackpush
            push!(exprs, Expr(:call, :push!, stmt.stack, stmt.value))
        elseif stmt.kind == :assign
            push!(exprs, Expr(:(=), stmt.lhs, stmt.rhs))
        elseif stmt.kind == :if
            push!(exprs, emit_if(stmt.cond, cgen_body(stmt.then, kernels, owner, backend), cgen_body(stmt.els, kernels, owner, backend)))
        elseif stmt.kind == :for
            if !stmt.sequential && !cgen_contains_stackop(stmt.body)
                idx = length(kernels) + 1
                fargs = cgen_free_vars(stmt, stmt.var)
                push!(kernels, cgen_kernel_def(stmt, owner, idx, fargs, backend))
                push!(exprs, cgen_launch_expr(stmt, owner, idx, fargs, backend))
            else
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, cgen_body(stmt.body, kernels, owner, backend)))
            end
        end
    end
    return exprs
end

# ---- device kernel + launch construction --------------------------

# prefixed with both the backend and the owning function's name --
# stade_gpu_file bundles kernels from every function in an input file
# (potentially processed for more than one backend over time) into
# one output file, so a bare running index alone would collide
cgen_kernel_fname(owner::Symbol, idx::Int, backend) =
    Symbol(backend.kernel_tag * "_kernel_" * string(owner) * "_" * string(idx) * "!")

cgen_trip_count(lo, step, hi) = Expr(:call, :+, Expr(:call, :div, Expr(:call, :-, hi, lo), step), 1)

cgen_loopvar_from_tid(lo, step, tid) =
    step == 1 ? Expr(:call, :+, lo, Expr(:call, :-, tid, 1)) :
                Expr(:call, :+, lo, Expr(:call, :*, Expr(:call, :-, tid, 1), step))

function cgen_kernel_def(stmt, owner::Symbol, idx::Int, fargs::Vector{Symbol}, backend)
    tid = :__tid
    n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)
    body = Any[
        Expr(:(=), tid, backend.tid_rhs),
        Expr(:if, Expr(:call, :>, tid, n_iter), Expr(:block, emit_return_nothing())),
        Expr(:(=), stmt.var, cgen_loopvar_from_tid(stmt.lo, stmt.step, tid)),
    ]
    append!(body, cgen_device_body(stmt.body, stmt.var, backend))
    push!(body, emit_return_nothing())
    return Expr(:function, Expr(:call, cgen_kernel_fname(owner, idx, backend), fargs...), Expr(:block, body...))
end

function cgen_launch_expr(stmt, owner::Symbol, idx::Int, fargs::Vector{Symbol}, backend)
    n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)
    nblocks = Expr(:call, :cld, n_iter, :nthread_per_block)
    call = Expr(:call, cgen_kernel_fname(owner, idx, backend), fargs...)
    return Expr(:macrocall, backend.launch_macro, nothing,
                Expr(:(=), backend.threads_kw, :nthread_per_block),
                Expr(:(=), backend.blocks_kw, nblocks),
                call)
end

# device-side body walk -- never sees :stackpush or a pop!-rhs assign,
# since cgen_body only reaches here for a loop cgen_contains_stackop
# already confirmed is clean at every depth
function cgen_device_body(body::Vector{NamedTuple}, thread_var::Symbol, backend)
    exprs = Any[]
    for stmt in body
        if stmt.kind == :assign
            push!(exprs, cgen_device_assign(stmt, thread_var, backend))
        elseif stmt.kind == :if
            push!(exprs, emit_if(stmt.cond, cgen_device_body(stmt.then, thread_var, backend), cgen_device_body(stmt.els, thread_var, backend)))
        elseif stmt.kind == :for
            push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, cgen_device_body(stmt.body, thread_var, backend)))
        end
    end
    return exprs
end

# a write races across threads unless the enclosing device loop's own
# thread-mapped variable occurs (at any depth) in the write's index --
# a real occurs-check, not shallow top-level membership
function cgen_device_assign(stmt, thread_var::Symbol, backend)
    if stmt.lhs isa Expr && stmt.lhs.head == :ref && !cgen_expr_contains(stmt.lhs.args[2:end], thread_var)
        terms = cgen_flatten_sum(stmt.rhs)
        self_idx = findfirst(t -> t == stmt.lhs, terms)
        if self_idx !== nothing
            other = cgen_sum_excluding(terms, self_idx)
            return Expr(:macrocall, backend.atomic_macro, nothing,
                        Expr(:(+=), stmt.lhs, other))
        end
    end
    return Expr(:(=), stmt.lhs, stmt.rhs)
end

function cgen_expr_contains(x, sym::Symbol)
    x isa Symbol && return x == sym
    x isa Expr && return any(a -> cgen_expr_contains(a, sym), x.args)
    return false
end
cgen_expr_contains(xs::Vector, sym::Symbol) = any(x -> cgen_expr_contains(x, sym), xs)

# flattens a +/- spine into signed terms -- e.g. `x + a - b` -> [x, a, -b] --
# so the self-reference check below works regardless of how many terms
# there are or where the write's own prior value falls among them
function cgen_flatten_sum(expr)
    if expr isa Expr && expr.head == :call && expr.args[1] == :+ && length(expr.args) >= 2
        terms = Any[]
        for a in expr.args[2:end]
            append!(terms, cgen_flatten_sum(a))
        end
        return terms
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :- && length(expr.args) == 3
        return vcat(cgen_flatten_sum(expr.args[2]), cgen_negate_terms(cgen_flatten_sum(expr.args[3])))
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :- && length(expr.args) == 2
        return cgen_negate_terms(cgen_flatten_sum(expr.args[2]))
    else
        return Any[expr]
    end
end

cgen_negate_terms(terms) = Any[Expr(:call, :-, t) for t in terms]

function cgen_sum_excluding(terms, skip_idx::Int)
    rest = [terms[i] for i in eachindex(terms) if i != skip_idx]
    isempty(rest) && return 0.0
    length(rest) == 1 && return rest[1]
    return Expr(:call, :+, rest...)
end

# ---- top-level emit: unifies both ingestion paths, any backend -----

cgen_host_fname(name::Symbol, backend) = Symbol(string(name) * backend.suffix)

function cgen_emit(gk, backend)
    kernels = Expr[]
    host_body = cgen_body(gk.body, kernels, gk.name, backend)
    isempty(kernels) || pushfirst!(host_body, :(nthread_per_block = 256))
    push!(host_body, emit_return_scalars(gk.ret))
    host = Expr(:function, Expr(:call, cgen_host_fname(gk.name, backend), gk.args...), Expr(:block, host_body...))
    return (host = host, kernels = kernels)
end

# opt-in, applied only if the caller wants single precision on the
# generated device/host code -- never applied to a primal copy, since
# a blanket downcast of the reference kernel would silently change
# what it's meant to validate against
function cgen_to_f32(expr)
    expr isa AbstractFloat && return Float32(expr)
    expr isa Expr && return Expr(expr.head, [cgen_to_f32(a) for a in expr.args]...)
    return expr
end


# ==================== val_* =======================================
# Correctness oracle: <y, J*x> == <J'*y, x>, checked against random
# seed vectors.

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

# for a file that may hold several function defs -- e.g. the output
# of stade_tangent_file (tangent + primal) or stade_adjoint_file
# (initstacks_ + adjoint + primal) -- returned in file order
function io_read_kernels(path::String)
    src = read(path, String)
    parsed = Meta.parseall(src)
    defs = [e for e in parsed.args if e isa Expr && e.head == :function]
    isempty(defs) && error("expected at least one function definition in $path, found none")
    return defs
end

function io_expr_to_source(expr::Expr)
    return string(expr) * "\n"
end

# bundles initstacks_foo_b, foo_b, and a copy of foo itself, in that order
function io_write_kernel_file(path::String, primal_expr::Expr, generated::Vector{Expr})
    parts = [io_expr_to_source(e) for e in vcat(generated, [primal_expr])]
    open(path, "w") do f
        write(f, join(parts, "\n"))
    end
    return nothing
end

# flat writer for stade_gpu_file -- unlike io_write_kernel_file there
# is no single designated "primal" to append last: a multi-function
# input may hand back several independent host functions (one per
# input def), so the caller decides the full ordered list itself
function io_write_gpu_file(path::String, exprs::Vector{Expr}; preamble::String = "")
    parts = [io_expr_to_source(e) for e in exprs]
    open(path, "w") do f
        isempty(preamble) || write(f, preamble * "\n")
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

# Expr in, cuda_plan out -- accepts either a plain skill-jade kernel
# or one of STADE's own generated functions (see cgen_ingest), for
# whichever GPU backend descriptor is passed in
function stade_gpu(expr::Expr, backend)
    return cgen_emit(cgen_ingest(expr), backend)
end

stade_cuda(expr::Expr) = stade_gpu(expr, cgen_backend_cuda())
stade_amdgpu(expr::Expr) = stade_gpu(expr, cgen_backend_amdgpu())

# path in, path out. Reads every function def in in_path (a plain
# kernel file, or a stade_tangent_file/stade_adjoint_file output) and
# writes one file: every device kernel first, then every host
# function in original file order.
function stade_gpu_file(in_path::String, out_path::String, backend)
    defs = io_read_kernels(in_path)
    kernels = Expr[]
    hosts = Expr[]
    for expr in defs
        plan = stade_gpu(expr, backend)
        append!(kernels, plan.kernels)
        push!(hosts, plan.host)
    end
    io_write_gpu_file(out_path, vcat(kernels, hosts); preamble = backend.preamble)
    return out_path
end

stade_cuda_file(in_path::String, out_path::String) = stade_gpu_file(in_path, out_path, cgen_backend_cuda())
stade_amdgpu_file(in_path::String, out_path::String) = stade_gpu_file(in_path, out_path, cgen_backend_amdgpu())


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

    cuda_primal = stade_cuda(trivial)
    cuda_tangent = stade_cuda(tangent_out)
    cuda_adjoint = stade_cuda(adjoint_out.adjoint)
    cuda_initstacks = stade_cuda(adjoint_out.initstacks)
    @assert cuda_primal.host isa Expr && cuda_tangent.host isa Expr
    @assert cuda_adjoint.host isa Expr && cuda_initstacks.host isa Expr
    println("cgen_* round-tripped the stub kernel's primal/tangent/adjoint/initstacks forms OK (CUDA backend)")

    rocm_primal = stade_amdgpu(trivial)
    @assert rocm_primal.host isa Expr
    @assert String(rocm_primal.host.args[1].args[1]) == "stub_amdgpu"
    println("cgen_* round-tripped the stub kernel's primal form OK (AMDGPU backend)")
end