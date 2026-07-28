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
#     (kind=:for, var::Symbol, lo, hi, step, sequential::Bool, body)  # body :: statement_list
#     (kind=:if, cond, then::statement_list, els::statement_list)
#     -- :while intentionally unsupported for now, see skill-stade.md
#     -- BREAKING CHANGE (verified against the corpus's mg_vcycle,
#        which needs a `hi:-1:lo` up-sweep): :for gained a `step`
#        field, added between `hi` and `sequential`. lo/hi/step are
#        all `Expr|Symbol|Number`; a plain `lo:hi` header (no
#        explicit step) parses with `step` set to the Int64 literal
#        `1`. Every stage that pattern-matches or destructures a
#        :for statement needs to account for this key before relying
#        on it.
# statement_list :: Vector{NamedTuple}
# active_map    :: Dict{Symbol,Bool}                # var/array name -> is-active
# snapshot_site :: (kind=:value|:array|:branch|:tripcount, array::Symbol, at::Int)
# snapshot_plan :: Vector{snapshot_site}
# lin_node      :: (op::Symbol, args::Vector, darg_exprs::Vector)
# lin_plan      :: Vector{lin_node}
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
# `ncg = div(nl, 2)` makes ncg Int64 even though ncg itself is never
# directly a range bound / div argument / subscript (forward: rhs's
# status decides lhs). A bare copy `lhs = rhs` (no arithmetic at all)
# is stronger than that: since assignment can't change a variable's
# kind, lhs and rhs must share one kind, so evidence has to flow
# *both* ways -- e.g. `nl = nfine` where nl is later a `div`
# argument (int) has to make nfine int too, even though nfine itself
# is never directly used as an index/bound/div-argument anywhere.

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