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
#   hvp_    Hessian-vec   forward-over-reverse: a second forward-mode
#                         pass over agen_'s own generated code, for
#                         Hessian-vector products.
#   val_    Validate      correctness checking against ground truth
#                         (finite differences; later the adjoint
#                         identity once tangent codegen is real).
#   io_     File I/O      the ONLY stage touching the filesystem --
#                         reads a kernel .jl file, writes a
#                         generated .jl file.
#   stade_  Public API    stade_tangent / stade_adjoint / stade_hvp
#                         (Expr in, Expr out) and stade_*_file
#                         wrappers (path in, path out), wiring every
#                         stage above together for an end user.
#
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


# ==================== hvp_* =======================================
# Forward-over-reverse Hessian-vector products: a SECOND, independent
# application of forward-mode differentiation, this time to the
# ALREADY-GENERATED adjoint kernel's own statement list rather than
# to a hand-written primal. Computes Hv = d(grad f)/dv by seeding a
# tangent on the kernel's own inputs and propagating it straight
# through agen_'s forward sweep (which recomputes the primal) and
# backward sweep (which computes the gradient) -- exactly tgen_'s own
# "shadow line right before primal line" strategy, applied to a
# second piece of code instead of to the original primal.
#
# This works as a genuinely general "one more forward layer" pass
# because reverse-mode differentiation, once carried out, leaves
# behind straight-line, order-preserving code: no further reversal,
# only replay. agen_'s output is exactly the kind of code forward-
# mode differentiation already knows how to handle. The only new
# mechanics are for push!/pop!, which never arise in a hand-written
# primal: a push of an active value gets a paired push onto a shadow
# stack; a pop gets a paired pop, in the same position, so LIFO order
# is inherited automatically rather than re-derived.
#
# There is no lin_plan for generated code (agen_ emits Expr directly,
# not a statement/lin_node tree), so this differentiates by walking
# the concrete Expr the earlier stage produced, rather than a
# separately-built tree -- the "IR" this stage adds a layer to is
# just agen_forward_body/agen_backward_body's own output, taken as
# input. No new derivative rules are needed either: an accumulation
# statement's rhs is built entirely from the same whitelisted
# intrinsics as the primal, so differentiating it a second time via
# der_tangent_generic already gives the correct second-order term.
#
# Every float arg's own tangent (xd, the seed direction v the caller
# picks) AND its adjoint-shadow's tangent (xbd) are both function
# parameters -- for an array-kinded arg this is unavoidable (nothing
# in this design ever allocates an array locally), and doing the same
# for scalars keeps the convention uniform: xbd represents "does the
# OUTER seed itself vary with v", which is 0.0 for a standard HVP,
# but making the caller pass that explicitly is clearer than hard-
# coding it, and costs nothing. Every local (non-argument) scalar
# gets its own shadow AND its adjoint-shadow's shadow declared and
# zeroed, exactly one layer past what agen_local_primal_inits /
# agen_local_shadow_inits already do. Shadow stacks are declared
# locally (never returned or inspected afterward) -- the original
# stacks still come from the primal's own initstacks_foo_b, reused
# unchanged; only the NEW shadow stacks are this stage's concern, and
# they never need to outlive one call.

function hvp_emit(kernel, active_map, lin_plan, sites)
    sig = kernel.sig
    stacks = agen_stack_map(sites)
    read_anywhere = agen_collect_reads(kernel.body)
    reassigned = agen_collect_reassigned(kernel.body)
    unsafe = agen_unsafe_int_vars(kernel)

    fwd = agen_forward_body(kernel.body, active_map, false, read_anywhere, reassigned, stacks)
    bwd = agen_backward_body(lin_plan, sig.kinds, unsafe, false, read_anywhere, reassigned, stacks)

    shadow_of = hvp_shadow_map(kernel, sites)

    fname = hvp_fname(sig.name)
    seed_args = Symbol[]
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        push!(seed_args, tgen_shadow(a))
        push!(seed_args, tgen_shadow(agen_shadow(a)))
    end
    fargs = vcat(agen_signature_args(sig), seed_args, agen_stack_names(sites))

    body = Any[]
    append!(body, hvp_shadow_stack_inits(sites, shadow_of))
    append!(body, agen_local_primal_inits(kernel, active_map))
    append!(body, agen_local_shadow_inits(kernel, active_map))
    append!(body, hvp_local_second_inits(kernel, shadow_of))
    append!(body, hvp_double_body(fwd, shadow_of))
    append!(body, hvp_double_body(bwd, shadow_of))

    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    ret = Symbol[]
    for a in scalar_args
        ab = agen_shadow(a)
        push!(ret, ab)
        push!(ret, tgen_shadow(ab))
    end
    push!(body, emit_return_scalars(ret))

    return Expr(:function, Expr(:call, fname, fargs...), Expr(:block, body...))
end

hvp_fname(name::Symbol) = Symbol(string(name) * "_hv")
hvp_stack_shadow(stack::Symbol) = Symbol(string(stack) * "_d")

# every float variable this stage will encounter: primal args/locals
# (shadow = tgen_shadow, the same "d" convention tgen_ already uses)
# and agen_'s own adjoint shadows (shadow = tgen_shadow of THOSE --
# e.g. xb's own second-layer shadow is xbd, via the same function,
# unchanged, since it only ever appends "d"); plus every Float64-
# holding stack (a paired shadow stack). Int64 stacks -- branch/
# tripcount bookkeeping -- get none; they're never differentiated.
function hvp_shadow_map(kernel, sites)
    kinds = kernel.sig.kinds
    m = Dict{Symbol,Symbol}()
    for (v, k) in kinds
        if k in (:scalar_float, :array_float)
            m[v] = tgen_shadow(v)
            m[agen_shadow(v)] = tgen_shadow(agen_shadow(v))
        end
    end
    for s in sites
        s.kind in (:array, :value) || continue
        nm = agen_site_stack_name(s)
        haskey(m, nm) || (m[nm] = hvp_stack_shadow(nm))
    end
    return m
end

function hvp_shadow_stack_inits(sites, shadow_of)
    exprs = Any[]
    seen = Set{Symbol}()
    for s in sites
        s.kind in (:array, :value) || continue
        nm = agen_site_stack_name(s)
        nm in seen && continue
        push!(seen, nm)
        push!(exprs, Expr(:(=), shadow_of[nm], Expr(:call, Expr(:curly, :Vector, :Float64))))
    end
    return exprs
end

# zero-initialize every local (non-argument) scalar's own second-
# layer shadow AND its adjoint-shadow's shadow -- exactly
# agen_local_primal_inits/agen_local_shadow_inits's job, one layer
# further out. Arrays can never be local (skill-jade rule 8), so
# there's never an array case to handle here; every float arg's own
# xd/xbd, by contrast, is a function parameter (see hvp_emit), never
# locally initialized. Unlike agen_'s own local-init functions, this
# does NOT gate on active_map: the forward sweep always replays every
# primal statement regardless of activity, so this stage's shadow of
# an "inactive" local can still be written to, and needs to exist.
function hvp_local_second_inits(kernel, shadow_of)
    sig = kernel.sig
    arg_set = Set(sig.args)
    kinds = sig.kinds
    exprs = Any[]
    for v in sort(collect(keys(kinds)))
        kinds[v] == :scalar_float || continue
        v in arg_set && continue
        push!(exprs, Expr(:(=), shadow_of[v], 0.0))
        push!(exprs, Expr(:(=), shadow_of[agen_shadow(v)], 0.0))
    end
    return exprs
end

# the general "add one forward layer" transform: differentiate a
# statement list agen_ already produced, emitting each shadow line
# immediately before its primal line -- tgen_'s own strategy, applied
# to Expr agen_ built rather than to a lin_node tree, and extended to
# recognize push!/pop! alongside assignment/for/if.
function hvp_double_body(exprs, shadow_of)
    out = Any[]
    for e in exprs
        append!(out, hvp_double_stmt(e, shadow_of))
    end
    return out
end

function hvp_double_stmt(e::Expr, shadow_of)
    if e.head == :call && e.args[1] == :push!
        stack, val = e.args[2], e.args[3]
        out = Any[]
        haskey(shadow_of, stack) && push!(out, Expr(:call, :push!, shadow_of[stack], hvp_tangent_expr(val, shadow_of)))
        push!(out, e)
        return out
    elseif e.head == :(=)
        lhs = e.args[1]
        var = lhs isa Symbol ? lhs : lhs.args[1]
        out = Any[]
        haskey(shadow_of, var) && push!(out, Expr(:(=), hvp_shadow_lvalue(lhs, shadow_of), hvp_tangent_expr(e.args[2], shadow_of)))
        push!(out, e)
        return out
    elseif e.head == :for
        inner = hvp_double_body(e.args[2].args, shadow_of)
        return Any[Expr(:for, e.args[1], Expr(:block, inner...))]
    elseif e.head == :if
        then_inner = hvp_double_body(e.args[2].args, shadow_of)
        if length(e.args) == 3
            els_inner = hvp_double_body(e.args[3].args, shadow_of)
            return Any[Expr(:if, e.args[1], Expr(:block, then_inner...), Expr(:block, els_inner...))]
        end
        return Any[Expr(:if, e.args[1], Expr(:block, then_inner...))]
    end
    return Any[e]
end

# lhs of the shadow assignment -- a bare Symbol shadows to a bare
# Symbol; an array-ref shadows to the same indices on the shadow array
function hvp_shadow_lvalue(lhs, shadow_of)
    lhs isa Symbol && return shadow_of[lhs]
    return Expr(:ref, shadow_of[lhs.args[1]], lhs.args[2:end]...)
end

# recursively differentiate an arbitrary primal-valued Expr -- fuses
# what lin_build_expr + tgen_tangent_expr do in two phases into one,
# since there is no retained tree for generated code to sweep a
# second time. A pop! differentiates to a pop from the paired shadow
# stack; everything else is the same chain-rule contraction tgen_
# already performs, via der_tangent_generic -- no new derivative
# rules, since agen_'s own accumulation statements are built entirely
# from the same whitelisted intrinsics as the primal.
function hvp_tangent_expr(expr, shadow_of)
    if expr isa Symbol
        return get(shadow_of, expr, 0.0)
    elseif expr isa Number
        return 0.0
    elseif expr isa Expr && expr.head == :ref
        haskey(shadow_of, expr.args[1]) || return 0.0
        return Expr(:ref, shadow_of[expr.args[1]], expr.args[2:end]...)
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :pop!
        stack = expr.args[2]
        haskey(shadow_of, stack) || return 0.0
        return Expr(:call, :pop!, shadow_of[stack])
    elseif expr isa Expr && expr.head == :call
        args = expr.args[2:end]
        dargs = [hvp_tangent_expr(a, shadow_of) for a in args]
        return der_tangent_generic(expr.args[1], args, dargs)
    end
    error("hvp_tangent_expr: unsupported expression $expr")
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


# ==================== val_* (baseline-driven FD/JVP/VJP validation) =
# Extends the val_ oracle above to work generically on any
# skill-jade kernel's generated _d/_b/_hv code, not just a hand-built
# fixture closure. Three identities, each reusing the same two
# oracles (val_finite_diff_check / val_finite_diff_check_jvp):
#   tangent (_d): direct JVP check -- f_d(x0,d) vs central FD of the
#                 primal itself, for random directions d.
#   adjoint (_b): dot-product identity <y,Jx> == <J'y,x> -- reuses
#                 val_finite_diff_check on the scalar closure
#                 f_eval(x) = y.primal(x), f_grad(x) = adjoint(x;seed=y).
#   hvp (_hv):    JVP check one layer out -- f_hv(x0,v) vs central FD
#                 of "gradient(x) at fixed seed y", the exact same
#                 val_finite_diff_check_jvp oracle applied to the
#                 adjoint instead of the primal.
# independents/dependents are always every float arg (parse_kernel's
# own rule), so "x" and "y" share one flattened space of dimension n;
# a single random baseline vector serves as both the input point and
# (separately drawn) the seed.

# ---- direct JVP oracle (new; val_finite_diff_check above already
#      covers the dot-product/gradient oracle unmodified) -----------

# unlike val_finite_diff_check (one x0-only gradient reused across
# many directions), the analytic side here is direction-dependent --
# f_jvp(x0,d) is called fresh each trial -- so it can't reuse that
# function's structure.
function val_finite_diff_check_jvp(f_eval_vec::Function, f_jvp::Function, x0::Vector{Float64};
                                    epsilon::Float64 = 1e-6, trials::Int = 10, rtol::Float64 = 1e-3)
    n = length(x0)
    results = NamedTuple[]
    worst_rel_err = 0.0
    for t in 1:trials
        d = randn(n)
        d = d ./ sqrt(sum(d .^ 2))
        fd = (f_eval_vec(x0 .+ epsilon .* d) .- f_eval_vec(x0 .- epsilon .* d)) ./ (2 * epsilon)
        jv = f_jvp(x0, d)
        denom = max(maximum(abs.(fd)), maximum(abs.(jv)), 1e-12)
        rel_err = maximum(abs.(fd .- jv)) / denom
        worst_rel_err = max(worst_rel_err, rel_err)
        push!(results, (direction = d, finite_diff = fd, jvp = jv, rel_err = rel_err))
    end
    return (ok = worst_rel_err <= rtol, max_rel_err = worst_rel_err, trials = results)
end

# ---- static helpers over a parsed kernel (no execution) ------------

# static scan: the largest number of :ref indices used anywhere for
# `arr` -- its dimensionality, never hardcoded to a fixed count.
function val_arg_ndims(kernel, arr::Symbol)
    found = Ref(1)
    val_scan_ndims!(kernel.body, arr, found)
    return found[]
end

function val_scan_ndims!(body, arr::Symbol, found::Ref{Int})
    for stmt in body
        if stmt.kind == :assign
            val_scan_expr_ndims!(stmt.lhs, arr, found)
            val_scan_expr_ndims!(stmt.rhs, arr, found)
        elseif stmt.kind == :for
            val_scan_expr_ndims!(stmt.lo, arr, found)
            val_scan_expr_ndims!(stmt.hi, arr, found)
            val_scan_ndims!(stmt.body, arr, found)
        elseif stmt.kind == :if
            val_scan_expr_ndims!(stmt.cond, arr, found)
            val_scan_ndims!(stmt.then, arr, found)
            val_scan_ndims!(stmt.els, arr, found)
        end
    end
    return nothing
end

function val_scan_expr_ndims!(expr, arr::Symbol, found::Ref{Int})
    if expr isa Expr && expr.head == :ref && expr.args[1] == arr
        found[] = max(found[], length(expr.args) - 1)
    end
    if expr isa Expr
        for a in expr.args
            val_scan_expr_ndims!(a, arr, found)
        end
    end
    return nothing
end

# duplicated from tgen_reassigned_scalar_args's logic rather than
# calling it directly (skill-stade purity rule: rely on another
# stage's documented shapes, not its private helpers) -- val_ needs
# to know exactly which scalar_float args a generated tangent file
# returns, to correctly capture its output.
function val_reassigned_scalar_float_args(kernel)
    arg_set = Set(kernel.sig.args)
    out = Symbol[]
    val_collect_reassigned_scalar_float!(kernel.body, arg_set, kernel.sig.kinds, out)
    return out
end

function val_collect_reassigned_scalar_float!(body, arg_set, kinds, out)
    for stmt in body
        if stmt.kind == :assign
            if stmt.lhs isa Symbol && stmt.lhs in arg_set && kinds[stmt.lhs] == :scalar_float && !(stmt.lhs in out)
                push!(out, stmt.lhs)
            end
        elseif stmt.kind == :for
            val_collect_reassigned_scalar_float!(stmt.body, arg_set, kinds, out)
        elseif stmt.kind == :if
            val_collect_reassigned_scalar_float!(stmt.then, arg_set, kinds, out)
            val_collect_reassigned_scalar_float!(stmt.els, arg_set, kinds, out)
        end
    end
    return nothing
end

# rebuilds the primal with an appended `return` of every scalar_float
# arg's final value -- skill-jade kernels never contain their own
# `return` statement (only :assign/:for/:if are recognized statement
# kinds), so appending one at the very end is always safe, and it's
# the only way a caller can observe a reassigned scalar argument the
# same way it already observes array arguments (in-place mutation).
function val_primal_observing_expr(kernel, primal_expr::Expr)
    sig = kernel.sig
    fname = Symbol(string(sig.name) * "_valobs")
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    body = Expr(:block, primal_expr.args[2].args..., emit_return_scalars(scalar_args))
    return Expr(:function, Expr(:call, fname, sig.args...), body)
end

# ---- execution helpers (the one place val_ steps outside pure Expr
#      manipulation: running dynamically generated code requires
#      evaluating it. Not filesystem access -- still permitted outside
#      io_ -- but it does define a transient global method; an
#      accepted, narrowly scoped exception, since there is no other
#      way to numerically execute generated Julia code.) --------------

function val_compile(expr::Expr)
    fname = expr.args[1].args[1]
    Base.eval(Main, expr)
    return getfield(Main, fname)
end

function val_init_stacks(initstacks_fn::Function)
    r = Base.invokelatest(initstacks_fn)
    r === nothing && return ()
    r isa Tuple && return r
    return (r,)
end

# ---- random baseline generation -------------------------------------

function val_random_int_args(sig; lo::Int = 2, hi::Int = 5)
    return Dict{Symbol,Int}(a => rand(lo:hi) for a in sig.args if sig.kinds[a] == :scalar_int)
end

function val_random_values(kernel, shapes::Dict, int_args::Dict{Symbol,Int}; scale::Float64 = 1.0)
    sig = kernel.sig
    values = Dict{Symbol,Any}()
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_float
            values[a] = scale * randn()
        elseif k == :array_float
            values[a] = scale .* randn(shapes[a]...)
        elseif k == :array_int
            values[a] = rand(1:3, shapes[a]...)
        end
    end
    return values
end

# grows one trial size N (same N along every dimension, for every
# array_float/array_int arg) until the primal runs without a
# BoundsError. Oversized dimensions are harmless -- only undersized
# ones are wrong -- so this always converges on a *safe* (if not
# minimal) size without any static index-range analysis, and stays
# fully general across arbitrarily shaped index expressions.
function val_grow_shapes(kernel, primal_fn::Function, int_args::Dict{Symbol,Int};
                          start::Int = 4, growth::Int = 2, max_size::Int = 512)
    sig = kernel.sig
    ndims_of = Dict(a => val_arg_ndims(kernel, a) for a in sig.args
                     if sig.kinds[a] in (:array_float, :array_int))
    N = start
    while N <= max_size
        shapes = Dict(a => ntuple(_ -> N, ndims_of[a]) for a in keys(ndims_of))
        trial = val_random_values(kernel, shapes, int_args; scale = 1.0)
        ok = try
            call_args = Any[sig.kinds[a] == :scalar_int ? int_args[a] : deepcopy(trial[a]) for a in sig.args]
            Base.invokelatest(primal_fn, call_args...)
            true
        catch e
            e isa BoundsError || rethrow(e)
            false
        end
        ok && return shapes
        N *= growth
    end
    error("val_grow_shapes: could not find a working array size up to $max_size for $(sig.name)")
end

# orchestrates a full random baseline: random ints, a compiled primal
# probed to find safe array sizes, then final random Float64/Int
# content at those sizes. A few retries with fresh int draws guard
# against a rare unlucky combination (e.g. a degenerate range) tripping
# an error unrelated to array sizing.
function val_generate_baseline(kernel, primal_expr::Expr;
                                scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                grow_start::Int = 4, grow_max::Int = 512, attempts::Int = 5)
    primal_fn = val_compile(primal_expr)
    last_err = nothing
    for attempt in 1:attempts
        int_args = val_random_int_args(kernel.sig; lo = int_lo, hi = int_hi)
        try
            shapes = val_grow_shapes(kernel, primal_fn, int_args; start = grow_start, max_size = grow_max)
            values = val_random_values(kernel, shapes, int_args; scale = scale)
            return (int_args = int_args, values = values)
        catch e
            last_err = e
            continue
        end
    end
    error("val_generate_baseline: failed after $attempts attempts for $(kernel.sig.name): $last_err")
end

# ---- flatten/unflatten over the shared x/y space (all float args,
#      in signature order; arrays row-major via Julia's own `vec`) ---

val_float_arg_order(sig) = [a for a in sig.args if sig.kinds[a] in (:scalar_float, :array_float)]

function val_flatten(kernel, values::Dict)
    x = Float64[]
    for a in val_float_arg_order(kernel.sig)
        v = values[a]
        v isa Number ? push!(x, v) : append!(x, vec(v))
    end
    return x
end

# rebuilds a values Dict (fresh array copies, never aliasing the
# template) with every float arg replaced by x's content, in the same
# order val_flatten used; non-float entries are copied through from
# `template` unchanged (needed for array_int workspace args).
function val_unflatten(kernel, int_args::Dict, template::Dict, x::Vector{Float64})
    sig = kernel.sig
    out = Dict{Symbol,Any}()
    for a in sig.args
        sig.kinds[a] == :array_int && (out[a] = template[a])
    end
    i = 1
    for a in val_float_arg_order(sig)
        v = template[a]
        if v isa Number
            out[a] = x[i]; i += 1
        else
            n = length(v)
            out[a] = reshape(x[i:i+n-1], size(v))
            i += n
        end
    end
    return out
end

function val_zeros_like(kernel, values::Dict)
    out = Dict{Symbol,Any}()
    for a in val_float_arg_order(kernel.sig)
        v = values[a]
        out[a] = v isa Number ? 0.0 : zeros(size(v))
    end
    return out
end

function val_random_values_like(kernel, values::Dict; scale::Float64 = 1.0)
    out = Dict{Symbol,Any}()
    for a in val_float_arg_order(kernel.sig)
        v = values[a]
        out[a] = v isa Number ? scale * randn() : scale .* randn(size(v)...)
    end
    return out
end

# ---- positional call-argument builders -- duplicate
#      tgen_signature_args/agen_signature_args's documented convention
#      (float arg immediately followed by its shadow; int args
#      unchanged) rather than reaching into those stages' private
#      helpers, per the same purity rule agen_ itself follows when
#      duplicating snap_'s TBR predicate. --------------------------

function val_primal_call_args(sig, int_args::Dict, values::Dict)
    return Any[sig.kinds[a] == :scalar_int ? int_args[a] : deepcopy(values[a]) for a in sig.args]
end

function val_tangent_call_args(sig, int_args::Dict, values::Dict, dvalues::Dict)
    call = Any[]
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(call, int_args[a])
        elseif k in (:scalar_float, :array_float)
            push!(call, deepcopy(values[a]))
            push!(call, deepcopy(dvalues[a]))
        else
            push!(call, deepcopy(values[a]))
        end
    end
    return call
end

function val_adjoint_call_args(sig, int_args::Dict, values::Dict, seed::Dict, stacks)
    call = Any[]
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(call, int_args[a])
        elseif k in (:scalar_float, :array_float)
            push!(call, deepcopy(values[a]))
            push!(call, deepcopy(seed[a]))
        else
            push!(call, deepcopy(values[a]))
        end
    end
    append!(call, stacks)
    return call
end

function val_hvp_call_args(sig, int_args::Dict, values::Dict, seed::Dict, dvalues::Dict, dseed::Dict, stacks)
    call = Any[]
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(call, int_args[a])
        elseif k in (:scalar_float, :array_float)
            push!(call, deepcopy(values[a]))
            push!(call, deepcopy(seed[a]))
        else
            push!(call, deepcopy(values[a]))
        end
    end
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        push!(call, deepcopy(dvalues[a]))
        push!(call, deepcopy(dseed[a]))
    end
    append!(call, stacks)
    return call
end

# ---- call + extract-output helpers, one per generated file kind ----

function val_call_primal_observed(primal_obs_fn::Function, kernel, int_args::Dict, values::Dict)
    sig = kernel.sig
    call_args = val_primal_call_args(sig, int_args, values)
    ret = Base.invokelatest(primal_obs_fn, call_args...)
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    ret_tuple = isempty(scalar_args) ? () : (length(scalar_args) == 1 ? (ret,) : ret)
    scalar_of = Dict(zip(scalar_args, ret_tuple))
    out = Dict{Symbol,Any}()
    for (a, v) in zip(sig.args, call_args)
        k = sig.kinds[a]
        k == :scalar_float && (out[a] = scalar_of[a])
        k == :array_float && (out[a] = v)
    end
    return val_flatten(kernel, out)
end

function val_call_tangent(tangent_fn::Function, kernel, int_args::Dict, values::Dict, dvalues::Dict)
    sig = kernel.sig
    call_args = val_tangent_call_args(sig, int_args, values, dvalues)
    ret = Base.invokelatest(tangent_fn, call_args...)
    reassigned = val_reassigned_scalar_float_args(kernel)
    ret_tuple = isempty(reassigned) ? () : (length(reassigned) == 1 ? (ret,) : ret)
    ret_of = Dict(zip(reassigned, ret_tuple))
    out = Dict{Symbol,Any}()
    pos = 1
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            pos += 1
        elseif k in (:scalar_float, :array_float)
            shadow = call_args[pos + 1]
            out[a] = k == :scalar_float ? get(ret_of, a, dvalues[a]) : shadow
            pos += 2
        else
            pos += 1
        end
    end
    return val_flatten(kernel, out)
end

function val_call_adjoint(adjoint_fn::Function, initstacks_fn::Function, kernel,
                           int_args::Dict, values::Dict, seed::Dict)
    sig = kernel.sig
    stacks = val_init_stacks(initstacks_fn)
    call_args = val_adjoint_call_args(sig, int_args, values, seed, stacks)
    ret = Base.invokelatest(adjoint_fn, call_args...)
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    ret_tuple = isempty(scalar_args) ? () : (length(scalar_args) == 1 ? (ret,) : ret)
    ret_of = Dict(zip(scalar_args, ret_tuple))
    out = Dict{Symbol,Any}()
    pos = 1
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            pos += 1
        elseif k in (:scalar_float, :array_float)
            shadow = call_args[pos + 1]
            out[a] = k == :scalar_float ? ret_of[a] : shadow
            pos += 2
        else
            pos += 1
        end
    end
    return val_flatten(kernel, out)
end

function val_call_hv(hv_fn::Function, initstacks_fn::Function, kernel, int_args::Dict,
                      values::Dict, seed::Dict, dvalues::Dict, dseed::Dict)
    sig = kernel.sig
    stacks = val_init_stacks(initstacks_fn)
    call_args = val_hvp_call_args(sig, int_args, values, seed, dvalues, dseed, stacks)
    ret = Base.invokelatest(hv_fn, call_args...)
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    # each scalar arg contributes TWO return values (ab, abd), so the
    # generated function's own bare-value-vs-tuple rule (emit_return_scalars)
    # only ever produces a bare (non-tuple) value in the impossible case of
    # a single combined return -- with 0 scalar args there are no returns
    # at all (nothing), otherwise it's always a genuine tuple of length 2*n.
    ret_tuple = isempty(scalar_args) ? () : ret
    hv_of = Dict{Symbol,Any}(a => ret_tuple[2i] for (i, a) in enumerate(scalar_args))
    n_lead = count(a -> sig.kinds[a] == :scalar_int, sig.args) +
             count(a -> sig.kinds[a] == :array_int, sig.args) +
             2 * count(a -> sig.kinds[a] in (:scalar_float, :array_float), sig.args)
    out = Dict{Symbol,Any}()
    i = 0
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        xbd_pos = n_lead + 2 * i + 2
        out[a] = sig.kinds[a] == :scalar_float ? hv_of[a] : call_args[xbd_pos]
        i += 1
    end
    return val_flatten(kernel, out)
end

# ---- top-level orchestration: one function per generated file kind -

function val_validate_tangent(kernel, primal_expr::Expr, tangent_expr::Expr, baseline;
                               trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3)
    obs_fn = val_compile(val_primal_observing_expr(kernel, primal_expr))
    tangent_fn = val_compile(tangent_expr)
    int_args = baseline.int_args
    x0 = val_flatten(kernel, baseline.values)
    f_eval_vec = x -> val_call_primal_observed(obs_fn, kernel, int_args,
                                                val_unflatten(kernel, int_args, baseline.values, x))
    f_jvp = (x, d) -> begin
        vals = val_unflatten(kernel, int_args, baseline.values, x)
        dvals = val_unflatten(kernel, int_args, val_zeros_like(kernel, baseline.values), d)
        val_call_tangent(tangent_fn, kernel, int_args, vals, dvals)
    end
    return val_finite_diff_check_jvp(f_eval_vec, f_jvp, x0; epsilon = epsilon, trials = trials, rtol = rtol)
end

function val_validate_adjoint(kernel, primal_expr::Expr, adjoint_out, baseline;
                               trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3)
    obs_fn = val_compile(val_primal_observing_expr(kernel, primal_expr))
    adjoint_fn = val_compile(adjoint_out.adjoint)
    initstacks_fn = val_compile(adjoint_out.initstacks)
    int_args = baseline.int_args
    x0 = val_flatten(kernel, baseline.values)
    seed = val_random_values_like(kernel, baseline.values)
    seed_flat = val_flatten(kernel, seed)
    f_eval = x -> begin
        vals = val_unflatten(kernel, int_args, baseline.values, x)
        y = val_call_primal_observed(obs_fn, kernel, int_args, vals)
        sum(y .* seed_flat)
    end
    f_grad = x0_ -> begin
        vals = val_unflatten(kernel, int_args, baseline.values, x0_)
        val_call_adjoint(adjoint_fn, initstacks_fn, kernel, int_args, vals, seed)
    end
    return val_finite_diff_check(f_eval, f_grad, x0; epsilon = epsilon, trials = trials, rtol = rtol)
end

function val_validate_hvp(kernel, primal_expr::Expr, adjoint_out, hvp_out, baseline;
                           trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3)
    adjoint_fn = val_compile(adjoint_out.adjoint)
    initstacks_fn = val_compile(adjoint_out.initstacks)
    hv_fn = val_compile(hvp_out.hvp)
    int_args = baseline.int_args
    x0 = val_flatten(kernel, baseline.values)
    seed = val_random_values_like(kernel, baseline.values)
    g_of = x -> begin
        vals = val_unflatten(kernel, int_args, baseline.values, x)
        val_call_adjoint(adjoint_fn, initstacks_fn, kernel, int_args, vals, seed)
    end
    hv_of = (x, v) -> begin
        vals = val_unflatten(kernel, int_args, baseline.values, x)
        dvals = val_unflatten(kernel, int_args, val_zeros_like(kernel, baseline.values), v)
        dseed = val_zeros_like(kernel, baseline.values)
        val_call_hv(hv_fn, initstacks_fn, kernel, int_args, vals, seed, dvals, dseed)
    end
    return val_finite_diff_check_jvp(g_of, hv_of, x0; epsilon = epsilon, trials = trials, rtol = rtol)
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

# bundles initstacks_foo_b, foo_b, and a copy of foo itself, in that order
function io_write_kernel_file(path::String, primal_expr::Expr, generated::Vector{Expr})
    parts = [io_expr_to_source(e) for e in vcat(generated, [primal_expr])]
    open(path, "w") do f
        write(f, join(parts, "\n"))
    end
    return nothing
end

io_path_exists(path::String) = isfile(path)
io_default_yaml_path(in_path::String) = splitext(in_path)[1] * ".yaml"

# a minimal, hand-written YAML subset -- no external dependency. Two
# top-level mappings: `int_args:` (bare integers) and `values:` (a
# scalar is a bare float, a 1-D array is a flow list `[a, b, c]`, a
# 2-D array is a block sequence of flow-list rows). Simple enough for
# a user to hand-edit their own baseline file in the same shape.
function io_write_baseline_yaml(path::String, kernel_name::Symbol, int_args::Dict, values::Dict)
    lines = String["# baseline values for kernel: $(kernel_name)", "int_args:"]
    for k in sort(collect(keys(int_args)); by = string)
        push!(lines, "  $(k): $(int_args[k])")
    end
    push!(lines, "values:")
    for k in sort(collect(keys(values)); by = string)
        v = values[k]
        if v isa Number
            push!(lines, "  $(k): $(v)")
        elseif ndims(v) == 1
            push!(lines, "  $(k): [" * join(v, ", ") * "]")
        elseif ndims(v) == 2
            push!(lines, "  $(k):")
            for i in 1:size(v, 1)
                push!(lines, "    - [" * join(v[i, :], ", ") * "]")
            end
        else
            error("io_write_baseline_yaml: unsupported array rank for $(k)")
        end
    end
    open(path, "w") do f
        write(f, join(lines, "\n") * "\n")
    end
    return path
end

function io_read_baseline_yaml(path::String)
    int_args = Dict{Symbol,Int}()
    values = Dict{Symbol,Any}()
    section = :none
    pending_key = nothing
    pending_rows = Vector{Float64}[]
    function flush_pending!()
        if pending_key !== nothing
            values[pending_key] = reduce(vcat, [permutedims(r) for r in pending_rows])
            pending_key = nothing
            pending_rows = Vector{Float64}[]
        end
    end
    for raw in readlines(path)
        line = rstrip(raw)
        (isempty(line) || startswith(strip(line), "#")) && continue
        if line == "int_args:"
            flush_pending!(); section = :int_args; continue
        elseif line == "values:"
            flush_pending!(); section = :values; continue
        end
        if section == :int_args
            m = match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?\d+)\s*$", line)
            m === nothing && error("io_read_baseline_yaml: malformed int_args line: $line")
            int_args[Symbol(m.captures[1])] = parse(Int, m.captures[2])
        elseif section == :values
            mrow = match(r"^\s*-\s*\[(.*)\]\s*$", line)
            if mrow !== nothing
                pending_key === nothing && error("io_read_baseline_yaml: matrix row with no preceding key: $line")
                push!(pending_rows, [parse(Float64, strip(t)) for t in split(mrow.captures[1], ",")])
                continue
            end
            flush_pending!()
            mkeyonly = match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$", line)
            if mkeyonly !== nothing
                pending_key = Symbol(mkeyonly.captures[1])
                continue
            end
            mlist = match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[(.*)\]\s*$", line)
            if mlist !== nothing
                values[Symbol(mlist.captures[1])] = [parse(Float64, strip(t)) for t in split(mlist.captures[2], ",")]
                continue
            end
            mscalar = match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?[0-9.eE+\-]+)\s*$", line)
            mscalar === nothing && error("io_read_baseline_yaml: malformed values line: $line")
            values[Symbol(mscalar.captures[1])] = parse(Float64, mscalar.captures[2])
        end
    end
    flush_pending!()
    return (int_args = int_args, values = values)
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

function stade_hvp(expr::Expr; independents::Union{Vector{Symbol},Nothing}=nothing,
                    dependents::Union{Vector{Symbol},Nothing}=nothing)
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    active_map = act_analyze(kernel)
    snapshot_plan = snap_plan(kernel, active_map)
    lin_plan = lin_build(kernel, active_map)
    hvp_expr = hvp_emit(kernel, active_map, lin_plan, snapshot_plan)
    initstacks_expr = agen_init_emit(kernel, snapshot_plan)
    return (hvp = hvp_expr, initstacks = initstacks_expr)
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

function stade_hvp_file(in_path::String, out_path::String)
    primal_expr = io_read_kernel(in_path)
    hvp_out = stade_hvp(primal_expr)
    io_write_kernel_file(out_path, primal_expr, [hvp_out.initstacks, hvp_out.hvp])
    return out_path
end


# ==================== stade_* baseline validation (public API) =====
# Numerically validates a generated tangent/adjoint/hvp file against
# central finite differences of the primal, using a baseline that is
# auto-generated once and cached to a YAML file next to the kernel (or
# a user-supplied one, read via the same public entry point). See the
# val_* banner above for what each mode actually checks.

# the function that reads a baseline YAML and performs the check --
# usable directly by a caller pointing at their own hand-written file.
function stade_validate_from_baseline(mode::Symbol, in_path::String, yaml_path::String;
                                       trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3)
    mode in (:tangent, :adjoint, :hvp) ||
        error("stade_validate_from_baseline: mode must be :tangent, :adjoint, or :hvp, got $mode")
    primal_expr = io_read_kernel(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = io_read_baseline_yaml(yaml_path)
    if mode == :tangent
        tangent_expr = stade_tangent(primal_expr)
        return val_validate_tangent(kernel, primal_expr, tangent_expr, baseline;
                                     trials = trials, epsilon = epsilon, rtol = rtol)
    elseif mode == :adjoint
        adjoint_out = stade_adjoint(primal_expr)
        return val_validate_adjoint(kernel, primal_expr, adjoint_out, baseline;
                                     trials = trials, epsilon = epsilon, rtol = rtol)
    else
        adjoint_out = stade_adjoint(primal_expr)
        hvp_out = stade_hvp(primal_expr)
        return val_validate_hvp(kernel, primal_expr, adjoint_out, hvp_out, baseline;
                                 trials = trials, epsilon = epsilon, rtol = rtol)
    end
end

# generates a random baseline for `in_path` and writes it to a YAML
# file sharing its basename (`foo.jl` -> `foo.yaml`), or to
# `yaml_path` if given. Exposed standalone so a caller can generate
# once, hand-edit the result, then validate repeatedly against it.
function stade_generate_baseline_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                       scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5)
    primal_expr = io_read_kernel(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = val_generate_baseline(kernel, primal_expr; scale = scale, int_lo = int_lo, int_hi = int_hi)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_write_baseline_yaml(yp, kernel.sig.name, baseline.int_args, baseline.values)
    return yp
end

function stade_validate_tangent_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                      scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                      trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi)
    return stade_validate_from_baseline(:tangent, in_path, yp; trials = trials, epsilon = epsilon, rtol = rtol)
end

function stade_validate_adjoint_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                      scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                      trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi)
    return stade_validate_from_baseline(:adjoint, in_path, yp; trials = trials, epsilon = epsilon, rtol = rtol)
end

function stade_validate_hvp_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                  scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                  trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi)
    return stade_validate_from_baseline(:hvp, in_path, yp; trials = trials, epsilon = epsilon, rtol = rtol)
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
    hvp_out     = stade_hvp(trivial)
    @assert tangent_out isa Expr
    @assert adjoint_out.adjoint isa Expr && adjoint_out.initstacks isa Expr
    @assert hvp_out.hvp isa Expr && hvp_out.initstacks isa Expr
    println("STADE.jl Phase 0 skeleton loaded and round-tripped a stub kernel OK")
end