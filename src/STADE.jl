# ============================================================
# STADE.jl -- source-to-source AD for skill-stade-compliant Julia
# kernels. See skill-stade-dev.md for the full house-style contract.
#
# Pipeline stages, in the order data flows through them:
#
#   inl_    Inline       multi-kernel-only: splices callee bodies into
#                         caller bodies (raw Expr, before parse_kernel
#                         ever runs) so nested call graphs reduce to
#                         the same single-function shape every stage
#                         below already understands. No-op for a
#                         single kernel with no calls.
#   parse_  Parse        raw Expr -> validated (sig, body) kernel;
#                         rejects anything violating skill-stade.
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
#   cgen_   GPU codegen   loop-nest transform (not a derivative): splits
#                         each iteration-independent (non-i_seq_) loop
#                         into a device kernel + launch call, for
#                         whichever GPU vendor a `gpu_backend` describes
#                         (CUDA/AMDGPU/Metal). Consumes a plain
#                         skill-stade kernel OR one of tgen_/agen_'s own
#                         generated functions (see cgen_ingest) -- never
#                         a multi-kernel corpus with un-inlined calls,
#                         see the cgen_* section header for that gap.
#   jgen_   JACC codegen  sibling to cgen_, not a gpu_backend value:
#                         JACC.jl's dispatch-deferred-to-runtime model
#                         (JACC.@parallel_for range=N + a plain indexed
#                         function, vendor chosen later via
#                         Preferences.jl) isn't the same programming
#                         model as a launch-macro backend, so it gets
#                         its own prefix, reusing cgen_'s shared
#                         parsing/free-var/atomic-detection helpers
#                         directly.
#   val_    Validate      correctness checking against ground truth
#                         (finite differences; later the adjoint
#                         identity once tangent codegen is real).
#   io_     File I/O      the ONLY stage touching the filesystem --
#                         reads one or more kernel definitions out of
#                         a .jl file, writes a generated .jl file.
#   stade_  Public API    stade_tangent / stade_adjoint / stade_hvp
#                         (Expr in, Expr out), their _corpus siblings
#                         (multi-kernel Dict in/out, run inl_ first),
#                         stade_*_file wrappers (path in, path out,
#                         single- or multi-kernel), and the GPU-porting
#                         siblings stade_cuda/stade_amdgpu/stade_metal/
#                         stade_jacc (+ their _file forms), wiring every
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
#     -- :while intentionally unsupported for now, see skill-stade-dev.md
#     -- lo/hi/step are all Expr|Symbol|Number; a plain `lo:hi` header
#        parses with `step` set to the Int64 literal `1`.
# statement_list :: Vector{NamedTuple}
# active_map    :: Dict{Symbol,Bool}                # var/array name -> is-active
# snapshot_site :: (kind=:value|:array|:branch|:tripcount, array::Symbol, at::Int)
# snapshot_plan :: Vector{snapshot_site}
# lin_node/lin_plan :: internal-only shape, documented at the lin_*
#     section header below.
# inl_*'s call graph / finalized-callee bookkeeping is internal-only
#     too (built and discarded inside inl_inline_calls) -- not part of
#     the frozen shape list. inl_* is the one stage that runs before
#     parse_kernel and therefore operates on raw Expr, not on the
#     kernel/statement shapes above.
# der_rule_pair :: (tangent::Function, adjoint::Function)
# cuda_plan     :: (host::Expr, kernels::Vector{Expr}) -- cgen_emit's
#     output, for any gpu_backend (the name predates Metal/AMDGPU
#     support but the shape is vendor-neutral).
# gpu_backend   :: (suffix, kernel_tag, launch_macro::Symbol,
#     threads_kw::Symbol, blocks_kw::Symbol, tid_rhs::Expr,
#     atomic_macro::Expr, allowscalar_macro::Expr, preamble::String,
#     default_precision::Type{<:AbstractFloat}, precision_locked::Bool,
#     precision_lock_reason::String) -- see cgen_backend_cuda/
#     cgen_backend_amdgpu/cgen_backend_metal.
# cgen_kernel   :: (name::Symbol, args::Vector{Symbol}, body::statement_list,
#     ret::Vector{Symbol}) -- cgen_-local, not the frozen `kernel` shape
#     above (a generated function has no kinds/independents/dependents
#     to carry). Built by cgen_from_kernel or cgen_parse_generated;
#     empty ret means `return nothing`. jgen_emit consumes the same shape.


module STADE

export stade_tangent_file
export stade_adjoint_file
export stade_hvp_file

export stade_cuda_file
export stade_amdgpu_file
export stade_metal_file
export stade_jacc_file


# ==================== inl_* ====================================
# Multi-kernel call graphs, inlined before parse_kernel runs. A call is a
# bare `callee_name(args...)` statement with symbol-only args. Inlining
# splices each callee body into its call sites, so later stages see only a
# flat body. A call cycle is a hard error.

# inl_inline_calls(kernels) -> Dict{Symbol,Expr}
#   kernels :: Dict{Symbol,Expr}, one `function ... end` Expr per kernel
# name from a corpus supplied up front. Pure in its input (rule 7); see
# inl_rename_map for naming. Returns one fully-inlined Expr per kernel name,
# every user-kernel call expanded away.
function inl_inline_calls(kernels::Dict{Symbol,Expr})
    graph, parsed = inl_build_call_graph(kernels)
    order = inl_topo_sort(graph)
    finalized = Dict{Symbol,Any}()
    result = Dict{Symbol,Expr}()
    for name in order
        args, body_block = parsed[name]
        confirmed = inl_confirmed_kinds(name, args, body_block)
        final_stmts = inl_inline_body(name, args, confirmed, body_block.args, finalized, Ref(0))
        final_body = Expr(:block, final_stmts...)
        final_expr = Expr(:function, Expr(:call, name, args...), final_body)
        final_kernel = parse_kernel(final_expr)
        finalized[name] = (kernel = final_kernel, body_block = final_body)
        result[name] = final_expr
    end
    return result
end

# ---- call graph + cycle/unknown-callee detection ----

# name -> (args, flattened-and-delined body :: Expr(:block, ...))
function inl_kernel_parts(kernel_expr::Expr)
    kernel_expr.head == :function ||
        error("inl_inline_calls: expected a `function ... end` definition for a kernel, got Expr(:$(kernel_expr.head), ...)")
    name, args = parse_signature(kernel_expr.args[1])
    body_block = Expr(:block, inl_flatten_block(kernel_expr.args[2])...)
    return name, args, body_block
end

# same as parse_strip_lines, but recurses into nested :for/:if bodies
# too (staying in raw Expr the whole way, per rule 3.5) so every
# block at every nesting depth, all the way down, is LineNumberNode-
# free and directly splice-able.
function inl_flatten_block(block_expr)
    return Any[inl_flatten_stmt(s) for s in parse_strip_lines(block_expr)]
end

function inl_flatten_stmt(stmt::Expr)
    if stmt.head == :for
        return Expr(:for, stmt.args[1], Expr(:block, inl_flatten_block(stmt.args[2])...))
    elseif stmt.head == :if
        if length(stmt.args) == 3
            return Expr(:if, stmt.args[1],
                        Expr(:block, inl_flatten_block(stmt.args[2])...),
                        Expr(:block, inl_flatten_block(stmt.args[3])...))
        else
            return Expr(:if, stmt.args[1], Expr(:block, inl_flatten_block(stmt.args[2])...))
        end
    else
        return stmt   # :(=) or a bare :call statement -- left as-is
    end
end

function inl_build_call_graph(kernels::Dict{Symbol,Expr})
    parsed = Dict{Symbol,Tuple{Vector{Symbol},Expr}}()
    graph = Dict{Symbol,Set{Symbol}}()
    for name in sort(collect(keys(kernels)))
        kname, kargs, body_block = inl_kernel_parts(kernels[name])
        kname == name ||
            error("inl_inline_calls: kernel dict key :$(name) doesn't match its own function name :$(kname)")
        callees = Set{Symbol}()
        inl_collect_calls!(body_block, name, callees, kernels)
        parsed[name] = (kargs, body_block)
        graph[name] = callees
    end
    return graph, parsed
end

# every bare-statement call anywhere in body_block (recursing through
# :for/:if), naming caller+callee on an unresolved reference -- a
# bare :call statement is never legal skill-stade input on its own, so
# by construction the only thing it can be is a call to another
# kernel in the corpus
function inl_collect_calls!(body_block::Expr, caller_name::Symbol, callees::Set{Symbol}, kernels::Dict{Symbol,Expr})
    for stmt in body_block.args
        if stmt isa Expr && stmt.head == :call
            callee = stmt.args[1]
            haskey(kernels, callee) ||
                error("inl_inline_calls: kernel :$(caller_name) calls unresolved callee :$(callee) (not a kernel in the supplied corpus)")
            push!(callees, callee)
        elseif stmt isa Expr && stmt.head == :for
            inl_collect_calls!(stmt.args[2], caller_name, callees, kernels)
        elseif stmt isa Expr && stmt.head == :if
            inl_collect_calls!(stmt.args[2], caller_name, callees, kernels)
            length(stmt.args) == 3 && inl_collect_calls!(stmt.args[3], caller_name, callees, kernels)
        end
    end
    return nothing
end

# postorder DFS: a kernel is only appended to `order` after every
# kernel it calls has been -- so processing `order` in sequence
# always has each callee already finalized before its caller needs
# it. Hard-errors with the full cycle trace, never an iteration cap.
function inl_topo_sort(graph::Dict{Symbol,Set{Symbol}})
    order = Symbol[]
    state = Dict{Symbol,Symbol}(k => :unvisited for k in keys(graph))
    stack = Symbol[]
    function inl_visit!(n::Symbol)
        state[n] == :done && return nothing
        if state[n] == :visiting
            idx = findfirst(==(n), stack)
            error("inl_inline_calls: cycle detected in kernel call graph: " * join(vcat(stack[idx:end], n), " -> "))
        end
        state[n] = :visiting
        push!(stack, n)
        for callee in sort(collect(graph[n]))
            inl_visit!(callee)
        end
        pop!(stack)
        state[n] = :done
        push!(order, n)
        return nothing
    end
    for n in sort(collect(keys(graph)))
        inl_visit!(n)
    end
    return order
end

# ---- per-caller inlining: scan, kind-check, rename, substitute, splice ----

# The caller's confirmed kind map, computed once from its original body with
# every bare-call statement dropped (at any nesting depth). A call site
# gives no syntactic evidence either way, so dropping it matches never
# having seen it, independent of inlining order. Reused for every call-site
# check in this caller.
function inl_confirmed_kinds(caller_name::Symbol, caller_args::Vector{Symbol}, body_block::Expr)
    pruned = inl_strip_calls(body_block)
    kernel_expr = Expr(:function, Expr(:call, caller_name, caller_args...), Expr(:block, pruned...))
    return parse_kernel(kernel_expr).sig.kinds
end

function inl_strip_calls(body_block::Expr)
    result = Any[]
    for stmt in body_block.args
        if stmt isa Expr && stmt.head == :call
            continue
        elseif stmt isa Expr && stmt.head == :for
            push!(result, Expr(:for, stmt.args[1], Expr(:block, inl_strip_calls(stmt.args[2])...)))
        elseif stmt isa Expr && stmt.head == :if
            if length(stmt.args) == 3
                push!(result, Expr(:if, stmt.args[1],
                                    Expr(:block, inl_strip_calls(stmt.args[2])...),
                                    Expr(:block, inl_strip_calls(stmt.args[3])...)))
            else
                push!(result, Expr(:if, stmt.args[1], Expr(:block, inl_strip_calls(stmt.args[2])...)))
            end
        else
            push!(result, stmt)
        end
    end
    return result
end

# Processes one (possibly nested) statement_list of `caller_name`, expanding
# each bare-call statement against `finalized` callees and returning the
# replacement statements. `confirmed` is the caller's kind map (see
# inl_confirmed_kinds), static for the whole caller. `counter` is the per-
# caller call-site counter, shared recursively.
function inl_inline_body(caller_name::Symbol, caller_args::Vector{Symbol}, confirmed::Dict{Symbol,Symbol},
                          stmts::Vector{Any}, finalized::Dict{Symbol,Any}, counter::Ref{Int})
    result = Any[]
    for stmt in stmts
        if stmt isa Expr && stmt.head == :call
            callee_name = stmt.args[1]
            haskey(finalized, callee_name) ||
                error("inl_inline_calls: kernel :$(caller_name) calls :$(callee_name) before it has been inlined -- not reachable in topological order")
            callee_kernel = finalized[callee_name].kernel
            callee_body = finalized[callee_name].body_block
            call_args = inl_check_call_kinds(caller_name, confirmed, stmt, callee_kernel)
            counter[] += 1
            rename_subst = inl_rename_map(callee_kernel, callee_body, counter[])
            renamed_body = inl_substitute_expr(callee_body, rename_subst)
            params_subst = Dict{Symbol,Symbol}(zip(callee_kernel.sig.args, call_args))
            substituted_body = inl_substitute_expr(renamed_body, params_subst)
            append!(result, substituted_body.args)
        elseif stmt isa Expr && stmt.head == :for
            inner = inl_inline_body(caller_name, caller_args, confirmed, stmt.args[2].args, finalized, counter)
            push!(result, Expr(:for, stmt.args[1], Expr(:block, inner...)))
        elseif stmt isa Expr && stmt.head == :if
            then_inner = inl_inline_body(caller_name, caller_args, confirmed, stmt.args[2].args, finalized, counter)
            if length(stmt.args) == 3
                els_inner = inl_inline_body(caller_name, caller_args, confirmed, stmt.args[3].args, finalized, counter)
                push!(result, Expr(:if, stmt.args[1], Expr(:block, then_inner...), Expr(:block, els_inner...)))
            else
                push!(result, Expr(:if, stmt.args[1], Expr(:block, then_inner...)))
            end
        else
            push!(result, stmt)
        end
    end
    return result
end

# Kind-checks a call site against the callee's declared signature before
# substitution -- what pure inlining would otherwise lose. `:scalar_float`
# is skipped: it is shape_infer's no-evidence default, matching a pass-
# through arg. `:array_float`/`:array_int`/`:scalar_int` are never defaults,
# so they are always enforced.
function inl_check_call_kinds(caller_name::Symbol, confirmed::Dict{Symbol,Symbol},
                               call_stmt::Expr, callee_kernel)
    call_args = call_stmt.args[2:end]
    all(a -> a isa Symbol, call_args) ||
        error("inl_inline_calls: call to :$(callee_kernel.sig.name) inside :$(caller_name) must pass bare symbol arguments only, got `$(call_stmt)`")
    length(call_args) == length(callee_kernel.sig.args) ||
        error("inl_inline_calls: call to :$(callee_kernel.sig.name) inside :$(caller_name) passes $(length(call_args)) argument(s), expected $(length(callee_kernel.sig.args))")
    for (i, a) in enumerate(call_args)
        haskey(confirmed, a) ||
            error("inl_inline_calls: call to :$(callee_kernel.sig.name) inside :$(caller_name) passes undefined variable :$(a)")
        caller_kind = confirmed[a]
        caller_kind == :scalar_float && continue   # no confirmed local evidence -- can't be a caught conflict
        callee_param = callee_kernel.sig.args[i]
        callee_kind = callee_kernel.sig.kinds[callee_param]
        caller_kind == callee_kind ||
            error("inl_inline_calls: call to :$(callee_kernel.sig.name) inside :$(caller_name): argument $(i) (:$(a)) has kind $(caller_kind), but parameter :$(callee_param) declares $(callee_kind)")
    end
    return call_args
end

# ---- deterministic rename (no `rand`) + pure symbol substitution ----

# every genuine local of the callee -- assigned-to scalars and
# for-loop variables that aren't one of the callee's own params --
# gets a deterministic `_<callee_name>_c<call_site_id>` suffix so
# repeated inlining of the same callee can never collide.
function inl_rename_map(callee_kernel, callee_body::Expr, call_site_id::Int)
    params = Set(callee_kernel.sig.args)
    locals = Set{Symbol}()
    inl_collect_locals!(callee_body, params, locals)
    subst = Dict{Symbol,Symbol}()
    for v in locals
        new_name = Symbol(String(v) * "_" * String(callee_kernel.sig.name) * "_c" * string(call_site_id))
        subst[v] = new_name
    end
    return subst
end

function inl_collect_locals!(body_block::Expr, params::Set{Symbol}, locals::Set{Symbol})
    for stmt in body_block.args
        if stmt.head == :(=)
            lhs = stmt.args[1]
            v = lhs isa Symbol ? lhs : lhs.args[1]
            v isa Symbol && !(v in params) && push!(locals, v)
        elseif stmt.head == :for
            var = stmt.args[1].args[1]
            var isa Symbol && !(var in params) && push!(locals, var)
            inl_collect_locals!(stmt.args[2], params, locals)
        elseif stmt.head == :if
            inl_collect_locals!(stmt.args[2], params, locals)
            length(stmt.args) == 3 && inl_collect_locals!(stmt.args[3], params, locals)
        end
    end
    return nothing
end

# pure Symbol->Symbol substitution over a raw Expr tree, used both
# for local renaming and for param->call-arg substitution -- a
# :call's own operator/function name (args[1]) is always left alone,
# every other position (including a :ref's array name) is fair game
function inl_substitute_expr(expr, subst::Dict{Symbol,Symbol})
    if expr isa Symbol
        return get(subst, expr, expr)
    elseif expr isa Expr
        if expr.head == :call
            return Expr(:call, expr.args[1], (inl_substitute_expr(a, subst) for a in expr.args[2:end])...)
        else
            return Expr(expr.head, (inl_substitute_expr(a, subst) for a in expr.args)...)
        end
    else
        return expr
    end
end



# ==================== parse_* ================================
# Raw Expr -> validated kernel. Hard-errors on every skill-stade rule visible
# at the Expr level: keyword args, the four variable shapes, indirect
# indexing, broadcasting, i_seq_ prefix, div-not-÷, and the intrinsic
# whitelist; comment-header and `for`-style are source-text-only.

function parse_kernel(expr::Expr)
    expr.head == :function ||
        error("parse_kernel: expected a `function ... end` definition, got Expr(:$(expr.head), ...)")
    length(expr.args) == 2 ||
        error("parse_kernel: malformed function Expr (expected signature + body)")
    name, args = parse_signature(expr.args[1])
    body = parse_statements(parse_strip_lines(expr.args[2]))
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

# the reserved i_seq_ prefix marks a genuinely sequential loop --
# the prefix and the sequential flag must agree. Case/style is
# otherwise unconstrained: STADE identifies variables by symbol
# identity only, never by how the name is spelled.
function parse_check_loop_prefix(var::Symbol, sequential::Bool)
    has_prefix = startswith(String(var), "i_seq_")
    return has_prefix == sequential
end

# ---- function signature: name + positional arguments ----
# (type annotations are allowed but inert -- shapes come from usage,
# never from a declared type, so the annotation is stripped and dropped)

function parse_signature(sig_expr)
    sig_expr isa Expr && sig_expr.head == :call ||
        error("parse_kernel: function signature must be a plain `name(args...)` call -- no where-clauses or return-type annotations")
    name_expr = sig_expr.args[1]
    name_expr isa Symbol ||
        error("parse_kernel: function name must be a plain identifier")

    args = Symbol[]
    for a in sig_expr.args[2:end]
        if a isa Symbol
            push!(args, a)
        elseif a isa Expr && a.head == :parameters
            error("parse_kernel: keyword arguments aren't allowed (found a `;` section in the signature of :$(name_expr))")
        elseif a isa Expr && a.head == :(::)
            arg_name = a.args[1]
            arg_name isa Symbol ||
                error("parse_kernel: argument `$(a)` must annotate a plain identifier")
            push!(args, arg_name)
        elseif a isa Expr && a.head == :kw
            error("parse_kernel: argument `$(a)` has a default value -- every argument must be positional")
        else
            error("parse_kernel: unsupported argument form `$(a)` in the signature of :$(name_expr)")
        end
    end
    return name_expr, args
end

# ---- statement-list plumbing ----

# Drops LineNumberNodes. If the last statement is a bare `return nothing`
# (the only body-level return skill-stade allows), drops it too; any other
# `return` is a hard error. `nothing` in source parses to the Symbol
# :nothing, not the singleton, so both forms are checked -- a caller could
# hand parse_kernel a programmatically built Expr.
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
            error("parse_kernel: the only body-level `return` skill-stade allows is exactly `return nothing`")
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
        error("parse_kernel: `while` loops aren't supported yet (see skill-stade-dev.md)")
    elseif stmt.head in (:+=, :-=, :*=, :/=, :^=)
        return parse_assign(parse_desugar_compound(stmt))
    elseif stmt.head in (:÷=, :%=, Symbol("\\="), :.=)
        error("parse_kernel: `$(stmt)` isn't allowed -- `÷=`/`%=`/`\\=` have no registered operator rule and `.=` is broadcast assignment")
    else
        error("parse_kernel: unsupported statement form `Expr(:$(stmt.head), ...)`")
    end
end

# compound assignment (`+=`, `-=`, `*=`, `/=`, `^=`) desugars to the
# same thing as writing it out in full -- rewrite to a plain `:(=)`
# Expr with an explicit binary-op call and hand off to parse_assign,
# rather than teaching every downstream stage a second statement shape
function parse_desugar_compound(stmt::Expr)
    op = Symbol(String(stmt.head)[1:end-1])   # :+= -> :+, etc.
    lhs, rhs = stmt.args[1], stmt.args[2]
    return Expr(:(=), lhs, Expr(:call, op, lhs, rhs))
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
# skill-stade-dev.md rule 3.

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
# condition) and hard-error on anything skill-stade forbids.
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

# ==================== shape_* =================================
# Infers each variable's kind from usage, not a stripped type annotation.
# Array vs scalar: is it ever indexed? Int vs float: is there evidence of
# index/size use (loop var, range bound, div arg, subscript)? Anything not
# provably Int64 defaults to Float64; loop/size vars are the exception.

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

# ---- int evidence, pass 2: propagate through assignments ---------
# An assignment can't change kind, so evidence flows both ways: rhs
# int-ness forces lhs int (forward), and lhs int-ness forces whatever
# rhs operands it depends on int too (backward).

function shape_propagate_int!(body::Vector{NamedTuple}, is_int::Dict{Symbol,Bool})
    changed = false
    for stmt in body
        if stmt.kind == :assign && stmt.lhs isa Symbol
            if !is_int[stmt.lhs] && shape_expr_int_status(stmt.rhs, is_int) == :int
                is_int[stmt.lhs] = true
                changed = true
            end
            # backward: an lhs kind discovered from evidence elsewhere
            # forces every rhs operand that shape_expr_int_status
            # would need int to reach that same verdict (the forward
            # direction is already covered by the check above)
            if is_int[stmt.lhs]
                changed = shape_force_int_expr!(stmt.rhs, is_int) || changed
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

# operators whose result is Int64 exactly when every operand is --
# shared with shape_expr_int_status's forward reasoning so the two
# directions can't drift apart
function shape_int_preserving_ops()
    return Set{Symbol}([:+, :-, :*, :^, :abs, :sign, :max, :min, :mod, :floor, :ceil, :trunc])
end

# given expr is already known to evaluate to Int64, force every
# operand that fact depends on to Int64 too -- recurses through
# int-preserving ops (mirrors shape_expr_int_status in reverse);
# a bare copy is just the single-leaf case of this same rule
function shape_force_int_expr!(expr, is_int::Dict{Symbol,Bool})
    if expr isa Symbol
        (haskey(is_int, expr) && !is_int[expr]) || return false
        is_int[expr] = true
        return true
    elseif expr isa Expr && expr.head == :call && expr.args[1] in shape_int_preserving_ops()
        changed = false
        for a in expr.args[2:end]
            changed = shape_force_int_expr!(a, is_int) || changed
        end
        return changed
    elseif expr isa Expr && expr.head == :ref
        # An indexed array's element type follows from how the indexing
        # result is used, not just its own subscripts. If `X = A[idx...]`
        # and X is Int64, A must be Int64-elemented too -- this classifies
        # a gather/permutation-table array as array_int, not array_float.
        base = expr.args[1]
        (base isa Symbol && haskey(is_int, base) && !is_int[base]) || return false
        is_int[base] = true
        return true
    end
    return false
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
        elseif op in shape_int_preserving_ops()
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
# Derivative rule table, built fresh per call, never stored. Each rule pair comes
# from a per-operator local-partials list, one partial derivative per argument.
# tangent sums partials*dargs; adjoint returns one contribution per arg, for agen_
# to accumulate, simplifying *1/*0/+0.

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

# ---- local partials, one function per skill-stade-whitelisted op ------

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
# skill-stade kernels, so that branch is usually dropped upstream
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
# a.e. w.r.t. both args (skill-stade rule 12: div is for Int64
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

# Builds `for var = lo:hi ... end`, or `lo:step:hi` when step isn't 1 (e.g.
# a descending sweep). step is required, not optional: it mirrors a field
# the frozen :for shape always carries, so a caller pulling from a parsed
# statement never decides whether to pass it.
function emit_forloop(var::Symbol, lo, hi, step, body_exprs::Vector)
    range_expr = step == 1 ? Expr(:call, :(:), lo, hi) :
                              Expr(:call, :(:), lo, step, hi)
    return Expr(:for, Expr(:(=), var, range_expr), Expr(:block, body_exprs...))
end

# if cond ... else ... end -- omits the else clause entirely when
# else_exprs is empty (rather than emitting an empty `else end`
# block), matching skill-stade's "keep if/else to the strict minimum"
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

# `return nothing` when there is nothing to hand back; `return v` for one var,
# `return v1,v2,...` for several -- the only way a scalar argument's new value
# escapes the call (arrays mutate in place, no return needed). Shared by
# tgen_/agen_ for a reassigned scalar's shadow, each scalar-float arg's adjoint,
# and agen_'s initstacks_ stack(s).
function emit_return_scalars(vars::Vector{Symbol})
    isempty(vars) && return emit_return_nothing()
    length(vars) == 1 && return Expr(:return, vars[1])
    return Expr(:return, Expr(:tuple, vars...))
end

# Builds the skill-stade rule-14 #-comment header: a `# name(args...)` line, a
# blank #, the summary (possibly multi-line), a blank #, then one `# arg: doc`
# line per argument. `args` and `arg_docs` must be the same length and order.
# Returns a String (an Expr can't hold comments); prepend it at file-write time.
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
# Forward taint analysis from independents through assignments, swept to a fixed
# point. Whole-variable, monotonic: once reachable, a variable stays active past
# a later overwrite. A single pass misses taint a loop carries into its own
# earlier statements, so it reruns until stable (at most #variables+1 passes).

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
# carry a derivative at all (skill-stade rule 7).
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
# Push/pop site analysis (the TBR-equivalent): decides, per write, whether the reverse sweep needs a snapshot,
# numbered forward and popped in exact reverse. :array/:value marks a write whose old value is needed (see
# snap_value_needed_vars); :branch marks every `if`; :tripcount marks a reassignable loop bound. A write with no
# sequential-loop ancestor, one assign site, and no read-before-write needs no site.

function snap_plan(kernel, active_map; site_needed = nothing)
    reassigned = snap_collect_reassigned(kernel.body)
    value_needed = snap_value_needed_vars(kernel)
    assign_counts = snap_count_assign_sites(kernel.body)
    sites = NamedTuple[]
    counter = Ref(0)
    snap_walk!(kernel.body, active_map, kernel.sig.kinds, reassigned, value_needed, sites, counter,
               false, assign_counts, kernel.body, site_needed)
    return sites
end

# Does `var`'s VALUE (not just its syntactic presence) feed some local partial derivative reachable
# from `expr`'s root? Top-down, mirroring ADTBRAnalyzer.collectZonesUsedByDiffRhs: `needed` starts
# false and stays false through nested +/- (a constant partial never needs an operand's value), so a
# copy or sum/difference chain marks nothing needed. Any other call is nonlinear: `needed` flips true
# and stays true for everything inside it.
function snap_var_value_needed!(expr, acc, needed)
    if expr isa Expr && expr.head == :call
        op = expr.args[1]
        args = expr.args[2:end]
        child_needed = (op == :+ || op == :-) ? needed : true
        for a in args
            snap_var_value_needed!(a, acc, child_needed)
        end
    elseif expr isa Expr && expr.head == :ref
        needed && push!(acc, expr.args[1])
        for a in expr.args[2:end]
            snap_var_value_needed!(a, acc, needed)
        end
    elseif expr isa Symbol
        needed && push!(acc, expr)
    end
    return nothing
end

# Every variable whose value is needed somewhere in the kernel: the union, over
# every :assign rhs (from `needed=false`) and every :if condition (from
# `needed=true`, deliberately not refined -- the reverse sweep never re-
# evaluates a condition), of snap_var_value_needed!'s result.
function snap_value_needed_vars(kernel)
    acc = Set{Symbol}()
    snap_collect_value_needed!(kernel.body, acc)
    return acc
end

function snap_collect_value_needed!(body, acc)
    for stmt in body
        if stmt.kind == :assign
            snap_var_value_needed!(stmt.rhs, acc, false)
        elseif stmt.kind == :for
            snap_collect_value_needed!(stmt.body, acc)
        elseif stmt.kind == :if
            snap_var_value_needed!(stmt.cond, acc, true)
            snap_collect_value_needed!(stmt.then, acc)
            snap_collect_value_needed!(stmt.els, acc)
        end
    end
    return nothing
end

# every var's source-level assign-site count, anywhere in the kernel
# (how many :assign STATEMENTS write it, not how many times any of
# them executes at runtime) -- used by the iteration-independent
# snapshot elision below
function snap_count_assign_sites(body)
    counts = Dict{Symbol,Int}()
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            counts[var] = get(counts, var, 0) + 1
        elseif stmt.kind == :for
            for (k, v) in snap_count_assign_sites(stmt.body)
                counts[k] = get(counts, k, 0) + v
            end
        elseif stmt.kind == :if
            for (k, v) in snap_count_assign_sites(stmt.then)
                counts[k] = get(counts, k, 0) + v
            end
            for (k, v) in snap_count_assign_sites(stmt.els)
                counts[k] = get(counts, k, 0) + v
            end
        end
    end
    return counts
end

# does `var` get read anywhere in body strictly before reaching
# statement `target` (forward textual order, at any nesting depth)?
# Returns (found_read, reached_target) so a caller partway through a
# nested block knows whether to keep scanning later siblings.
function snap_read_before_walk(body, target, var)
    for stmt in body
        stmt === target && return (false, true)
        if stmt.kind == :assign
            snap_count_var_refs(stmt.rhs, var) > 0 && return (true, true)
        elseif stmt.kind == :for
            (found, reached) = snap_read_before_walk(stmt.body, target, var)
            (found || reached) && return (found, true)
        elseif stmt.kind == :if
            snap_count_var_refs(stmt.cond, var) > 0 && return (true, true)
            (found_t, reached_t) = snap_read_before_walk(stmt.then, target, var)
            (found_t || reached_t) && return (found_t, true)
            (found_e, reached_e) = snap_read_before_walk(stmt.els, target, var)
            (found_e || reached_e) && return (found_e, true)
        end
    end
    return (false, false)
end

snap_read_before(body, target, var) = snap_read_before_walk(body, target, var)[1]

# Every variable assigned via a scalar :assign somewhere inside a loop (any depth), gated on
# in_loop since only a write that can execute more than once can leave a different value at
# different points. A var assigned only at top level is a one-shot constant and can never
# need trip-count-style snapshot/restore -- flagging it would wrongly force a push/pop that
# corrupts the loop's own value (see keep_push_pop history).
function snap_collect_reassigned(body, in_loop = false)
    reassigned = Set{Symbol}()
    for stmt in body
        if stmt.kind == :assign
            in_loop && stmt.lhs isa Symbol && push!(reassigned, stmt.lhs)
        elseif stmt.kind == :for
            union!(reassigned, snap_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :if
            union!(reassigned, snap_collect_reassigned(stmt.then, in_loop))
            union!(reassigned, snap_collect_reassigned(stmt.els, in_loop))
        end
    end
    return reassigned
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

# One forward-order pass, emitting sites as found. A write needs a site whenever `var` is
# read anywhere else in the kernel, regardless of loop nesting: what matters is only whether
# some other read could see a different value once forward and backward walk it in opposite
# orders. Over-snapshotting costs a harmless extra push/pop pair, never a correctness bug --
# push and pop always occupy the mirrored position, so nesting stays self-consistent.
function snap_walk!(body, active_map, kinds, reassigned, value_needed, sites, counter, in_loop, assign_counts, full_body, site_needed = nothing)
    for idx in eachindex(body)
        stmt = body[idx]
        if stmt.kind == :assign
            snap_check_assign!(stmt, active_map, kinds, value_needed, sites, counter, in_loop, assign_counts, full_body, body, idx, site_needed)
        elseif stmt.kind == :for
            snap_check_tripcount!(stmt, reassigned, sites, counter)
            snap_walk!(stmt.body, active_map, kinds, reassigned, value_needed, sites, counter,
                       true, assign_counts, full_body, site_needed)
        elseif stmt.kind == :if
            counter[] = counter[] + 1
            push!(sites, (kind = :branch, array = :cond, at = counter[]))
            snap_walk!(stmt.then, active_map, kinds, reassigned, value_needed, sites, counter, in_loop, assign_counts, full_body, site_needed)
            snap_walk!(stmt.els, active_map, kinds, reassigned, value_needed, sites, counter, in_loop, assign_counts, full_body, site_needed)
        end
    end
    return nothing
end

# A write needs a site iff `var in value_needed` (see snap_value_needed_vars), subsuming the
# old self-reference and cross-statement rules in one test. Also gated by iteration-
# independent elision below: sole write, no enclosing loop at all (any :for, sequential or
# not, still re-executes and needs restoring), no self-reference, never read before it runs.
function snap_check_assign!(stmt, active_map, kinds, value_needed, sites, counter, in_loop, assign_counts, full_body, body = nothing, idx = nothing, site_needed = nothing)
    var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
    if site_needed !== nothing
        # site-level TBR path: consult the precomputed per-site Dict
        # (snap_value_needed_sites) instead of the whole-variable test
        # -- already folds in the active_map gate, so no separate
        # check needed here
        needed = get(site_needed, agen_site_key(body, idx), false)
        kind = kinds[var] == :array_float ? :array : :value
    else
        _, needed, kind = snap_assign_site_decision(stmt, active_map, kinds, value_needed, in_loop, assign_counts, full_body)
    end
    needed || return nothing
    counter[] = counter[] + 1
    push!(sites, (kind = kind, array = var, at = counter[]))
    return nothing
end

# the array/value site decision at a single :assign statement, factored
# out of snap_check_assign! so the identical predicate can be reused by
# comparison/instrumentation code (e.g. the site-level shadow analysis)
# without duplicating the logic. Behavior-preserving refactor -- callers
# of snap_check_assign! see no change.
function snap_assign_site_decision(stmt, active_map, kinds, value_needed, in_loop, assign_counts, full_body)
    var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
    active_map[var] || return (var, false, nothing)
    var in value_needed || return (var, false, nothing)
    self_ref = snap_count_var_refs(stmt.rhs, var) > 0
    if !self_ref && !in_loop && get(assign_counts, var, 0) == 1 && !snap_read_before(full_body, stmt, var)
        return (var, false, nothing)
    end
    kind = kinds[var] == :array_float ? :array : :value
    return (var, true, kind)
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

# Is `lhs = rhs` an identity-preserving self-update (d(new)/d(old)=1), so old-lhs and new-
# lhs are the same quantity for the shadow's purposes (agen_backward_assign can skip
# resetting it to 0)? This is unrelated to the snapshot decision, which is
# snap_value_needed_vars's job. Two shapes qualify: lhs as a top-level `+` operand, or as
# the minuend of a binary `-`; lhs's exact slot must not appear anywhere else in rhs.
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

# ---- site-level (forward-seen) TBR: shadow analysis ----
# `snap_value_needed_vars` decides whole-variable necessity; `snap_fwd_walk!` refines to a per-write
# decision: w needs its pre-write value iff (a) w self-references, or (b) a nonlinear read of var
# occurred strictly before w in forward order. Shadow-only; cross-checked against snap_plan (must be a
# subset), not yet wired into codegen.
function snap_fwd_walk!(body, seen, active_map, decisions)
    for idx in eachindex(body)
        stmt = body[idx]
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            # local_reads: vars read nonlinearly within this rhs alone
            # (needed=false at the root). Self-reference only forces a
            # snapshot when var's own occurrence is one of these, not merely
            # present under a linear +/- (pure accumulation needs no old
            # value: d(new)/d(old)=1).
            local_reads = Set{Symbol}()
            snap_var_value_needed!(stmt.rhs, local_reads, false)
            decisions[agen_site_key(body, idx)] = active_map[var] && (var in local_reads || var in seen)
            seen = union(seen, local_reads)
        elseif stmt.kind == :if
            snap_var_value_needed!(stmt.cond, seen, true)   # cond always conservative
            seen_then = snap_fwd_walk!(stmt.then, copy(seen), active_map, decisions)
            seen_els  = snap_fwd_walk!(stmt.els,  copy(seen), active_map, decisions)
            seen = union(seen_then, seen_els)
        elseif stmt.kind == :for
            seen = snap_fwd_walk_loop!(stmt.body, seen, active_map, decisions)
        end
    end
    return seen
end

# Loop forward fixed point: a read near the top of a loop body is, for every iteration but
# the first, preceded by the previous iteration's later reads -- so a write near the top can
# need a snapshot due to a read near the bottom, one iteration back. `seen` only grows
# across passes, so this terminates; a final pass re-walks with the converged set so
# `decisions` reflects the fixed point.
function snap_fwd_walk_loop!(body, in_seen, active_map, decisions)
    seen = copy(in_seen)
    while true
        scratch = Dict{Any,Bool}()
        out_seen = snap_fwd_walk!(body, union(seen, in_seen), active_map, scratch)
        out_seen == seen && break
        seen = out_seen
    end
    final_in = union(seen, in_seen)
    return snap_fwd_walk!(body, final_in, active_map, decisions)
end

# public entry point for the shadow analysis: per-site (not
# per-variable) TBR decisions for every :assign in `kernel`, keyed by
# agen_site_key(body, idx). Not yet consumed by snap_plan or
# agen_needs_snapshot.
function snap_value_needed_sites(kernel)
    decisions = Dict{Any,Bool}()
    snap_fwd_walk!(kernel.body, Set{Symbol}(), act_analyze(kernel), decisions)
    return decisions
end


# ==================== lin_* =====================================
# Shared derivative-tree representation, swept differently by each codegen direction, built with
# der_partials. `lin_node` mirrors a primal sub-expression: `:leaf` (var/literal read) or `:op`
# (rebuilt expr, its `partials`, one child lin_node per arg), each carrying `active::Bool`. `lin_stmt`
# parallels the frozen `statement` shape; only `:assign` gains a built `tree`.

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
# Forward-mode codegen: single sweep, original order, no snapshot stacks. Every active statement gets
# a shadow ("d"-suffixed) line before its primal line, from current pre-statement values -- always
# safe, since a tangent never depends on its own lhs's new value. Emitted even when it collapses to
# 0.0, so a later active read sees the reset, not a stale value.

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
# Reverse-mode codegen: forward sweep with pushes, reversed backward sweep with pops, plus initstacks_*. Forward
# sweep replays the primal, pushing before any write that needs it. Backward sweep reverses statement order (and a
# sequential loop's direction), distributing each seed via der_rule(op).adjoint; pure accumulation skips resetting
# the shadow, else it resets to 0.0. A snapshotted write pops its old value first.

function agen_emit(kernel, lin_plan, snapshot_plan; keep_push_pop::Bool = true, push_pop = nothing, ii_plan = nothing)
    active_map = act_analyze(kernel)
    layout = nothing
    value_needed = exempt = stacks = nothing
    if !keep_push_pop
        # Tier B kernels (a ragged/data-dependent loop bound) are no longer refused: agen_layout resolves as
        # much as possible into closed-form ragged-block tables, falling back to plain :stack semantics only
        # for what a block can't resolve. NOTE: ii_plan/fuse_ii_loops is untested with keep_push_pop=false --
        # a fused var's stack is still allocated as if unfused, harmlessly unused; treat this as unsupported
        # until exercised.
        value_needed = agen_value_needed_vars(kernel)
        reassigned = agen_collect_reassigned(kernel.body)
        exempt = agen_exempt_vars(kernel, value_needed)
        stacks = agen_stack_map(snapshot_plan)
        layout = agen_layout(kernel, kernel.sig.kinds, active_map, value_needed, reassigned, exempt, stacks; push_pop = push_pop)
    end
    (initstacks_expr, table_names, tot_names, val_names) = agen_init_emit(kernel, snapshot_plan; keep_push_pop = keep_push_pop, layout = layout,
                                      active_map = active_map, value_needed = value_needed, exempt = exempt, stacks = stacks, push_pop = push_pop)
    tier_b_extra_args = vcat(table_names, tot_names, val_names)
    adjoint_expr = agen_adjoint_emit(kernel, active_map, lin_plan, snapshot_plan; keep_push_pop = keep_push_pop, layout = layout, push_pop = push_pop,
                                      tier_b_extra_args = tier_b_extra_args, ii_plan = ii_plan)
    # fuse_ii_loops can leave a stack with zero remaining push!/pop! calls anywhere in the
    # generated body (every write-site that would have used it got fused away) -- checked from
    # the actual generated code, never predicted ahead of time, since a var's ii_plan coverage
    # does NOT by itself guarantee its stack is fully unused. Scoped to keep_push_pop=true,
    # matching fuse_ii_loops's existing scope.
    if keep_push_pop && ii_plan !== nothing
        used = agen_used_stack_names(adjoint_expr)
        unused_stacks = Set(nm for nm in agen_stack_names(snapshot_plan) if !(nm in used))
        if !isempty(unused_stacks)
            adjoint_expr = agen_drop_unused_stack_args(adjoint_expr, unused_stacks)
            initstacks_expr = agen_drop_unused_stack_allocs(initstacks_expr, unused_stacks)
        end
    end
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

function agen_adjoint_emit(kernel, active_map, lin_plan, sites; keep_push_pop::Bool = true, layout = nothing, push_pop = nothing,
                            tier_b_extra_args::Vector{Symbol} = Symbol[], ii_plan = nothing)
    fname = agen_fname(kernel.sig.name)
    stacks = agen_stack_map(sites)
    # `tier_b_extra_args` (Phase D) is the same table/total/value-table name list agen_init_emit
    # returned, appended in the same order so it lines up with initstacks_*'s return tuple,
    # keeping val_init_stacks' splat-the-whole-tuple convention working. Empty under
    # keep_push_pop=true or a kernel with no ragged block.
    fargs = vcat(agen_signature_args(kernel.sig), agen_stack_names(sites), tier_b_extra_args)

    value_needed = agen_value_needed_vars(kernel)
    reassigned = agen_collect_reassigned(kernel.body)
    unsafe = agen_unsafe_int_vars(kernel)
    exempt = agen_exempt_vars(kernel, value_needed)

    ectx = (keep_push_pop = keep_push_pop, loop_ctx = Any[], layout = layout, push_pop = push_pop, ii_plan = ii_plan)

    body = Any[]
    append!(body, agen_local_primal_inits(kernel, active_map))
    append!(body, agen_local_shadow_inits(kernel, active_map))
    append!(body, agen_forward_body(kernel.body, kernel.sig.kinds, active_map, value_needed, reassigned, stacks, exempt; ectx = ectx, lin_body = lin_plan, unsafe = unsafe))
    append!(body, agen_backward_body(lin_plan, kernel.body, kernel.sig.kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = ectx))

    scalar_args = [a for a in kernel.sig.args if kernel.sig.kinds[a] == :scalar_float]
    push!(body, emit_return_scalars([agen_shadow(a) for a in scalar_args]))

    return Expr(:function, Expr(:call, fname, fargs...), Expr(:block, body...))
end

# Int64 variables reassigned at more than one :assign site are evolving state depending on the full
# control-flow history -- a Julia `for` loop assigning an outer variable leaks past it, so recomputing
# from a fixed start is wrong. Anything depending on such a variable is excluded too. The fix: never
# hoist/recompute this set; loop bounds are already restored via :tripcount. The set is seeded by
# self-reference, not mere multiplicity, so a fresh per-loop variable stays hoistable.
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

# One `nm = Vector{T}(...)` init statement: growable for `keep_push_pop` or a tainted (Tier
# B fallback) stack; presized from `layout.sizes` for a pure Tier A stack, or from
# `layout.block_totals` for a Tier B ragged-block table (see agen_layout's docs). Factored
# out of agen_init_emit to keep that function's comprehension a plain one-call-per-element
# form.
function agen_init_alloc_stmt(nm, kind, keep_push_pop::Bool, layout)
    tainted = layout !== nothing && nm in layout.tainted_stacks
    grow = keep_push_pop || tainted
    size_expr = grow ? nothing : (haskey(layout.sizes, nm) ? layout.sizes[nm] : layout.block_totals[nm])
    return Expr(:(=), nm, agen_stack_alloc_expr(kind, grow, size_expr))
end

# Returns `(expr, table_names, tot_names, val_names)`: empty unless there is a ragged block, else the extra
# per-stack tables agen_tier_b_kernel_skeleton builds, appended in lockstep to initstacks_*'s return and
# `<name>_b`/`<name>_hv`'s args. A stack-size formula can reference a scalar derived in the body (e.g. `n_d
# = n*d`); such assignments are hoisted, reporting the kernel arguments they need. Tier B locals are never
# hoisted; a name assigned twice is skipped, leaving the original UndefVarError rather than a wrong size.
function agen_collect_assigned_syms!(e, acc)
    if e isa Expr
        e.head == :(=) && e.args[1] isa Symbol && push!(acc, e.args[1])
        for a in e.args
            agen_collect_assigned_syms!(a, acc)
        end
    end
    return nothing
end

function agen_init_derived_stmts(kernel, wanted, already_defined)
    defs = Dict{Symbol,Any}()
    counts = agen_count_assign_sites(kernel.body)
    for stmt in kernel.body
        stmt.kind == :assign || continue
        stmt.lhs isa Symbol || continue
        get(counts, stmt.lhs, 0) == 1 || continue
        haskey(defs, stmt.lhs) || (defs[stmt.lhs] = stmt.rhs)
    end
    args = Set{Symbol}(kernel.sig.args)
    order = Symbol[]
    seen = Set{Symbol}()
    extra = Set{Symbol}()
    function visit(v)
        (v in seen || v in args || v in already_defined) && return
        push!(seen, v)
        haskey(defs, v) || return
        rd = Set{Symbol}()
        agen_collect_expr_vars!(defs[v], rd)
        for r in rd
            visit(r)
            r in args && push!(extra, r)
        end
        push!(order, v)
    end
    for v in wanted
        visit(v)
    end
    return (Any[Expr(:(=), v, defs[v]) for v in order], extra)
end

function agen_init_emit(kernel, sites; keep_push_pop::Bool = true, layout = nothing,
                         active_map = nothing, value_needed = nothing, exempt = nothing, stacks = nothing, push_pop = nothing)
    fname = agen_init_fname(kernel.sig.name)
    names = agen_stack_names(sites)
    kind_of = Dict{Symbol,Symbol}()
    for s in sites
        kind_of[agen_site_stack_name(s)] = s.kind
    end
    # under keep_push_pop=false, `initstacks_*`'s signature grows to
    # accept the minimal set of kernel arguments any stack's size
    # expression actually references -- see skill-stade-dev.md's
    # keep_push_pop entry for why the minimal set (rather than the
    # kernel's full argument list) was chosen
    fargs = keep_push_pop ? Symbol[] : layout.free_vars
    sizing_stmts = Any[]
    total_of = Dict{Symbol,Symbol}()
    if !keep_push_pop && !isempty(layout.tainted_stacks)
        (sizing_stmts, total_of) = agen_tier_b_sizing_stmts(kernel, active_map, value_needed, exempt, stacks, layout.tainted_stacks; push_pop = push_pop)
        # The sizing skeleton may reference kernel arguments the Tier A size formulas alone never
        # would (e.g. a multigrid smoother's iteration-count arguments). Widen the signature to
        # match, using the same free-vars-of-what-we-actually-reference principle as the Tier A case
        # above.
        sizing_free = Set{Symbol}()
        for s in sizing_stmts
            agen_collect_expr_vars!(s, sizing_free)
        end
        fargs = sort(union(fargs, intersect(sizing_free, kernel.sig.args)); by = string)
    end
    # Tier B ragged-block tables (Phase D): built once here, like the fallback sizing pass
    # above, but for stacks resolved into one or more agen_layout blocks instead of falling back
    # entirely. agen_tier_b_kernel_skeleton already interleaves each block's table-construction
    # loop with the surrounding scalar prelude (Phase B); nothing more is needed beyond widening
    # the signature.
    table_stmts = Any[]
    table_names = Symbol[]; tot_names = Symbol[]; val_names = Symbol[]
    if !keep_push_pop && !isempty(layout.blocks)
        table_stmts = agen_tier_b_kernel_skeleton(kernel.body, kernel.sig.kinds, layout)
        for bid in sort(collect(keys(layout.blocks)))
            blk = layout.blocks[bid]
            for s in sort(collect(keys(blk.local_sizes)); by = string)
                push!(table_names, Symbol("prefix_" * string(s) * "_" * string(bid)))
                push!(tot_names, blk.total_sym[s])
            end
            for v in sort(collect(blk.value_vars); by = string)
                push!(val_names, Symbol("val_" * string(v) * "_" * string(bid)))
            end
        end
        fargs = sort(union(fargs, Set(agen_tier_b_skeleton_free_vars(table_stmts, kernel.sig.args))); by = string)
    end
    # Tier B: a tainted stack (see agen_use_stack_push) has no size formula --
    # layout.sizes/layout.block_totals deliberately omit it -- so it always allocates growable,
    # like keep_push_pop's true-case, regardless of the kernel-wide flag. A computed `__sz_*`
    # total additionally gets a sizehint! right after, avoiding push!'s repeated reallocation.
    alloc_stmts = Any[]
    for nm in names
        push!(alloc_stmts, agen_init_alloc_stmt(nm, kind_of[nm], keep_push_pop, layout))
        haskey(total_of, nm) && push!(alloc_stmts, Expr(:call, :sizehint!, nm, total_of[nm]))
    end
    # names the Tier B passes define for themselves must not be hoisted
    derived_stmts = Any[]
    if !keep_push_pop && !isempty(layout.derived_vars)
        already = Set{Symbol}()
        for st in vcat(sizing_stmts, table_stmts)
            st isa Expr && st.head == :(=) && st.args[1] isa Symbol && push!(already, st.args[1])
            st isa Expr && st.head == :for && agen_collect_assigned_syms!(st, already)
        end
        (derived_stmts, extra) = agen_init_derived_stmts(kernel, layout.derived_vars, already)
        fargs = sort(union(fargs, extra); by = string)
    end
    body = vcat(derived_stmts, sizing_stmts, table_stmts, alloc_stmts)
    push!(body, emit_return_scalars(vcat(names, table_names, tot_names, val_names)))
    return (Expr(:function, Expr(:call, fname, fargs...), Expr(:block, body...)), table_names, tot_names, val_names)
end

# :array/:value stacks hold the popped Float64 scalar itself (every push is one exact lhs
# reference, never a whole-array copy); :branch/:tripcount stacks hold Int64 flags/bounds.
# Under keep_push_pop=false, `size_expr` sizes the allocation up front instead of growing
# via push! -- either way every stack holds the same element type.
function agen_stack_alloc_expr(kind, keep_push_pop::Bool = true, size_expr = nothing)
    T = kind in (:array, :value) ? :Float64 : :Int64
    keep_push_pop && return Expr(:call, Expr(:curly, :Vector, T))
    return Expr(:call, Expr(:curly, :Vector, T), :undef, size_expr)
end

# ---- TBR predicate, duplicated (agen_-prefixed) from snap_*'s own
#      logic rather than calling it directly -- skill-stade's purity
#      rule only allows relying on another stage's *documented*
#      input/output shape (here: the sites list itself), not
#      reaching into its private helpers ------------------------------

# gated on in_loop -- see snap_collect_reassigned's comment; kept as
# a separate agen_-prefixed duplicate for the same reason every other
# agen_/snap_ pair in this file is duplicated rather than shared.
function agen_collect_reassigned(body, in_loop = false)
    reassigned = Set{Symbol}()
    for stmt in body
        if stmt.kind == :assign
            in_loop && stmt.lhs isa Symbol && push!(reassigned, stmt.lhs)
        elseif stmt.kind == :for
            union!(reassigned, agen_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :if
            union!(reassigned, agen_collect_reassigned(stmt.then, in_loop))
            union!(reassigned, agen_collect_reassigned(stmt.els, in_loop))
        end
    end
    return reassigned
end

# ---- block-boundary scalar restoration ----
# A scalar var written only inside a nested sub-:for/:if of `body` has no restore reaching a later sibling
# statement's read in the same `body` when the enclosing loop repeats. Per-write push/pop restores a var to its pre-
# write value -- right for a read within the same loop that writes it, wrong for a sibling loop's read, since nothing
# re-establishes the post-loop value first. `agen_nested_write_vars` finds these candidates.
function agen_nested_write_vars(body, kinds)
    vars = Set{Symbol}()
    for stmt in body
        if stmt.kind == :for
            union!(vars, agen_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :if
            # This function's concern (a var written only inside a sub-
            # loop/sub-if) is unrelated to agen_collect_reassigned's in_loop
            # gating, which restricts tripcount-snapshot candidates to vars
            # that vary across a loop's iterations. Force in_loop=true here so
            # this call keeps collecting every assign unconditionally.
            union!(vars, agen_collect_reassigned(stmt.then, true))
            union!(vars, agen_collect_reassigned(stmt.els, true))
        end
    end
    return Set(v for v in vars if kinds[v] == :scalar_float)
end

# True iff every assignment to `var` within `body` (recursively) sits inside an ii_plan-
# covered (:independent or :reduction) loop. Once inside such a loop, everything nested
# further inside it counts as covered too, matching how ii_plan's own classification already
# covers a whole stmt.body recursively -- no separate re-proof needed for nested :for/:if
# structure.
function agen_ii_covered_write_check(body, var, ii_plan, in_covered)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            if stmt.lhs isa Symbol && stmt.lhs == var
                in_covered || return false
            end
        elseif stmt.kind == :for
            key = agen_site_key(body, idx)
            covered_here = in_covered || get(ii_plan, key, nothing) in (:independent, :reduction, :mixed, :recompute)
            agen_ii_covered_write_check(stmt.body, var, ii_plan, covered_here) || return false
        elseif stmt.kind == :if
            agen_ii_covered_write_check(stmt.then, var, ii_plan, in_covered) || return false
            agen_ii_covered_write_check(stmt.els, var, ii_plan, in_covered) || return false
        end
    end
    return true
end

# The subset needing an extra push (end of `body`, forward) and matching pop (start of `body`'s backward): value-
# needed, not exempt, with an allocated (:value, var) stack, sorted for deterministic order. `ii_plan` (nothing by
# default) excludes a var whenever agen_ii_covered_write_check proves every write-site is covered -- required, since
# this function's own criterion has no knowledge of ii_plan's stricter proof. A var with only some write-sites
# covered still correctly stays a candidate.
function agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks; ii_plan = nothing)
    cand = agen_nested_write_vars(body, kinds)
    return sort(collect(v for v in cand if v in value_needed && !(v in exempt) && haskey(stacks, (:value, v)) &&
                          !(ii_plan !== nothing && agen_ii_covered_write_check(body, v, ii_plan, false))); by = string)
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
# agen_-prefixed pair in this file. Feeds only the shadow-reset
# decision now; see agen_value_needed_vars for the snapshot decision.
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

# see snap_var_value_needed!'s comment -- identical logic, duplicated
# here for the same purity-rule reason as every other agen_-prefixed
# pair in this file.
function agen_var_value_needed!(expr, acc, needed)
    if expr isa Expr && expr.head == :call
        op = expr.args[1]
        args = expr.args[2:end]
        child_needed = (op == :+ || op == :-) ? needed : true
        for a in args
            agen_var_value_needed!(a, acc, child_needed)
        end
    elseif expr isa Expr && expr.head == :ref
        needed && push!(acc, expr.args[1])
        for a in expr.args[2:end]
            agen_var_value_needed!(a, acc, needed)
        end
    elseif expr isa Symbol
        needed && push!(acc, expr)
    end
    return nothing
end

function agen_value_needed_vars(kernel)
    acc = Set{Symbol}()
    agen_collect_value_needed!(kernel.body, acc)
    return acc
end

function agen_collect_value_needed!(body, acc)
    for stmt in body
        if stmt.kind == :assign
            agen_var_value_needed!(stmt.rhs, acc, false)
        elseif stmt.kind == :for
            agen_collect_value_needed!(stmt.body, acc)
        elseif stmt.kind == :if
            agen_var_value_needed!(stmt.cond, acc, true)
            agen_collect_value_needed!(stmt.then, acc)
            agen_collect_value_needed!(stmt.els, acc)
        end
    end
    return nothing
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

# A write to `var` needs a push/pop-restore whenever `var in value_needed` (see
# agen_value_needed_vars); this already subsumes self-reference, no separate check needed.
# Polymorphic on `value_needed`'s type: a `Set{Symbol}` (default) behaves as before,
# ignoring `key`; a `Dict{Any,Bool}` (site-level TBR, keyed by agen_site_key) looks up this
# statement's own decision instead.
function agen_needs_snapshot(lhs, rhs, var, value_needed, key = nothing)
    value_needed isa AbstractDict && return get(value_needed, key, false)
    return var in value_needed
end

# Identical logic to snap_fwd_walk!, duplicated here for the same purity-rule reason as
# every agen_/snap_ pair in this file. Shadow analysis only; must be checked against
# snap_value_needed_sites for exact Dict equality (not just subset) before wiring into
# codegen -- forward push and backward pop must decide identically at every site or push/pop
# counts desync.
function agen_fwd_walk!(body, seen, active_map, decisions)
    for idx in eachindex(body)
        stmt = body[idx]
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            local_reads = Set{Symbol}()
            agen_var_value_needed!(stmt.rhs, local_reads, false)
            decisions[agen_site_key(body, idx)] = active_map[var] && (var in local_reads || var in seen)
            seen = union(seen, local_reads)
        elseif stmt.kind == :if
            agen_var_value_needed!(stmt.cond, seen, true)
            seen_then = agen_fwd_walk!(stmt.then, copy(seen), active_map, decisions)
            seen_els  = agen_fwd_walk!(stmt.els,  copy(seen), active_map, decisions)
            seen = union(seen_then, seen_els)
        elseif stmt.kind == :for
            seen = agen_fwd_walk_loop!(stmt.body, seen, active_map, decisions)
        end
    end
    return seen
end

function agen_fwd_walk_loop!(body, in_seen, active_map, decisions)
    seen = copy(in_seen)
    while true
        scratch = Dict{Any,Bool}()
        out_seen = agen_fwd_walk!(body, union(seen, in_seen), active_map, scratch)
        out_seen == seen && break
        seen = out_seen
    end
    final_in = union(seen, in_seen)
    return agen_fwd_walk!(body, final_in, active_map, decisions)
end

function agen_value_needed_sites(kernel)
    decisions = Dict{Any,Bool}()
    agen_fwd_walk!(kernel.body, Set{Symbol}(), act_analyze(kernel), decisions)
    return decisions
end


# bound-variables of a :for statement that are reassigned somewhere
# else in the kernel -- the same set snap_plan's :tripcount sites are
# keyed on
function agen_tripcount_bound_vars(stmt, reassigned)
    bound_vars = agen_for_bound_vars(stmt)
    return [bv for bv in bound_vars if bv in reassigned]
end

# A :for statement's own lo/hi/step free variables, factored out so agen_tier_b_walk's
# detection and agen_layout_walk!'s taint-marking can never drift apart on what counts as
# this loop's bound variables -- they must agree exactly, since taint-marking implements
# Tier B support for loops agen_tier_b_offender would otherwise refuse.
function agen_for_bound_vars(stmt)
    bound_vars = Set{Symbol}()
    agen_collect_expr_vars!(stmt.lo, bound_vars)
    agen_collect_expr_vars!(stmt.hi, bound_vars)
    agen_collect_expr_vars!(stmt.step, bound_vars)
    return bound_vars
end

# True if `expr` contains a ref (`arr[...]`) to an array-kinded var, used by the Tier B
# sizing pass to decide if a scalar assign is safe to replicate into a data-free skeleton:
# an array-free RHS is exactly the set of assigns that can matter to a loop bound or branch
# condition downstream, since skill-stade never lets a bound/condition reference an array
# directly.
function agen_expr_reads_array(expr, kinds)
    if expr isa Expr
        expr.head == :ref && get(kinds, expr.args[1], nothing) in (:array_float, :array_int) && return true
        return any(a -> agen_expr_reads_array(a, kinds), expr.args)
    end
    return false
end

function agen_negate_step(step)
    step isa Number && return -step
    return Expr(:call, :-, step)
end

# ---- iteration-independent snapshot elision, duplicated
#      (agen_-prefixed) from snap_*'s own logic for the same purity-
#      rule reason as every other agen_-prefixed pair in this file --
#      see snap_check_assign!'s comment for the correctness argument.

function agen_count_assign_sites(body)
    counts = Dict{Symbol,Int}()
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            counts[var] = get(counts, var, 0) + 1
        elseif stmt.kind == :for
            for (k, v) in agen_count_assign_sites(stmt.body)
                counts[k] = get(counts, k, 0) + v
            end
        elseif stmt.kind == :if
            for (k, v) in agen_count_assign_sites(stmt.then)
                counts[k] = get(counts, k, 0) + v
            end
            for (k, v) in agen_count_assign_sites(stmt.els)
                counts[k] = get(counts, k, 0) + v
            end
        end
    end
    return counts
end

function agen_read_before_walk(body, target, var)
    for stmt in body
        stmt === target && return (false, true)
        if stmt.kind == :assign
            agen_count_var_refs(stmt.rhs, var) > 0 && return (true, true)
        elseif stmt.kind == :for
            (found, reached) = agen_read_before_walk(stmt.body, target, var)
            (found || reached) && return (found, true)
        elseif stmt.kind == :if
            agen_count_var_refs(stmt.cond, var) > 0 && return (true, true)
            (found_t, reached_t) = agen_read_before_walk(stmt.then, target, var)
            (found_t || reached_t) && return (found_t, true)
            (found_e, reached_e) = agen_read_before_walk(stmt.els, target, var)
            (found_e || reached_e) && return (found_e, true)
        end
    end
    return (false, false)
end

agen_read_before(body, target, var) = agen_read_before_walk(body, target, var)[1]

# Vars whose sole write anywhere in the kernel qualifies for the elision (mirrors
# snap_check_assign!'s per-statement test, collected as a Set{Symbol} instead of gating site
# creation directly). `in_loop` must be true beneath any :for ancestor, not just a
# sequential one -- see snap_check_assign!'s comment.
function agen_collect_exempt_vars!(body, value_needed, assign_counts, full_body, in_loop, exempt)
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            self_ref = agen_count_var_refs(stmt.rhs, var) > 0
            if !self_ref && var in value_needed && !in_loop && get(assign_counts, var, 0) == 1 &&
               !agen_read_before(full_body, stmt, var)
                push!(exempt, var)
            end
        elseif stmt.kind == :for
            agen_collect_exempt_vars!(stmt.body, value_needed, assign_counts, full_body, true, exempt)
        elseif stmt.kind == :if
            agen_collect_exempt_vars!(stmt.then, value_needed, assign_counts, full_body, in_loop, exempt)
            agen_collect_exempt_vars!(stmt.els, value_needed, assign_counts, full_body, in_loop, exempt)
        end
    end
    return nothing
end

function agen_exempt_vars(kernel, value_needed)
    assign_counts = agen_count_assign_sites(kernel.body)
    exempt = Set{Symbol}()
    agen_collect_exempt_vars!(kernel.body, value_needed, assign_counts, kernel.body, false, exempt)
    return exempt
end

# ---- keep_push_pop=false: Tier A/B sizing + :indexed emission ----
# Every snapshot site gets a runtime index (`base_offset + local_position`) into its stack, computed by agen_indexed_layout
# with the same traversal agen_forward_body's pushes use, keyed structurally since branch-scalar hoisting reorders backward
# emission. Tier A: closed-form trip counts, single offset/size formulas. Tier B: a loop whose bound is reassigned in an
# ancestor sequential loop taints its stack, falling back to growable push!/pop! regardless of keep_push_pop.

# a snapshot site's unique identity for :indexed offset lookup -- see
# the section comment above for why this is a key, not a count
agen_site_key(body::Vector, idx::Int) = (objectid(body), idx, nothing)
agen_site_key(body::Vector, idx::Int, bv::Symbol) = (objectid(body), idx, bv)

# literal-folding expression builders -- avoid emitting a degenerate
# `x + 0`/`x * 1`/single-term `+()` call, mirroring the pattern
# cgen_sum_excluding already uses for the same reason
agen_add_exprs(a, b) = a == 0 ? b : (b == 0 ? a : Expr(:call, :+, a, b))
agen_mul_exprs(a, b) = a == 1 ? b : (b == 1 ? a : Expr(:call, :*, a, b))
agen_sum_exprs(terms) = isempty(terms) ? 0 : (length(terms) == 1 ? terms[1] : Expr(:call, :+, terms...))
agen_prod_exprs(terms) = isempty(terms) ? 1 : (length(terms) == 1 ? terms[1] : Expr(:call, :*, terms...))

# a loop_ctx frame's 0-based position: (var - lo)/step, using plain
# subtraction when step == 1 (the common case) to keep generated
# expressions simple
agen_pos0(frame) = frame.step == 1 ? Expr(:call, :-, frame.var, frame.lo) :
                                       Expr(:call, :div, Expr(:call, :-, frame.var, frame.lo), frame.step)

# product of trip counts of every loop_ctx frame STRICTLY more nested
# than index i (i.e. i+1..end) -- frame i's row-major stride
function agen_stride(loop_ctx, i)
    terms = Any[cgen_trip_count(loop_ctx[j].lo, loop_ctx[j].step, loop_ctx[j].hi) for j in (i + 1):length(loop_ctx)]
    return agen_prod_exprs(terms)
end

# product of trip counts of every frame in loop_ctx -- one
# occurrence's own local multiplicity (§5's term), or 1 for a
# non-loop site
agen_local_multiplicity(loop_ctx) = agen_prod_exprs(Any[cgen_trip_count(f.lo, f.step, f.hi) for f in loop_ctx])

# 1-based row-major flat position within one occurrence's own local block, from the current
# enclosing loop nest (outermost first) -- degenerates to the literal 1 for a non-loop site.
# Verified against a hand-derived nested-loop offset formula (a single global +1, not one
# per level).
function agen_local_position(loop_ctx)
    isempty(loop_ctx) && return 1
    terms = Any[agen_mul_exprs(agen_pos0(loop_ctx[i]), agen_stride(loop_ctx, i)) for i in eachindex(loop_ctx)]
    return agen_add_exprs(agen_sum_exprs(terms), 1)
end

# ---- Tier B detection ------------------------------------------------
# A loop's bound-determining symbol is ever an assignment target inside an ancestor
# sequential loop (a multigrid solver's ragged level-size halving sequence is the confirmed
# real instance). Returns the offending bound var, or `nothing` if the kernel is fully Tier
# A.
function agen_tier_b_offender(kernel)
    return agen_tier_b_walk(kernel.body, Set{Symbol}())
end

function agen_tier_b_walk(body, seq_reassigned)
    for stmt in body
        if stmt.kind == :for
            bound_vars = agen_for_bound_vars(stmt)
            for bv in bound_vars
                bv in seq_reassigned && return bv
            end
            inner_seq = stmt.sequential ? union(seq_reassigned, agen_collect_reassigned(stmt.body, true)) : seq_reassigned
            found = agen_tier_b_walk(stmt.body, inner_seq)
            found === nothing || return found
        elseif stmt.kind == :if
            found = agen_tier_b_walk(stmt.then, seq_reassigned)
            found === nothing || return found
            found = agen_tier_b_walk(stmt.els, seq_reassigned)
            found === nothing || return found
        end
    end
    return nothing
end

# ---- layout construction ----
# Walks kernel.body once, in exactly agen_forward_body's push-gating order, recording each occurrence's
# stack, key, and local multiplicity, then folds those into per-stack running-sum base offsets and total
# sizes. `seq0`/`in_ragged0` seed the walk's state instead of starting fresh, letting agen_ragged_block
# reuse this whole function verbatim as the sub-engine for one ragged block's own body.
function agen_indexed_layout(kernel, kinds, active_map, value_needed, reassigned, exempt, stacks; push_pop = nothing, seq0 = Set{Symbol}(), in_ragged0 = false)
    occ_mult = Dict{Symbol,Vector{Any}}()
    key_order = Dict{Symbol,Vector{Any}}()
    tainted_stacks = Set{Symbol}()
    loop_ctx = Any[]
    agen_layout_walk!(kernel.body, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, occ_mult, key_order, tainted_stacks, seq0, in_ragged0, push_pop)
    offsets = Dict{Any,Any}()
    sizes = Dict{Symbol,Any}()
    for (stack, mults) in occ_mult
        # Tier B: a tainted stack gets neither an offset formula nor a
        # size entry -- it falls back to :stack (push!/pop!, growable)
        # semantics wholesale instead, via agen_use_stack_push -- see
        # the "Tier B (implemented...)" comment above agen_local_position.
        stack in tainted_stacks && continue
        keys = key_order[stack]
        running = 0
        for (k, m) in zip(keys, mults)
            offsets[k] = (stack, running)
            running = agen_add_exprs(running, m)
        end
        sizes[stack] = running
    end
    free = Set{Symbol}()
    for (_, sz) in sizes
        agen_collect_expr_vars!(sz, free)
    end
    return (offsets = offsets, sizes = sizes, free_vars = sort(collect(free); by = string),
            tainted_stacks = tainted_stacks, derived_vars = Symbol[])
end

# `seq_reassigned`/`in_ragged` mirror agen_tier_b_walk's recursion exactly, but instead of
# stopping at the first offender, every occurrence recorded while `in_ragged` is true taints its
# stack. `in_ragged` starts false and is OR'd in, never cleared, on descent, so a loop nested
# inside a ragged ancestor stays tainted regardless of its own bound.
function agen_layout_walk!(body, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, occ_mult, key_order, tainted_stacks, seq_reassigned, in_ragged, push_pop = nothing)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            # Gate on the lhs var's own activity, not the rhs's, so a
            # destructive inactive-rhs write still gets a slot sized here
            # exactly when snap_plan would create a site for it. `push_pop`
            # (site-level TBR), when given, takes over from `value_needed`
            # here, matching agen_forward_body's own push gate.
            site_source = push_pop === nothing ? value_needed : push_pop
            if kinds[var] in (:scalar_float, :array_float) && get(active_map, var, false) &&
               agen_needs_snapshot(stmt.lhs, stmt.rhs, var, site_source, agen_site_key(body, idx)) && !(var in exempt)
                nm = agen_site_stack_name((kind = agen_snapshot_kind(stmt.lhs), array = var, at = 0))
                agen_layout_record!(occ_mult, key_order, tainted_stacks, nm, loop_ctx, agen_site_key(body, idx), in_ragged)
            end
        elseif stmt.kind == :for
            # this loop's OWN tripcount-site registration uses the
            # OUTER in_ragged (whatever was passed in), not this
            # loop's own raggedness: its occurrence count is governed
            # by loop_ctx's CURRENT (not-yet-extended) frames alone --
            # see the Tier B comment above agen_local_position.
            for bv in agen_tripcount_bound_vars(stmt, reassigned)
                agen_layout_record!(occ_mult, key_order, tainted_stacks, :tripcount_stack, loop_ctx, agen_site_key(body, idx, bv), in_ragged)
            end
            bound_vars = agen_for_bound_vars(stmt)
            this_ragged = in_ragged || any(bv -> bv in seq_reassigned, bound_vars)
            inner_seq = stmt.sequential ? union(seq_reassigned, agen_collect_reassigned(stmt.body, true)) : seq_reassigned
            push!(loop_ctx, (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step))
            agen_layout_walk!(stmt.body, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, occ_mult, key_order, tainted_stacks, inner_seq, this_ragged, push_pop)
            pop!(loop_ctx)
        elseif stmt.kind == :if
            agen_layout_record!(occ_mult, key_order, tainted_stacks, :branch_stack, loop_ctx, agen_site_key(body, idx), in_ragged)
            agen_layout_walk!(stmt.then, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, occ_mult, key_order, tainted_stacks, seq_reassigned, in_ragged, push_pop)
            agen_layout_walk!(stmt.els, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, occ_mult, key_order, tainted_stacks, seq_reassigned, in_ragged, push_pop)
        end
    end
    # Block-boundary restoration (see agen_block_boundary_vars). Mirrors the extra push
    # agen_forward_body emits at the end of this same `body`, using the current loop_ctx
    # (already including this body's enclosing loop frame), giving exactly the once-per-
    # enclosing-iteration multiplicity this occurrence needs.
    for var in agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks)
        agen_layout_record!(occ_mult, key_order, tainted_stacks, stacks[(:value, var)], loop_ctx, agen_site_key(body, 0, var), in_ragged)
    end
    return nothing
end

function agen_layout_record!(occ_mult, key_order, tainted_stacks, stack_name, loop_ctx, key, tainted)
    tainted && push!(tainted_stacks, stack_name)
    mult = agen_local_multiplicity(loop_ctx)
    push!(get!(() -> Any[], occ_mult, stack_name), mult)
    push!(get!(() -> Any[], key_order, stack_name), key)
    return nothing
end

# ---- Tier B ragged-block layout (closed-form, GPU-eligible) ----
# A 'ragged block' = an ancestor sequential loop AL whose body contains a ragged descendant it alone governs. Returns
# `nothing` if `stmt` isn't a genuine AL. AL's body is laid out via agen_indexed_layout reused as the AL-scoped sub-
# engine, seeded with the incoming reassignments (not AL's own top-level ones), so only genuine AL-within-AL
# raggedness taints. Returns `(header, local_offsets, local_sizes, ineligible_stacks)`, scoped to AL's own frame.
function agen_ragged_block(stmt, kinds, active_map, value_needed, reassigned, exempt, stacks, seq_reassigned; push_pop = nothing)
    stmt.sequential || return nothing
    own_reassigned = agen_collect_reassigned(stmt.body, true)
    agen_tier_b_walk(stmt.body, own_reassigned) === nothing && return nothing
    sub = agen_indexed_layout((body = stmt.body,), kinds, active_map, value_needed, reassigned, exempt, stacks;
                               push_pop = push_pop, seq0 = seq_reassigned, in_ragged0 = false)
    return (header = (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step), body = stmt.body,
            local_offsets = sub.offsets, local_sizes = sub.sizes, ineligible_stacks = sub.tainted_stacks)
end

# ---- Tier B: top-level layout with ragged blocks (Phase A) ----
# New top-level counterpart to agen_indexed_layout, not yet called by agen_emit/stade_hvp. Walks kernel.body once, tracking a
# per-stack running offset that may become a runtime expression referencing a ragged block's own computed total, so multiple
# ragged blocks on one stack chain correctly. A ragged `:for` is delegated to agen_ragged_block's sub-engine; any other loop is
# walked as agen_layout_walk! does. Returns `(offsets, sizes, blocks, block_totals, tainted_stacks, free_vars)`.
function agen_layout(kernel, kinds, active_map, value_needed, reassigned, exempt, stacks; push_pop = nothing)
    offsets = Dict{Any,Any}()
    current = Dict{Symbol,Any}()
    blocks = Dict{Any,Any}()
    block_of = Dict{Any,Int}()
    block_touched = Set{Symbol}()
    ineligible = Set{Symbol}()
    block_counter = Ref(0)
    agen_layout_walk_top!(kernel.body, kinds, active_map, value_needed, reassigned, exempt, stacks, Any[], offsets, current, blocks, block_of, block_touched, ineligible, Set{Symbol}(), block_counter, push_pop)
    # A block-local scalar in a local_offset/local_size formula isn't safe to read bare at an arbitrary push/pop site:
    # agen_backward_body reverses per-statement order too, so a block-boundary occurrence's forward-last statement can
    # run first in reverse, before its scalar recompute. Fix (agen_tier_b_value_tables_stmts): give each such scalar a
    # per-iteration value table, looked up instead of read bare. A kernel argument is normally safe to exclude, except
    # when also in `reassigned`, which stays a value_var too.
    safe_args = setdiff(Set(kernel.sig.args), reassigned)
    offset_exprs_by_block = Dict{Int,Vector{Any}}()
    for (_, entry) in offsets
        entry[1] === :ragged || continue
        (_, _, block_id, off) = entry
        push!(get!(() -> Any[], offset_exprs_by_block, block_id), off)
    end
    blocks2 = Dict{Any,Any}()
    for (block_id, blk) in blocks
        free = Set{Symbol}()
        for (_, sz) in blk.local_sizes
            agen_collect_expr_vars!(sz, free)
        end
        for off in get(offset_exprs_by_block, block_id, Any[])
            agen_collect_expr_vars!(off, free)
        end
        value_vars = setdiff(free, safe_args)
        blocks2[block_id] = merge(blk, (value_vars = value_vars,))
    end
    blocks = blocks2
    sizes = Dict{Symbol,Any}()
    block_totals = Dict{Symbol,Any}()
    for (stack, expr) in current
        stack in ineligible && continue
        if stack in block_touched
            block_totals[stack] = expr
        else
            sizes[stack] = expr
        end
    end
    free = Set{Symbol}()
    for (_, sz) in sizes
        agen_collect_expr_vars!(sz, free)
    end
    for (_, blk) in blocks
        for (stack, sz) in blk.local_sizes
            stack in ineligible && continue
            agen_collect_expr_vars!(sz, free)
        end
        # A block's own header (lo/hi/step) is what Phase B's table allocation and its own `for
        # <header>` loop need -- e.g. a multigrid solver's own level count never appears in any
        # local_size formula, so it would otherwise be silently missing from initstacks_*'s
        # eventual signature.
        agen_collect_expr_vars!(blk.header.lo, free)
        agen_collect_expr_vars!(blk.header.hi, free)
        agen_collect_expr_vars!(blk.header.step, free)
    end
# A block's own local_size formula legitimately references internal, ragged-controlling locals valid only inside code that
# has already run the sizing/table pass defining them -- never real kernel arguments, so never valid as a signature
# parameter. Filtered once here. But a pure Tier A size formula doesn't always reference only kernel arguments: a loop
# bound can be a scalar derived at the kernel body's top level, and dropping it left initstacks_* referencing an undefined
# name. Reported separately so agen_init_emit can hoist their defining assignments instead.
    derived = sort(collect(setdiff(free, Set(kernel.sig.args))); by = string)
    free = intersect(free, Set(kernel.sig.args))
    return (offsets = offsets, sizes = sizes, blocks = blocks, block_of = block_of, block_totals = block_totals,
            tainted_stacks = ineligible, free_vars = sort(collect(free); by = string),
            derived_vars = derived)
end

function agen_layout_walk_top!(body, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, offsets, current, blocks, block_of, block_touched, ineligible, seq_reassigned, block_counter, push_pop)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            site_source = push_pop === nothing ? value_needed : push_pop
            if kinds[var] in (:scalar_float, :array_float) && get(active_map, var, false) &&
               agen_needs_snapshot(stmt.lhs, stmt.rhs, var, site_source, agen_site_key(body, idx)) && !(var in exempt)
                nm = agen_site_stack_name((kind = agen_snapshot_kind(stmt.lhs), array = var, at = 0))
                agen_layout_static_record!(offsets, current, nm, loop_ctx, agen_site_key(body, idx))
            end
        elseif stmt.kind == :for
            for bv in agen_tripcount_bound_vars(stmt, reassigned)
                agen_layout_static_record!(offsets, current, :tripcount_stack, loop_ctx, agen_site_key(body, idx, bv))
            end
            blk = agen_ragged_block(stmt, kinds, active_map, value_needed, reassigned, exempt, stacks, seq_reassigned; push_pop = push_pop)
            if blk === nothing
                inner_seq = stmt.sequential ? union(seq_reassigned, agen_collect_reassigned(stmt.body, true)) : seq_reassigned
                push!(loop_ctx, (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step))
                agen_layout_walk_top!(stmt.body, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, offsets, current, blocks, block_of, block_touched, ineligible, inner_seq, block_counter, push_pop)
                pop!(loop_ctx)
            else
                union!(ineligible, blk.ineligible_stacks)
                block_counter[] += 1
                block_id = block_counter[]
                block_of[agen_site_key(body, idx)] = block_id
                base = Dict{Symbol,Any}()
                total_sym = Dict{Symbol,Symbol}()
                for (stack, _) in blk.local_sizes
                    stack in blk.ineligible_stacks && continue
                    push!(block_touched, stack)
                    base[stack] = get(current, stack, 0)
                    total_sym[stack] = Symbol("__tot_" * string(stack) * "_" * string(block_id))
                    for (key, (s, off)) in blk.local_offsets
                        s == stack || continue
                        offsets[key] = (:ragged, stack, block_id, off)
                    end
                    current[stack] = agen_add_exprs(base[stack], total_sym[stack])
                end
                blocks[block_id] = (header = blk.header, body = blk.body, base = base, local_sizes = blk.local_sizes, total_sym = total_sym, depth = length(loop_ctx))
            end
        elseif stmt.kind == :if
            agen_layout_static_record!(offsets, current, :branch_stack, loop_ctx, agen_site_key(body, idx))
            agen_layout_walk_top!(stmt.then, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, offsets, current, blocks, block_of, block_touched, ineligible, seq_reassigned, block_counter, push_pop)
            agen_layout_walk_top!(stmt.els, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, offsets, current, blocks, block_of, block_touched, ineligible, seq_reassigned, block_counter, push_pop)
        end
    end
    for var in agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks)
        agen_layout_static_record!(offsets, current, stacks[(:value, var)], loop_ctx, agen_site_key(body, 0, var))
    end
    return nothing
end

# exactly agen_layout_record!'s own base/advance split, just without
# the occ_mult/key_order two-pass deferral -- `current[stack]` IS the
# running sum, updated in place as we go, so a ragged block's own
# entry (agen_layout in the caller above) can read the right "so far"
# value out of it directly instead of needing a second folding pass.
function agen_layout_static_record!(offsets, current, stack_name, loop_ctx, key)
    base = get(current, stack_name, 0)
    offsets[key] = (stack_name, base)
    current[stack_name] = agen_add_exprs(base, agen_local_multiplicity(loop_ctx))
    return nothing
end

# Tier B sizing pass: a tainted stack stays push!/pop!-based; this only lets initstacks_*/hvp's shadow-stack init
# `sizehint!` the buffer ahead of time, avoiding repeated reallocation. Since sizehint! is a pure hint, this pass has
# no correctness bar -- a reasonable estimate suffices, built via a fresh walk rather than agen_layout_walk!'s
# bookkeeping. Returns `(stmts, total_of)`: a data-free kernel.body replica that increments `__sz_<stack>` at each
# tainted occurrence; `total_of` maps stack name to its count variable.
function agen_tier_b_sizing_stmts(kernel, active_map, value_needed, exempt, stacks, tainted_stacks; push_pop = nothing)
    isempty(tainted_stacks) && return (Any[], Dict{Symbol,Symbol}())
    total_of = Dict(nm => Symbol("__sz_" * string(nm)) for nm in tainted_stacks)
    decls = Any[Expr(:(=), total_of[nm], 0) for nm in sort(collect(tainted_stacks); by = string)]
    reassigned = agen_collect_reassigned(kernel.body)
    body = agen_tier_b_sizing_walk(kernel.body, kernel.sig.kinds, active_map, value_needed, reassigned, exempt, stacks, total_of, push_pop)
    return (vcat(decls, body), total_of)
end

function agen_tier_b_sizing_walk(body, kinds, active_map, value_needed, reassigned, exempt, stacks, total_of, push_pop)
    out = Any[]
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            site_source = push_pop === nothing ? value_needed : push_pop
            if kinds[var] in (:scalar_float, :array_float) && get(active_map, var, false) &&
               agen_needs_snapshot(stmt.lhs, stmt.rhs, var, site_source, agen_site_key(body, idx)) && !(var in exempt)
                nm = agen_site_stack_name((kind = agen_snapshot_kind(stmt.lhs), array = var, at = 0))
                haskey(total_of, nm) && push!(out, Expr(:(+=), total_of[nm], 1))
            end
            if kinds[var] in (:scalar_float, :scalar_int) && !agen_expr_reads_array(stmt.rhs, kinds)
                push!(out, Expr(:(=), var, stmt.rhs))
            end
        elseif stmt.kind == :for
            for bv in agen_tripcount_bound_vars(stmt, reassigned)
                haskey(total_of, :tripcount_stack) && push!(out, Expr(:(+=), total_of[:tripcount_stack], 1))
            end
            inner = agen_tier_b_sizing_walk(stmt.body, kinds, active_map, value_needed, reassigned, exempt, stacks, total_of, push_pop)
            push!(out, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
        elseif stmt.kind == :if
            haskey(total_of, :branch_stack) && push!(out, Expr(:(+=), total_of[:branch_stack], 1))
            then_inner = agen_tier_b_sizing_walk(stmt.then, kinds, active_map, value_needed, reassigned, exempt, stacks, total_of, push_pop)
            els_inner = agen_tier_b_sizing_walk(stmt.els, kinds, active_map, value_needed, reassigned, exempt, stacks, total_of, push_pop)
            push!(out, emit_if(stmt.cond, then_inner, els_inner))
        end
    end
    for var in agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks)
        nm = stacks[(:value, var)]
        haskey(total_of, nm) && push!(out, Expr(:(+=), total_of[nm], 1))
    end
    return out
end

# ---- Tier B ragged-block table construction (Phase B) ----
# Builds the per-stack prefix table for one ragged block: one entry per AL iteration holding the cumulative offset before that
# iteration, plus the block's final total. Replicates AL's own scalar control state (agen_tier_b_block_skeleton) but stops at
# any nested `:for` -- Phase A's closed-form local_size already covers what's inside a ragged loop. Recurses through `:if`,
# keeps a scalar assign unless its RHS reads an array, drops every array assign and all `:for` structure.
function agen_tier_b_block_skeleton(body, kinds)
    out = Any[]
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            if kinds[var] in (:scalar_float, :scalar_int) && !agen_expr_reads_array(stmt.rhs, kinds)
                push!(out, Expr(:(=), var, stmt.rhs))
            end
        elseif stmt.kind == :if
            then_inner = agen_tier_b_block_skeleton(stmt.then, kinds)
            els_inner = agen_tier_b_block_skeleton(stmt.els, kinds)
            push!(out, emit_if(stmt.cond, then_inner, els_inner))
        end
    end
    return out
end

# `blk` is one entry of agen_layout(...).blocks. For every stack this block touches, in one shared loop over AL's header:
# allocates a `prefix_<stack>_<block_id>` table and `__tot_<stack>_<block_id>`, writes the pre-iteration total at
# `agen_pos0(header)+1` (Phase C's read uses the same index, guaranteeing agreement), then updates the total. Also builds a
# per-iteration `val_<var>_<block_id>` table for every block-local scalar a sizing formula depends on, so agen_site_index can
# resolve it by lookup instead of an unsafe bare reference.
function agen_tier_b_block_stmts(block_id, blk, kinds)
    header = blk.header
    tripcount_expr = cgen_trip_count(header.lo, header.step, header.hi)
    stack_names = sort(collect(keys(blk.local_sizes)); by = string)
    value_vars = sort(collect(blk.value_vars); by = string)
    table_name(s) = Symbol("prefix_" * string(s) * "_" * string(block_id))
    value_table_name(v) = Symbol("val_" * string(v) * "_" * string(block_id))
    decls = Any[]
    for s in stack_names
        push!(decls, Expr(:(=), table_name(s), Expr(:call, Expr(:curly, :Vector, :Int), :undef, tripcount_expr)))
        push!(decls, Expr(:(=), blk.total_sym[s], 0))
    end
    for v in value_vars
        vt = kinds[v] == :scalar_int ? :Int64 : :Float64
        push!(decls, Expr(:(=), value_table_name(v), Expr(:call, Expr(:curly, :Vector, vt), :undef, tripcount_expr)))
    end
    idx_expr = agen_add_exprs(agen_pos0(header), 1)
# Value tables and size accumulation must see each local as it stood when this iteration's own inner loops
# ran, so the block body splits at the first RETIRE reassignment: an assignment to a size-relevant local a
# loop has already consumed as a bound. That statement belongs to the next iteration. Splitting at the
# first `:for` instead is too early (mg_vcycle defines a later loop's bound after its first loop); emitting
# everything first mis-sizes blocks whenever a bound is reassigned after the loop it governs.
    relevant = Set{Symbol}(value_vars)
    for s_ in stack_names
        agen_collect_expr_vars!(blk.local_sizes[s_], relevant)
    end
    consumed = Set{Symbol}()
    split_at = nothing
    for (i, st) in enumerate(blk.body)
        if st.kind == :assign && st.lhs isa Symbol && st.lhs in relevant && st.lhs in consumed
            split_at = i
            break
        elseif st.kind == :for
            agen_collect_expr_vars!(st.lo, consumed)
            agen_collect_expr_vars!(st.hi, consumed)
            agen_collect_expr_vars!(st.step, consumed)
        end
    end
    pre_body = split_at === nothing ? blk.body : blk.body[1:split_at - 1]
    post_body = split_at === nothing ? blk.body[1:0] : blk.body[split_at:end]
    loop_body = Any[Expr(:(=), Expr(:ref, table_name(s), idx_expr), blk.total_sym[s]) for s in stack_names]
    append!(loop_body, agen_tier_b_block_skeleton(pre_body, kinds))
    for v in value_vars
        push!(loop_body, Expr(:(=), Expr(:ref, value_table_name(v), idx_expr), v))
    end
    for s in stack_names
        push!(loop_body, Expr(:(=), blk.total_sym[s], agen_add_exprs(blk.total_sym[s], blk.local_sizes[s])))
    end
    append!(loop_body, agen_tier_b_block_skeleton(post_body, kinds))
    push!(decls, emit_forloop(header.var, header.lo, header.hi, header.step, loop_body))
    return decls
end

# Full-kernel scalar skeleton with ragged blocks spliced in -- the actual Phase B deliverable. Mirrors agen_layout_walk_top!'s
# own traversal exactly (same agen_ragged_block-classified `:for`s, via `layout.block_of`) so the skeleton can never drift from
# what agen_layout used to build `blocks`/`offsets`. Where agen_layout_walk_top! delegated a `:for` to agen_ragged_block, this
# emits that block's table-construction statements instead; elsewhere it keeps scalars/drops arrays like
# agen_tier_b_block_skeleton, except a non-AL `:for` is kept and recursed into.
function agen_tier_b_kernel_skeleton(body, kinds, layout)
    out = Any[]
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            if kinds[var] in (:scalar_float, :scalar_int) && !agen_expr_reads_array(stmt.rhs, kinds)
                push!(out, Expr(:(=), var, stmt.rhs))
            end
        elseif stmt.kind == :for
            key = agen_site_key(body, idx)
            if haskey(layout.block_of, key)
                block_id = layout.block_of[key]
                append!(out, agen_tier_b_block_stmts(block_id, layout.blocks[block_id], kinds))
            else
                inner = agen_tier_b_kernel_skeleton(stmt.body, kinds, layout)
                push!(out, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
            end
        elseif stmt.kind == :if
            then_inner = agen_tier_b_kernel_skeleton(stmt.then, kinds, layout)
            els_inner = agen_tier_b_kernel_skeleton(stmt.els, kinds, layout)
            push!(out, emit_if(stmt.cond, then_inner, els_inner))
        end
    end
    return out
end

# The correct, complete way to get initstacks_*'s eventual Tier B argument list (Phase D):
# every kernel argument referenced anywhere in the actual skeleton code that will run,
# including prelude statements outside any block that agen_layout's own free_vars can't see.
# Computed from the skeleton itself rather than enumerated case-by-case, so it can't
# silently miss an arg that only appears in an unblocked prelude statement.
function agen_tier_b_skeleton_free_vars(skeleton_stmts, kernel_args)
    free = Set{Symbol}()
    for s in skeleton_stmts
        agen_collect_expr_vars!(s, free)
    end
    return sort(collect(intersect(free, Set(kernel_args))); by = string)
end

# ---- push/pop emission strategy ----
# `ectx` is the small thread-through-the-recursion value, exactly analogous to cgen_body's
# own owner/kernels threading. `loop_ctx` is temporarily extended around a `:for`'s own
# recursion, in both agen_forward_body and agen_backward_body. Under keep_push_pop=true,
# `layout` is never consulted (ectx.keep_push_pop short-circuits first).
agen_ectx_stack() = (keep_push_pop = true, loop_ctx = Any[], layout = nothing, push_pop = nothing, ii_plan = nothing)

# Resolves which value_needed-like info the per-statement push/pop gate should consult: the
# site-level Dict (ectx.push_pop) when one was computed, else the ordinary whole-variable
# Set (`value_needed`) -- exactly today's behavior when the flag is off. Never affects
# agen_block_boundary_vars/agen_exempt_vars/agen_if_branch_scalar_vars, which stay on the
# plain Set always.
agen_push_pop_source(value_needed, ectx) = ectx.push_pop === nothing ? value_needed : ectx.push_pop

# pure AST substitution -- replaces every occurrence of a Symbol key
# in `subst` with its mapped replacement Expr, leaving everything else
# (including OTHER symbols, numbers, and the overall structure)
# unchanged. Used by agen_site_index (Phase C) to swap a block-local
# scalar reference for a table lookup -- see its own comment for why.
function agen_substitute_vars(expr, subst::Dict{Symbol,Any})
    if expr isa Symbol
        return get(subst, expr, expr)
    elseif expr isa Expr
        return Expr(expr.head, [agen_substitute_vars(a, subst) for a in expr.args]...)
    end
    return expr
end

function agen_site_index(exprs, ectx, key)
    entry = ectx.layout.offsets[key]
    if entry[1] === :ragged
        # Tier B (Phase C): a ragged-block occurrence. All three index terms are pure
        # expressions (table lookups keyed by the current loop variable, plus an ordinary
        # position formula) -- never a mutating index, since an embedded mutation would be
        # evaluated twice by hvp_double_stmt (it reuses the same index sub-Expr for both shadow
        # and primal writes).
        (_, stack, block_id, local_offset) = entry
        blk = ectx.layout.blocks[block_id]
        table_name = Symbol("prefix_" * string(stack) * "_" * string(block_id))
        table_idx = agen_add_exprs(agen_pos0(blk.header), 1)
        # `local_offset` may reference a block-local scalar, unsafe to read bare here:
        # agen_backward_body's per-statement reversal can reach a push/pop before that scalar's
        # own recompute (see the value_vars comment in agen_layout). Substituting a table
        # lookup removes the program-order dependency, using the same `table_idx` as the offset
        # table (both written at the same point by agen_tier_b_block_stmts).
        subst = Dict{Symbol,Any}(v => Expr(:ref, Symbol("val_" * string(v) * "_" * string(block_id)), table_idx) for v in blk.value_vars)
        local_offset = agen_substitute_vars(local_offset, subst)
        inner_ctx = ectx.loop_ctx[(blk.depth + 2):end]
        # agen_local_position's own formula can ALSO reference a
        # block-local scalar directly -- an inner frame's `hi` is
        # literally the loop bound symbol (e.g. n, for `for
        # i_seq_j = 1:n`) -- so it needs the identical substitution,
        # not just `local_offset` above.
        local_position = agen_substitute_vars(agen_local_position(inner_ctx), subst)
        # `blk.base[stack]` is the offset this whole block starts at (0, or a prior block's
        # total, for a stack touched by more than one block). The prefix table is only relative
        # to within this block alone, so omitting `base` would make later blocks silently
        # overlap earlier ones' index range. `base` is always a plain symbol/kernel-arg
        # expression, never a block-local scalar, so it needs no substitution.
        full_index = agen_add_exprs(agen_add_exprs(agen_add_exprs(get(blk.base, stack, 0), Expr(:ref, table_name, table_idx)), local_offset), local_position)
        # `full_index` embeds a table lookup as a sub-expression of an outer `stack[...]` ref's
        # index -- exactly the a[b[i]]-style indirect indexing parse_check_no_indirect_indexing
        # forbids (STADE's front end rejecting its own generated code). Reading it into a
        # scalar on its own line first, right here, satisfies that rule the way a hand-written
        # kernel would.
        parse_contains_ref(full_index) || return full_index
        tmp = Symbol("__idx_", string(stack), "_", block_id, "_", length(exprs))
        push!(exprs, Expr(:(=), tmp, full_index))
        return tmp
    end
    (_, offset) = entry
    return agen_add_exprs(offset, agen_local_position(ectx.loop_ctx))
end

# True whenever `stack_name` should use plain push!/pop! rather than an :indexed write/read: either
# the whole kernel is in :stack mode, or this stack is in `ectx.layout.tainted_stacks` (genuine AL-
# within-AL, Phase A's single-level scope restriction). `ectx.layout` is only `nothing` when
# keep_push_pop is already true, so the `!== nothing` guard is defensive.
agen_use_stack_push(ectx, stack_name) = ectx.keep_push_pop ||
    (ectx.layout !== nothing && stack_name in ectx.layout.tainted_stacks)

# `key` is unused (and may be `nothing`) whenever
# agen_use_stack_push(ectx, stack_name) is true, matching every call
# site below that only ever computes a real key inside the :indexed
# branch's own guard
function agen_emit_push!(exprs, stack_name::Symbol, value, ectx, key)
    if agen_use_stack_push(ectx, stack_name)
        push!(exprs, Expr(:call, :push!, stack_name, value))
        return nothing
    end
    idx = agen_site_index(exprs, ectx, key)
    push!(exprs, Expr(:(=), Expr(:ref, stack_name, idx), value))
    return nothing
end

# Returns the RHS expr only -- caller wraps `lhs = <this>`, matching how a plain
# `pop!(stack)` was always just an rhs expr too. `exprs` is the caller's in-construction
# statement list: agen_site_index may push a hoisted index assignment onto it before
# returning, landing ahead of whatever statement the caller builds from this call's return
# value.
function agen_emit_pop(stack_name::Symbol, ectx, key, exprs)
    agen_use_stack_push(ectx, stack_name) && return Expr(:call, :pop!, stack_name)
    return Expr(:ref, stack_name, agen_site_index(exprs, ectx, key))
end

# ---- Phase 3 cleanup: drop stack args left unused by fusion ----
# A var covered by an ii_plan site does not always mean its stack becomes fully unused -- agen_block_boundary_vars
# can still need it. A blanket drop would be a real bug, so the safe approach generates the adjoint body first, then
# drops only what provably has zero remaining push!/pop! calls in the actual output. Scoped to keep_push_pop=true,
# matching fuse_ii_loops's own scope; under keep_push_pop=false a push/pop is an indexed ref, not a push!/pop! call.
function agen_collect_used_stacks!(expr, used)
    if expr isa Expr
        if expr.head == :call && length(expr.args) >= 2 && expr.args[1] in (:push!, :pop!) && expr.args[2] isa Symbol
            push!(used, expr.args[2])
        end
        for a in expr.args
            agen_collect_used_stacks!(a, used)
        end
    end
    return nothing
end

function agen_used_stack_names(expr)
    used = Set{Symbol}()
    agen_collect_used_stacks!(expr, used)
    return used
end

# drops `unused` symbols from a `function f(args...)` Expr's own call
# signature, leaving the body untouched (it was already correct --
# these names simply never appear in it any more).
function agen_drop_unused_stack_args(fn_expr, unused)
    call_expr = fn_expr.args[1]
    new_args = filter(a -> !(a isa Symbol && a in unused), call_expr.args)
    return Expr(:function, Expr(:call, new_args...), fn_expr.args[2])
end

# drops each unused stack's own `nm = Vector{T}()` allocation
# statement from initstacks_*'s body, and drops it from the returned
# tuple, re-deriving the correct emit_return_scalars shape (bare
# return, single-value return, or tuple return) for whatever remains.
function agen_drop_unused_stack_allocs(initstacks_expr, unused)
    body_block = initstacks_expr.args[2]
    new_stmts = Any[]
    for stmt in body_block.args
        if stmt isa Expr && stmt.head == :(=) && stmt.args[1] isa Symbol && stmt.args[1] in unused
            continue
        elseif stmt isa Expr && stmt.head == :call && stmt.args[1] == :sizehint! && stmt.args[2] isa Symbol && stmt.args[2] in unused
            continue
        elseif stmt isa Expr && stmt.head == :return
            ret_val = stmt.args[1]
            remaining = ret_val isa Expr && ret_val.head == :tuple ? Symbol[a for a in ret_val.args if !(a isa Symbol && a in unused)] :
                        (ret_val isa Symbol && ret_val in unused ? Symbol[] : Symbol[ret_val])
            push!(new_stmts, emit_return_scalars(remaining))
        else
            push!(new_stmts, stmt)
        end
    end
    return Expr(:function, initstacks_expr.args[1], Expr(:block, new_stmts...))
end

# ---- ii_* Phase 3 codegen: :independent fusion only ----
# Builds one un-reversed loop fusing the primal recompute of stmt.body with its backward differentiation, for `:independent`
# sites only (`:reduction`/`:mixed` use separate handling, since fusing reads an undertotaled reduction too early). Fused vars
# are excluded via a locally-scoped value_needed Set, not kernel-wide. `agen_ii_override_ectx` is required: site-level TBR
# consults `ectx.push_pop`, so excluding a var from value_needed alone leaves an unmatched push.
function agen_ii_force_no_snapshot!(body, vn_local, override)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            var in vn_local && (override[agen_site_key(body, idx)] = false)
        elseif stmt.kind == :for
            agen_ii_force_no_snapshot!(stmt.body, vn_local, override)
        elseif stmt.kind == :if
            agen_ii_force_no_snapshot!(stmt.then, vn_local, override)
            agen_ii_force_no_snapshot!(stmt.els, vn_local, override)
        end
    end
    return nothing
end

function agen_ii_override_ectx(ectx, body, vn_local)
    (ectx.push_pop === nothing || isempty(vn_local)) && return ectx
    override = copy(ectx.push_pop)
    agen_ii_force_no_snapshot!(body, vn_local, override)
    return (keep_push_pop = ectx.keep_push_pop, loop_ctx = ectx.loop_ctx, layout = ectx.layout, push_pop = override, ii_plan = ectx.ii_plan)
end

# ---- filtered primal recompute (backward-position fusion only) ----
# `fwd` is the real primal at the forward position (`:independent`), but only a recompute at the backward one, re-establishing
# fused scalars never snapshotted. Re-executing the whole body there is wrong, not wasteful: an accumulating array write
# applies twice, doubling its forward value -- gradients stay bit-identical while the array diverges, so no FD test catches it.
# The filter keeps scalar assigns and drops array writes; sound only when no recomputed scalar reads an array this body writes.
function agen_ii_recompute_stmts(body)
    exprs = Any[]
    for stmt in body
        if stmt.kind == :assign
            stmt.lhs isa Symbol && push!(exprs, Expr(:(=), stmt.lhs, stmt.rhs))
        elseif stmt.kind == :for
            inner = agen_ii_recompute_stmts(stmt.body)
            isempty(inner) || push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
        elseif stmt.kind == :if
            then_e = agen_ii_recompute_stmts(stmt.then)
            els_e = agen_ii_recompute_stmts(stmt.els)
            (isempty(then_e) && isempty(els_e)) || push!(exprs, emit_if(stmt.cond, then_e, els_e))
        end
    end
    return exprs
end

function agen_emit_ii_loop(stmt, lin_stmt, kinds, active_map, value_needed, reassigned, stacks, exempt, unsafe; ectx, recompute::Bool = false)
    local_names = cgen_locally_assigned_scalars(stmt.body)
    redvars = cgen_scalar_reduction_vars(stmt.body)
    vn_local = Set(v for v in intersect(value_needed, union(local_names, redvars)) if get(active_map, v, false))
    local_value_needed = setdiff(value_needed, vn_local)
    local_ectx = agen_ii_override_ectx(ectx, stmt.body, vn_local)
    fwd = recompute ?
        agen_ii_recompute_stmts(stmt.body) :
        agen_forward_body(stmt.body, kinds, active_map, local_value_needed, reassigned, stacks, exempt;
                           ectx = local_ectx, lin_body = lin_stmt.body, unsafe = unsafe)
    bwd = agen_backward_body(lin_stmt.body, stmt.body, kinds, active_map, unsafe, local_value_needed, reassigned, stacks, exempt; ectx = local_ectx)
    return emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, vcat(fwd, bwd))
end

function agen_forward_body(body, kinds, active_map, value_needed, reassigned, stacks, exempt; ectx = agen_ectx_stack(), lin_body = nothing, unsafe = nothing)
    exprs = Any[]
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            # Gate on the lhs var's own activity, not this statement's rhs activity: a write
            # with an inactive rhs can still destroy a value an earlier statement needs for
            # its own nonlinear derivative, matching snap_check_assign!'s own gate. An int-
            # kinded lhs never owns a :value/:array site; `exempt` skips a push snap_plan
            # itself would elide.
            if kinds[var] in (:scalar_float, :array_float) && get(active_map, var, false) && agen_needs_snapshot(stmt.lhs, stmt.rhs, var, agen_push_pop_source(value_needed, ectx), agen_site_key(body, idx)) && !(var in exempt)
                agen_emit_push!(exprs, stacks[(agen_snapshot_kind(stmt.lhs), var)], stmt.lhs, ectx, agen_site_key(body, idx))
            end
            push!(exprs, Expr(:(=), stmt.lhs, stmt.rhs))
        elseif stmt.kind == :for
            # ii_plan-covered site: `:independent` is built as one fused, un-reversed loop
            # (agen_emit_ii_loop) instead of push-then-reverse. `:reduction` keeps its
            # ordinary forward position, its reduction var(s) excluded from value_needed so
            # nothing pushes their old value. `:mixed` gets the same treatment: excluded from
            # push, nothing fused here -- see the `:mixed` branch below.
            key = agen_site_key(body, idx)
            ii_kind = ectx.ii_plan === nothing ? nothing : get(ectx.ii_plan, key, nothing)
            if ii_kind === :independent && lin_body !== nothing && unsafe !== nothing
                push!(exprs, agen_emit_ii_loop(stmt, lin_body[idx], kinds, active_map, value_needed, reassigned, stacks, exempt, unsafe; ectx = ectx))
            elseif ii_kind === :mixed && lin_body !== nothing && unsafe !== nothing
                # NOT split across positions: a var safe to fuse here (vn_ind) can be read by a
                # var deferred to the backward position (vn_red)'s accumulation, and vn_ind's
                # collect-then-distribute step needs both contributions first. So `:mixed` is
                # treated like `:reduction`: exclude vn_local from push here, but defer all
                # differentiation to the backward position.
                local_names = cgen_locally_assigned_scalars(stmt.body)
                redvars = cgen_scalar_reduction_vars(stmt.body)
                vn_local = Set(v for v in intersect(value_needed, union(local_names, redvars)) if get(active_map, v, false))
                loop_value_needed = setdiff(value_needed, vn_local)
                loop_ectx = agen_ii_override_ectx(ectx, stmt.body, vn_local)
                for bv in agen_tripcount_bound_vars(stmt, reassigned)
                    agen_emit_push!(exprs, stacks[(:tripcount, bv)], bv, ectx, agen_site_key(body, idx, bv))
                end
                push!(ectx.loop_ctx, (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step))
                inner = agen_forward_body(stmt.body, kinds, active_map, loop_value_needed, reassigned, stacks, exempt;
                                           ectx = loop_ectx,
                                           lin_body = lin_body[idx].body,
                                           unsafe = unsafe)
                pop!(ectx.loop_ctx)
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
            else
                loop_value_needed = value_needed
                loop_ectx = ectx
                if ii_kind === :reduction || ii_kind === :recompute
                    local_names = cgen_locally_assigned_scalars(stmt.body)
                    redvars = cgen_scalar_reduction_vars(stmt.body)
                    vn_red = Set(v for v in intersect(value_needed, union(local_names, redvars)) if get(active_map, v, false))
                    loop_value_needed = setdiff(value_needed, vn_red)
                    loop_ectx = agen_ii_override_ectx(ectx, stmt.body, vn_red)
                end
                for bv in agen_tripcount_bound_vars(stmt, reassigned)
                    agen_emit_push!(exprs, stacks[(:tripcount, bv)], bv, ectx, agen_site_key(body, idx, bv))
                end
                push!(ectx.loop_ctx, (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step))
                inner = agen_forward_body(stmt.body, kinds, active_map, loop_value_needed, reassigned, stacks, exempt;
                                           ectx = loop_ectx,
                                           lin_body = lin_body === nothing ? nothing : lin_body[idx].body,
                                           unsafe = unsafe)
                pop!(ectx.loop_ctx)
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
            end
        elseif stmt.kind == :if
            nm = stacks[(:branch, :cond)]
            key = agen_site_key(body, idx)
            # Built as a fresh list, not the previous vcat one-liner, so
            # agen_emit_push! has this branch's own statement list to hoist a
            # ragged index assignment into, ahead of the branch-flag push --
            # hoisting into the outer `exprs` would be wrong, since it isn't
            # guarded by `stmt.cond`.
            then_exprs = Any[]
            agen_emit_push!(then_exprs, nm, 1, ectx, key)
            append!(then_exprs, agen_forward_body(stmt.then, kinds, active_map, value_needed, reassigned, stacks, exempt;
                                                    ectx = ectx,
                                                    lin_body = lin_body === nothing ? nothing : lin_body[idx].then,
                                                    unsafe = unsafe))
            els_exprs = Any[]
            agen_emit_push!(els_exprs, nm, 0, ectx, key)
            append!(els_exprs, agen_forward_body(stmt.els, kinds, active_map, value_needed, reassigned, stacks, exempt;
                                                   ectx = ectx,
                                                   lin_body = lin_body === nothing ? nothing : lin_body[idx].els,
                                                   unsafe = unsafe))
            push!(exprs, emit_if(stmt.cond, then_exprs, els_exprs))
        end
    end
    # block-boundary restoration (see agen_block_boundary_vars above):
    # snapshot each such var's value once here, at the end of `body`,
    # capturing whatever this ENCLOSING iteration's own nested writes
    # left it at -- restored symmetrically at the start of this same
    # body's own backward processing in agen_backward_body.
    for var in agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks; ii_plan = ectx.ii_plan)
        agen_emit_push!(exprs, stacks[(:value, var)], var, ectx, agen_site_key(body, 0, var))
    end
    return exprs
end

# an active lhs is always scalar_float or array_float -- :ref means
# the array kind (:array), a bare Symbol means the scalar kind
# (:value), matching how snap_plan classified the same write
agen_snapshot_kind(lhs) = lhs isa Symbol ? :value : :array

# find the top-level :assign to `var` in a lin_plan body, if any --
# only looks at direct children, matching how the :if handling below
# only ever needs a branch's OWN immediate write, not one buried
# inside a further-nested :for/:if
function agen_find_assign(body, var)
    for s in body
        s.kind == :assign || continue
        v = s.lhs isa Symbol ? s.lhs : s.lhs.args[1]
        v == var && return s
    end
    return nothing
end

# The constant a branch-snapshotted scalar falls back to on whichever side of the :if
# doesn't assign it: the nearest preceding sibling assign to the same var in this block.
# Only a literal-Number rhs is trusted, since recompute must not depend on any other
# variable's current value, which the reverse sweep could already have disturbed.
function agen_branch_scalar_fallback(plan, idx, var)
    for k in (idx - 1):-1:1
        s = plan[k]
        s.kind == :assign || continue
        v = s.lhs isa Symbol ? s.lhs : s.lhs.args[1]
        v == var || continue
        return s.tree.expr isa Number ? s.tree.expr : nothing
    end
    return nothing
end

# scalar_float vars this :if assigns (on either branch) purely
# because they're read elsewhere -- NOT because the write is a
# self-recurrence, which the normal pre-write restore already handles
# correctly (it needs the OLD value at exactly this write's own
# reverse position, unlike the read-elsewhere case below).
function agen_if_branch_scalar_vars(stmt, kinds, value_needed)
    vars = Symbol[]
    for sub in vcat(stmt.then, stmt.els)
        sub.kind == :assign || continue
        var = sub.lhs isa Symbol ? sub.lhs : sub.lhs.args[1]
        (sub.lhs isa Symbol && kinds[var] == :scalar_float) || continue
        agen_count_var_refs(sub.tree.expr, var) == 0 || continue
        agen_needs_snapshot(sub.lhs, sub.tree.expr, var, value_needed) || continue
        var in vars || push!(vars, var)
    end
    return vars
end

# ---- backward sweep (walks lin_plan, whose :for/:if fields mirror
#      the primal's own exactly -- only :assign carries a built tree) -

# `skip_restore`: vars whose pre-write restore has already been done
# by an enclosing :if's hoist (see below) -- popping again here would
# consume the wrong stack entry. Only ever non-empty for the direct
# then/els body of a hoisted :if; every other call uses the default.
function agen_backward_body(plan, primal_body, kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt, skip_restore = Set{Symbol}(); ectx = agen_ectx_stack())
    exprs = Any[]
    # Block-boundary restoration: restore each such var here, at the very start of this
    # body's own backward processing, from the matching push agen_forward_body emitted at
    # the end of this same body. Must run before anything else below, since this is what
    # makes this body's own nested-loop-written value visible again instead of a later
    # sibling iteration's.
    for var in agen_block_boundary_vars(primal_body, kinds, value_needed, exempt, stacks; ii_plan = ectx.ii_plan)
        push!(exprs, Expr(:(=), var, agen_emit_pop(stacks[(:value, var)], ectx, agen_site_key(primal_body, 0, var), exprs)))
    end
    # Int-kinded local assignments never carry gradients, so agen_backward_assign emits nothing for them -- but the array
    # indices they compute can still be needed by other statements once everything else is reversed. A plain reversal
    # would put the needing statement before the computing one, so all of them (except `unsafe` ones) are recomputed up
    # front, in original forward order. A var reassigned at more than one site elsewhere can't be safely reconstructed
    # this way -- what matters downstream (a loop bound) is already restored via :tripcount.
    for stmt in plan
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            if kinds[var] in (:scalar_int, :array_int) && !(var in unsafe)
                push!(exprs, Expr(:(=), stmt.lhs, stmt.tree.expr))
            end
        end
    end
# Branch-snapshotted scalars whose own primal value is read by a different, later statement in this block for a nonlinear
# term's partial. A plain reverse walk restores such a scalar only when it reaches the :if itself, one iteration too late.
# Recomputed here instead, forward-style, from the freshly popped branch flag: safe since a snapshot site's rhs never depends
# on anything the reverse sweep has touched yet. Walked in reverse-plan order so multiple qualifying :if's pop branch_stack in
# LIFO order.
    branch_flags = Dict{Int,Symbol}()
    hoisted_vars = Dict{Int,Set{Symbol}}()
    for idx in length(plan):-1:1
        stmt = plan[idx]
        stmt.kind == :if || continue
        vars = agen_if_branch_scalar_vars(stmt, kinds, value_needed)
        isempty(vars) && continue
        resolved = Any[]
        for var in vars
            then_stmt = agen_find_assign(stmt.then, var)
            els_stmt = agen_find_assign(stmt.els, var)
            then_expr = then_stmt === nothing ? agen_branch_scalar_fallback(plan, idx, var) : then_stmt.tree.expr
            els_expr = els_stmt === nothing ? agen_branch_scalar_fallback(plan, idx, var) : els_stmt.tree.expr
            (then_expr === nothing || els_expr === nothing) && continue   # can't safely recompute -- leave to the normal (imperfectly-timed) restore
            # a branch only pushed onto the value stack (forward sweep)
            # if its own rhs was active -- a literal-constant branch
            # (like the `then_expr`/`els_expr` fallback case) never
            # does, matching agen_forward_body's own push gate exactly
            then_pushed = then_stmt !== nothing && then_stmt.active
            els_pushed = els_stmt !== nothing && els_stmt.active
            push!(resolved, (var, then_expr, els_expr, then_pushed, els_pushed))
        end
        isempty(resolved) && continue
        flag = Symbol("__branch_pre_", idx)
        push!(exprs, Expr(:(=), flag, agen_emit_pop(stacks[(:branch, :cond)], ectx, agen_site_key(primal_body, idx), exprs)))
        branch_flags[idx] = flag
        hoisted_vars[idx] = Set(v for (v, _, _, _, _) in resolved)
        for (var, then_expr, els_expr, then_pushed, els_pushed) in resolved
            snm = stacks[(:value, var)]
            # the discard-pop below exists ONLY to keep push!/pop!'s
            # single shared stack pointer in sync -- :indexed mode has
            # no such pointer, so it's simply omitted there; the
            # pushed slot goes unread, a harmless over-snapshot (see
            # skill-stade-dev.md's keep_push_pop entry)
            if ectx.keep_push_pop
                if then_pushed && els_pushed
                    push!(exprs, Expr(:(=), :__snap_discard, agen_emit_pop(snm, ectx, nothing, exprs)))
                elseif then_pushed
                    push!(exprs, emit_if(Expr(:call, :(==), flag, 1), Any[Expr(:(=), :__snap_discard, agen_emit_pop(snm, ectx, nothing, exprs))], Any[]))
                elseif els_pushed
                    push!(exprs, emit_if(Expr(:call, :(==), flag, 0), Any[Expr(:(=), :__snap_discard, agen_emit_pop(snm, ectx, nothing, exprs))], Any[]))
                end
            end
            # declared with a dummy numeric value first, then assigned
            # in each branch, rather than `var = if cond; a; else; b;
            # end` -- both forms are equivalent here (both branches of
            # a plain if/else always assign var exactly once), this
            # form is just easier to scan in the generated file
            push!(exprs, Expr(:(=), var, 0.0))
            push!(exprs, emit_if(Expr(:call, :(==), flag, 1), Any[Expr(:(=), var, then_expr)], Any[Expr(:(=), var, els_expr)]))
        end
    end
    for idx in length(plan):-1:1
        stmt = plan[idx]
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            kinds[var] in (:scalar_int, :array_int) && continue   # hoisted above, or unsafe (skipped entirely)
            append!(exprs, agen_backward_assign(stmt, kinds, active_map, value_needed, reassigned, stacks, exempt, skip_restore; ectx = ectx, key = agen_site_key(primal_body, idx)))
        elseif stmt.kind == :for
            key = agen_site_key(primal_body, idx)
            ii_kind = ectx.ii_plan === nothing ? nothing : get(ectx.ii_plan, key, nothing)
            if ii_kind === :independent
                # already fully emitted by agen_forward_body's own
                # agen_emit_ii_loop call -- nothing to do here,
                # including no tripcount pop (the fused loop's single
                # un-reversed header never pushed one).
            elseif ii_kind === :reduction
                # This loop's own reduction var(s) never needed a push (excluded from value_needed above) --
                # their adjoint is emitted here, reusing agen_emit_ii_loop as :independent does, but
                # appended at this loop's ordinary, unfused backward position. Safe since by the time the
                # reverse sweep reaches here, everything after it has already run its backward code, so the
                # shadow distributed here is already complete. No tripcount pop, as with :independent.
                push!(exprs, agen_emit_ii_loop(primal_body[idx], stmt, kinds, active_map, value_needed, reassigned, stacks, exempt, unsafe; ectx = ectx, recompute = true))
            elseif ii_kind === :recompute
                # Same dispatch as :reduction -- the loop keeps its ordinary
                # forward and backward positions; only its snapshots are
                # replaced by the filtered recompute. Nothing moves, which is
                # what lets this kind exist for a loop with an escaping array
                # write at all.
                push!(exprs, agen_emit_ii_loop(primal_body[idx], stmt, kinds, active_map, value_needed, reassigned, stacks, exempt, unsafe; ectx = ectx, recompute = true))
            elseif ii_kind === :mixed
                # NOT split -- see agen_forward_body's own :mixed
                # handling for the full reasoning. Both halves get
                # their actual differentiation HERE, together, in one
                # un-reversed loop -- deferred to this position instead
                # of some of it running at the forward position.
                push!(exprs, agen_emit_ii_loop(primal_body[idx], stmt, kinds, active_map, value_needed, reassigned, stacks, exempt, unsafe; ectx = ectx, recompute = true))
            else
                push!(ectx.loop_ctx, (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step))
                inner = agen_backward_body(stmt.body, primal_body[idx].body, kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = ectx)
                pop!(ectx.loop_ctx)
                reverse_it = stmt.sequential || agen_body_has_snapshot(stmt.body, primal_body[idx].body, kinds, active_map, value_needed, reassigned, exempt, stacks, ectx)
                loop_expr = reverse_it ?
                    emit_forloop(stmt.var, stmt.hi, stmt.lo, agen_negate_step(stmt.step), inner) :
                    emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner)
                for bv in agen_tripcount_bound_vars(stmt, reassigned)
                    push!(exprs, Expr(:(=), bv, agen_emit_pop(stacks[(:tripcount, bv)], ectx, agen_site_key(primal_body, idx, bv), exprs)))
                end
                push!(exprs, loop_expr)
            end
        elseif stmt.kind == :if
            skip = get(hoisted_vars, idx, Set{Symbol}())
            then_exprs = agen_backward_body(stmt.then, primal_body[idx].then, kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt, skip; ectx = ectx)
            els_exprs = agen_backward_body(stmt.els, primal_body[idx].els, kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt, skip; ectx = ectx)
            if haskey(branch_flags, idx)
                push!(exprs, emit_if(Expr(:call, :(==), branch_flags[idx], 1), then_exprs, els_exprs))
            else
                nm = stacks[(:branch, :cond)]
                push!(exprs, Expr(:(=), :__branch, agen_emit_pop(nm, ectx, agen_site_key(primal_body, idx), exprs)))
                push!(exprs, emit_if(Expr(:call, :(==), :__branch, 1), then_exprs, els_exprs))
            end
        end
    end
    return exprs
end

# A loop must be reversed in the backward sweep whenever any push happens inside it, at any nesting
# depth -- not just when this loop is itself sequential. LIFO stack discipline requires every
# enclosing loop to run in exact reverse, full stop; reversal is about stack order, not mathematical
# dependency. Also true whenever `body` itself has a block-boundary var (see
# agen_block_boundary_vars).
function agen_body_has_snapshot(body, primal_body, kinds, active_map, value_needed, reassigned, exempt, stacks, ectx)
    !isempty(agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks; ii_plan = ectx.ii_plan)) && return true
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            # gate on active_map[var], matching agen_forward_body's own
            # push gate (not stmt.active) -- see its comment: a write
            # can need a push (and hence force this loop to reverse)
            # even when its own rhs is a plain inactive literal.
            if get(active_map, var, false) && agen_needs_snapshot(stmt.lhs, stmt.tree.expr, var, agen_push_pop_source(value_needed, ectx), agen_site_key(primal_body, idx)) && !(var in exempt)
                return true
            end
        elseif stmt.kind == :for
            !isempty(agen_tripcount_bound_vars(stmt, reassigned)) && return true
            agen_body_has_snapshot(stmt.body, primal_body[idx].body, kinds, active_map, value_needed, reassigned, exempt, stacks, ectx) && return true
        elseif stmt.kind == :if
            return true   # every `if` pushes a branch flag, unconditionally
        end
    end
    return false
end

function agen_backward_assign(stmt, kinds, active_map, value_needed, reassigned, stacks, exempt, skip_restore = Set{Symbol}(); ectx = agen_ectx_stack(), key = nothing)
    var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
    is_accum = agen_is_pure_accumulation(stmt.lhs, stmt.tree.expr, var)
    exprs = Any[]
    # int-kinded lhs can't own a shadow, or a pop, at all
    if kinds[var] in (:scalar_float, :array_float)
        # Restore this write's overwritten old value whenever active_map[var] does, matching
        # agen_forward_body's push gate -- not stmt.active, since a write can destroy a value
        # another statement's nonlinear derivative still needs even when this write's own rhs is a
        # plain inactive literal. `exempt`/`skip_restore` mirror the forward sweep's own skips -- no
        # push happened for those, so there's nothing to pop.
        if get(active_map, var, false) && agen_needs_snapshot(stmt.lhs, stmt.tree.expr, var, agen_push_pop_source(value_needed, ectx), key) && !(var in exempt) && !(var in skip_restore)
            nm = stacks[(agen_snapshot_kind(stmt.lhs), var)]
            push!(exprs, Expr(:(=), stmt.lhs, agen_emit_pop(nm, ectx, key, exprs)))
        end
        if stmt.active
            lhsb = agen_shadow(stmt.lhs)
            if is_accum
                agen_distribute!(stmt.tree, lhsb, exprs; skip_expr = stmt.lhs)
            else
                # A leaf whose slot exactly matches lhs is a genuine, non-identity self-reference --
                # is_accum only catches the identity case. Its contribution can't accumulate into lhsb
                # normally: lhsb is both the seed read from and the target written to, so `lhsb = lhsb +
                # contribution` reads its own not-yet-updated self, and the next line's reset then throws
                # that sum away. Collected separately and applied as a replacement of lhsb instead.
                self_terms = Any[]
                agen_distribute!(stmt.tree, lhsb, exprs; self_expr = stmt.lhs, self_terms = self_terms)
                if isempty(self_terms)
                    push!(exprs, Expr(:(=), lhsb, 0.0))
                else
                    push!(exprs, Expr(:(=), lhsb, der_sum_terms(self_terms)))
                end
            end
        elseif !is_accum
            # This write's rhs carries no active leaf, so there's nothing to distribute, but
            # the shadow this write 'produced' still needs resetting. Skipping the reset
            # because this write is a constant would let an earlier-processed statement's
            # accumulation leak into the next (chronologically earlier) iteration's
            # contribution instead of starting fresh.
            push!(exprs, Expr(:(=), agen_shadow(stmt.lhs), 0.0))
        end
    end
    return exprs
end

# Recursively distributes `seed` through a lin_node tree, accumulating `target = target + contribution` at every active leaf
# whose slot differs from both `skip_expr` and `self_expr`. `skip_expr` is honored only against a direct child of the node it's
# passed to -- the top-level slot a pure accumulation's lhs occupies. `self_expr`/`self_terms` propagate to any depth, since a
# self-reference can appear anywhere; matching leaves push onto `self_terms` instead of an accumulate statement.
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

# Every local (non-argument) scalar needs to already exist before the forward sweep runs --
# normally its first primal assignment establishes that, but a local flagged for
# snapshotting can get pushed (reading its pre-overwrite value) as early as the first loop
# iteration, before any primal assignment. Without this, that first push would be an
# UndefVarError. Harmless for locals that don't need it.
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
# it -- arrays can never be local under skill-stade (rule 8: no
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
# Forward-over-reverse Hessian-vector products: a second forward-mode pass applied to the already-generated adjoint kernel's
# own Expr (no lin_plan -- walks agen_'s output directly). Computes Hv = d(grad f)/dv by seeding a tangent and propagating
# through agen_'s forward+backward sweep, which works since reverse-mode output is straight-line, replay-only code. New
# mechanics: a push!/pop! of an active value gets a paired push/pop on a shadow stack.

function hvp_emit(kernel, active_map, lin_plan, sites; keep_push_pop::Bool = true, layout = nothing, push_pop = nothing,
                   tier_b_extra_args::Vector{Symbol} = Symbol[], ii_plan = nothing)
    sig = kernel.sig
    stacks = agen_stack_map(sites)
    value_needed = agen_value_needed_vars(kernel)
    reassigned = agen_collect_reassigned(kernel.body)
    unsafe = agen_unsafe_int_vars(kernel)
    exempt = agen_exempt_vars(kernel, value_needed)

    ectx = (keep_push_pop = keep_push_pop, loop_ctx = Any[], layout = layout, push_pop = push_pop, ii_plan = ii_plan)

    fwd = agen_forward_body(kernel.body, sig.kinds, active_map, value_needed, reassigned, stacks, exempt; ectx = ectx, lin_body = lin_plan, unsafe = unsafe)
    bwd = agen_backward_body(lin_plan, kernel.body, sig.kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = ectx)

    shadow_of = hvp_shadow_map(kernel, sites)

    fname = hvp_fname(sig.name)
    seed_args = Symbol[]
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        push!(seed_args, tgen_shadow(a))
        push!(seed_args, tgen_shadow(agen_shadow(a)))
    end
    fargs = vcat(agen_signature_args(sig), seed_args, agen_stack_names(sites), tier_b_extra_args)

    body = Any[]
    sizing_stmts = Any[]
    total_of = Dict{Symbol,Symbol}()
    if !keep_push_pop && layout !== nothing && !isempty(layout.tainted_stacks)
        (sizing_stmts, total_of) = agen_tier_b_sizing_stmts(kernel, active_map, value_needed, exempt, stacks, layout.tainted_stacks; push_pop = push_pop)
    end
    append!(body, sizing_stmts)
    append!(body, hvp_shadow_stack_inits(sites, shadow_of, keep_push_pop, layout, total_of))
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

# Every float variable this stage will encounter: primal args/locals (shadow = tgen_shadow, the same
# "d" convention tgen_ already uses) and agen_'s own adjoint shadows (shadow = tgen_shadow of those --
# e.g. xb's second-layer shadow is xbd, unchanged, since it only appends "d"); plus every
# Float64-holding stack. Int64 stacks get none; they're never differentiated.
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

function hvp_shadow_stack_inits(sites, shadow_of, keep_push_pop::Bool = true, layout = nothing, total_of = Dict{Symbol,Symbol}())
    exprs = Any[]
    seen = Set{Symbol}()
    for s in sites
        s.kind in (:array, :value) || continue
        nm = agen_site_stack_name(s)
        nm in seen && continue
        push!(seen, nm)
# A shadow stack is exactly as large as its primal counterpart, folded directly into this `_hv` function's body
# (never a separate initstacks_*_hv). Uses `length(nm)` on the primal stack -- already allocated and passed in as
# `_hv`'s argument -- rather than re-evaluating `layout.sizes`/`layout.block_totals`: a Tier B size formula can
# reference a block-local scalar only safe to read when the real forward sweep computes it, never at this function's
# top. `length(nm)` sidesteps that entirely.
        grow = keep_push_pop || (layout !== nothing && nm in layout.tainted_stacks)
        alloc = agen_stack_alloc_expr(:value, grow, grow ? nothing : Expr(:call, :length, nm))
        push!(exprs, Expr(:(=), shadow_of[nm], alloc))
        haskey(total_of, nm) && push!(exprs, Expr(:call, :sizehint!, shadow_of[nm], total_of[nm]))
    end
    return exprs
end

# Zero-initialize every local scalar's second-layer shadow and its adjoint-shadow's shadow -- exactly
# agen_local_primal_inits/agen_local_shadow_inits's job, one layer further out. Arrays can never be
# local (skill-stade rule 8); every float arg's own xd/xbd is a function parameter, never locally
# initialized. Unlike agen_'s own local-init functions, this does not gate on active_map: the forward
# sweep always replays every primal statement regardless of activity.
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

# Recursively differentiates an arbitrary primal-valued Expr -- fuses what lin_build_expr +
# tgen_tangent_expr do in two phases into one, since there's no retained tree for generated
# code to sweep a second time. A pop! differentiates to a pop from the paired shadow stack;
# everything else is the same chain-rule contraction tgen_ already performs, via
# der_tangent_generic -- no new derivative rules needed.
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


# ==================== cgen_* =====================================
# CUDA codegen: turns a validated kernel, or a STADE-generated function, into a host launcher plus one `@cuda` device kernel per data-parallel
# (sequential=false) loop -- a loop-nest transform, independent of act_/snap_/lin_. Ingests via cgen_from_kernel (plain skill-stade) or
# cgen_parse_generated (STADE's own vocabulary). Stack safety: a loop with push!/pop! anywhere is never split, since LIFO order can't survive
# concurrent threads. Race safety: a split write is atomic-free only if the thread var occurs in its index, else CUDA.@atomic.

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

# Matches agen_stack_alloc_expr's own output: the keep_push_pop=true empty form
# Vector{Float64}()/Vector{Int64}() (1 arg), and the keep_push_pop=false pre-sized form
# Vector{Float64}(undef, size_expr)/Vector{Int64}(undef, size_expr) (3 args) -- size_expr
# itself is never validated here, just carried verbatim into the emitted rhs.
cgen_is_stack_alloc(rhs) = rhs isa Expr && rhs.head == :call &&
    rhs.args[1] isa Expr && rhs.args[1].head == :curly && rhs.args[1].args[1] == :Vector &&
    (length(rhs.args) == 1 || (length(rhs.args) == 3 && rhs.args[2] == :undef))

# The pre-sized subset of cgen_is_stack_alloc -- the only form ever read/written from inside
# a split device kernel. The empty, growing keep_push_pop=true form never needs this: every
# loop touching it contains a push!/pop!, and the stack safety rule means such a loop is
# never split, so that Vector stays a plain host Vector.
cgen_is_sized_stack_alloc(rhs) = cgen_is_stack_alloc(rhs) && length(rhs.args) == 3

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
        error("cgen_ingest: `$(expr.args[1])` is neither a valid skill-stade kernel ($(kernel_err)) nor recognizable STADE-generated code ($(generated_err))")
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
#      own copies rather than calling them -- see skill-stade-dev.md rule
#      7 and agen_collect_expr_vars!'s own comment for precedent) ----

# Collects from the loop's bounds as well as its body -- a device kernel's bounds check
# needs whatever variables lo/hi/step reference, not just what the body touches -- then
# subtracts every scalar the body assigns locally (see cgen_locally_assigned_scalars), since
# those are per-iteration temporaries, not caller-supplied arguments.
function cgen_free_vars(stmt, exclude::Symbol)
    vars = Set{Symbol}()
    cgen_collect_expr_vars!(stmt.lo, vars)
    cgen_collect_expr_vars!(stmt.hi, vars)
    cgen_collect_expr_vars!(stmt.step, vars)
    cgen_collect_vars!(stmt.body, vars)
    delete!(vars, exclude)
    setdiff!(vars, cgen_locally_assigned_scalars(stmt.body))
    return sort(collect(vars); by = string)
end

# A scalar assigned anywhere inside a loop's body is always a local temporary, never a caller-supplied argument -- true because the loop is
# iteration-independent, so it's guaranteed fresh-initialized before any read. Array names never qualify (skill-stade forbids in-kernel allocation).
# EXCEPTION: a self-referencing assignment (`cb = cb + ...`) is a cross-thread scalar reduction, the pattern cgen_device_assign special-cases for
# an array-indexed lhs; excluding it keeps it a free var, passed as a kernel argument, with the atomic rewrite in
# cgen_device_assign/jgen_device_assign.
function cgen_locally_assigned_scalars(body::Vector{NamedTuple})
    names = Set{Symbol}()
    cgen_collect_locally_assigned!(body, names)
    return names
end

function cgen_self_referencing_assign(stmt)
    stmt.kind == :assign && stmt.lhs isa Symbol || return false
    terms = cgen_flatten_sum(stmt.rhs)
    return any(t -> t == stmt.lhs, terms)
end

function cgen_collect_locally_assigned!(body::Vector{NamedTuple}, names::Set{Symbol})
    for stmt in body
        if stmt.kind == :assign
            if stmt.lhs isa Symbol && !cgen_self_referencing_assign(stmt)
                push!(names, stmt.lhs)
            end
        elseif stmt.kind == :for
            push!(names, stmt.var)
            cgen_collect_locally_assigned!(stmt.body, names)
        elseif stmt.kind == :if
            cgen_collect_locally_assigned!(stmt.then, names)
            cgen_collect_locally_assigned!(stmt.els, names)
        end
    end
    return nothing
end

# The subset of a to-be-device-split loop's free vars that are scalar cross-thread reductions -- the vars needing a 1-element device box and
# an atomic-add rewrite (cgen_device_assign/jgen_device_assign). Not every self-referencing assignment qualifies: a per-thread running sum,
# reinitialized within the same iteration before it's read, is thread-private and safe as an ordinary local. Only a var self-referenced with
# no fresh init anywhere is a genuine cross-thread accumulator: self-referenced vars, minus what cgen_locally_assigned_scalars deems local.
function cgen_scalar_reduction_vars(body::Vector{NamedTuple})
    self_ref = Set{Symbol}()
    cgen_collect_scalar_reductions!(body, self_ref)
    return setdiff(self_ref, cgen_locally_assigned_scalars(body))
end

function cgen_collect_scalar_reductions!(body::Vector{NamedTuple}, names::Set{Symbol})
    for stmt in body
        if stmt.kind == :assign
            cgen_self_referencing_assign(stmt) && push!(names, stmt.lhs)
        elseif stmt.kind == :if
            cgen_collect_scalar_reductions!(stmt.then, names)
            cgen_collect_scalar_reductions!(stmt.els, names)
        elseif stmt.kind == :for
            cgen_collect_scalar_reductions!(stmt.body, names)
        end
    end
    return nothing
end

# ---- sequential-loop reduction-eligibility check --------------------
# A sequential (i_seq_) loop is normally kept host-side, but is split when its only cross-iteration coupling is a commutative
# accumulation at a fixed index (safe, like an independent loop's atomic add), never a genuine value recurrence (same array read/written
# at different loop-var-dependent indices, e.g. `u[i]=c*u[i-1]`, always refused). A scatter-style accumulation with a data-dependent
# target is allowed through unproven, the same trade-off cgen_device_assign accepts for independent loops.
function cgen_all_assigned_scalars(body::Vector{NamedTuple})
    names = Set{Symbol}()
    cgen_collect_all_assigned!(body, names)
    return names
end

function cgen_collect_all_assigned!(body::Vector{NamedTuple}, names::Set{Symbol})
    for stmt in body
        if stmt.kind == :assign && stmt.lhs isa Symbol
            push!(names, stmt.lhs)
        elseif stmt.kind == :if
            cgen_collect_all_assigned!(stmt.then, names)
            cgen_collect_all_assigned!(stmt.els, names)
        elseif stmt.kind == :for
            cgen_collect_all_assigned!(stmt.body, names)
        end
    end
end

# ---- array-privacy proof: is a written array confined to a private, non-overlapping
#      slice per outer iteration, regardless of how many sub-loops touch it? ----
# The pairwise write/read-index check below (cgen_reduction_only_loop's own tail) treats every
# access to an array as one undifferentiated set, so it cannot tell "three sub-loops each write
# their own disjoint slice of a per-edge scratch buffer, a fourth reads the assembled whole" apart
# from a genuine value recurrence -- both look like "a write index that doesn't match some read
# index of the same array". This section adds a narrower, additive proof for exactly the safe
# shape, scoped per sub-loop rather than globally: an array is accepted here only when every
# access decomposes cleanly into (an outer-loop-invariant base, an inner loop's own additive index),
# and the group structure matches one of two provably-safe patterns (see cgen_array_private_to_loop).
# Anything that doesn't decompose this way falls straight back to the pairwise check, unchanged.

# canonicalizes an additive/subtractive expression into a sign-tagged multiset of leaf terms --
# `+`/`-` flatten at any depth, and a literal-integer multiplier expands into repeated leaves
# (`2*n` becomes two copies of `n`), so `n_in_msg` and `n_node_feat+n_node_feat+n_edge_feat` can be
# compared on equal footing once `n_in_msg`'s own definition is substituted in (see
# cgen_expand_additive_terms). Bounded to a sane literal range so a stray large constant can't blow
# this up.
function cgen_flatten_additive_terms(expr, sign::Int = 1, out::Vector{Tuple{Int,Any}} = Tuple{Int,Any}[])
    if expr isa Expr && expr.head == :call && expr.args[1] == :+
        for a in expr.args[2:end]
            cgen_flatten_additive_terms(a, sign, out)
        end
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :- && length(expr.args) == 2
        cgen_flatten_additive_terms(expr.args[2], -sign, out)
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :- && length(expr.args) == 3
        cgen_flatten_additive_terms(expr.args[2], sign, out)
        cgen_flatten_additive_terms(expr.args[3], -sign, out)
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :* && length(expr.args) == 3 &&
           expr.args[2] isa Integer && 0 <= expr.args[2] <= 64
        for _ in 1:expr.args[2]
            cgen_flatten_additive_terms(expr.args[3], sign, out)
        end
    elseif expr isa Integer && expr == 0
        # drop zero terms
    else
        push!(out, (sign, expr))
    end
    return out
end

cgen_additive_key(terms) = sort([(s, string(t)) for (s, t) in terms])

# a scalar's traced definition, for substitution below -- only ever a plain `x = expr` with no
# array read and no self-reference, so substituting it can never hide an aliasing hazard behind a
# name; anything else (conditional, array-derived, self-referential) is simply absent from the map,
# and cgen_expand_additive_terms leaves such a symbol as an opaque leaf instead of guessing.
function cgen_scalar_def_map(body::Vector{NamedTuple})
    defs = Dict{Symbol,Any}()
    for stmt in body
        stmt.kind == :assign && stmt.lhs isa Symbol || continue
        var = stmt.lhs
        if cgen_expr_has_ref(stmt.rhs) || cgen_expr_contains(stmt.rhs, var)
            delete!(defs, var)
        else
            defs[var] = stmt.rhs
        end
    end
    return defs
end

# fully expands expr into a canonical multiset, substituting any leaf symbol that has a traced
# definition, recursively, to a bounded depth (mirrors shape_propagate_int!'s own bounded-iteration
# convention elsewhere in this file)
function cgen_expand_additive_terms(expr, defs::Dict{Symbol,Any}, depth::Int = 8)
    out = Tuple{Int,Any}[]
    for (sign, term) in cgen_flatten_additive_terms(expr)
        if depth > 0 && term isa Symbol && haskey(defs, term)
            for (s2, t2) in cgen_expand_additive_terms(defs[term], defs, depth - 1)
                push!(out, (sign * s2, t2))
            end
        else
            push!(out, (sign, term))
        end
    end
    return out
end

cgen_additive_terms_equal(a, b, defs::Dict{Symbol,Any}) =
    cgen_additive_key(cgen_expand_additive_terms(a, defs)) == cgen_additive_key(cgen_expand_additive_terms(b, defs))

# strips `localvar` as a clean top-level additive term from expr (e.g. `in_off + k` strips to
# `in_off` for localvar=k), refusing if any OTHER chain variable also appears -- a mixed index like
# `base + i_loc + i_k` naming two different sub-loops' variables in one expression isn't a shape this
# proof attempts to reason about, so it's left to the pairwise fallback instead of guessed at.
function cgen_strip_local_additive_term(expr, localvar::Symbol, other_chain_vars::Vector{Symbol})
    any(v -> cgen_expr_contains(expr, v), other_chain_vars) && return (nothing, false)
    if expr === localvar
        return (0, true)
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :+
        args = expr.args[2:end]
        idx = findfirst(a -> a === localvar, args)
        idx === nothing && return (nothing, false)
        rest = [args[i] for i in eachindex(args) if i != idx]
        isempty(rest) && return (0, true)
        length(rest) == 1 && return (rest[1], true)
        return (Expr(:call, :+, rest...), true)
    else
        return (nothing, false)
    end
end

# collects every occurrence of `arr` within one top-level statement of the candidate loop, at any
# nesting depth, tagged with the full chain of enclosing loop vars paired with their own `.hi`
# (never just the variable name -- sibling sub-loops routinely reuse the same loop-variable name
# with different trip counts, e.g. three separate `for k = 1:...` loops in the same kernel, so a
# name-keyed trip-count table would silently collide). Bails (returns `nothing`) the moment `arr`
# is touched inside an `:if` -- reasoning through a conditional's effect on which slice gets
# written is out of scope here, so that case is left to the pairwise fallback -- or a for-loop
# whose bounds aren't a literal `1:hi` step-1 range, since the additive stripping below assumes
# that shape.
function cgen_deep_array_occurrences!(stmts, arr::Symbol, chain::Vector{Tuple{Symbol,Any}}, out)
    for stmt in stmts
        if stmt.kind == :assign
            if stmt.lhs isa Expr && stmt.lhs.head == :ref && stmt.lhs.args[1] == arr
                length(stmt.lhs.args) == 2 || return nothing
                push!(out, (:write, stmt.lhs.args[2], copy(chain)))
            end
            refs = Dict{Any,Vector{Any}}()
            cgen_collect_refs!(stmt.rhs, refs)
            for idxs in get(refs, arr, Any[])
                length(idxs) == 1 || return nothing
                push!(out, (:read, idxs[1], copy(chain)))
            end
        elseif stmt.kind == :for
            (stmt.lo isa Integer && stmt.lo == 1 && stmt.step isa Integer && stmt.step == 1) || return nothing
            cgen_deep_array_occurrences!(stmt.body, arr, vcat(chain, [(stmt.var, stmt.hi)]), out) === nothing && return nothing
        elseif stmt.kind == :if
            writes = Dict{Any,Vector{Any}}(); reads = Dict{Any,Vector{Any}}()
            cgen_collect_array_accesses!(NamedTuple[stmt], writes, reads)
            (haskey(writes, arr) || haskey(reads, arr)) && return nothing
        end
    end
    return out
end

# picks the one chain variable actually present in idx_expr (requiring exactly one -- an index
# mixing two different sub-loop variables isn't attempted here) and strips it, returning
# (base, trip) -- trip is `nothing` for a top-level occurrence with no enclosing sub-loop at all.
function cgen_classify_array_occurrence(idx_expr, chain::Vector{Tuple{Symbol,Any}})
    isempty(chain) && return (idx_expr, nothing)
    present = [(v, t) for (v, t) in chain if cgen_expr_contains(idx_expr, v)]
    length(present) == 1 || return nothing
    (v, trip) = present[1]
    others = [c for (c, _) in chain if c != v]
    (base, ok) = cgen_strip_local_additive_term(idx_expr, v, others)
    ok || return nothing
    return (base, trip)
end

# The proof itself. Groups every occurrence of `arr` within the candidate loop's body by which
# top-level statement it came from (a plain assignment, or an immediately- or more-deeply-nested
# sub-loop), requires each group to be internally self-consistent (its own write and read indices,
# after stripping any local sub-loop variable, all agree -- the same requirement the pairwise check
# already makes, just scoped per group instead of globally), then accepts the whole array under
# either of two provably-safe patterns:
#
#   1. No group is a pure reader (every group either only writes, or reads back exactly what it
#      itself just wrote). Cross-group relationships never matter for thread safety in this case --
#      each group is independently either a self-contained accumulation or a write nothing else in
#      this loop depends on -- so whether different groups' bases could coincide across different
#      outer iterations is cgen_device_assign's job (atomic vs plain), never this proof's.
#   2. Every group shares the exact same (base, trip) -- repeatedly touching one already-established
#      sub-region (write, then read-modify-write, then plain read, all at the same offset).
#   3. A cumulative-assembly pattern: one reading group (optionally also writing, already
#      self-consistent) consumes a base plus some trip-sized range; zero or more write-only groups,
#      each earlier in program order, each add their own trip to a running total starting from that
#      same base; the reader's own range must equal that running total exactly. This is the shape
#      that lets `msg_input[in_off+k]`, `msg_input[in_off+n_node_feat+k]`, and
#      `msg_input[in_off+2n_node_feat+k]` (three private sub-ranges) be proven to add up to exactly
#      what `msg_input[in_off+i_seq_i]` (for i_seq_i = 1:n_in_msg) reads back, once n_in_msg's own
#      definition is expanded and matches term-for-term.
#
# Anything not covered -- multiple pure-reader groups, a reader before some of its writers, a
# group whose own accesses don't line up, a multi-dimensional ref, arr touched inside an `:if` --
# returns `nothing`, meaning "not proven either way", and the caller falls back to the pairwise
# check for that array exactly as before this proof existed.
function cgen_array_private_to_loop(body::Vector{NamedTuple}, arr::Symbol, defs::Dict{Symbol,Any})
    occ = Tuple{Symbol,Any,Any,Int}[]   # (kind, base, trip_or_nothing, group_id)
    for (gi, stmt) in enumerate(body)
        raw = Tuple{Symbol,Any,Vector{Tuple{Symbol,Any}}}[]
        if stmt.kind in (:assign, :for)
            cgen_deep_array_occurrences!([stmt], arr, Tuple{Symbol,Any}[], raw) === nothing && return nothing
        elseif stmt.kind == :if
            writes = Dict{Any,Vector{Any}}(); reads = Dict{Any,Vector{Any}}()
            cgen_collect_array_accesses!(NamedTuple[stmt], writes, reads)
            (haskey(writes, arr) || haskey(reads, arr)) && return nothing
            continue
        end
        isempty(raw) && continue
        bases = Any[]; trips = Any[]; kinds = Symbol[]
        for (k, idx, chain) in raw
            cls = cgen_classify_array_occurrence(idx, chain)
            cls === nothing && return nothing
            (base, trip) = cls
            push!(bases, base); push!(trips, trip); push!(kinds, k)
        end
        length(unique(cgen_additive_key(cgen_expand_additive_terms(b, defs)) for b in bases)) == 1 || return nothing
        length(unique(t === nothing ? nothing : cgen_additive_key(cgen_expand_additive_terms(t, defs)) for t in trips)) == 1 || return nothing
        haswrite = :write in kinds; hasread = :read in kinds
        push!(occ, (haswrite && hasread ? :both : haswrite ? :write : :read, bases[1], trips[1], gi))
    end
    isempty(occ) && return true

    # pattern 1: no pure-reader group
    any(o -> o[1] == :read, occ) || return true

    # pattern 2: every group is the same sub-region
    same_base = unique(cgen_additive_key(cgen_expand_additive_terms(o[2], defs)) for o in occ)
    same_trip = unique(o[3] === nothing ? nothing : cgen_additive_key(cgen_expand_additive_terms(o[3], defs)) for o in occ)
    (length(same_base) == 1 && length(same_trip) == 1) && return true

    # pattern 3: cumulative assembly, one reader, writers strictly earlier
    readers = filter(o -> o[1] in (:read, :both), occ)
    length(readers) == 1 || return nothing
    R = readers[1]
    writers = filter(o -> o[1] == :write, occ)
    all(w -> w[4] < R[4], writers) || return nothing
    sort!(writers, by = w -> w[4])
    cum = 0
    for w in writers
        expected_base = cum == 0 ? R[2] : Expr(:call, :+, R[2], cum)
        cgen_additive_terms_equal(w[2], expected_base, defs) || return nothing
        cum = cum == 0 ? w[3] : Expr(:call, :+, cum, w[3])
    end
    if R[3] === nothing
        cum == 0 || return nothing
    else
        cgen_additive_terms_equal(R[3], cum, defs) || return nothing
    end
    return true
end

function cgen_reduction_only_loop(body::Vector{NamedTuple}, loopvar::Symbol, known_consts::Dict{Symbol,Any} = Dict{Symbol,Any}(), outer_defs::Dict{Symbol,Any} = Dict{Symbol,Any}())
    local_names = cgen_locally_assigned_scalars(body)
# A locally-assigned scalar that fails the plain use-before-def walk below may still be safe if it provably converges to
# the same literal on every control-flow path, PROVIDED that constant matches the value known coming into the loop
# (known_consts, from a top-level literal assignment immediately preceding this loop). By induction every iteration then
# starts at that value, so injecting it as the kernel body's first statement reproduces sequential semantics. Without a
# matching known_consts entry, the loop stays refused.
    synth = Dict{Symbol,Any}()
    for v in local_names
        haskey(known_consts, v) || continue
        c = cgen_loop_convergent_constant(body, v)
        c !== nothing && c == known_consts[v] && (synth[v] = c)
    end
    cgen_use_before_def(body, local_names, synth)[1] || return nothing
    writes = Dict{Any,Vector{Any}}()
    reads = Dict{Any,Vector{Any}}()
    cgen_collect_array_accesses!(body, writes, reads)
# Deliberately not scoped to index expressions mentioning the loop's own variable: a sweep-style
# recurrence can carry its cross-iteration coupling through an inner loop's index, with no mention of
# the outer i_seq_ variable at all. The loop var doesn't have to appear in an index for reading and
# writing the same array at two indices to be a genuine recurrence -- any structural mismatch between
# a write and read index of the same array disqualifies the loop, unconditionally -- UNLESS
# cgen_array_private_to_loop can prove that array's whole access pattern is confined to a private,
# non-overlapping per-outer-iteration region despite the mismatch (see that function's own comment).
# Only tried for a written array; a purely-read array never reaches this loop at all, and never needed
# proving in the first place, since concurrent reads of shared memory are always safe on their own.
    array_defs = merge(outer_defs, cgen_scalar_def_map(body))
    for (arr, widxs) in writes
        cgen_array_private_to_loop(body, arr, array_defs) === true && continue
        ridxs = get(reads, arr, Any[])
        for w in widxs, r in ridxs
            w == r && continue
            return nothing
        end
    end
    return synth
end

# Is `var` assigned anywhere within body, at any nesting depth? Used only to bail out of the
# convergence analysis below when a nested loop touches the same variable -- proving
# convergence through a sub-loop's own repeated execution needs its own fixed-point
# argument, not needed for any corpus kernel today, so it's simplest and safest to decline.
function cgen_var_assigned_anywhere(body::Vector{NamedTuple}, var::Symbol)
    for stmt in body
        if stmt.kind == :assign && stmt.lhs === var
            return true
        elseif stmt.kind == :if
            (cgen_var_assigned_anywhere(stmt.then, var) || cgen_var_assigned_anywhere(stmt.els, var)) && return true
        elseif stmt.kind == :for
            cgen_var_assigned_anywhere(stmt.body, var) && return true
        end
    end
    return false
end

# The value `var` provably holds at the end of one traversal of body, if it's the same
# literal on every control-flow path, else `nothing`. Walks in program order threading a
# running state: :unchanged (not touched yet), :unknown (touched but not provably a single
# literal), or a Number (every touch since the last branch point agrees on this exact
# literal).
function cgen_loop_convergent_constant(body::Vector{NamedTuple}, var::Symbol)
    state = cgen_terminal_value_walk(body, var, :unchanged)
    return state isa Number ? state : nothing
end

function cgen_terminal_value_walk(body::Vector{NamedTuple}, var::Symbol, state)
    for stmt in body
        if stmt.kind == :assign && stmt.lhs === var
            state = stmt.rhs isa Number ? stmt.rhs : :unknown
        elseif stmt.kind == :if
            then_state = cgen_terminal_value_walk(stmt.then, var, state)
            els_state = cgen_terminal_value_walk(stmt.els, var, state)
            state = if then_state isa Number && els_state isa Number && then_state == els_state
                then_state
            elseif then_state == :unchanged && els_state == :unchanged
                :unchanged
            else
                :unknown
            end
        elseif stmt.kind == :for
            if cgen_var_assigned_anywhere(stmt.body, var)
                # A var touched inside a nested loop isn't automatically :unknown -- if that nested loop's
                # own body provably converges `var` to the same literal on every one of its control-flow
                # paths too (recursing exactly as the top-level caller does), that literal is what `var`
                # holds after the nested loop finishes, regardless of what it held entering it. Needed once
                # this proof runs on loops whose reset lives one level deeper than the reduction itself.
                inner = cgen_loop_convergent_constant(stmt.body, var)
                state = inner === nothing ? :unknown : inner
            end
        end
    end
    return state
end

# Program-order walk over vars cgen_locally_assigned_scalars calls thread-private. Requires every self-referencing use to be
# preceded, within this loop's body, by an assignment -- or listed in `synth` (cgen_reduction_only_loop's convergent-constant
# proof), since cgen_kernel_def injects `var = synth[var]` first. Narrower than cgen_locally_assigned_scalars, correct only for
# iteration-independent loops: catches a var whose sole reset sits before the loop, which would otherwise read undefined in the
# per-thread kernel -- confirmed on a live GPU run, now recovered via `synth`.
cgen_read_vars(e) = (s = Set{Symbol}(); cgen_collect_expr_vars!(e, s); s)

# Generalizes cgen_reduction_only_scalar_walk's self-referencing-only check to any read of a local_names var, anywhere in an
# expression -- not just on its own rhs. A locally-assigned scalar can carry a cross-iteration dependency without ever self-
# referencing on its own lhs, e.g. a stack-restored value consumed by an earlier statement and only refreshed by a later one (a
# prefetch-for-next-iteration pattern valid only under strict sequential order). This subsumes the self-assign-only check,
# replacing it as cgen_reduction_only_loop's safety gate.
function cgen_use_before_def(body::Vector{NamedTuple}, local_names::Set{Symbol}, synth::Dict{Symbol,Any}, defined_in::Set{Symbol} = Set{Symbol}())
    defined = union(defined_in, Set{Symbol}(keys(synth)))
    flagged(vars) = any(v -> v in local_names && !(v in defined), vars)
    for stmt in body
        if stmt.kind == :assign
            flagged(cgen_read_vars(stmt.rhs)) && return (false, defined)
            if stmt.lhs isa Expr && stmt.lhs.head == :ref
                flagged(cgen_read_vars(stmt.lhs.args[2:end])) && return (false, defined)
            elseif stmt.lhs isa Symbol
                push!(defined, stmt.lhs)
            end
        elseif stmt.kind == :if
            flagged(cgen_read_vars(stmt.cond)) && return (false, defined)
            ok_then, defined_then = cgen_use_before_def(stmt.then, local_names, synth, defined)
            ok_then || return (false, defined)
            ok_els, defined_els = cgen_use_before_def(stmt.els, local_names, synth, defined)
            ok_els || return (false, defined)
            defined = intersect(defined_then, defined_els)
        elseif stmt.kind == :for
            flagged(cgen_read_vars(stmt.lo)) && return (false, defined)
            flagged(cgen_read_vars(stmt.hi)) && return (false, defined)
            flagged(cgen_read_vars(stmt.step)) && return (false, defined)
            # the loop's own iteration variable is bound by its header,
            # hence always "defined" throughout its body -- not a read-
            # before-def case at all, unlike a genuine local scalar.
            ok, _ = cgen_use_before_def(stmt.body, local_names, synth, union(defined, Set([stmt.var])))
            ok || return (false, defined)
        end
    end
    return (true, defined)
end

# collects every array read/write index expression
function cgen_collect_array_accesses!(body::Vector{NamedTuple}, writes::Dict, reads::Dict)
    for stmt in body
        if stmt.kind == :assign
            if stmt.lhs isa Expr && stmt.lhs.head == :ref
                push!(get!(writes, stmt.lhs.args[1], Any[]), stmt.lhs.args[2:end])
                for a in stmt.lhs.args[2:end]
                    cgen_collect_refs!(a, reads)
                end
            end
            cgen_collect_refs!(stmt.rhs, reads)
        elseif stmt.kind == :if
            cgen_collect_refs!(stmt.cond, reads)
            cgen_collect_array_accesses!(stmt.then, writes, reads)
            cgen_collect_array_accesses!(stmt.els, writes, reads)
        elseif stmt.kind == :for
            cgen_collect_array_accesses!(stmt.body, writes, reads)
        end
    end
end

function cgen_collect_refs!(e, refs::Dict)
    if e isa Expr
        if e.head == :ref
            push!(get!(refs, e.args[1], Any[]), e.args[2:end])
        end
        for a in e.args
            cgen_collect_refs!(a, refs)
        end
    end
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
# gpu_backend :: (suffix, kernel_tag, launch_macro, threads_kw, blocks_kw, tid_rhs, atomic_macro, allowscalar_macro, preamble, default_precision,
# precision_locked, precision_lock_reason). Only the launch macro, launch keywords, thread-index intrinsic, atomic/allowscalar modules, and preamble differ
# between CUDA.jl/AMDGPU.jl/Metal.jl. precision_locked exists because Metal has no FP64 hardware and refuses Float64; it makes stade_gpu refuse a non-default
# request at codegen time rather than emit source guaranteed to fail on real hardware.

function cgen_backend_cuda()
    return (
        suffix = "_cuda",
        kernel_tag = "cuda",
        launch_macro = Symbol("@cuda"),
        threads_kw = :threads,
        blocks_kw = :blocks,
        tid_rhs = :((blockIdx().x - 1) * blockDim().x + threadIdx().x),
        atomic_macro = Expr(:., :CUDA, QuoteNode(Symbol("@atomic"))),
        allowscalar_macro = Expr(:., :CUDA, QuoteNode(Symbol("@allowscalar"))),
        arrtype = :CuArray,
        preamble = "import Pkg\nhaskey(Pkg.project().dependencies, \"CUDA\") || Pkg.add(\"CUDA\")\nusing CUDA\nusing LinearAlgebra\nCUDA.allowscalar(false)\n",
        default_precision = Float64,
        precision_locked = false,
        precision_lock_reason = "",
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
        allowscalar_macro = Expr(:., :AMDGPU, QuoteNode(Symbol("@allowscalar"))),
        arrtype = :ROCArray,
        preamble = "import Pkg\nhaskey(Pkg.project().dependencies, \"AMDGPU\") || Pkg.add(\"AMDGPU\")\nusing AMDGPU\nusing LinearAlgebra\nAMDGPU.allowscalar(false)\n",
        default_precision = Float64,
        precision_locked = false,
        precision_lock_reason = "",
    )
end

# thread_position_in_grid().x is already a global 1-based index (no manual affine combination needed, unlike CUDA/AMDGPU);
# `groups=` is the current launch keyword (the _1d/_2d/_3d-suffixed intrinsics are deprecated as of Metal.jl v1.9).
# Metal.@atomic works the same way CUDA.jl's does. `^` is not accounted for even though Metal has a real compiler bug
# (Float32^Integer computed in double precision internally, JuliaGPU/Metal.jl#552) -- a backend bug, not something an Expr-
# level rewrite could fix, so it's a caveat: avoid `^` in Metal-bound kernels' innermost loops until confirmed fixed.
function cgen_backend_metal()
    return (
        suffix = "_metal",
        kernel_tag = "metal",
        launch_macro = Symbol("@metal"),
        threads_kw = :threads,
        blocks_kw = :groups,
        tid_rhs = :(thread_position_in_grid().x),
        atomic_macro = Expr(:., :Metal, QuoteNode(Symbol("@atomic"))),
        allowscalar_macro = Expr(:., :Metal, QuoteNode(Symbol("@allowscalar"))),
        arrtype = :MtlArray,
        preamble = "import Pkg\nhaskey(Pkg.project().dependencies, \"Metal\") || Pkg.add(\"Metal\")\nusing Metal\nusing LinearAlgebra\nMetal.allowscalar(false)\n",
        default_precision = Float32,
        precision_locked = true,
        precision_lock_reason = "Apple GPUs have no FP64 hardware -- Metal.jl disallows constructing Float64 arrays at all, and any kernel whose arithmetic touches a Float64 fails to compile",
    )
end

# Device residency for a keep_push_pop=false stack: agen_init_emit always writes it as a plain host `Vector{T}(undef,
# size_expr)`, since agen_ has no notion of a GPU backend -- turning that into a real on-device allocation is cgen_'s
# job. Only fires for the pre-sized :indexed form, the only stack shape a split device kernel ever touches directly.
# `undef` is preserved rather than zero-filled: every element gets written by a push before its first read, so a
# device-side `undef` allocation costs nothing.
function cgen_stack_device_expr(rhs::Expr, backend)
    T = rhs.args[1].args[2]
    size_expr = rhs.args[3]
    return Expr(:call, Expr(:curly, backend.arrtype, T), :undef, size_expr)
end

# ---- idiomatic scalar-reduction detection (keep_all_atomic=false) ----
# Recognizes the narrow shape `target = target +/- f(arr_1[loopvar], ..., free scalars)` as the whole and only statement in a loop body
# already proven safe to split -- a bare dot-product-style `loss[1]=loss[1]+u[i]*v[i]`, not a case with an :if picking the term (a second
# statement, so this declines, falling back to today's atomic-kernel codegen). Returns `(target, op, arrs, term)` or `nothing`; the caller
# turns `(arrs, term, loopvar)` into a replacement expression, which differs by target.
function cgen_idiomatic_scalar_reduction(body::Vector{NamedTuple}, loopvar::Symbol)
    length(body) == 1 || return nothing
    stmt = body[1]
    stmt.kind == :assign || return nothing
    target = stmt.lhs
    target isa Expr && target.head == :ref || return nothing
    cgen_expr_contains(target, loopvar) && return nothing   # target must be thread-invariant
    rhs = stmt.rhs
    rhs isa Expr && rhs.head == :call && length(rhs.args) == 3 || return nothing
    op = rhs.args[1]
    op in (:+, :-) || return nothing
    a, b = rhs.args[2], rhs.args[3]
    if a == target
        term = b
    elseif b == target && op == :+   # b - a with a==target would mean `target - term`, not `term - target` -- only the `+` (commutative) form is safe to read this way
        term = a
    else
        return nothing
    end
    refs = Dict{Any,Vector{Any}}()
    cgen_collect_refs!(term, refs)
    isempty(refs) && return nothing
    arrs = Symbol[]
    for (arr, idxs) in refs
        arr isa Symbol || return nothing
        for idx in idxs
            length(idx) == 1 && idx[1] === loopvar || return nothing
        end
        push!(arrs, arr)
    end
    cgen_bare_var_outside_index(term, loopvar) && return nothing
    sort!(arrs; by = string)   # deterministic output only, doesn't affect correctness
    return (target, op, arrs, term)
end

# True iff `loopvar` occurs anywhere in `e` outside of a :ref's index position -- the per-
# iteration term uses the loop variable for something other than indexing one of the arrays
# it reads (cgen_idiomatic_scalar_reduction already verified every :ref's index IS exactly
# loopvar; this only rules out a bare use elsewhere, e.g. a hypothetical `u[i] + i`).
function cgen_bare_var_outside_index(e, loopvar::Symbol)
    if e isa Symbol
        return e === loopvar
    elseif e isa Expr
        e.head == :ref && return false
        return any(a -> cgen_bare_var_outside_index(a, loopvar), e.args)
    end
    return false
end

# Rewrites every `arr[loopvar]` occurrence in `e` (arr a key of subst)
# to the corresponding closure-argument symbol, leaving everything else
# untouched -- used to turn a proven idiomatic-reduction term into a
# mapreduce closure body operating on plain scalar values rather than
# array elements.
function cgen_substitute_indexed_refs(e, subst::Dict{Symbol,Symbol}, loopvar::Symbol)
    if e isa Expr && e.head == :ref && length(e.args) == 2 &&
       e.args[1] isa Symbol && e.args[2] === loopvar && haskey(subst, e.args[1])
        return subst[e.args[1]]
    elseif e isa Expr
        return Expr(e.head, [cgen_substitute_indexed_refs(a, subst, loopvar) for a in e.args]...)
    else
        return e
    end
end

# CUDA/AMDGPU/Metal: `dot`/`sum(abs2, ·)` and `mapreduce` all dispatch on the array TYPE of their
# argument (GPUArrays.jl's generic machinery, or a vendor cuBLAS/rocBLAS dot method), so the same call
# works on every backend -- they pick whichever implementation is fastest at runtime, no backend-
# conditional codegen needed. Both come from `using LinearAlgebra`/Base, present in every GPU
# backend's preamble.
function cgen_idiomatic_reduction_value(arrs::Vector{Symbol}, term, loopvar::Symbol)
    if length(arrs) == 2 &&
       term.head == :call && term.args[1] == :* && length(term.args) == 3 &&
       Set(Any[term.args[2], term.args[3]]) == Set(Any[Expr(:ref, arrs[1], loopvar), Expr(:ref, arrs[2], loopvar)])
        return Expr(:call, :dot, arrs[1], arrs[2])
    elseif length(arrs) == 1 && term == Expr(:call, :^, Expr(:ref, arrs[1], loopvar), 2)
        return Expr(:call, :sum, :abs2, arrs[1])
    else
        closure_args = [Symbol(:__mr_, i) for i in eachindex(arrs)]
        subst = Dict(arrs[i] => closure_args[i] for i in eachindex(arrs))
        closure_body = cgen_substitute_indexed_refs(term, subst, loopvar)
        return Expr(:call, :mapreduce, Expr(:->, Expr(:tuple, closure_args...), closure_body), :+, arrs...)
    end
end

# JACC: no confirmed BLAS-level acceleration to special-case, so every matched shape goes through the
# same `JACC.@parallel_reduce range=N f(args...)` primitive, replacing the `@parallel_for` +
# `Atomix.@atomic` kernel it would otherwise get. Unlike CUDA/AMDGPU/Metal, `term` needs no
# substitution: JACC's convention is a closure whose first parameter IS the loop index, so reusing
# loopvar's name and leaving `arr[loopvar]` as written is already correct.
function jgen_idiomatic_reduction_value(arrs::Vector{Symbol}, term, loopvar::Symbol, n_iter)
    closure = Expr(:->, Expr(:tuple, loopvar, arrs...), term)
    return Expr(:macrocall, Expr(:., :JACC, QuoteNode(Symbol("@parallel_reduce"))), nothing,
                Expr(:(=), :range, n_iter), Expr(:call, closure, arrs...))
end

# ---- JACC idiomatic-reduction write-back (keep_all_atomic=false) ---
# `target` (an `arr[idx...]` ref) must never be read or written from the host once `arr` may be a JACC device array. This accumulates `target
# += value` inside a trivial one-thread device kernel instead, reusing the `Atomix.@atomic target += ...` shape jgen_device_assign already
# emits for reduce_vars. `__jgen_redval` is a synthetic parameter for the reduction result, passed through as a `CuArray{Float64,1}` --
# `JACC.@parallel_reduce` already returns one directly, not a host Float64 -- and read via `[1]` on-device, like any other boxed scalar.
function jgen_reduction_writeback_kernel(owner::Symbol, idx::Int, target::Expr, wfargs::Vector{Symbol})
    jidx = :__jacc_i   # unused (range=1), but every JACC kernel takes the index as its first param -- see jgen_kernel_def's comment
    redval = :__jgen_redval
    body = Any[
        Expr(:macrocall, Expr(:., :Atomix, QuoteNode(Symbol("@atomic"))), nothing,
             Expr(:(+=), target, Expr(:ref, redval, 1))),
        emit_return_nothing(),
    ]
    return Expr(:function, Expr(:call, jgen_kernel_fname(owner, idx), jidx, wfargs..., redval), Expr(:block, body...))
end

function jgen_reduction_writeback_launch(owner::Symbol, idx::Int, wfargs::Vector{Symbol}, value)
    return Expr(:macrocall, Expr(:., :JACC, QuoteNode(Symbol("@parallel_for"))), nothing,
                Expr(:(=), :range, 1),
                Expr(:call, jgen_kernel_fname(owner, idx), wfargs..., value))
end


# True iff `e` contains an `Expr(:ref, ...)` anywhere -- an element-wise array index -- at
# any depth. Used by cgen_body to spot host-side statements that touch what may be a device
# array; scalar arguments/locals never appear as a :ref's base, so this is exactly the 'is
# this legal under allowscalar(false)' test, no kind/type lookup needed.
function cgen_expr_has_ref(e)
    e isa Expr || return false
    e.head == :ref && return true
    return any(cgen_expr_has_ref, e.args)
end

# ---- host-side body walk: splits device kernels off ----
# Host-side body walk: splits off one device kernel per eligible iteration-independent loop; anything left over runs as ordinary host-side Julia. A
# left-over statement touching a device array is what CUDA.allowscalar(false) rejects -- confirmed as a live-GPU crash. Fix: each run of consecutive
# :assign statements is wrapped in one allowscalar_macro block if any touches an array, else emitted unwrapped. keep_all_atomic=false additionally
# offers a splittable loop to cgen_idiomatic_scalar_reduction, replacing it with one dot/sum/mapreduce call when it matches that narrow shape.
# outer_defs: scalar definitions from the whole kernel's own top-level body (see
# cgen_scalar_def_map), built once by cgen_emit and threaded through unchanged at every
# recursion level -- cgen_reduction_only_loop's array-privacy proof needs it to see through a
# kernel-level size relationship like `n_in_msg = 2 * n_node_feat + n_edge_feat`, defined well
# outside any candidate loop's own body.
function cgen_body(body::Vector{NamedTuple}, kernels::Vector{Expr}, owner::Symbol, backend, reduce_vars::Set{Symbol}, fn_args::Set{Symbol}; keep_all_atomic::Bool = true, outer_defs::Dict{Symbol,Any} = Dict{Symbol,Any}())
    exprs = Any[]
    known_consts = Dict{Symbol,Any}()
    pending = Any[]
    pending_has_array = false
    flush_pending! = () -> begin
        if !isempty(pending)
            if pending_has_array
                push!(exprs, Expr(:macrocall, backend.allowscalar_macro, nothing, Expr(:block, pending...)))
            else
                append!(exprs, pending)
            end
            empty!(pending)
            pending_has_array = false
        end
    end
    for stmt in body
        if stmt.kind == :stackpush
            flush_pending!()
            push!(exprs, Expr(:call, :push!, stmt.stack, stmt.value))
        elseif stmt.kind == :assign
            rhs = cgen_is_sized_stack_alloc(stmt.rhs) ? cgen_stack_device_expr(stmt.rhs, backend) : stmt.rhs
            push!(pending, Expr(:(=), stmt.lhs, rhs))
            pending_has_array = pending_has_array || cgen_expr_has_ref(stmt.lhs) || cgen_expr_has_ref(rhs)
            # Tracked purely so a later sequential loop in this same body can prove a locally-
            # assigned scalar's true entering value (cgen_reduction_only_loop's convergent-
            # constant check) -- see how a `wb = 0.0` right before its reverse sweep gets used
            # this way. Only a bare literal counts; anything else invalidates the symbol,
            # since we have no way to prove it stayed constant either.
            if stmt.lhs isa Symbol
                if stmt.rhs isa Number
                    known_consts[stmt.lhs] = stmt.rhs
                else
                    delete!(known_consts, stmt.lhs)
                end
            end
        elseif stmt.kind == :if
            flush_pending!()
            cond = cgen_expr_has_ref(stmt.cond) ? Expr(:macrocall, backend.allowscalar_macro, nothing, stmt.cond) : stmt.cond
            push!(exprs, emit_if(cond, cgen_body(stmt.then, kernels, owner, backend, reduce_vars, fn_args; keep_all_atomic, outer_defs), cgen_body(stmt.els, kernels, owner, backend, reduce_vars, fn_args; keep_all_atomic, outer_defs)))
            for v in cgen_all_assigned_scalars(vcat(stmt.then, stmt.els))
                delete!(known_consts, v)
            end
        elseif stmt.kind == :for
            flush_pending!()
            # Previously only a loop already marked sequential ran this check; an independent loop got an unconditional
            # empty synth. That's false whenever the loop's body carries a scalar whose true entering value crosses the
            # loop boundary (a reverse-mode reduction shadow reset each iteration but never freshly initialized at the
            # first). Running the check unconditionally costs nothing when no such scalar exists, and otherwise either
            # recovers it via synth or refuses to split -- never silently emitting a kernel reading an undefined value.
            synth = cgen_reduction_only_loop(stmt.body, stmt.var, known_consts, outer_defs)
            # A var cgen_scalar_reduction_vars calls reduction-worthy is only safe for cgen_emit's global box-once/unbox-once treatment when it's a
            # whole-function-lifetime entity (a top-level argument like `cb`). An internal scratch scalar can satisfy the same shape by accident when
            # an enclosing loop demotes for an unrelated reason, its true reset ending up a sibling statement one level up. Globally boxing it is
            # wrong two ways: the box precedes its own host-side reset (UndefVarError), and even fixed, one shared box accumulates across launches
            # meant to start fresh. Refusing to split is the same conservative choice as any unprovable case: slower, never wrong.
            loop_reduce_vars = synth === nothing ? Set{Symbol}() : cgen_scalar_reduction_vars(stmt.body)
            safe_scope = issubset(loop_reduce_vars, fn_args)
            if synth !== nothing && safe_scope && !cgen_contains_stackop(stmt.body)
                red = keep_all_atomic ? nothing : cgen_idiomatic_scalar_reduction(stmt.body, stmt.var)
                if red !== nothing
                    target, op, arrs, term = red
                    value = cgen_idiomatic_reduction_value(arrs, term, stmt.var)
                    push!(pending, Expr(:(=), target, Expr(:call, op, target, value)))
                    pending_has_array = true
                else
                    idx = length(kernels) + 1
                    fargs = cgen_free_vars(stmt, stmt.var)
                    union!(reduce_vars, loop_reduce_vars)
                    push!(kernels, cgen_kernel_def(stmt, owner, idx, fargs, backend, loop_reduce_vars, synth))
                    push!(exprs, cgen_launch_expr(stmt, owner, idx, fargs, backend))
                end
            else
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, cgen_body(stmt.body, kernels, owner, backend, reduce_vars, fn_args; keep_all_atomic, outer_defs)))
            end
            delete!(known_consts, stmt.var)
        end
    end
    flush_pending!()
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

function cgen_kernel_def(stmt, owner::Symbol, idx::Int, fargs::Vector{Symbol}, backend, reduce_vars::Set{Symbol}, synth::Dict{Symbol,Any} = Dict{Symbol,Any}())
    tid = :__tid
    n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)
    body = Any[
        Expr(:(=), tid, backend.tid_rhs),
        Expr(:if, Expr(:call, :>, tid, n_iter), Expr(:block, emit_return_nothing())),
        Expr(:(=), stmt.var, cgen_loopvar_from_tid(stmt.lo, stmt.step, tid)),
    ]
    # cgen_reduction_only_loop's convergent-constant proof: a
    # locally-assigned scalar with no in-body fresh initializer (its
    # true one sits outside the loop entirely) gets its provably-true
    # entering value injected here, before the loop's own body runs --
    # see the convergent-constant discussion above.
    for (v, c) in synth
        push!(body, Expr(:(=), v, c))
    end
    append!(body, cgen_device_body(stmt.body, stmt.var, backend, reduce_vars))
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

# True iff `x` provably depends on the thread var only through pure arithmetic -- safe to trust as per-thread-unique.
# Two things break that proof: (1) any array read anywhere in the expression, since a gather can return the same
# value for two different thread-var values, tainting everything downstream; (2) any symbol that's thread-dependent
# but not already proven injective, since the taint propagates through a chained let-binding. A thread-independent
# symbol is always fine.
function cgen_expr_injective_ok(x, injective_dep::Set{Symbol}, thread_dep::Set{Symbol})
    x isa Symbol && return !(x in thread_dep) || (x in injective_dep)
    x isa Expr || return true
    x.head == :ref && return false
    # div/mod/rem/fld/cld collapse multiple thread-index values onto the same result by construction -- e.g. a 2x upsample's `div(oim1,
    # scale)` maps every pair of adjacent output positions to the same source row, so a write index built from it collides across
    # threads. Recursing into their args asks the wrong question, since div/mod throw injectivity away regardless of their operands, and
    # there's no general way to prove a specific division is exact -- confirmed live: a plain write here silently dropped a fraction of
    # the correct accumulation on a real GPU run.
    x.head == :call && x.args[1] in (:div, :mod, :rem, :fld, :cld, :÷, :%) && return false
    return all(a -> cgen_expr_injective_ok(a, injective_dep, thread_dep), x.args)
end
cgen_expr_injective_ok(xs::Vector, injective_dep::Set{Symbol}, thread_dep::Set{Symbol}) =
    all(x -> cgen_expr_injective_ok(x, injective_dep, thread_dep), xs)

# device-side body walk -- never sees :stackpush or a pop!-rhs assign,
# since cgen_body only reaches here for a loop cgen_contains_stackop
# already confirmed is clean at every depth
function cgen_device_body(body::Vector{NamedTuple}, thread_var::Symbol, backend, reduce_vars::Set{Symbol}, thread_dep::Set{Symbol} = Set([thread_var]), injective_dep::Set{Symbol} = Set([thread_var]))
    exprs = Any[]
    for stmt in body
        if stmt.kind == :assign
            push!(exprs, cgen_device_assign(stmt, thread_var, thread_dep, injective_dep, backend, reduce_vars))
            # A write's index can be computed through a same-body scalar let-binding one or more hops from the thread variable, not just written literally
            # in the index -- extend thread_dep transitively so cgen_device_assign's occurs-check sees through it, tracking CURRENT dependency per
            # variable (removed on a reassignment breaking the chain). injective_dep is tracked the same way, one level stricter: a chain stays injective
            # only while every hop is pure arithmetic on already-injective symbols with no array read in the chain.
            if stmt.lhs isa Symbol
                if cgen_expr_contains_any(stmt.rhs, thread_dep)
                    push!(thread_dep, stmt.lhs)
                    if cgen_expr_injective_ok(stmt.rhs, injective_dep, thread_dep)
                        push!(injective_dep, stmt.lhs)
                    else
                        delete!(injective_dep, stmt.lhs)
                    end
                else
                    delete!(thread_dep, stmt.lhs)
                    delete!(injective_dep, stmt.lhs)
                end
            end
        elseif stmt.kind == :if
            push!(exprs, emit_if(stmt.cond, cgen_device_body(stmt.then, thread_var, backend, reduce_vars, copy(thread_dep), copy(injective_dep)), cgen_device_body(stmt.els, thread_var, backend, reduce_vars, copy(thread_dep), copy(injective_dep))))
        elseif stmt.kind == :for
            push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, cgen_device_body(stmt.body, thread_var, backend, reduce_vars, copy(thread_dep), copy(injective_dep))))
        end
    end
    return exprs
end

# A write races across threads unless the enclosing device loop's thread-mapped variable occurs in the write's index, or through a chain of let-
# bindings (`thread_dep`) -- a real occurs-check. Any thread-invariant-indexed write is a race regardless of pattern: an accumulation is fixable
# with an atomic +=, a plain replacement is not, so it's refused outright. A thread-DEPENDENT index isn't automatically race-free either: a gather-
# derived index is thread-dependent but not injective, and gets the same atomic treatment as the invariant case; a non-additive write through it is
# left untouched, since an atomic can't fix a replacement race either way.
function cgen_device_assign(stmt, thread_var::Symbol, thread_dep::Set{Symbol}, injective_dep::Set{Symbol}, backend, reduce_vars::Set{Symbol})
    if stmt.lhs isa Symbol && stmt.lhs in reduce_vars
        # Scalar cross-thread reduction (e.g. an adjoint accumulator like `cb`): cgen_emit
        # already boxed this free var into a 1-element device array before any kernel
        # launch, so the accumulation becomes an atomic add against index 1 -- the same
        # treatment as the array-indexed self-reference case below, just pre-boxed to a
        # known-size-1 array.
        terms = cgen_flatten_sum(stmt.rhs)
        self_idx = findfirst(t -> t == stmt.lhs, terms)
        self_idx === nothing && error("cgen_device_assign: `$(stmt.lhs)` was classified as a scalar reduction target (see cgen_scalar_reduction_vars) but `$(stmt.lhs) = $(stmt.rhs)` doesn't self-reference -- this should be unreachable")
        other = cgen_sum_excluding(terms, self_idx)
        return Expr(:macrocall, backend.atomic_macro, nothing,
                    Expr(:(+=), Expr(:ref, stmt.lhs, 1), other))
    end
    if stmt.lhs isa Expr && stmt.lhs.head == :ref
        thread_invariant = !cgen_expr_contains_any(stmt.lhs.args[2:end], thread_dep)
        needs_atomic_check = thread_invariant || !cgen_expr_injective_ok(stmt.lhs.args[2:end], injective_dep, thread_dep)
        if needs_atomic_check
            terms = cgen_flatten_sum(stmt.rhs)
            self_idx = findfirst(t -> t == stmt.lhs, terms)
            if self_idx !== nothing
                other = cgen_sum_excluding(terms, self_idx)
                return Expr(:macrocall, backend.atomic_macro, nothing,
                            Expr(:(+=), stmt.lhs, other))
            end
            thread_invariant && error("cgen_device_assign: write to `$(stmt.lhs)` inside a GPU-split loop has an index that doesn't depend on the loop's own thread variable (`$thread_var`), even transitively through same-body scalar let-bindings, and isn't an additive accumulation -- this is a data race across threads, not something an atomic wrapper can fix. See skill-stade-dev.md's cgen_device_assign hardening note.")
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

# same occurs-check, generalized to a set of symbols -- used for
# cgen_device_assign's thread_dep (thread_var plus every scalar
# transitively derived from it in the current device-loop body)
function cgen_expr_contains_any(x, syms::Set{Symbol})
    x isa Symbol && return x in syms
    x isa Expr && return any(a -> cgen_expr_contains_any(a, syms), x.args)
    return false
end
cgen_expr_contains_any(xs::Vector, syms::Set{Symbol}) = any(x -> cgen_expr_contains_any(x, syms), xs)

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

function cgen_emit(gk, backend; keep_all_atomic::Bool = true)
    kernels = Expr[]
    reduce_vars = Set{Symbol}()
    fn_args = Set{Symbol}(gk.args)
    outer_defs = cgen_scalar_def_map(gk.body)
    host_body = cgen_body(gk.body, kernels, gk.name, backend, reduce_vars, fn_args; keep_all_atomic, outer_defs)
    isempty(kernels) || pushfirst!(host_body, :(nthread_per_block = 256))
    # Every scalar cross-thread reduction free var must already be a 1-element device array
    # before the first kernel launch that atomically writes it, and must be unboxed to a plain
    # host scalar before it can be returned -- both are whole-array host<->device copies, never
    # an element-wise scalar index into a device-resident array, so this is legal under
    # allowscalar(false). Sorted for deterministic codegen output.
    reduce_vars_sorted = sort(collect(reduce_vars); by = string)
    for v in reverse(reduce_vars_sorted)
        pushfirst!(host_body, Expr(:(=), v, Expr(:call, backend.arrtype, Expr(:vect, v))))
    end
    for v in reduce_vars_sorted
        v in gk.ret || continue
        push!(host_body, Expr(:(=), v, Expr(:ref, Expr(:call, :Array, v), 1)))
    end
    push!(host_body, emit_return_scalars(gk.ret))
    host = Expr(:function, Expr(:call, cgen_host_fname(gk.name, backend), gk.args...), Expr(:block, host_body...))
    return (host = host, kernels = kernels)
end

# Opt-in, applied only when the caller passes precision=T: converts every Float64 literal to T, leaving Int-typed
# loop/index arithmetic untouched. Just a literal walk, no operand-forcing rewrite, since every array/scalar is caller-
# supplied (skill-stade rule 8) -- precision is the caller's job. Caveat: a handful of Base operations return Float64
# unconditionally when both operands are Integer (true division, transcendentals), so index arithmetic can stay Float64
# even under precision=Float32 -- on Metal this fails to compile rather than silently running in double precision.
function cgen_convert_precision(expr, ::Type{T}) where {T<:AbstractFloat}
    tname = Symbol(string(T))
    if expr isa AbstractFloat
        return T(expr)
    # STADE never emits the bare Symbol :Float64 anywhere except as a stack's element-type marker (a device stack
    # allocation's curly type, or JACC.zeros' first argument). These are STADE's own allocations, never caller-supplied,
    # so retyping them is still STADE's job -- without it, a Metal.jl kernel reading an un-retyped Float64 stack fails to
    # compile. `:Int64` is left untouched: a :branch/:tripcount stack's element type is never precision-converted.
    elseif expr === :Float64
        return tname
    elseif expr isa Expr
        return Expr(expr.head, [cgen_convert_precision(a, T) for a in expr.args]...)
    end
    return expr
end


# ==================== jgen_* =======================================
# JACC.jl codegen: replaces cgen_'s vendor launch macro/thread-index model with a plain function taking the loop index as its first argument,
# dispatched via `JACC.@parallel_for range=N f(args...)`; the vendor backend is chosen per-project via Preferences.jl, deferred past generation.
# Reuses cgen_'s backend-agnostic front end directly; atomics use Atomix.@atomic. Precision isn't locked, since STADE can't know which backend runs
# the output; bounds checking is omitted, per JACC's documented contract (unverified against hardware).

jgen_kernel_fname(owner::Symbol, idx::Int) = Symbol("jacc_kernel_" * string(owner) * "_" * string(idx) * "!")

# Same device-residency need as cgen_stack_device_expr, but JACC has no vendor-specific array
# constructor and can't know at generation time which vendor will run the output. `JACC.zeros(T,
# N)` is JACC.jl's own documented, portable allocator -- the only safe call, since there's no
# `undef`-style JACC allocator, so this zero-fills instead. Harmless in practice (every element
# is overwritten by a push before its first read), just a slightly more expensive allocation.
function jgen_stack_device_expr(rhs::Expr)
    T = rhs.args[1].args[2]
    size_expr = rhs.args[3]
    return Expr(:call, Expr(:., :JACC, QuoteNode(:zeros)), T, size_expr)
end

# keep_all_atomic: same meaning as cgen_body's -- a matched loop is replaced by one `JACC.@parallel_reduce` call instead of a synthesized per-
# element atomic kernel. CORRECTED: a live GPU run proved allowscalar-style handling IS needed here too: `JACC.@parallel_reduce` returns a
# plain host scalar, but writing it back into `target` (a device-array-backed ref at runtime) via a bare host assignment is a host setindex!
# against a device array. Unlike CUDA/AMDGPU/Metal, JACC has no allowscalar escape hatch, raising 'Scalar indexing is disallowed'. Fix:
# accumulate on-device via a synthesized range=1 kernel instead.
# outer_defs: see cgen_body's identical parameter -- kept in sync for the JACC target.
function jgen_body(body::Vector{NamedTuple}, kernels::Vector{Expr}, owner::Symbol, reduce_vars::Set{Symbol}, fn_args::Set{Symbol}, allowscalar_macro; keep_all_atomic::Bool = true, outer_defs::Dict{Symbol,Any} = Dict{Symbol,Any}())
    exprs = Any[]
    known_consts = Dict{Symbol,Any}()
    pending = Any[]
    pending_has_array = false
    flush_pending! = () -> begin
        if !isempty(pending)
            if pending_has_array && allowscalar_macro !== nothing
                push!(exprs, Expr(:macrocall, allowscalar_macro, nothing, Expr(:block, pending...)))
            else
                append!(exprs, pending)
            end
            empty!(pending)
            pending_has_array = false
        end
    end
    for stmt in body
        if stmt.kind == :stackpush
            flush_pending!()
            push!(exprs, Expr(:call, :push!, stmt.stack, stmt.value))
        elseif stmt.kind == :assign
            rhs = cgen_is_sized_stack_alloc(stmt.rhs) ? jgen_stack_device_expr(stmt.rhs) : stmt.rhs
            push!(pending, Expr(:(=), stmt.lhs, rhs))
            pending_has_array = pending_has_array || cgen_expr_has_ref(stmt.lhs) || cgen_expr_has_ref(rhs)
            if stmt.lhs isa Symbol
                if stmt.rhs isa Number
                    known_consts[stmt.lhs] = stmt.rhs
                else
                    delete!(known_consts, stmt.lhs)
                end
            end
        elseif stmt.kind == :if
            flush_pending!()
            push!(exprs, emit_if(stmt.cond, jgen_body(stmt.then, kernels, owner, reduce_vars, fn_args, allowscalar_macro; keep_all_atomic, outer_defs), jgen_body(stmt.els, kernels, owner, reduce_vars, fn_args, allowscalar_macro; keep_all_atomic, outer_defs)))
            for v in cgen_all_assigned_scalars(vcat(stmt.then, stmt.els))
                delete!(known_consts, v)
            end
        elseif stmt.kind == :for
            flush_pending!()
            synth = cgen_reduction_only_loop(stmt.body, stmt.var, known_consts, outer_defs)
            loop_reduce_vars = synth === nothing ? Set{Symbol}() : cgen_scalar_reduction_vars(stmt.body)
            safe_scope = issubset(loop_reduce_vars, fn_args)
            if synth !== nothing && safe_scope && !cgen_contains_stackop(stmt.body)
                red = keep_all_atomic ? nothing : cgen_idiomatic_scalar_reduction(stmt.body, stmt.var)
                if red !== nothing
                    target, op, arrs, term = red
                    n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)
                    value = jgen_idiomatic_reduction_value(arrs, term, stmt.var, n_iter)
                    # `target - value` (op==:-) is normalized to a signed
                    # additive term here, matching cgen_flatten_sum's/
                    # jgen_device_assign's own convention of only ever
                    # emitting `Atomix.@atomic ... += ...` (never `-=`) --
                    # see jgen_reduction_writeback_kernel's comment.
                    signed_value = op == :+ ? value : Expr(:call, :-, value)
                    widx = length(kernels) + 1
                    wvars = Set{Symbol}()
                    cgen_collect_expr_vars!(target, wvars)   # arr + any free vars inside target's index expr(s)
                    wfargs = sort(collect(wvars); by = string)
                    # `JACC.@parallel_reduce(...)` must be its own top-level host statement, never nested inside another
                    # JACC launch macro's call-argument list: `JACC.@parallel_for` doesn't guarantee host-side pre-
                    # evaluation of a macro-call argument. Nesting the reduce in the launch call (an earlier fix) produced
                    # `GPUCompiler.InvalidIRError` on a live GPU run. `widx` is unique per write-back site, so the temp
                    # name can't collide with another reduction in the same body.
                    redvar = Symbol("__jgen_redval_", widx)
                    push!(exprs, Expr(:(=), redvar, signed_value))
                    # NOT boxed via JACC.array here, despite that being jgen_emit's reduce_vars convention: a live GPU
                    # diagnostic confirmed `JACC.@parallel_reduce` already returns a device-resident `CuArray{Float64,1}`
                    # directly, not a host Float64. Wrapping it again via `JACC.array([redvar])` (an earlier fix) tried to
                    # build a CuArray whose element type is itself a CuArray -- illegal, confirmed live. `redvar` is
                    # passed straight through as the kernel argument; the kernel indexes `[1]` on it on-device.
                    push!(kernels, jgen_reduction_writeback_kernel(owner, widx, target, wfargs))
                    push!(exprs, jgen_reduction_writeback_launch(owner, widx, wfargs, redvar))
                else
                    idx = length(kernels) + 1
                    fargs = cgen_free_vars(stmt, stmt.var)
                    union!(reduce_vars, loop_reduce_vars)
                    push!(kernels, jgen_kernel_def(stmt, owner, idx, fargs, loop_reduce_vars, synth))
                    push!(exprs, jgen_launch_expr(stmt, owner, idx, fargs))
                end
            else
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, jgen_body(stmt.body, kernels, owner, reduce_vars, fn_args, allowscalar_macro; keep_all_atomic, outer_defs)))
            end
            delete!(known_consts, stmt.var)
        end
    end
    flush_pending!()
    return exprs
end

# JACC hands the loop index in directly as the split-off function's first parameter -- no
# thread-index intrinsic to bind, unlike cgen_kernel_def, since JACC.@parallel_for range=N
# already guarantees the index range. cgen_loopvar_from_tid still does the affine lo/step
# remapping, same as for cgen_.
function jgen_kernel_def(stmt, owner::Symbol, idx::Int, fargs::Vector{Symbol}, reduce_vars::Set{Symbol}, synth::Dict{Symbol,Any} = Dict{Symbol,Any}())
    jidx = :__jacc_i
    body = Any[Expr(:(=), stmt.var, cgen_loopvar_from_tid(stmt.lo, stmt.step, jidx))]
    # see cgen_kernel_def's matching comment: cgen_reduction_only_loop's
    # convergent-constant proof injects each synthesized initializer
    # here, before the loop's own body runs.
    for (v, c) in synth
        push!(body, Expr(:(=), v, c))
    end
    append!(body, jgen_device_body(stmt.body, stmt.var, reduce_vars))
    push!(body, emit_return_nothing())
    return Expr(:function, Expr(:call, jgen_kernel_fname(owner, idx), jidx, fargs...), Expr(:block, body...))
end

# JACC.jl v1.x: JACC.@parallel_for range=N f(args...) -- a macro with a `range=` keyword-
# style argument, not a plain function call (that was the pre-1.0/v0.0.x API). The
# underlying kernel function's own signature is unchanged (index still its first parameter,
# supplied internally by the macro); only the launch call's shape moved.
function jgen_launch_expr(stmt, owner::Symbol, idx::Int, fargs::Vector{Symbol})
    n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)
    return Expr(:macrocall, Expr(:., :JACC, QuoteNode(Symbol("@parallel_for"))), nothing,
                Expr(:(=), :range, n_iter),
                Expr(:call, jgen_kernel_fname(owner, idx), fargs...))
end

function jgen_device_body(body::Vector{NamedTuple}, thread_var::Symbol, reduce_vars::Set{Symbol}, thread_dep::Set{Symbol} = Set([thread_var]), injective_dep::Set{Symbol} = Set([thread_var]))
    exprs = Any[]
    for stmt in body
        if stmt.kind == :assign
            push!(exprs, jgen_device_assign(stmt, thread_var, thread_dep, injective_dep, reduce_vars))
            # Mirrors cgen_device_body's thread_dep/injective_dep tracking
            # exactly -- see that function's comment. This brings the JACC
            # target up to the same transitive occurs-check parity
            # cgen_device_assign already has, rather than jgen_device_assign's
            # previous bare (non-transitive) thread_var check.
            if stmt.lhs isa Symbol
                if cgen_expr_contains_any(stmt.rhs, thread_dep)
                    push!(thread_dep, stmt.lhs)
                    if cgen_expr_injective_ok(stmt.rhs, injective_dep, thread_dep)
                        push!(injective_dep, stmt.lhs)
                    else
                        delete!(injective_dep, stmt.lhs)
                    end
                else
                    delete!(thread_dep, stmt.lhs)
                    delete!(injective_dep, stmt.lhs)
                end
            end
        elseif stmt.kind == :if
            push!(exprs, emit_if(stmt.cond, jgen_device_body(stmt.then, thread_var, reduce_vars, copy(thread_dep), copy(injective_dep)), jgen_device_body(stmt.els, thread_var, reduce_vars, copy(thread_dep), copy(injective_dep))))
        elseif stmt.kind == :for
            push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, jgen_device_body(stmt.body, thread_var, reduce_vars, copy(thread_dep), copy(injective_dep))))
        end
    end
    return exprs
end

# Identical decision to cgen_device_assign, reusing cgen_'s occurs-check, injectivity-check, and sum-flattening helpers --
# only the atomic macro's target module is fixed. The scalar-reduction branch mirrors cgen_device_assign's exactly:
# reduce_vars was boxed into a 1-element JACC.array by jgen_emit, so `cb = cb + other` becomes an atomic add against index
# 1. The array-ref branch now also mirrors the thread-invariant refusal and injective_dep distinction -- previously this
# used a bare, non-transitive thread_var check with no refusal path for a genuine non-additive race.
function jgen_device_assign(stmt, thread_var::Symbol, thread_dep::Set{Symbol}, injective_dep::Set{Symbol}, reduce_vars::Set{Symbol})
    if stmt.lhs isa Symbol && stmt.lhs in reduce_vars
        terms = cgen_flatten_sum(stmt.rhs)
        self_idx = findfirst(t -> t == stmt.lhs, terms)
        self_idx === nothing && error("jgen_device_assign: `$(stmt.lhs)` was classified as a scalar reduction target (see cgen_scalar_reduction_vars) but `$(stmt.lhs) = $(stmt.rhs)` doesn't self-reference -- this should be unreachable")
        other = cgen_sum_excluding(terms, self_idx)
        return Expr(:macrocall, Expr(:., :Atomix, QuoteNode(Symbol("@atomic"))), nothing,
                    Expr(:(+=), Expr(:ref, stmt.lhs, 1), other))
    end
    if stmt.lhs isa Expr && stmt.lhs.head == :ref
        thread_invariant = !cgen_expr_contains_any(stmt.lhs.args[2:end], thread_dep)
        needs_atomic_check = thread_invariant || !cgen_expr_injective_ok(stmt.lhs.args[2:end], injective_dep, thread_dep)
        if needs_atomic_check
            terms = cgen_flatten_sum(stmt.rhs)
            self_idx = findfirst(t -> t == stmt.lhs, terms)
            if self_idx !== nothing
                other = cgen_sum_excluding(terms, self_idx)
                return Expr(:macrocall, Expr(:., :Atomix, QuoteNode(Symbol("@atomic"))), nothing,
                            Expr(:(+=), stmt.lhs, other))
            end
            thread_invariant && error("jgen_device_assign: write to `$(stmt.lhs)` inside a GPU-split loop has an index that doesn't depend on the loop's own thread variable (`$thread_var`), even transitively through same-body scalar let-bindings, and isn't an additive accumulation -- this is a data race across threads, not something an atomic wrapper can fix. See skill-stade-dev.md's cgen_device_assign hardening note.")
        end
    end
    return Expr(:(=), stmt.lhs, stmt.rhs)
end

jgen_host_fname(name::Symbol) = Symbol(string(name) * "_jacc")

# JACC has no vendor-neutral scalar-indexing escape hatch of its own (unlike CUDA.jl/AMDGPU.jl/Metal.jl, each shipping their
# own @allowscalar). A loop jgen_body demotes to host-sequential can still need to touch a JACC.array-boxed argument element-
# wise, with nothing to wrap it in -- confirmed live, a crash before this existed. On this image, JACC.set_backend("CUDA")
# means JACC.array returns a CuArray, so CUDA.@allowscalar is a real fix here too. A CUDA-specific stand-in, not a portable
# JACC API: a future AMDGPU/Metal-backed image needs a different default.
jgen_default_allowscalar_macro() = Expr(:., :CUDA, QuoteNode(Symbol("@allowscalar")))

function jgen_emit(gk; keep_all_atomic::Bool = true, allowscalar_macro = jgen_default_allowscalar_macro())
    kernels = Expr[]
    reduce_vars = Set{Symbol}()
    fn_args = Set{Symbol}(gk.args)
    outer_defs = cgen_scalar_def_map(gk.body)
    host_body = jgen_body(gk.body, kernels, gk.name, reduce_vars, fn_args, allowscalar_macro; keep_all_atomic, outer_defs)
    # Mirrors cgen_emit's box/unbox handling, JACC v1.x API: JACC.array
    # for a host->device whole-array transfer, JACC.to_host for the reverse.
    reduce_vars_sorted = sort(collect(reduce_vars); by = string)
    for v in reverse(reduce_vars_sorted)
        pushfirst!(host_body, Expr(:(=), v, Expr(:call, Expr(:., :JACC, QuoteNode(:array)), Expr(:vect, v))))
    end
    for v in reduce_vars_sorted
        v in gk.ret || continue
        push!(host_body, Expr(:(=), v, Expr(:ref, Expr(:call, Expr(:., :JACC, QuoteNode(:to_host)), v), 1)))
    end
    push!(host_body, emit_return_scalars(gk.ret))
    host = Expr(:function, Expr(:call, jgen_host_fname(gk.name), gk.args...), Expr(:block, host_body...))
    return (host = host, kernels = kernels)
end

function jgen_preamble()
    # `using CUDA` alongside JACC/Atomix: jgen_emit's default allowscalar_macro emits
    # CUDA.@allowscalar around any host-side JACC-array touch a demoted loop needs -- every
    # jgen_-generated file needs CUDA loaded for that to resolve, not just this validator's own
    # script.
    return "import Pkg\nhaskey(Pkg.project().dependencies, \"JACC\") || Pkg.add(name = \"JACC\", version = \"1\")\nhaskey(Pkg.project().dependencies, \"Atomix\") || Pkg.add(\"Atomix\")\nusing CUDA\nimport JACC\nimport Atomix\nJACC.@init_backend\n"
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
# Extends the val_ oracle to work generically on any skill-stade kernel's generated _d/_b/_hv code, not just a hand-built fixture. Three identities
# reuse the same two oracles: tangent (_d) is a direct JVP check against central FD of the primal; adjoint (_b) is <y,Jx>==<J'y,x>, reusing
# val_finite_diff_check on a scalar closure; hvp (_hv) is a JVP check one layer out, the same val_finite_diff_check_jvp oracle applied to the
# adjoint instead of the primal. x and y always share one flattened space of dimension n.

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

# Rebuilds the primal with an appended `return` of every scalar_float arg's final value --
# skill-stade kernels never contain their own `return` (only :assign/:for/:if), so appending
# one at the end is safe, and it's the only way a caller can observe a reassigned scalar the
# way it already observes array arguments (in-place mutation).
function val_primal_observing_expr(kernel, primal_expr::Expr)
    sig = kernel.sig
    fname = Symbol(string(sig.name) * "_valobs")
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    raw_stmts = [s for s in primal_expr.args[2].args if !(s isa LineNumberNode)]
    # parse_kernel already validated that the only body-level `return` a raw
    # kernel Expr may contain is a trailing `return nothing` -- strip it here
    # too, or it fires before our appended return and this wrapper always
    # observes `nothing` instead of the scalar args' final values.
    if !isempty(raw_stmts) && raw_stmts[end] isa Expr && raw_stmts[end].head == :return
        raw_stmts = raw_stmts[1:end-1]
    end
    body = Expr(:block, raw_stmts..., emit_return_scalars(scalar_args))
    return Expr(:function, Expr(:call, fname, sig.args...), body)
end

# ---- execution helpers ----
# The one place val_ steps outside pure Expr manipulation: running dynamically generated
# code requires evaluating it. Not filesystem access (still permitted outside io_), but it
# does define a transient global method -- an accepted, narrowly scoped exception, since
# there's no other way to numerically execute generated Julia code.

function val_compile(expr::Expr)
    fname = expr.args[1].args[1]
    Base.eval(Main, expr)
    return getfield(Main, fname)
end

# Replicates stade_tangent's own three-call pipeline (act_analyze -> lin_build -> tgen_emit)
# directly, rather than calling stade_tangent itself -- val_generate_baseline needs a
# tangent to self-check a candidate baseline, and calling into stade_ (which itself composes
# val_'s machinery) would make the two layers mutually dependent. Calling the same pipeline
# stages keeps val_ resting only on the codegen pipeline.
function val_build_tangent(kernel)
    active_map = act_analyze(kernel)
    lin_plan = lin_build(kernel, active_map)
    return tgen_emit(kernel, lin_plan)
end

# finds the definition named `name` within a bundle read by
# io_read_kernel_bundle (e.g. picking out `foo_b` and `initstacks_foo_b`
# from a third-party file that also carries a trailing copy of `foo`).
function val_find_def(defs::Vector{Expr}, name::Symbol)
    for e in defs
        e.args[1].args[1] == name && return e
    end
    error("val_find_def: no function named $name found among $(length(defs)) definitions")
end

# the positional argument names of a raw function-definition Expr, as
# parsed straight from source (not a skill-stade sig) -- used to learn
# what a third-party initstacks function expects (STADE's own always
# takes zero args; other tools' may take one or more primal arrays).
val_def_arg_names(expr::Expr) = Symbol[a for a in expr.args[1].args[2:end]]

# `extra_args` supports initstacks functions that need one or more
# primal arrays to size/type their stacks (e.g. Tapenade's own
# `initstacks_foo_b(du) = Vector{typeof(du)}()` convention) -- STADE's
# own agen_init_emit always takes zero args, so existing call sites
# are unaffected by the default.
function val_init_stacks(initstacks_fn::Function, extra_args::Vector = [])
    r = Base.invokelatest(initstacks_fn, extra_args...)
    r === nothing && return ()
    r isa Tuple && return r
    return (r,)
end

# ---- random baseline generation -------------------------------------

function val_random_int_args(sig; lo::Int = 2, hi::Int = 5, divisible_by::Dict{Symbol,Int} = Dict{Symbol,Int}())
    return Dict{Symbol,Int}(a => val_random_int_arg(lo, hi, get(divisible_by, a, 1)) for a in sig.args if sig.kinds[a] == :scalar_int)
end

# Plain rand(lo:hi) when `k <= 1`. Otherwise draws uniformly among multiples of `k` within
# [lo, hi] -- and since a narrow range can contain none or exactly one, falls back to the
# single nearest multiple of `k` at or above `lo` rather than erroring: a valid draw outside
# the requested range is more useful than none, matching val_generate_baseline's own
# attempts-loop philosophy.
function val_random_int_arg(lo::Int, hi::Int, k::Int)
    k <= 1 && return rand(lo:hi)
    candidates = [m for m in (k * cld(lo, k)):k:hi]
    isempty(candidates) && return k * cld(lo, k)
    return rand(candidates)
end

# A scalar_int kernel argument can be a divisor a corpus kernel's own comment documents a hard constraint on
# (e.g. unet.jl: h/w must both be divisible by 4) that plain rand(lo:hi) can't know about -- a bad draw produces
# a degenerate derived dimension that crashes deep inside a GPU kernel launch, harder to debug than respecting
# the constraint up front. Deliberately a small, explicit lookup, not a general scanner inferring 'must divide
# evenly' from every div() call, since exactly one corpus kernel needs it today.
function val_corpus_int_constraints(kernel_name::Symbol)
    kernel_name === :unet && return Dict{Symbol,Int}(:h => 4, :w => 4)
    return Dict{Symbol,Int}()
end

# ---- avoiding near-zero/negative divisors in random baselines ----
# A float argument used anywhere as a `/` divisor has no guarantee of staying away from zero, or even positive,
# under plain randn(). An iterative relaxation dividing by a near-zero or sign-flipped value on every step is a
# classic divergence trigger, unrelated to whether the generated derivative code is correct. Kernel-agnostic,
# detected like val_arg_ndims detects usage shape: a static scan of kernel.body.
function val_divisor_args(kernel)
    found = Set{Symbol}()
    val_scan_divisors!(kernel.body, found)
    return found
end

function val_scan_divisors!(body, found::Set{Symbol})
    for stmt in body
        if stmt.kind == :assign
            val_scan_expr_divisors!(stmt.lhs, found)
            val_scan_expr_divisors!(stmt.rhs, found)
        elseif stmt.kind == :for
            val_scan_expr_divisors!(stmt.lo, found)
            val_scan_expr_divisors!(stmt.hi, found)
            val_scan_divisors!(stmt.body, found)
        elseif stmt.kind == :if
            val_scan_expr_divisors!(stmt.cond, found)
            val_scan_divisors!(stmt.then, found)
            val_scan_divisors!(stmt.els, found)
        end
    end
    return nothing
end

function val_scan_expr_divisors!(expr, found::Set{Symbol})
    if expr isa Expr && expr.head == :call && expr.args[1] == :/ && length(expr.args) == 3
        divisor = expr.args[3]
        base = divisor isa Expr && divisor.head == :ref ? divisor.args[1] : divisor
        base isa Symbol && push!(found, base)
    end
    if expr isa Expr
        for a in expr.args
            val_scan_expr_divisors!(a, found)
        end
    end
    return nothing
end

function val_random_values(kernel, shapes::Dict, int_args::Dict{Symbol,Int};
                            scale::Float64 = 1.0, idx_cap::Union{Int,Nothing} = nothing)
    sig = kernel.sig
    values = Dict{Symbol,Any}()
# val_grow_shapes sizes every array_float/array_int arg's every dimension to the same N. array_int
# content is drawn from 1:idx_cap rather than 1:N: a direct-indexing arg is safely in-bounds with
# idx_cap==N, but a compressed id needs idx_cap far below N instead -- val_grow_shapes searches both
# independently. Falls back to N for a caller that never passed idx_cap, and to 1 if there are no
# array args at all.
    N = isempty(shapes) ? 1 : minimum(minimum(s) for s in Base.values(shapes))
    cap = idx_cap === nothing ? N : idx_cap
    divisors = val_divisor_args(kernel)
    # A positive-but-wide-spread divisor still lets the ratio between two independent divisor-
    # like args land far from 1 -- and an iterative loop amplifies whatever that ratio is on
    # every step. Narrowing the spread specifically for divisor args (still positive, still
    # random, just closer to a common scale) keeps that per-step gain closer to 1 without hard-
    # coding anything about what the ratio means.
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_float
            values[a] = a in divisors ? scale * (1.0 + 0.1 * randn()) : scale * randn()
        elseif k == :array_float
            values[a] = a in divisors ? scale .* (1.0 .+ 0.1 .* randn(shapes[a]...)) : scale .* randn(shapes[a]...)
        elseif k == :array_int
            values[a] = rand(1:cap, shapes[a]...)
        end
    end
    return values
end

# Grows one trial size N until the primal runs without a BoundsError. Oversized dimensions are harmless, so this
# always converges on a safe (if not minimal) size without static index-range analysis. idx_cap (the array_int
# content range) is searched independently of N, outer loop around the inner N loop -- tying them together would
# make a compressed-id argument scaled by a stride before indexing unsatisfiable at any N. Starting idx_cap
# small and growing it outward still finds direct-indexing kernels' previous idx_cap==N solution immediately.
function val_grow_shapes(kernel, primal_fn::Function, int_args::Dict{Symbol,Int};
                          start::Int = 4, growth::Int = 2, max_size::Int = 512)
    sig = kernel.sig
    ndims_of = Dict(a => val_arg_ndims(kernel, a) for a in sig.args
                     if sig.kinds[a] in (:array_float, :array_int))
    idx_cap = start
    while idx_cap <= max_size
        N = start
        while N <= max_size
            shapes = Dict(a => ntuple(_ -> N, ndims_of[a]) for a in keys(ndims_of))
            trial = val_random_values(kernel, shapes, int_args; scale = 1.0, idx_cap = idx_cap)
            ok = try
                call_args = Any[sig.kinds[a] == :scalar_int ? int_args[a] : deepcopy(trial[a]) for a in sig.args]
                Base.invokelatest(primal_fn, call_args...)
                true
            catch e
                e isa BoundsError || rethrow(e)
                false
            end
            ok && return (shapes = shapes, idx_cap = idx_cap)
            N *= growth
        end
        idx_cap *= growth
    end
    error("val_grow_shapes: could not find a working array size/index range up to $max_size for $(sig.name)")
end

# Orchestrates a full random baseline: random ints, a compiled primal probed to find safe array sizes, then
# final random Float64/Int content at those sizes. A few retries guard against a rare unlucky combination
# tripping an unrelated error -- and, when self_check is on, against a combination that runs cleanly but is
# semantically degenerate. No general way to detect that up front; instead a tangent-vs-FD check runs against
# the candidate, discarding and redrawing the whole triple if it fails.
function val_generate_baseline(kernel, primal_expr::Expr;
                                scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                grow_start::Int = 4, grow_max::Int = 512, attempts::Int = 16,
                                self_check::Bool = true, self_check_trials::Int = 2,
                                self_check_epsilon::Float64 = 1e-6, self_check_rtol::Float64 = 1e-3,
                                max_output_magnitude::Float64 = 1e6,
                                divisible_by::Union{Dict{Symbol,Int},Nothing} = nothing)
    divisible_by = divisible_by === nothing ? val_corpus_int_constraints(kernel.sig.name) : divisible_by
    primal_fn = val_compile(primal_expr)
    obs_fn = self_check ? val_compile(val_primal_observing_expr(kernel, primal_expr)) : nothing
    tangent_fn = self_check ? val_compile(val_build_tangent(kernel)) : nothing
# The self-check below exercises the primal and tangent, both keep_push_pop-agnostic -- so it never touches the one place a kernel's
# integer arguments must be mutually coherent: keep_push_pop=false's initstacks_*, which sizes every stack from those arguments. A
# multigrid kernel drawn with nfine=4, num_levels=4 asks for four coarsenings of a 4-point grid and computes a negative length; :stack
# mode computes no sizes at all, so the baseline looked fine and only failed years later. Running it here turns that into just another
# rejected, redrawn candidate.
    sizing_fn = nothing
    sizing_arg_names = Symbol[]
    if self_check
        try
            init_expr = stade_adjoint(primal_expr; keep_push_pop = false).initstacks
            sizing_arg_names = Symbol[a for a in init_expr.args[1].args[2:end]]
            sizing_fn = val_compile(init_expr)
        catch
            sizing_fn = nothing
        end
    end
    last_err = nothing
# A scalar_int arg is very often an iteration count -- more iterations means more chances for per-step amplification to
# compound into a blow-up under otherwise reasonable random data. Narrowing the upper end of the range after a divergence --
# rather than redrawing at the same range -- targets that directly, without assuming what any particular arg means. Allowed to
# fall below the caller's own int_lo down to 1 (never 0, since a zero-iteration loop would trivially pass): a smaller working
# baseline beats none. Reset for a non-divergence failure.
    cur_hi = int_hi
    for attempt in 1:attempts
        int_args = val_random_int_args(kernel.sig; lo = min(int_lo, cur_hi), hi = cur_hi, divisible_by = divisible_by)
        try
            grown = val_grow_shapes(kernel, primal_fn, int_args; start = grow_start, max_size = grow_max)
            values = val_random_values(kernel, grown.shapes, int_args; scale = scale, idx_cap = grown.idx_cap)
            if sizing_fn !== nothing
                # same arg-source convention as val_call_adjoint's
                # stack_extra_args: an initstacks_* parameter is an int
                # arg when there is one, otherwise a float scalar value.
                Base.invokelatest(sizing_fn,
                                   [haskey(int_args, a) ? int_args[a] : deepcopy(values[a])
                                    for a in sizing_arg_names]...)
            end
            if self_check
                x0 = val_flatten(kernel, values)
                f_eval_vec = x -> val_call_primal_observed(obs_fn, kernel, int_args,
                                                            val_unflatten(kernel, int_args, values, x))
# A candidate whose own output has blown up (e.g. a diverging relaxation loop for a random,
# physically meaningless input) makes h=epsilon finite differences numerically meaningless
# -- their step's own contribution falls below floating-point precision at that magnitude,
# so the FD reference itself is garbage, not the exactly-computed adjoint/tangent being
# compared against it. Reject and redraw before even reaching the tangent self-check below.
                y0 = f_eval_vec(x0)
                if !(all(isfinite, y0) && maximum(abs.(y0); init = 0.0) <= max_output_magnitude)
                    cur_hi = max(1, div(cur_hi + 1, 2))
                    error("candidate baseline's own primal output is non-finite or too large " *
                          "(max|y|=$(maximum(abs.(y0); init = 0.0))) -- likely a diverging relaxation " *
                          "under random, physically meaningless inputs")
                end
                f_jvp = (x, d) -> begin
                    vals = val_unflatten(kernel, int_args, values, x)
                    dvals = val_unflatten(kernel, int_args, val_zeros_like(kernel, values), d)
                    val_call_tangent(tangent_fn, kernel, int_args, vals, dvals)
                end
                check = val_finite_diff_check_jvp(f_eval_vec, f_jvp, x0;
                                                   epsilon = self_check_epsilon, trials = self_check_trials,
                                                   rtol = self_check_rtol)
                check.ok || error("candidate baseline failed tangent self-check (max_rel_err=$(check.max_rel_err))")
            end
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
    # val_unflatten unconditionally reads template[a] for every
    # :array_int arg (to pass its real, non-perturbed data through
    # unchanged) -- any dict handed to it as `template` must carry
    # those keys too, not just the float ones this function's own
    # loop above covers.
    for a in kernel.sig.args
        kernel.sig.kinds[a] == :array_int && (out[a] = values[a])
    end
    return out
end

function val_random_values_like(kernel, values::Dict; scale::Float64 = 1.0)
    out = Dict{Symbol,Any}()
    for a in val_float_arg_order(kernel.sig)
        v = values[a]
        out[a] = v isa Number ? scale * randn() : scale .* randn(size(v)...)
    end
    # see val_zeros_like's own comment just above -- same contract.
    for a in kernel.sig.args
        kernel.sig.kinds[a] == :array_int && (out[a] = values[a])
    end
    return out
end

# ---- positional call-argument builders ----
# Positional call-argument builders duplicate tgen_signature_args/agen_signature_args's
# documented convention (float arg immediately followed by its shadow; int args unchanged)
# rather than reaching into those stages' private helpers, per the same purity rule agen_
# itself follows duplicating snap_'s TBR predicate.

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
                           int_args::Dict, values::Dict, seed::Dict;
                           stack_arg_names::Vector{Symbol} = Symbol[])
    sig = kernel.sig
    # a stack_arg_name is either a Tapenade-style float-array sizing
    # arg (in `values`) or, for STADE's own keep_push_pop=false
    # Tier A sizing (see agen_indexed_layout's `free_vars`), a
    # kernel-level int loop-bound arg (in `int_args`) -- check both
    stack_extra_args = [haskey(int_args, n) ? int_args[n] : deepcopy(values[n]) for n in stack_arg_names]
    stacks = val_init_stacks(initstacks_fn, stack_extra_args)
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
                      values::Dict, seed::Dict, dvalues::Dict, dseed::Dict;
                      stack_arg_names::Vector{Symbol} = Symbol[])
    sig = kernel.sig
    stack_extra_args = [haskey(int_args, n) ? int_args[n] : deepcopy(values[n]) for n in stack_arg_names]
    stacks = val_init_stacks(initstacks_fn, stack_extra_args)
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
                               trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3,
                               stack_arg_names::Vector{Symbol} = Symbol[])
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
        val_call_adjoint(adjoint_fn, initstacks_fn, kernel, int_args, vals, seed;
                          stack_arg_names = stack_arg_names)
    end
    return val_finite_diff_check(f_eval, f_grad, x0; epsilon = epsilon, trials = trials, rtol = rtol)
end

# ---- exact tangent-vs-adjoint dot-product oracle -------------------
# The three oracles above bottom out in finite differences, capped by FD truncation (~1e-8). This compares the two derivative codes
# against each other instead, via the adjoint's defining identity <Yb,J*Xd>==<J'*Yb,Xd>, computed by _d and _b at the same point with the
# same random Xd/Yb. No epsilon: agreement should be at machine precision (~1e-14), catching adjoint errors two orders smaller than FD
# can see. It does NOT validate the tangent, since a bug shared by both codes cancels.
function val_validate_dotprod(kernel, tangent_expr::Expr, adjoint_out, baseline;
                               trials::Int = 10, rtol::Float64 = 1e-11,
                               stack_arg_names::Vector{Symbol} = Symbol[])
    tangent_fn = val_compile(tangent_expr)
    adjoint_fn = val_compile(adjoint_out.adjoint)
    initstacks_fn = val_compile(adjoint_out.initstacks)
    int_args = baseline.int_args
    vals = baseline.values
    zeros_t = val_zeros_like(kernel, vals)
    results = NamedTuple[]
    for _ in 1:trials
        xd = val_random_values_like(kernel, vals)
        yb = val_random_values_like(kernel, vals)
        yd = val_call_tangent(tangent_fn, kernel, int_args, deepcopy(vals), xd)
        xb = val_call_adjoint(adjoint_fn, initstacks_fn, kernel, int_args, deepcopy(vals), yb;
                               stack_arg_names = stack_arg_names)
        lhs = sum(val_flatten(kernel, yb) .* yd)
        rhs = sum(xb .* val_flatten(kernel, xd))
        scale = max(abs(lhs), abs(rhs), 1.0)
        push!(results, (lhs = lhs, rhs = rhs, rel_err = abs(lhs - rhs) / scale))
    end
    max_rel_err = maximum(r.rel_err for r in results)
    return (ok = max_rel_err <= rtol, max_rel_err = max_rel_err, results = results)
end

function val_validate_hvp(kernel, primal_expr::Expr, adjoint_out, hvp_out, baseline;
                           trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3,
                           stack_arg_names::Vector{Symbol} = Symbol[])
    adjoint_fn = val_compile(adjoint_out.adjoint)
    initstacks_fn = val_compile(adjoint_out.initstacks)
    hv_fn = val_compile(hvp_out.hvp)
    int_args = baseline.int_args
    x0 = val_flatten(kernel, baseline.values)
    seed = val_random_values_like(kernel, baseline.values)
    g_of = x -> begin
        vals = val_unflatten(kernel, int_args, baseline.values, x)
        val_call_adjoint(adjoint_fn, initstacks_fn, kernel, int_args, vals, seed; stack_arg_names = stack_arg_names)
    end
    hv_of = (x, v) -> begin
        vals = val_unflatten(kernel, int_args, baseline.values, x)
        dvals = val_unflatten(kernel, int_args, val_zeros_like(kernel, baseline.values), v)
        dseed = val_zeros_like(kernel, baseline.values)
        val_call_hv(hv_fn, initstacks_fn, kernel, int_args, vals, seed, dvals, dseed; stack_arg_names = stack_arg_names)
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
    return string(Base.remove_linenums!(deepcopy(expr))) * "\n"
end

# like io_read_kernel, but for files bundling more than one top-level
# function definition (e.g. a third-party `initstacks_foo_b`/`foo_b`
# pair alongside a trailing copy of the primal) -- returns every
# definition found, in file order, with no uniqueness check.
function io_read_kernel_bundle(path::String)
    src = read(path, String)
    parsed = Meta.parseall(src)
    return Expr[e for e in parsed.args if e isa Expr && e.head == :function]
end

# Unlike io_read_kernel_bundle (an unkeyed, order-preserving list for a mixed bag of
# differently-named generated artifacts), this reads a corpus of original kernel definitions
# that may call each other -- keyed by each kernel's own parsed name, erroring on a
# duplicate, ready to hand straight to inl_inline_calls. Works unchanged for a single-kernel
# file too.
function io_read_kernel_corpus(path::String)
    src = read(path, String)
    parsed = Meta.parseall(src)
    defs = [e for e in parsed.args if e isa Expr && e.head == :function]
    isempty(defs) && error("expected at least one function definition in $path, found 0")
    kernels = Dict{Symbol,Expr}()
    for def in defs
        name, _ = parse_signature(def.args[1])
        haskey(kernels, name) &&
            error("duplicate kernel name :$(name) in $path")
        kernels[name] = def
    end
    return kernels
end

# Shared by io_read_corpus_entry and every stade_*_file writer below: resolves which kernel
# in a corpus is the entry -- the one whose overall behavior the file is about. For a
# single-kernel file that's just its one kernel. For a multi-kernel corpus file, the
# convention every single-kernel file already follows (a file named foo.jl defines a kernel
# named foo) extends to say the entry kernel is named after the file.
function io_corpus_entry_name(path::String, kernels::Dict{Symbol,Expr})
    length(kernels) == 1 && return first(keys(kernels))
    entry_name = Symbol(splitext(basename(path))[1])
    haskey(kernels, entry_name) ||
        error("io_corpus_entry_name: $path defines $(length(kernels)) kernels but none is named :$(entry_name) (the file's own basename) -- a multi-kernel file needs an entry kernel named after the file")
    return entry_name
end

# Every val_*/stade_validate_* function below only takes a single primal Expr -- this is the
# one place that bridges a possibly-multi-kernel file down to that single Expr, so nothing
# downstream needs to know whether the file had one kernel or several. For a single-kernel
# file, returns that kernel's Expr exactly as io_read_kernel always did. For a multi-kernel
# corpus file, inlines the whole call graph and returns the entry kernel.
function io_read_corpus_entry(path::String)
    kernels = io_read_kernel_corpus(path)
    entry_name = io_corpus_entry_name(path, kernels)
    length(kernels) == 1 && return kernels[entry_name]
    inlined = inl_inline_calls(kernels)
    return inlined[entry_name]
end

# bundles initstacks_foo_b, foo_b, and a copy of foo itself, in that order
function io_write_kernel_file(path::String, primal_expr::Expr, generated::Vector{Expr})
    parts = [io_expr_to_source(e) for e in vcat(generated, [primal_expr])]
    open(path, "w") do f
        write(f, join(parts, "\n"))
    end
    return nothing
end

# Corpus counterpart of io_write_kernel_file: for every name in primal_exprs (sorted for deterministic layout), bundles
# that name's generated parts (already assembled by the caller) followed by its primal_exprs entry, exactly as handed in
# -- no opinion on whether that primal is original or inlined. stade_tangent_file et al. pass the entry kernel's fully-
# inlined primal, so the written file is just that kernel's derivative plus its own primal. Reduces to
# io_write_kernel_file's output for a single-entry call.
function io_write_kernel_corpus_file(path::String, primal_exprs::Dict{Symbol,Expr}, generated_parts::Dict{Symbol,Vector{Expr}})
    parts = String[]
    for name in sort(collect(keys(primal_exprs)); by = string)
        for e in get(generated_parts, name, Expr[])
            push!(parts, io_expr_to_source(e))
        end
        push!(parts, io_expr_to_source(primal_exprs[name]))
    end
    open(path, "w") do f
        write(f, join(parts, "\n"))
    end
    return nothing
end

io_path_exists(path::String) = isfile(path)

# flat writer for the cgen_/jgen_ GPU-porting entry points (stade_gpu_file
# et al.) -- unlike io_write_kernel_file/io_write_kernel_corpus_file there
# is no single designated "primal" to append last: a multi-function input
# may hand back several independent host functions (one per input def),
# so the caller decides the full ordered list itself
function io_write_gpu_file(path::String, exprs::Vector{Expr}; preamble::String = "")
    parts = [io_expr_to_source(e) for e in exprs]
    open(path, "w") do f
        isempty(preamble) || write(f, preamble * "\n")
        write(f, join(parts, "\n"))
    end
    return nothing
end

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

# io_read_baseline_yaml is kernel-agnostic (plain text in, numbers out) so it has no way to know
# which `values:` entries are really array_int args rather than array_float ones -- everything
# comes back parsed as Float64. Any caller with `kernel` in scope must coerce those entries back
# to Int before using them as array indices (round rather than raw Int(...) truncation, for
# exact-integer-valued Float64->Int robustness).
function val_coerce_int_arrays!(kernel, values::Dict)
    for a in kernel.sig.args
        if kernel.sig.kinds[a] == :array_int && haskey(values, a)
            values[a] = round.(Int, values[a])
        end
    end
    return values
end


# ==================== stade_* (public API) ========================
# independents/dependents auto-derived -- see parse_infer_indep_dep.
# Override kwargs exist only for the rare exclusion case.

function stade_tangent(expr::Expr; independents::Union{Vector{Symbol},Nothing}=nothing,
                        dependents::Union{Vector{Symbol},Nothing}=nothing,
                        keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    # Both kwargs accepted, documented, and otherwise ignored -- tgen_* never emits push!/pop!
    # at all (every active statement gets a shadow line directly, no stacks), so there's nothing
    # for either keep_push_pop's storage strategy or fuse_ii_loops' fusion to apply to. Pure
    # interface-consistency no-ops, letting a caller iterate uniformly over
    # stade_tangent/stade_adjoint/stade_hvp without special-casing tangent mode.
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    active_map = act_analyze(kernel)
    lin_plan = lin_build(kernel, active_map)
    return tgen_emit(kernel, lin_plan)
end

# Computes the site-level TBR decision from both independently duplicated implementations (snap_* and
# agen_*) and asserts they agree exactly before returning either -- the permanent guard, so a future
# silent divergence fails loudly here instead of corrupting a gradient. Returns (snap_sites,
# agen_sites): snap_plan consumes its own, agen_emit/hvp_emit consume theirs. Always run -- site-level
# TBR is no longer opt-in, it's how every snapshot decision is made.
function stade_site_level_tbr_check(kernel)
    snap_sites = snap_value_needed_sites(kernel)
    agen_sites = agen_value_needed_sites(kernel)
    @assert keys(snap_sites) == keys(agen_sites) "site_level_tbr: snap_* and agen_* site-level analyses produced different site keys for `$(kernel.sig.name)` -- this should never happen; the two stages walked the kernel differently somewhere"
    @assert snap_sites == agen_sites "site_level_tbr: snap_* and agen_* site-level analyses DISAGREE for `$(kernel.sig.name)` -- refusing to generate code that could desync push/pop counts. The two independently-duplicated implementations have drifted; see skill-stade's site-level TBR rollout notes."
    return snap_sites, agen_sites
end

function stade_adjoint(expr::Expr; independents::Union{Vector{Symbol},Nothing}=nothing,
                        dependents::Union{Vector{Symbol},Nothing}=nothing,
                        keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    active_map = act_analyze(kernel)
    snap_sites, agen_sites = stade_site_level_tbr_check(kernel)
    snapshot_plan = snap_plan(kernel, active_map; site_needed = snap_sites)
    lin_plan = lin_build(kernel, active_map)
    ii_plan = fuse_ii_loops ? stade_ii_plan_check(kernel) : nothing
    return agen_emit(kernel, lin_plan, snapshot_plan; keep_push_pop = keep_push_pop, push_pop = agen_sites, ii_plan = ii_plan)
end

function stade_hvp(expr::Expr; independents::Union{Vector{Symbol},Nothing}=nothing,
                    dependents::Union{Vector{Symbol},Nothing}=nothing,
                    keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    active_map = act_analyze(kernel)
    snap_sites, agen_sites = stade_site_level_tbr_check(kernel)
    snapshot_plan = snap_plan(kernel, active_map; site_needed = snap_sites)
    lin_plan = lin_build(kernel, active_map)
    ii_plan = fuse_ii_loops ? stade_ii_plan_check(kernel) : nothing
    layout = nothing
    value_needed = exempt = stacks = nothing
    if !keep_push_pop
        # Tier B: see the matching comment in agen_emit above.
        value_needed = agen_value_needed_vars(kernel)
        reassigned = agen_collect_reassigned(kernel.body)
        exempt = agen_exempt_vars(kernel, value_needed)
        stacks = agen_stack_map(snapshot_plan)
        layout = agen_layout(kernel, kernel.sig.kinds, active_map, value_needed, reassigned, exempt, stacks; push_pop = agen_sites)
    end
    (initstacks_expr, table_names, tot_names, val_names) = agen_init_emit(kernel, snapshot_plan; keep_push_pop = keep_push_pop, layout = layout,
                                      active_map = active_map, value_needed = value_needed, exempt = exempt, stacks = stacks, push_pop = agen_sites)
    tier_b_extra_args = vcat(table_names, tot_names, val_names)
    hvp_expr = hvp_emit(kernel, active_map, lin_plan, snapshot_plan; keep_push_pop = keep_push_pop, layout = layout, push_pop = agen_sites,
                         tier_b_extra_args = tier_b_extra_args, ii_plan = ii_plan)
    # Mirrors agen_emit's own post-hoc cleanup, applied here independently because hvp_expr, not adjoint_expr, is what this
    # function returns paired with initstacks_expr. Required for consistency: stade_adjoint's own initstacks_* already drops a
    # stack once fusion leaves it unused, and validation code sharing initstacks_* across both adjoint and hvp calls needs
    # hvp_expr's signature to match. hvp_expr's fwd/bwd use the same agen_forward_body/agen_backward_body calls as the adjoint
    # path, so scanning it independently lands on the same unused set.
    if keep_push_pop && ii_plan !== nothing
        used = agen_used_stack_names(hvp_expr)
        unused_stacks = Set(nm for nm in agen_stack_names(snapshot_plan) if !(nm in used))
        if !isempty(unused_stacks)
            hvp_expr = agen_drop_unused_stack_args(hvp_expr, unused_stacks)
            initstacks_expr = agen_drop_unused_stack_allocs(initstacks_expr, unused_stacks)
        end
    end
    return (hvp = hvp_expr, initstacks = initstacks_expr)
end

# Multi-kernel entry points: inline the whole corpus's call graph away (inl_inline_calls),
# then defer to the existing single-kernel function above, unchanged, per kernel.
# Independents/dependents overrides still don't belong here -- a caller who needs them can
# run inl_inline_calls directly and call stade_tangent/stade_adjoint/stade_hvp per kernel.
function stade_tangent_corpus(kernels::Dict{Symbol,Expr}; keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    inlined = inl_inline_calls(kernels)
    return Dict(name => stade_tangent(expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops) for (name, expr) in inlined)
end

function stade_adjoint_corpus(kernels::Dict{Symbol,Expr}; keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    inlined = inl_inline_calls(kernels)
    return Dict(name => stade_adjoint(expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops) for (name, expr) in inlined)
end

function stade_hvp_corpus(kernels::Dict{Symbol,Expr}; keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    inlined = inl_inline_calls(kernels)
    return Dict(name => stade_hvp(expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops) for (name, expr) in inlined)
end

# Reads any number of kernels from one file (a lone kernel, or a corpus that call each other),
# differentiates only the corpus's entry kernel against its whole call graph inlined away, and writes
# back out just that one kernel: its generated derivative parts, followed by its own inlined primal --
# not a bundle of every original definition. One code path handles both: a single-kernel file is a
# one-entry corpus, where inlining is a no-op.
function stade_tangent_file(in_path::String, out_path::String; keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    kernels = io_read_kernel_corpus(in_path)
    entry_name = io_corpus_entry_name(in_path, kernels)
    inlined = inl_inline_calls(kernels)
    generated = stade_tangent(inlined[entry_name]; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
    io_write_kernel_corpus_file(out_path, Dict(entry_name => inlined[entry_name]), Dict(entry_name => Expr[generated]))
    return out_path
end

function stade_adjoint_file(in_path::String, out_path::String; keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    kernels = io_read_kernel_corpus(in_path)
    entry_name = io_corpus_entry_name(in_path, kernels)
    inlined = inl_inline_calls(kernels)
    generated = stade_adjoint(inlined[entry_name]; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
    io_write_kernel_corpus_file(out_path, Dict(entry_name => inlined[entry_name]), Dict(entry_name => Expr[generated.initstacks, generated.adjoint]))
    return out_path
end

function stade_hvp_file(in_path::String, out_path::String; keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    kernels = io_read_kernel_corpus(in_path)
    entry_name = io_corpus_entry_name(in_path, kernels)
    inlined = inl_inline_calls(kernels)
    generated = stade_hvp(inlined[entry_name]; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
    io_write_kernel_corpus_file(out_path, Dict(entry_name => inlined[entry_name]), Dict(entry_name => Expr[generated.initstacks, generated.hvp]))
    return out_path
end

# Expr in, cuda_plan out -- accepts a plain skill-stade kernel or one of STADE's own generated functions, for
# whichever GPU backend descriptor is passed in. precision=nothing means use the backend's own default_precision
# (Float64 for CUDA/AMDGPU, Float32 for Metal); an explicit precision overrides that, except for a precision_locked
# backend, where anything else is a hard error at generation time rather than a silent guarantee that only surfaces
# once the caller tries to compile/run the result.
function stade_gpu(expr::Expr, backend; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true)
    p = precision === nothing ? backend.default_precision : precision
    if backend.precision_locked && p !== backend.default_precision
        error("stade_gpu: backend `$(backend.kernel_tag)` only supports precision=$(backend.default_precision) (got $(p)) -- $(backend.precision_lock_reason)")
    end
    plan = cgen_emit(cgen_ingest(expr), backend; keep_all_atomic)
    p === Float64 && return plan
    return (host = cgen_convert_precision(plan.host, p),
            kernels = Expr[cgen_convert_precision(k, p) for k in plan.kernels])
end

stade_cuda(expr::Expr; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true) =
    stade_gpu(expr, cgen_backend_cuda(); precision, keep_all_atomic)
stade_amdgpu(expr::Expr; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true) =
    stade_gpu(expr, cgen_backend_amdgpu(); precision, keep_all_atomic)
stade_metal(expr::Expr; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true) =
    stade_gpu(expr, cgen_backend_metal(); precision, keep_all_atomic)

# Path in, path out. Reads every function def in in_path and writes one file: every device kernel first, then every host
# function in original order. precision applies uniformly to every function converted in this call; for per-function
# control, call stade_gpu directly on each def. The input file is only ever read, never rewritten. keep_all_atomic
# (default true): pass false to let a pure scalar reduction generate as a dot/sum/mapreduce call instead of a hand-rolled
# atomic-accumulate kernel.
function stade_gpu_file(in_path::String, out_path::String, backend; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true)
    defs = io_read_kernel_bundle(in_path)
    kernels = Expr[]
    hosts = Expr[]
    for expr in defs
        plan = stade_gpu(expr, backend; precision, keep_all_atomic)
        append!(kernels, plan.kernels)
        push!(hosts, plan.host)
    end
    io_write_gpu_file(out_path, vcat(kernels, hosts); preamble = backend.preamble)
    return out_path
end

stade_cuda_file(in_path::String, out_path::String; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true) =
    stade_gpu_file(in_path, out_path, cgen_backend_cuda(); precision, keep_all_atomic)
stade_amdgpu_file(in_path::String, out_path::String; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true) =
    stade_gpu_file(in_path, out_path, cgen_backend_amdgpu(); precision, keep_all_atomic)
stade_metal_file(in_path::String, out_path::String; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true) =
    stade_gpu_file(in_path, out_path, cgen_backend_metal(); precision, keep_all_atomic)

# JACC has no gpu_backend value at all -- there's only one JACC target from cgen_/jgen_'s point
# of view, since which vendor it runs on is chosen later, outside this call. precision has no
# locked default here for the same reason: Float64 unless the caller asks otherwise.
# keep_all_atomic: same meaning as stade_gpu_file's; on JACC a matched reduction becomes one
# JACC.@parallel_reduce call instead of @parallel_for + Atomix.@atomic.
function stade_jacc(expr::Expr; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true)
    plan = jgen_emit(cgen_ingest(expr); keep_all_atomic)
    p = precision === nothing ? Float64 : precision
    p === Float64 && return plan
    return (host = cgen_convert_precision(plan.host, p),
            kernels = Expr[cgen_convert_precision(k, p) for k in plan.kernels])
end

function stade_jacc_file(in_path::String, out_path::String; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, keep_all_atomic::Bool = true)
    defs = io_read_kernel_bundle(in_path)
    kernels = Expr[]
    hosts = Expr[]
    for expr in defs
        plan = stade_jacc(expr; precision, keep_all_atomic)
        append!(kernels, plan.kernels)
        push!(hosts, plan.host)
    end
    io_write_gpu_file(out_path, vcat(kernels, hosts); preamble = jgen_preamble())
    return out_path
end


# ==================== stade_* baseline validation (public API) =====
# Numerically validates a generated tangent/adjoint/hvp file against central finite differences
# of the primal, using a baseline auto-generated once and cached to a YAML file next to the
# kernel (or a user-supplied one, read via the same public entry point). See the val_* banner
# above for what each mode checks.

# The function that reads a baseline YAML and performs the check -- usable directly by a
# caller pointing at their own hand-written file. Exact tangent-vs-adjoint check for one
# kernel file. Generates both derivative codes with the SAME flags (the defaults would
# otherwise validate a different mode's math than the one under test -- the same trap
# documented for keep_push_pop).
function stade_validate_dotprod_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                      scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                      trials::Int = 10, rtol::Float64 = 1e-11, self_check::Bool = true,
                                      keep_push_pop::Bool = true, fuse_ii_loops::Bool = false)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = io_read_baseline_yaml(yp)
    val_coerce_int_arrays!(kernel, baseline.values)
    tangent_expr = stade_tangent(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
    adjoint_out = stade_adjoint(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
    stack_names = Symbol[a for a in adjoint_out.initstacks.args[1].args[2:end]]
    return val_validate_dotprod(kernel, tangent_expr, adjoint_out, baseline;
                                 trials = trials, rtol = rtol, stack_arg_names = stack_names)
end

function stade_validate_from_baseline(mode::Symbol, in_path::String, yaml_path::String;
                                       trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3,
                                       keep_push_pop::Bool = true, fuse_ii_loops::Bool = false)
    mode in (:tangent, :adjoint, :hvp) ||
        error("stade_validate_from_baseline: mode must be :tangent, :adjoint, or :hvp, got $mode")
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = io_read_baseline_yaml(yaml_path)
    val_coerce_int_arrays!(kernel, baseline.values)
    if mode == :tangent
        tangent_expr = stade_tangent(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
        return val_validate_tangent(kernel, primal_expr, tangent_expr, baseline;
                                     trials = trials, epsilon = epsilon, rtol = rtol)
    elseif mode == :adjoint
        adjoint_out = stade_adjoint(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
# Under keep_push_pop=false, initstacks_*'s own signature is whatever agen_emit built it with -- read it back from
# the generated Expr rather than recomputing it here, so this can never drift from what agen_init_emit decided. NB:
# Symbol.(...) on an empty Any[] yields Vector{Any}, not Vector{Symbol}, so a kernel whose indexed-mode initstacks_*
# takes no arguments tripped the stack_arg_names::Vector{Symbol} type assertion. The typed comprehension is what
# val_def_arg_names already uses.
        stack_arg_names = keep_push_pop ? Symbol[] :
            Symbol[a for a in adjoint_out.initstacks.args[1].args[2:end]]
        return val_validate_adjoint(kernel, primal_expr, adjoint_out, baseline;
                                     trials = trials, epsilon = epsilon, rtol = rtol, stack_arg_names = stack_arg_names)
    else
        adjoint_out = stade_adjoint(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
        hvp_out = stade_hvp(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
# NB: Symbol.(...) on an empty Any[] yields Vector{Any}, not Vector{Symbol}, so a kernel
# whose indexed-mode initstacks_* takes no arguments (no stacks at all) tripped the
# stack_arg_names::Vector{Symbol} keyword's type assertion. The typed comprehension is what
# val_def_arg_names already uses; keep the two spellings the same.
        stack_arg_names = keep_push_pop ? Symbol[] :
            Symbol[a for a in adjoint_out.initstacks.args[1].args[2:end]]
        return val_validate_hvp(kernel, primal_expr, adjoint_out, hvp_out, baseline;
                                 trials = trials, epsilon = epsilon, rtol = rtol, stack_arg_names = stack_arg_names)
    end
end

# generates a random baseline for `in_path` and writes it to a YAML
# file sharing its basename (`foo.jl` -> `foo.yaml`), or to
# `yaml_path` if given. Exposed standalone so a caller can generate
# once, hand-edit the result, then validate repeatedly against it.
function stade_generate_baseline_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                       scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                       self_check::Bool = true,
                                       divisible_by::Union{Dict{Symbol,Int},Nothing} = nothing)
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = val_generate_baseline(kernel, primal_expr; scale = scale, int_lo = int_lo, int_hi = int_hi,
                                      self_check = self_check, divisible_by = divisible_by)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_write_baseline_yaml(yp, kernel.sig.name, baseline.int_args, baseline.values)
    return yp
end

function stade_validate_tangent_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                      scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                      trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3,
                                      self_check::Bool = true, keep_push_pop::Bool = true, fuse_ii_loops::Bool = false)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    return stade_validate_from_baseline(:tangent, in_path, yp; trials = trials, epsilon = epsilon, rtol = rtol,
                                         keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
end

function stade_validate_adjoint_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                      scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                      trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3,
                                      self_check::Bool = true, keep_push_pop::Bool = true, fuse_ii_loops::Bool = false)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    return stade_validate_from_baseline(:adjoint, in_path, yp; trials = trials, epsilon = epsilon, rtol = rtol,
                                         keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
end

function stade_validate_hvp_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                  scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                  trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3,
                                  self_check::Bool = true, keep_push_pop::Bool = true, fuse_ii_loops::Bool = false)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    return stade_validate_from_baseline(:hvp, in_path, yp; trials = trials, epsilon = epsilon, rtol = rtol,
                                         keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
end

# Sibling to stade_validate_adjoint_file for a third-party (not STADE-generated) adjoint, e.g. from Tapenade: same
# baseline machinery and val_validate_adjoint oracle, but the adjoint/initstacks come from adjoint_path instead of
# calling stade_adjoint on the primal. stade_validate_adjoint_file itself can't be reused as-is, since it always
# regenerates STADE's own adjoint internally -- this reuses everything beneath it instead. Expects adjoint_path to
# bundle a <name>_b function and an initstacks_<name>_b function, matching Tapenade's naming convention.
function stade_validate_adjoint_against_file(primal_path::String, adjoint_path::String;
                                              yaml_path::Union{String,Nothing} = nothing,
                                              scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                              trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3)
    primal_expr = io_read_corpus_entry(primal_path)
    kernel = parse_kernel(primal_expr)
    bname = string(kernel.sig.name)
    defs = io_read_kernel_bundle(adjoint_path)
    adjoint_expr = val_find_def(defs, Symbol(bname * "_b"))
    initstacks_expr = val_find_def(defs, Symbol("initstacks_" * bname * "_b"))
    adjoint_out = (adjoint = adjoint_expr, initstacks = initstacks_expr)
    stack_arg_names = val_def_arg_names(initstacks_expr)

    yp = yaml_path === nothing ? io_default_yaml_path(primal_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(primal_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi)
    baseline = io_read_baseline_yaml(yp)
    val_coerce_int_arrays!(kernel, baseline.values)

    return val_validate_adjoint(kernel, primal_expr, adjoint_out, baseline;
                                 trials = trials, epsilon = epsilon, rtol = rtol,
                                 stack_arg_names = stack_arg_names)
end

# ==================== stade_validate_gpu_file (Item 1) ==============
# Emits one self-contained Julia script bundling the keep_push_pop=false CPU adjoint and its stade_gpu-converted device counterpart, plus a
# baseline as literal Julia source -- no dependency on STADE.jl itself. It runs both, compares every mutated array/scalar return, and prints only a
# verdict on stdout. This sandbox has no GPU, so this function never executes the script -- it returns a NamedTuple like every other
# stade_validate_*_file plus skipped=true/script_path. Only keep_push_pop=false has a GPU target, so keep_push_pop=true is a hard error.

val_def_fn_name(expr::Expr) = expr.args[1].args[1]

# Array SHAPES always come from the caller's own baseline/seed Dicts
# (never hand-constructed here -- see the Item-1 trap about ttgc's
# `c`/`skx`/`i_cell_to_node` matrices), so this only ever needs to
# serialize whatever real value/shape it's handed.
val_julia_literal(v::Real) = string(Float64(v))
function val_julia_literal(v::AbstractVector{<:Real})
    T = eltype(v) <: Integer ? "Int64" : "Float64"
    return "$(T)[" * join(v, ", ") * "]"
end
function val_julia_literal(v::AbstractMatrix{<:Real})
    T = eltype(v) <: Integer ? "Int64" : "Float64"
    rows = [join(v[i, :], " ") for i in 1:size(v, 1)]
    return "$(T)[" * join(rows, "; ") * "]"
end

# positional call-argument NAMES (not values) for one device's copies,
# in kernel.sig.args order with each float-kind arg expanded to its
# (value, shadow) pair -- mirrors val_adjoint_call_args's own ordering
# rule exactly (array_int gets no shadow; scalar_int is a bare name,
# shared verbatim between cpu/gpu since it's never wrapped).
function val_gpu_call_args(sig, device::Symbol)
    parts = String[]
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(parts, string(a))
        elseif k in (:scalar_float, :array_float)
            push!(parts, "$(a)_val_$(device)", "$(a)_sh_$(device)")
        else
            push!(parts, "$(a)_val_$(device)")
        end
    end
    return parts
end

# a stack_arg_name is either a kernel-level scalar_int (already bound
# to a plain, device-independent variable above) or a values-Dict
# entry with its own per-device _val copy -- same two cases
# val_call_adjoint already distinguishes.
function val_gpu_stack_extra_ref(n::Symbol, int_args::Dict, device::Symbol)
    haskey(int_args, n) && return string(n)
    return "$(n)_val_$(device)"
end

# device-name text embedded in the script's own final JSON line --
# generic over backend rather than hardcoding CUDA, since stade_gpu
# itself is backend-parametric; falls back to a literal string for any
# backend this hasn't been taught a device-query for yet rather than
# emitting code that won't run.
function val_gpu_device_name_expr(backend)
    backend.kernel_tag == "cuda" && return "string(CUDA.name(CUDA.device()))"
    backend.kernel_tag == "amdgpu" && return "string(AMDGPU.device())"
    backend.kernel_tag == "metal" && return "string(Metal.current_device())"
    # JACC has no vendor-neutral device-name query of its own -- the image sets its backend at
    # build time via JACC.set_backend("CUDA"), so on this image JACC's device and CUDA's device
    # are the same physical GPU; val_jacc_backend()'s own preamble adds `using CUDA`
    # specifically so this query works.
    backend.kernel_tag == "jacc" && return "string(CUDA.name(CUDA.device()))"
    return "\"unknown-backend-$(backend.kernel_tag)\""
end

# Assembles the full script described above. `seeds` is a Vector of
# per-trial seed Dicts (same shape as `values`, one per trial) --
# `values` itself (the primal baseline) is shared across trials but
# freshly deepcopy'd every trial on both devices, since the call under
# test mutates its float arrays in place.
function val_gpu_parity_script(kernel, int_args::Dict, values::Dict, seeds::Vector,
                                adjoint_out, gpu_init, gpu_adj, stack_arg_names::Vector{Symbol},
                                backend; rtol::Float64 = 2.5e-14)
    sig = kernel.sig
    cpu_init_name = val_def_fn_name(adjoint_out.initstacks)
    cpu_adj_name  = val_def_fn_name(adjoint_out.adjoint)
    gpu_init_name = val_def_fn_name(gpu_init.host)
    gpu_adj_name  = val_def_fn_name(gpu_adj.host)
    arrtype = backend.arrtype

    lines = String[]
    isempty(backend.preamble) || push!(lines, backend.preamble)
    push!(lines, "using JSON3", "")

    push!(lines, io_expr_to_source(adjoint_out.initstacks))
    push!(lines, io_expr_to_source(adjoint_out.adjoint))
    push!(lines, io_expr_to_source(gpu_init.host))
    for k in gpu_init.kernels
        push!(lines, io_expr_to_source(k))
    end
    push!(lines, io_expr_to_source(gpu_adj.host))
    for k in gpu_adj.kernels
        push!(lines, io_expr_to_source(k))
    end

    push!(lines, "function _val_init_stacks_local(fn, extra_args)\n    r = fn(extra_args...)\n    r === nothing && return ()\n    r isa Tuple && return r\n    return (r,)\nend\n")
    push!(lines, "_relerr_scalar(a, b) = abs(a - b) / max(abs(a), abs(b), 1.0)\n" *
                 "_relerr_arr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1.0))\n")

    for a in sig.args
        sig.kinds[a] == :scalar_int || continue
        push!(lines, "$(a) = $(int_args[a])")
    end

    float_or_intarr_args = [a for a in sig.args if sig.kinds[a] in (:scalar_float, :array_float, :array_int)]
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    array_float_args = [a for a in sig.args if sig.kinds[a] == :array_float]
    array_int_args = [a for a in sig.args if sig.kinds[a] == :array_int]

    for a in float_or_intarr_args
        push!(lines, "__base_$(a) = $(val_julia_literal(values[a]))")
    end
    for a in scalar_args
        seed_list = join([val_julia_literal(sd[a]) for sd in seeds], ", ")
        push!(lines, "__seed_$(a) = [$(seed_list)]")
    end
    for a in array_float_args
        seed_list = join([val_julia_literal(sd[a]) for sd in seeds], ", ")
        push!(lines, "__seed_$(a) = [$(seed_list)]")
    end

    stack_extra_cpu = join([val_gpu_stack_extra_ref(n, int_args, :cpu) for n in stack_arg_names], ", ")
    stack_extra_gpu = join([val_gpu_stack_extra_ref(n, int_args, :gpu) for n in stack_arg_names], ", ")
    cpu_call = "$(cpu_adj_name)($(join(val_gpu_call_args(sig, :cpu), ", ")), stacks_cpu...)"
    gpu_call = "$(gpu_adj_name)($(join(val_gpu_call_args(sig, :gpu), ", ")), stacks_gpu...)"
    ret_cpu_tuple = isempty(scalar_args) ? "()" : (length(scalar_args) == 1 ? "(ret_cpu,)" : "ret_cpu")
    ret_gpu_tuple = isempty(scalar_args) ? "()" : (length(scalar_args) == 1 ? "(ret_gpu,)" : "ret_gpu")

    push!(lines, "max_rel_err = 0.0")
    push!(lines, "for __trial in 1:$(length(seeds))")
    for a in array_float_args
        push!(lines, "  $(a)_val_cpu = deepcopy(__base_$(a)); $(a)_sh_cpu = deepcopy(__seed_$(a)[__trial])")
        push!(lines, "  $(a)_val_gpu = $(arrtype)(deepcopy(__base_$(a))); $(a)_sh_gpu = $(arrtype)(deepcopy(__seed_$(a)[__trial]))")
    end
    for a in array_int_args
        push!(lines, "  $(a)_val_cpu = deepcopy(__base_$(a))")
        push!(lines, "  $(a)_val_gpu = $(arrtype)(deepcopy(__base_$(a)))")
    end
    for a in scalar_args
        push!(lines, "  $(a)_val_cpu = __base_$(a); $(a)_sh_cpu = __seed_$(a)[__trial]")
        push!(lines, "  $(a)_val_gpu = __base_$(a); $(a)_sh_gpu = __seed_$(a)[__trial]")
    end
    push!(lines, "  stacks_cpu = _val_init_stacks_local($(cpu_init_name), Any[$(stack_extra_cpu)])")
    push!(lines, "  stacks_gpu = _val_init_stacks_local($(gpu_init_name), Any[$(stack_extra_gpu)])")
    push!(lines, "  ret_cpu = $(cpu_call)")
    push!(lines, "  ret_gpu = $(gpu_call)")
    push!(lines, "  ret_cpu_t = $(ret_cpu_tuple)")
    push!(lines, "  ret_gpu_t = $(ret_gpu_tuple)")
    push!(lines, "  local_max = 0.0")
    for a in array_float_args
        push!(lines, "  local_max = max(local_max, _relerr_arr($(a)_sh_cpu, Array($(a)_sh_gpu)))")
        push!(lines, "  local_max = max(local_max, _relerr_arr($(a)_val_cpu, Array($(a)_val_gpu)))")
    end
    for a in array_int_args
        push!(lines, "  local_max = max(local_max, _relerr_arr(Float64.($(a)_val_cpu), Float64.(Array($(a)_val_gpu))))")
    end
    for (i, _) in enumerate(scalar_args)
        push!(lines, "  local_max = max(local_max, _relerr_scalar(ret_cpu_t[$(i)], ret_gpu_t[$(i)]))")
    end
    push!(lines, "  global max_rel_err = max(max_rel_err, local_max)")
    push!(lines, "end")
    push!(lines, "println(JSON3.write(Dict(\"ok\" => max_rel_err <= $(rtol), \"max_rel_err\" => max_rel_err, \"device\" => $(val_gpu_device_name_expr(backend)))))")

    return join(lines, "\n") * "\n"
end

# Counts every `@atomic ...` site (backend.atomic_macro) anywhere within
# a generated device kernel Expr, at any nesting depth -- used to scale
# stade_validate_gpu_file's default tolerance (see its own comment).
function val_count_atomic_sites(expr, atomic_macro)
    expr isa Expr || return 0
    n = (expr.head == :macrocall && !isempty(expr.args) && expr.args[1] == atomic_macro) ? 1 : 0
    for a in expr.args
        n += val_count_atomic_sites(a, atomic_macro)
    end
    return n
end

function stade_validate_gpu_file(in_path::String, out_path::String, backend;
                                  yaml_path::Union{String,Nothing} = nothing,
                                  scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                  trials::Int = 3, rtol::Union{Float64,Nothing} = nothing,
                                  self_check::Bool = true,
                                  keep_push_pop::Bool = false, fuse_ii_loops::Bool = false)
    keep_push_pop &&
        error("stade_validate_gpu_file: keep_push_pop=true has no GPU target -- :stack mode's push!/pop! is inherently host-only")
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = io_read_baseline_yaml(yp)
    val_coerce_int_arrays!(kernel, baseline.values)

    adjoint_out = stade_adjoint(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops)
    gpu_init = stade_gpu(adjoint_out.initstacks, backend)
    gpu_adj  = stade_gpu(adjoint_out.adjoint, backend)
    stack_arg_names = Symbol[a for a in adjoint_out.initstacks.args[1].args[2:end]]
    seeds = [val_random_values_like(kernel, baseline.values) for _ in 1:trials]

# IEEE float addition isn't associative, so an atomic accumulation sums its contributing threads in whatever order the
# scheduler runs them -- different from the CPU adjoint's sequential summation, though both compute the identical sum. A kernel
# with one or two atomic sites reproduces to within ~1e-16; unet's 35 sites accumulate that reordering noise repeatedly,
# measuring ~2.5e-13 -- correct, not a bug, just past a tolerance calibrated for the atomic-light case. Scaling linearly with
# atomic-site count is a conservative proxy, verified against unet's case. An explicit rtol always overrides this.
    resolved_rtol = rtol
    if resolved_rtol === nothing
        atomic_sites = sum(val_count_atomic_sites(k, backend.atomic_macro) for k in gpu_adj.kernels; init = 0)
        resolved_rtol = 2.5e-14 * max(1, atomic_sites)
    end

    script = val_gpu_parity_script(kernel, baseline.int_args, baseline.values, seeds,
                                    adjoint_out, gpu_init, gpu_adj, stack_arg_names, backend; rtol = resolved_rtol)
    open(out_path, "w") do f
        write(f, script)
    end
    return (ok = false, max_rel_err = NaN, skipped = true, script_path = out_path)
end

stade_validate_cuda_file(in_path::String, out_path::String; kwargs...) =
    stade_validate_gpu_file(in_path, out_path, cgen_backend_cuda(); kwargs...)

# ==================== stade_validate_jacc_file (Item 4) ==============
# The jgen_/JACC target was entirely unvalidated. This is that extension: the same val_gpu_parity_script used for
# stade_validate_gpu_file, just fed a JACC-generated plan (stade_jacc) instead of a cgen_backend_*-generated one. stade_jacc takes no
# backend argument (JACC picks its vendor at build/deploy time), so there's no real cgen_backend_jacc() to reuse. val_jacc_backend()
# below is NOT a real backend registry entry -- it exists purely to give val_gpu_parity_script the text fields it needs.
function val_jacc_backend()
    return (
        kernel_tag = "jacc",
        arrtype = "JACC.array",
        atomic_macro = Expr(:., :Atomix, QuoteNode(Symbol("@atomic"))),
        # jgen_preamble() now includes `using CUDA` itself (needed for
        # jgen_emit's own default @allowscalar wrapping -- see its
        # comment), which also happens to be exactly what val_gpu_
        # device_name_expr's JACC case needs to query the device name.
        preamble = jgen_preamble(),
    )
end

function stade_validate_jacc_file(in_path::String, out_path::String;
                                   yaml_path::Union{String,Nothing} = nothing,
                                   scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                   trials::Int = 3, rtol::Union{Float64,Nothing} = nothing,
                                   self_check::Bool = true,
                                   keep_push_pop::Bool = false, fuse_ii_loops::Bool = false)
    keep_push_pop &&
        error("stade_validate_jacc_file: keep_push_pop=true has no GPU target -- :stack mode's push!/pop! is inherently host-only")
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = io_read_baseline_yaml(yp)
    val_coerce_int_arrays!(kernel, baseline.values)

    adjoint_out = stade_adjoint(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops)
    backend = val_jacc_backend()
    jacc_init = stade_jacc(adjoint_out.initstacks)
    jacc_adj  = stade_jacc(adjoint_out.adjoint)
    stack_arg_names = Symbol[a for a in adjoint_out.initstacks.args[1].args[2:end]]
    seeds = [val_random_values_like(kernel, baseline.values) for _ in 1:trials]

    # same reasoning as stade_validate_gpu_file's own resolved_rtol --
    # Atomix.@atomic's cross-thread reordering is the same non-
    # associative-float-sum phenomenon regardless of which backend
    # emits the atomic.
    resolved_rtol = rtol
    if resolved_rtol === nothing
        atomic_sites = sum(val_count_atomic_sites(k, backend.atomic_macro) for k in jacc_adj.kernels; init = 0)
        resolved_rtol = 2.5e-14 * max(1, atomic_sites)
    end

    script = val_gpu_parity_script(kernel, baseline.int_args, baseline.values, seeds,
                                    adjoint_out, jacc_init, jacc_adj, stack_arg_names, backend; rtol = resolved_rtol)
    open(out_path, "w") do f
        write(f, script)
    end
    return (ok = false, max_rel_err = NaN, skipped = true, script_path = out_path)
end


# ============================================================
# ii_* : eligibility analysis and codegen for fusing iteration-independent-loop adjoint generation ("II-loop fusion", after Tapenade's II_LOOP),
# enabled via fuse_ii_loops=true. Classification is outermost-eligible-loop-first: a nest classifies as one fusion unit when the outer loop's whole
# body passes; only on failure does the walk recurse into direct :for children. This lets a value built by one sibling inner loop and consumed by
# another, both nested in the same outer iteration, classify as :independent without special-casing.

# ---- shared helpers ----
# Shared helpers, not duplicated -- pure structural walks, not a TBR/codegen decision, so Hard Rule 7's duplication rationale doesn't
# apply. Escape detection tracks program order rather than a flat 'read anywhere else' check, which is unsound: a variable name can be
# reused, unrelated, in a different loop nest, and a flat check would wrongly flag a later loop's own fresh overwrite as an escaping read
# of the earlier loop's value. A fresh overwrite kills the dependency.

# Walks `body` in forward order, threading `alive` (vars still carrying `target`'s contribution) the way snap_fwd_walk!/agen_fwd_walk_loop! thread
# `seen` -- returns the updated alive-set, accumulating escapes into `escaped`. A var is removed from `alive` on a fresh (non-self-referencing)
# assignment; a read of a still-alive var is an escape. `:if` is conservative: a var survives only if it survives both arms. Unlike
# agen_var_value_needed!/snap_var_value_needed!, this counts a read regardless of linearity, since a linear downstream read still contributes to a
# fused var's shadow, and that must land before the fused loop's own backward code.
function ii_expr_reads(expr, vars, acc)
    if expr isa Expr
        if expr.head == :ref
            (expr.args[1] in vars) && push!(acc, expr.args[1])
            for a in expr.args[2:end]
                ii_expr_reads(a, vars, acc)
            end
        else
            start = expr.head == :call ? 2 : 1
            for a in expr.args[start:end]
                ii_expr_reads(a, vars, acc)
            end
        end
    elseif expr isa Symbol
        (expr in vars) && push!(acc, expr)
    end
    return nothing
end

# Walks `body` in forward order, threading `alive` (vars still carrying `target`'s contribution) the way
# snap_fwd_walk!/agen_fwd_walk_loop! thread `seen` -- returns the updated alive-set, accumulating escapes into `escaped`. A var
# is removed from `alive` on a fresh (non-self-referencing) assignment; any read (linear or nonlinear) of a still-alive var is
# an escape. `:if` is conservative: a var survives only if it survives both arms. A write's own lhs index expressions are
# checked too -- an alive scalar used as an index elsewhere is still a genuine read.
function ii_kill_and_collect!(body, alive, escaped)
    for stmt in body
        isempty(alive) && return alive
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            local_reads = Set{Symbol}()
            ii_expr_reads(stmt.rhs, alive, local_reads)
            if stmt.lhs isa Expr
                for a in stmt.lhs.args[2:end]
                    ii_expr_reads(a, alive, local_reads)
                end
            end
            union!(escaped, local_reads)
            if stmt.lhs isa Symbol && var in alive && agen_count_var_refs(stmt.rhs, var) == 0
                delete!(alive, var)
            end
        elseif stmt.kind == :if
            cond_reads = Set{Symbol}()
            ii_expr_reads(stmt.cond, alive, cond_reads)
            union!(escaped, cond_reads)
            alive_then = ii_kill_and_collect!(stmt.then, copy(alive), escaped)
            alive_els  = ii_kill_and_collect!(stmt.els,  copy(alive), escaped)
            alive = intersect(alive_then, alive_els)
        elseif stmt.kind == :for
            alive = ii_kill_and_collect!(stmt.body, alive, escaped)
        end
    end
    return alive
end

# ---- target scope resolution: :for AND :if ancestors ----
# A sound scope check handles two cases: an ancestor that's the literal kernel top level (executes once) vs a repeating `:for` (a read before
# target can observe its contribution on the next iteration). `:if` ancestors: a branch never repeats, so repeating=false within it; the `:if`'s
# own position follows the ordinary rule one level out, and the sibling branch is never examined, since only one branch runs per evaluation.
# `ii_find_ancestor_path` returns the chain of (body, index, repeats) triples, innermost to outermost; `nothing` means target isn't reachable.
function ii_find_ancestor_path(body, target, body_repeats::Bool)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :for
            if stmt.body === target
                return Any[(body, idx, body_repeats)]
            end
            sub = ii_find_ancestor_path(stmt.body, target, true)
            if sub !== nothing
                push!(sub, (body, idx, body_repeats))
                return sub
            end
        elseif stmt.kind == :if
            for branch in (stmt.then, stmt.els)
                sub = ii_find_ancestor_path(branch, target, false)
                if sub !== nothing
                    push!(sub, (body, idx, body_repeats))
                    return sub
                end
            end
        end
    end
    return nothing
end

# vars from `vars` that escape `target`, handling target at any :for/:if-mixed nesting depth (falls back to the top-level-only case
# when target is already top-level). At each repeating level: a full wraparound pass (same-position 'after' siblings, then wrapping
# to 'before' siblings) detects every reachable read -- one pass suffices, since `alive` is fixed, not something that needs to
# converge. What propagates outward uses only the 'after' segment's kill effect: a consumer outside this level only observes what
# the last repetition's tail left behind. A non-repeating level skips the wraparound pass entirely.
function ii_escapes_nested(kernel_body, target, vars)
    path = ii_find_ancestor_path(kernel_body, target, false)
    path === nothing && return copy(vars)
    escaped = Set{Symbol}()
    alive = copy(vars)
    for i in eachindex(path)
        isempty(alive) && break
        (body, idx, repeating) = path[i]
        after = body[idx+1:end]
        if repeating
            before = body[1:idx-1]
            ii_kill_and_collect!(vcat(after, before), copy(alive), escaped)
            alive = ii_kill_and_collect!(after, alive, escaped)
        else
            alive = ii_kill_and_collect!(after, alive, escaped)
        end
    end
    return escaped
end

# ---- snap_ii_* / agen_ii_* pair (Hard Rule 7 duplication) ----

# collects every array name assigned (as arr[...] = ...) anywhere in
# body, recursively through :for/:if -- the candidate set to check
# for outside reads.
function ii_collect_array_writes!(body, kinds, active_map, acc)
    for stmt in body
        if stmt.kind == :assign && stmt.lhs isa Expr && stmt.lhs.head == :ref
            arr = stmt.lhs.args[1]
            get(kinds, arr, nothing) == :array_float && get(active_map, arr, false) && push!(acc, arr)
        elseif stmt.kind == :for
            ii_collect_array_writes!(stmt.body, kinds, active_map, acc)
        elseif stmt.kind == :if
            ii_collect_array_writes!(stmt.then, kinds, active_map, acc)
            ii_collect_array_writes!(stmt.els, kinds, active_map, acc)
        end
    end
    return nothing
end

# True iff `body` (recursively) writes to any active array read -- at all, linear or nonlinear -- anywhere in `kernel_body` outside `body`. Matters
# because agen_emit_ii_loop fuses a loop's entire backward differentiation; a write whose array is read elsewhere would get its backward code fused too
# early, silently wrong. Refuses the whole loop rather than fusing only proven-safe statements, since that needs agen_backward_body to skip individual
# statements, real future work. Deliberately models no 'kill' for arrays, unlike the scalar side: proving a later write safely overwrites this one needs
# index-equality proof this doesn't attempt, so an array stays alive to the kernel top level -- strictly more conservative than scalars.
function ii_body_has_escaping_array_write(kernel_body, body, kinds, active_map)
    arrs = Set{Symbol}()
    ii_collect_array_writes!(body, kinds, active_map, arrs)
    isempty(arrs) && return false
    path = ii_find_ancestor_path(kernel_body, body, false)
    path === nothing && return true   # couldn't locate -- conservative refusal
    escaped = Set{Symbol}()
    for i in eachindex(path)
        (level_body, idx, repeating) = path[i]
        after = level_body[idx+1:end]
        if repeating
            before = level_body[1:idx-1]
            ii_array_reads_walk!(vcat(after, before), nothing, arrs, escaped)
        else
            ii_array_reads_walk!(after, nothing, arrs, escaped)
        end
    end
    return !isempty(escaped)
end

function ii_array_reads_walk!(walk_body, target, arrs, acc)
    for stmt in walk_body
        if stmt.kind == :assign
            if stmt.lhs isa Expr
                for a in stmt.lhs.args[2:end]
                    ii_expr_reads(a, arrs, acc)
                end
            end
            ii_expr_reads(stmt.rhs, arrs, acc)
        elseif stmt.kind == :if
            ii_expr_reads(stmt.cond, arrs, acc)
            ii_array_reads_walk!(stmt.then, target, arrs, acc)
            ii_array_reads_walk!(stmt.els, target, arrs, acc)
        elseif stmt.kind == :for
            stmt.body === target && continue
            ii_array_reads_walk!(stmt.body, target, arrs, acc)
        end
    end
    return nothing
end

# ---- recomputability: can a value be rebuilt at backward time? -----
# A snapshot carries a primal value from forward to backward time; recomputing it there moves no differentiation, so unlike fusion it stays sound with
# an escaping array write. ii_recomputable is the proof obligation: re-executing loop_body's statements at backward time must reproduce var's forward
# value. An ARRAY is refused whenever assigned anywhere in the kernel, since proving it's this loop's own write needs index reasoning this doesn't
# attempt. A SCALAR is refused if live-in to loop_body, separating an accumulator reset outside the loop (refused) from one reset inside (accepted).
function ii_assigned_vars!(body, acc, skip = nothing)
    for stmt in body
        if stmt.kind == :assign
            push!(acc, stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1])
        elseif stmt.kind == :for
            stmt.body === skip && continue
            ii_assigned_vars!(stmt.body, acc, skip)
        elseif stmt.kind == :if
            ii_assigned_vars!(stmt.then, acc, skip)
            ii_assigned_vars!(stmt.els, acc, skip)
        end
    end
    return nothing
end

function ii_collect_defs!(body, defs, loop_vars)
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            push!(get!(defs, var, Any[]), stmt)
        elseif stmt.kind == :for
            push!(loop_vars, stmt.var)
            ii_collect_defs!(stmt.body, defs, loop_vars)
        elseif stmt.kind == :if
            ii_collect_defs!(stmt.then, defs, loop_vars)
            ii_collect_defs!(stmt.els, defs, loop_vars)
        end
    end
    return nothing
end

# Walks `body` in program order. Returns :live if a read of `var` can
# precede every write of it, :killed if a write provably comes first,
# :none if neither occurs. A write nested inside a :for/:if does NOT
# kill (the loop may run zero times, the branch may not be taken) --
# only a top-level write of this body does.
function ii_scalar_live_in(body, var)
    for stmt in body
        if stmt.kind == :assign
            reads = Set{Symbol}()
            agen_collect_expr_vars!(stmt.rhs, reads)
            if stmt.lhs isa Expr
                for a in stmt.lhs.args[2:end]
                    agen_collect_expr_vars!(a, reads)
                end
            end
            var in reads && return :live
            stmt.lhs isa Symbol && stmt.lhs == var && return :killed
        elseif stmt.kind == :for
            ii_scalar_live_in(stmt.body, var) === :live && return :live
        elseif stmt.kind == :if
            cond_reads = Set{Symbol}()
            agen_collect_expr_vars!(stmt.cond, cond_reads)
            var in cond_reads && return :live
            ii_scalar_live_in(stmt.then, var) === :live && return :live
            ii_scalar_live_in(stmt.els, var) === :live && return :live
        end
    end
    return :none
end

# ---- array intactness: mirror of the escaping-write check ----------
# ii_body_has_escaping_array_write asks if a read of this array is reachable after this loop. Intactness asks the dual: is a write reachable after? If
# none is, the array holds its forward contents at backward time, since a write before the loop has its restore run after the loop's backward code.
# Separates two cases the blunt 'assigned anywhere' rule conflated: an encoder output read once by a later decoder loop is intact, while an array
# rewritten by later passes is not. Same machinery as the read side, same wraparound distinction; a write inside loop_body itself also disqualifies it.
function ii_array_writes_walk!(walk_body, target, arrs, acc)
    for stmt in walk_body
        if stmt.kind == :assign
            if stmt.lhs isa Expr && stmt.lhs.head == :ref && stmt.lhs.args[1] in arrs
                push!(acc, stmt.lhs.args[1])
            end
        elseif stmt.kind == :if
            ii_array_writes_walk!(stmt.then, target, arrs, acc)
            ii_array_writes_walk!(stmt.els, target, arrs, acc)
        elseif stmt.kind == :for
            stmt.body === target && continue
            ii_array_writes_walk!(stmt.body, target, arrs, acc)
        end
    end
    return nothing
end

function ii_array_intact(kernel_body, loop_body, arr)
    arrs = Set{Symbol}([arr])
    own = Set{Symbol}()
    ii_array_writes_walk!(loop_body, nothing, arrs, own)
    isempty(own) || return false
    path = ii_find_ancestor_path(kernel_body, loop_body, false)
    path === nothing && return false   # couldn't locate -- conservative
    later = Set{Symbol}()
    for (level_body, idx, repeating) in path
        after = level_body[idx+1:end]
        if repeating
            before = level_body[1:idx-1]
            ii_array_writes_walk!(vcat(after, before), loop_body, arrs, later)
        else
            ii_array_writes_walk!(after, loop_body, arrs, later)
        end
    end
    return isempty(later)
end

function ii_recomputable(kernel, loop_body, var, restored = Set{Symbol}())
    written = Set{Symbol}()
    ii_assigned_vars!(kernel.body, written)
    defs = Dict{Symbol,Vector{Any}}()
    loop_vars = Set{Symbol}()
    ii_collect_defs!(loop_body, defs, loop_vars)
    seen = Set{Symbol}()
    frontier = Symbol[var]
    while !isempty(frontier)
        v = pop!(frontier)
        (v in seen || v in loop_vars) && continue
        push!(seen, v)
        is_array = get(kernel.sig.kinds, v, nothing) in (:array_float, :array_int)
        if is_array
            # intact arrays keep their forward contents at the backward
            # position -- see ii_array_intact.
            (v in written && !ii_array_intact(kernel.body, loop_body, v)) && return false
        elseif haskey(defs, v) || v in written
            # a scalar neither assigned in this body nor anywhere in the
            # kernel is a loop index or a read-only argument -- nothing
            # to prove, and it must not trip the live-in test below.
            if ii_scalar_live_in(loop_body, v) === :live
                # live-in: the recompute cannot rebuild it, so the
                # ordinary machinery must restore it instead (its own
                # stack pop, or an enclosing classified loop's own
                # recompute). Treated as a leaf either way.
                v in restored || return false
            else
                for stmt in get(defs, v, Any[])
                    stmt.lhs isa Symbol || return false
                    reads = Set{Symbol}()
                    agen_collect_expr_vars!(stmt.rhs, reads)
                    for r in reads
                        push!(frontier, r)
                    end
                end
            end
        end
    end
    return true
end

# True iff any var in `vars` is assigned inside a nested :for of `body`. agen_emit_ii_loop builds its fused loop as vcat(fwd, bwd) at the
# classified loop's own level only: the whole forward nest runs, then the whole backward nest. A fused scalar assigned inside a nested :for
# holds only its last inner-iteration value by the time the backward nest reads it -- every backward iteration then uses the wrong primal.
# Refusing here rather than teaching agen_emit_ii_loop to interleave per level is the conservative choice; interleaving is real, not-yet-
# attempted future work.
function ii_assigns_any(body, vars)
    for stmt in body
        if stmt.kind == :assign
            (stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]) in vars && return true
        elseif stmt.kind == :for
            ii_assigns_any(stmt.body, vars) && return true
        elseif stmt.kind == :if
            (ii_assigns_any(stmt.then, vars) || ii_assigns_any(stmt.els, vars)) && return true
        end
    end
    return false
end

function ii_fused_var_in_nested_for(body, vars, plan = nothing)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :for
            # a nested loop that is ITSELF classified re-establishes its
            # own scalars per inner iteration, so it is not a hazard.
            covered = plan !== nothing && haskey(plan, agen_site_key(body, idx))
            (!covered && ii_assigns_any(stmt.body, vars)) && return true
        elseif stmt.kind == :if
            (ii_fused_var_in_nested_for(stmt.then, vars, plan) ||
             ii_fused_var_in_nested_for(stmt.els, vars, plan)) && return true
        end
    end
    return false
end

# True iff some scalar assignment in `body` reads an array that `body` itself writes.
# agen_ii_recompute_stmts drops array writes from the backward-position recompute, so such a
# scalar would be rebuilt from the array's post-forward contents rather than the value it
# held when the statement first ran.
function ii_body_scalar_reads_own_array_write(body)
    arrs = Set{Symbol}()
    ii_collect_written_arrays!(body, arrs)
    isempty(arrs) && return false
    return ii_scalar_reads_arrays(body, arrs)
end

function ii_collect_written_arrays!(body, acc)
    for stmt in body
        if stmt.kind == :assign
            stmt.lhs isa Expr && stmt.lhs.head == :ref && push!(acc, stmt.lhs.args[1])
        elseif stmt.kind == :for
            ii_collect_written_arrays!(stmt.body, acc)
        elseif stmt.kind == :if
            ii_collect_written_arrays!(stmt.then, acc)
            ii_collect_written_arrays!(stmt.els, acc)
        end
    end
    return nothing
end

function ii_scalar_reads_arrays(body, arrs)
    for stmt in body
        if stmt.kind == :assign
            if stmt.lhs isa Symbol
                reads = Set{Symbol}()
                ii_expr_reads(stmt.rhs, arrs, reads)
                isempty(reads) || return true
            end
        elseif stmt.kind == :for
            ii_scalar_reads_arrays(stmt.body, arrs) && return true
        elseif stmt.kind == :if
            (ii_scalar_reads_arrays(stmt.then, arrs) ||
             ii_scalar_reads_arrays(stmt.els, arrs)) && return true
        end
    end
    return false
end

# True iff `body` still contains a push site after the vn_local exclusion -- a snapshot, branch flag, or tripcount agen_forward_body would emit anyway.
# This only matters for classifications executing the body TWICE: :reduction/:mixed keep the ordinary forward loop AND emit agen_emit_ii_loop at the
# backward position, so a surviving push runs in both, leaving the forward one orphaned. :independent replaces the forward loop, needing no gate. Keyed
# by the site-level TBR decisions: a whole-variable approximation is unusable, since a pure accumulation is value-needed yet emits no push. `exempt` is
# not consulted, since it never exempts a write inside a loop.
function ii_body_has_surviving_snapshot(body, kinds, active_map, sites, vn_local, reassigned)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            get(kinds, var, nothing) in (:scalar_float, :array_float) &&
                get(active_map, var, false) &&
                get(sites, agen_site_key(body, idx), false) &&
                !(var in vn_local) && return true
        elseif stmt.kind == :for
            isempty(agen_tripcount_bound_vars(stmt, reassigned)) || return true
            ii_body_has_surviving_snapshot(stmt.body, kinds, active_map, sites, vn_local, reassigned) && return true
        elseif stmt.kind == :if
            return true   # agen_forward_body always pushes a branch flag
        end
    end
    return false
end

function snap_ii_classify(stmt, kernel, value_needed, known_consts, active_map, plan = nothing)
    stmt.kind == :for || return :none
    agen_tier_b_walk(stmt.body, Set{Symbol}()) === nothing || return :none
    # An escaping array write rules out the three FUSING kinds, whose
    # codegen would move the array's adjoint to the forward position.
    # It does not rule out :recompute, which leaves every adjoint where
    # it was and only replaces snapshots with re-execution.
    has_escape = ii_body_has_escaping_array_write(kernel.body, stmt.body, kernel.sig.kinds, active_map)
    # kernel.body is already in scope here, so the array-privacy proof gets the same kernel-level
    # scalar defs (e.g. `n_in_msg = 2*n_node_feat+n_edge_feat`) as the cgen_/jgen_ splitting path --
    # no separate threading needed, unlike cgen_body/jgen_body's own outer_defs parameter.
    synth = cgen_reduction_only_loop(stmt.body, stmt.var, known_consts, cgen_scalar_def_map(kernel.body))
    synth === nothing && return :none
    local_names = cgen_locally_assigned_scalars(stmt.body)
    redvars = cgen_scalar_reduction_vars(stmt.body)
    # active_map gate first, matching snap_assign_site_decision's own
    # order (active_map[var] checked before value_needed at all) --
    # an int-kinded index var like i_loc/i_node can never be active,
    # so this is what actually excludes them, not an incidental side
    # effect of anything else here.
    vn_local = Set(v for v in intersect(value_needed, union(local_names, redvars)) if get(active_map, v, false))
    isempty(vn_local) && return :none
    # see ii_fused_var_in_nested_for -- vcat(fwd, bwd) is only valid
    # when no fused var is live across a nested loop boundary.
    ii_fused_var_in_nested_for(stmt.body, vn_local, plan) && return :none
    # A loop whose own trip count is a reassigned scalar (`for i = 1:cur` with `cur` retired
    # each pass) carries a tripcount snapshot, and every fusing kind re-runs the header at the
    # backward position against whatever `cur` holds there rather than the value this iteration
    # used. Gate 2 above only inspects the loop's body, so this shape slipped through and
    # produced wrong gradients in both stack modes whenever fusion was on.
    isempty(agen_tripcount_bound_vars(stmt, snap_collect_reassigned(kernel.body))) || return :none
    vn_red = intersect(vn_local, redvars)
    vn_ind = setdiff(vn_local, redvars)
    # Each half is checked independently against its own safety condition rather than requiring
    # the whole loop to be purely one shape or the other -- a loop can genuinely contain both a
    # pure reduction accumulator and a fully-contained independent chain at once, and there's no
    # reason to refuse the whole loop just because it isn't homogeneous.
    if !isempty(vn_red)
        inside_nonlinear = Set{Symbol}()
        snap_collect_value_needed!(stmt.body, inside_nonlinear)
        isempty(intersect(vn_red, inside_nonlinear)) || return :none
    end
    if !isempty(vn_ind)
        escaped = ii_escapes_nested(kernel.body, stmt.body, vn_ind)
        isempty(escaped) || return :none
    end
    # only :reduction/:mixed run the body twice -- see
    # ii_body_has_surviving_snapshot.
    if (!isempty(vn_red) || has_escape) &&
       (ii_body_has_surviving_snapshot(stmt.body, kernel.sig.kinds, active_map, snap_value_needed_sites(kernel),
                                        vn_local, snap_collect_reassigned(kernel.body)) ||
        # the backward-position recompute drops array writes -- see
        # ii_body_scalar_reads_own_array_write. :independent does not
        # recompute at all, so this does not apply to it.
        ii_body_scalar_reads_own_array_write(stmt.body))
        return :none
    end
    if has_escape
        # Nothing moves, so the snapshots must go instead: every fused
        # var has to be rebuildable at the backward position, with
        # live-ins supplied by the ordinary restore machinery.
        restored = setdiff(value_needed, vn_local)
        all(v -> ii_recomputable(kernel, stmt.body, v, restored), vn_local) || return :none
        return :recompute
    end
    if !isempty(vn_red) && !isempty(vn_ind)
        return :mixed
    elseif !isempty(vn_red)
        return :reduction
    else
        return :independent
    end
end

function agen_ii_classify(stmt, kernel, value_needed, known_consts, active_map, plan = nothing)
    stmt.kind == :for || return :none
    agen_tier_b_walk(stmt.body, Set{Symbol}()) === nothing || return :none
    # An escaping array write rules out the three FUSING kinds, whose
    # codegen would move the array's adjoint to the forward position.
    # It does not rule out :recompute, which leaves every adjoint where
    # it was and only replaces snapshots with re-execution.
    has_escape = ii_body_has_escaping_array_write(kernel.body, stmt.body, kernel.sig.kinds, active_map)
    # kernel.body is already in scope here, so the array-privacy proof gets the same kernel-level
    # scalar defs (e.g. `n_in_msg = 2*n_node_feat+n_edge_feat`) as the cgen_/jgen_ splitting path --
    # no separate threading needed, unlike cgen_body/jgen_body's own outer_defs parameter.
    synth = cgen_reduction_only_loop(stmt.body, stmt.var, known_consts, cgen_scalar_def_map(kernel.body))
    synth === nothing && return :none
    local_names = cgen_locally_assigned_scalars(stmt.body)
    redvars = cgen_scalar_reduction_vars(stmt.body)
    vn_local = Set(v for v in intersect(value_needed, union(local_names, redvars)) if get(active_map, v, false))
    isempty(vn_local) && return :none
    # see ii_fused_var_in_nested_for -- vcat(fwd, bwd) is only valid
    # when no fused var is live across a nested loop boundary.
    ii_fused_var_in_nested_for(stmt.body, vn_local, plan) && return :none
    # A loop whose own trip count is a reassigned scalar carries a tripcount snapshot, and every
    # fusing kind re-runs the header at the backward position against whatever the retired value
    # holds rather than what this iteration used. Gate 2 above only inspects the loop's body, so
    # this shape slipped through and produced wrong gradients in both stack modes whenever
    # fusion was on.
    isempty(agen_tripcount_bound_vars(stmt, agen_collect_reassigned(kernel.body))) || return :none
    vn_red = intersect(vn_local, redvars)
    vn_ind = setdiff(vn_local, redvars)
    if !isempty(vn_red)
        inside_nonlinear = Set{Symbol}()
        agen_collect_value_needed!(stmt.body, inside_nonlinear)
        isempty(intersect(vn_red, inside_nonlinear)) || return :none
    end
    if !isempty(vn_ind)
        escaped = ii_escapes_nested(kernel.body, stmt.body, vn_ind)
        isempty(escaped) || return :none
    end
    if (!isempty(vn_red) || has_escape) &&
       (ii_body_has_surviving_snapshot(stmt.body, kernel.sig.kinds, active_map, agen_value_needed_sites(kernel),
                                        vn_local, agen_collect_reassigned(kernel.body)) ||
        # the backward-position recompute drops array writes -- see
        # ii_body_scalar_reads_own_array_write. :independent does not
        # recompute at all, so this does not apply to it.
        ii_body_scalar_reads_own_array_write(stmt.body))
        return :none
    end
    if has_escape
        # Nothing moves, so the snapshots must go instead: every fused
        # var has to be rebuildable at the backward position, with
        # live-ins supplied by the ordinary restore machinery.
        restored = setdiff(value_needed, vn_local)
        all(v -> ii_recomputable(kernel, stmt.body, v, restored), vn_local) || return :none
        return :recompute
    end
    if !isempty(vn_red) && !isempty(vn_ind)
        return :mixed
    elseif !isempty(vn_red)
        return :reduction
    else
        return :independent
    end
end

# Recurses into a failed loop's `:for` children and `:if` branches
# looking for a smaller eligible unit (see ii_find_ancestor_path,
# which proves escape-safety correctly at any nesting depth, and
# ii_escapes_nested, which uses it).
function snap_ii_plan(kernel)
    value_needed = snap_value_needed_vars(kernel)
    active_map = act_analyze(kernel)
    plan = Dict{Any,Symbol}()
    snap_ii_plan_walk!(kernel.body, kernel, value_needed, active_map, plan)
    return plan
end

# `known_consts` is built locally, fresh at the top of every call (one per body-list), mirroring cgen_body's own convention
# rather than being threaded down from a caller -- each body-list gets its own Dict, populated only by literal scalar assigns
# preceding a candidate loop. Required, not just tidier: with known_consts always empty, an unrelated self-referencing-with-
# reset accumulator anywhere in a loop's body causes cgen_reduction_only_loop to refuse the entire loop, even when the var this
# analysis cares about has nothing to do with it.
function snap_ii_plan_walk!(body, kernel, value_needed, active_map, plan)
    known_consts = Dict{Symbol,Any}()
    for idx in eachindex(body)
        stmt = body[idx]
        if stmt.kind == :assign
            if stmt.lhs isa Symbol
                if stmt.rhs isa Number
                    known_consts[stmt.lhs] = stmt.rhs
                else
                    delete!(known_consts, stmt.lhs)
                end
            end
        elseif stmt.kind == :for
            # post-order: nested loops are classified FIRST, so this
            # loop's own ii_fused_var_in_nested_for check can see them.
            snap_ii_plan_walk!(stmt.body, kernel, value_needed, active_map, plan)
            kind = snap_ii_classify(stmt, kernel, value_needed, known_consts, active_map, plan)
            kind === :none || (plan[agen_site_key(body, idx)] = kind)
            delete!(known_consts, stmt.var)
        elseif stmt.kind == :if
            snap_ii_plan_walk!(stmt.then, kernel, value_needed, active_map, plan)
            snap_ii_plan_walk!(stmt.els, kernel, value_needed, active_map, plan)
            for v in cgen_all_assigned_scalars(vcat(stmt.then, stmt.els))
                delete!(known_consts, v)
            end
        end
    end
    return nothing
end

function agen_ii_plan(kernel)
    value_needed = agen_value_needed_vars(kernel)
    active_map = act_analyze(kernel)
    plan = Dict{Any,Symbol}()
    agen_ii_plan_walk!(kernel.body, kernel, value_needed, active_map, plan)
    return plan
end

function agen_ii_plan_walk!(body, kernel, value_needed, active_map, plan)
    known_consts = Dict{Symbol,Any}()
    for idx in eachindex(body)
        stmt = body[idx]
        if stmt.kind == :assign
            if stmt.lhs isa Symbol
                if stmt.rhs isa Number
                    known_consts[stmt.lhs] = stmt.rhs
                else
                    delete!(known_consts, stmt.lhs)
                end
            end
        elseif stmt.kind == :for
            # post-order: nested loops are classified FIRST, so this
            # loop's own ii_fused_var_in_nested_for check can see them.
            agen_ii_plan_walk!(stmt.body, kernel, value_needed, active_map, plan)
            kind = agen_ii_classify(stmt, kernel, value_needed, known_consts, active_map, plan)
            kind === :none || (plan[agen_site_key(body, idx)] = kind)
            delete!(known_consts, stmt.var)
        elseif stmt.kind == :if
            agen_ii_plan_walk!(stmt.then, kernel, value_needed, active_map, plan)
            agen_ii_plan_walk!(stmt.els, kernel, value_needed, active_map, plan)
            for v in cgen_all_assigned_scalars(vcat(stmt.then, stmt.els))
                delete!(known_consts, v)
            end
        end
    end
    return nothing
end

# cross-check: both hand-maintained copies must agree exactly,
# mirroring stade_site_level_tbr_check's own pattern.
function stade_ii_plan_check(kernel)
    a = snap_ii_plan(kernel)
    b = agen_ii_plan(kernel)
    @assert a == b "snap_ii_plan and agen_ii_plan disagree for kernel $(kernel.sig.name): $a vs $b"
    return a
end

end # module STADE