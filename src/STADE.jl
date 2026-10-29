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
#                         each provably iteration-independent loop
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
#     (kind=:for, var::Symbol, lo, hi, step, body)               # body :: statement_list
#     (kind=:if, cond, then::statement_list, els::statement_list)
#     (kind=:while, cond, body::statement_list)
#        cond follows the same rules as an :if condition. It is never
#        differentiated and takes no shadow: the trip count is recorded by
#        the forward sweep and replayed by the backward one, so the
#        gradient is the derivative at FIXED trip count.
#     -- lo/hi/step are all Expr|Symbol|Number; a plain `lo:hi` header
#        parses with `step` set to the Int64 literal `1`.
# statement_list :: Vector{NamedTuple}
# active_map    :: Dict{Symbol,Bool}                # var/array name -> is-active
# snapshot_site :: (kind=:value|:array|:branch|:tripcount|:whilecount, array::Symbol, at::Int)
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
        # A one-node cycle is still a cycle. The topo sort would find it, but
        # io_read_corpus_entry short-circuits a single-kernel file before the
        # inliner runs at all, so a self-call escaped to parse_kernel and was
        # reported as an unsupported STATEMENT FORM. The user had written a
        # recursive function and was told their syntax was unrecognised.
        name in callees && error(inl_recursion_message([name, name]))
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
        elseif stmt isa Expr && stmt.head == :while
            inl_collect_calls!(stmt.args[2], caller_name, callees, kernels)
        elseif stmt isa Expr && stmt.head == :if
            inl_collect_calls!(stmt.args[2], caller_name, callees, kernels)
            length(stmt.args) == 3 && inl_collect_calls!(stmt.args[3], caller_name, callees, kernels)
        end
    end
    return nothing
end

# Recursion is refused, and the message says what is unsupported rather
# than what the algorithm found. A user who writes a recursive kernel needs
# to know whether this is a permanent limit, an unimplemented feature, or a
# mistake of theirs. It is the first, for a concrete reason: recursive depth
# is unknown until the code runs, so the tape has no closed-form size, so
# `keep_push_pop = false` could not size its stacks and the GPU path could
# not run it. That is the same limit a `while` loop has.
#
# There is usually a rewrite. A recursion over a fixed depth is a loop with
# an explicit level index, which STADE handles today. `mg_vcycle` in the
# corpus is exactly that: a multigrid V-cycle written as loops.
function inl_recursion_message(chain::Vector{Symbol})
    what = length(chain) == 2 ? "`$(chain[1])` calls itself" :
                                "recursion: " * join(chain, " -> ")
    return "inl_inline_calls: " * what * ". STADE inlines every call, and a " *
           "recursive call has no depth known before it runs, so its tape has " *
           "no closed-form size. Rewrite the recursion as a loop over an " *
           "explicit level index, as `mg_vcycle` does"
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
            error(inl_recursion_message(vcat(stack[idx:end], n)))
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
            # An expression argument is bound to a temporary here rather than
            # substituted at every use. That keeps the substitution
            # symbol-to-symbol, so nothing downstream changes, and it avoids
            # duplicating the expression once per use of the parameter. It is
            # also the right semantics: the expression is evaluated once at
            # the call site, and once per iteration when the call sits in a
            # loop, exactly as a real call would.
            (call_args, prelude) = inl_bind_arg_exprs(callee_name, callee_kernel, call_args, counter[])
            append!(result, prelude)
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
# Which parameters MUST be a bare symbol at the call site. Two kinds:
# anything the callee assigns to, because the callee returns through it and
# an expression is not assignable, and every array, because an expression is
# not an array and a kernel has no view concept. A read-only scalar takes
# any expression.
function inl_must_be_symbol(callee_kernel)
    written = Set{Symbol}()
    parse_collect_written!(callee_kernel.body, written)
    return Set(a for a in callee_kernel.sig.args
               if a in written || callee_kernel.sig.kinds[a] in (:array_float, :array_int))
end

# Emits `__inl_<callee>_<n>_<i> = <expr>` for each expression argument and
# returns the argument list with those names in place.
function inl_bind_arg_exprs(callee_name::Symbol, callee_kernel, call_args, site::Int)
    prelude = Any[]
    out = Any[]
    for (i, a) in enumerate(call_args)
        if a isa Symbol
            push!(out, a)
        else
            tmp = Symbol("__inl_", callee_name, "_", site, "_", i)
            push!(prelude, Expr(:(=), tmp, a))
            push!(out, tmp)
        end
    end
    return (out, prelude)
end

function inl_check_call_kinds(caller_name::Symbol, confirmed::Dict{Symbol,Symbol},
                               call_stmt::Expr, callee_kernel)
    call_args = call_stmt.args[2:end]
    must = inl_must_be_symbol(callee_kernel)
    for (i, a) in enumerate(call_args)
        i <= length(callee_kernel.sig.args) || break
        param = callee_kernel.sig.args[i]
        (a isa Symbol || !(param in must)) ||
            error("inl_inline_calls: call to :$(callee_kernel.sig.name) inside :$(caller_name) " *
                  "passes `$(a)` for parameter :$(param), which must be a bare symbol. " *
                  (callee_kernel.sig.kinds[param] in (:array_float, :array_int) ?
                   "It is an array" : "The callee assigns to it, so the call returns through it"))
    end
    length(call_args) == length(callee_kernel.sig.args) ||
        error("inl_inline_calls: call to :$(callee_kernel.sig.name) inside :$(caller_name) passes $(length(call_args)) argument(s), expected $(length(callee_kernel.sig.args))")
    for (i, a) in enumerate(call_args)
        # an expression argument has no entry in the caller's kind map; its
        # kind is settled by shape_ on the temporary bound above
        a isa Symbol || continue
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
# indexing, broadcasting, div-not-÷, and the intrinsic
# whitelist; comment-header and `for`-style are source-text-only.

function parse_kernel(expr::Expr)
    expr.head == :function ||
        error("parse_kernel: expected a `function ... end` definition, got Expr(:$(expr.head), ...)")
    length(expr.args) == 2 ||
        error("parse_kernel: malformed function Expr (expected signature + body)")
    name, args = parse_signature(expr.args[1])
    body = parse_hoist_array_bounds(parse_statements(parse_strip_lines(expr.args[2])), Ref(0))
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

# A `for` bound that reads an array element (`for m = 1:cnt[1]`) is evaluated once, when the loop is entered. The
# reverse sweep rebuilds a bound from scalars it can restore, and cannot rebuild one from an array element that the
# loop body (or a later pass of an enclosing loop) rewrites: it would see the value after the loop. The bound is
# therefore read into a scalar just before the loop, which is exactly what Julia does, and the loop uses the scalar.
function parse_expr_reads_array(e)
    e isa Expr || return false
    e.head == :ref && return true
    return any(parse_expr_reads_array, e.args)
end

function parse_hoist_array_bounds(body::Vector, counter::Base.RefValue{Int})
    out = NamedTuple[]
    for stmt in body
        if stmt.kind == :for
            bounds = Any[stmt.lo, stmt.step, stmt.hi]
            for q in 1:3
                if parse_expr_reads_array(bounds[q])
                    counter[] += 1
                    tmp = Symbol("__bnd_", counter[])
                    push!(out, (kind = :assign, lhs = tmp, rhs = bounds[q]))
                    bounds[q] = tmp
                end
            end
            push!(out, merge(stmt, (lo = bounds[1], step = bounds[2], hi = bounds[3],
                                    body = parse_hoist_array_bounds(stmt.body, counter))))
        elseif stmt.kind == :while
            push!(out, merge(stmt, (body = parse_hoist_array_bounds(stmt.body, counter),)))
        elseif stmt.kind == :if
            push!(out, merge(stmt, (then = parse_hoist_array_bounds(stmt.then, counter),
                                    els = parse_hoist_array_bounds(stmt.els, counter))))
        else
            push!(out, stmt)
        end
    end
    return out
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
        return parse_while(stmt)
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

    body = parse_statements(parse_strip_lines(stmt.args[2]))
    return (kind = :for, var = var, lo = lo, hi = hi, step = step, body = body)
end

# ---- if statement (plain if/else only -- no elseif chains yet) ----

# A `while` carries no trip count. The forward sweep counts its own
# iterations and pushes the total; the backward sweep replays it as a
# plain `for`. So the adjoint of a `while` contains no `:while`, and every
# stage that ingests generated code stays unchanged.
#
# Two refusals are cheap and worth making here, at the only point that
# sees the user's source.
function parse_while(stmt::Expr)
    length(stmt.args) == 2 ||
        error("parse_kernel: unsupported `while` form `$(stmt)`")
    cond = stmt.args[1]
    # before parse_check_expr, which would reject an assignment with its
    # generic message. This one names the fix.
    parse_while_check_pure(cond)
    parse_check_expr(cond, true)

    body_block = stmt.args[2]
    body_block isa Expr && body_block.head == :block ||
        error("parse_kernel: malformed `while` body")
    body = parse_statements(parse_strip_lines(body_block))

    # The condition must read at least one name the body writes. STADE
    # cannot prove termination and does not try. This far weaker property
    # catches the loop that plainly cannot end, at generation time rather
    # than by hanging a job.
    read_vars = Set{Symbol}()
    parse_collect_expr_vars!(cond, read_vars)
    written = Set{Symbol}()
    parse_collect_written!(body, written)
    isempty(intersect(read_vars, written)) &&
        error("parse_kernel: this `while` condition reads `" *
              join(sort(collect(read_vars)), "`, `") *
              "`, and the body writes none of them, so the loop cannot end. " *
              "STADE does not prove termination; it checks only that the " *
              "condition can change")
    return (kind = :while, cond = cond, body = body)
end

# An assignment inside a condition has no place in either sweep: the
# forward sweep evaluates the condition once per iteration, and the
# backward sweep never evaluates it at all, so a side effect there would
# happen a different number of times in each.
function parse_while_check_pure(expr)
    expr isa Expr || return nothing
    expr.head in (:(=), :+=, :-=, :*=, :/=, :^=) &&
        error("parse_kernel: `while` condition `$(expr)` assigns. A condition " *
              "is an expression; move the assignment into the body")
    for a in expr.args
        parse_while_check_pure(a)
    end
    return nothing
end

function parse_collect_expr_vars!(expr, out::Set{Symbol})
    expr isa Symbol && (push!(out, expr); return out)
    if expr isa Expr
        if expr.head == :ref
            expr.args[1] isa Symbol && push!(out, expr.args[1])
            for a in expr.args[2:end]; parse_collect_expr_vars!(a, out); end
        elseif expr.head == :call
            for a in expr.args[2:end]; parse_collect_expr_vars!(a, out); end
        else
            for a in expr.args; parse_collect_expr_vars!(a, out); end
        end
    end
    return out
end

function parse_collect_written!(body::Vector{NamedTuple}, out::Set{Symbol})
    for st in body
        if st.kind == :assign
            st.lhs isa Symbol && push!(out, st.lhs)
            st.lhs isa Expr && st.lhs.head == :ref && push!(out, st.lhs.args[1])
        elseif st.kind == :for
            push!(out, st.var)
            parse_collect_written!(st.body, out)
        elseif st.kind == :while
            parse_collect_written!(st.body, out)
        elseif st.kind == :if
            parse_collect_written!(st.then, out)
            parse_collect_written!(st.els, out)
        end
    end
    return out
end

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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
    elseif expr isa Expr && expr.head == :ref
        # an element of an integer array is an integer: `c = cnt[i]` is an integer scalar even when `c` is only used as a number
        base = expr.args[1]
        return (base isa Symbol && get(is_int, base, false)) ? :int : :unknown
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

# while cond ... end. The condition is emitted verbatim: it is never
# differentiated, because the trip count is recorded by the forward sweep
# and replayed by the backward one.
function emit_while(cond, body_exprs::Vector)
    return Expr(:while, cond, Expr(:block, body_exprs...))
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

# ---- local common-subexpression elimination -------------------------
# A shared post-pass over an already-generated body, run by tgen_/agen_/hvp_ alike. The
# duplication it removes is structural, not accidental: der_adjoint_generic copies the incoming
# seed into every one of a node's contributions, so a node with k active children replicates its
# whole seed k times and each level multiplies against the one above. `advection_b`'s
# `(1.0 / dx) * -(ub[i_x])` appearing three times, with `-(ub[i_x])` a shared factor in a fourth
# place, is one `/` node and one `*` node's seeds.
#
# Scope is one straight-line run of statements inside a single block. A `:for` or `:if` ends the
# run, and its bodies are processed on their own -- nothing is ever hoisted across a loop header
# or into a branch, so a temporary can never outlive the control flow that computed it.

# Candidates are exactly the ops carrying a derivative rule (which is exactly what skill-stade
# lets a kernel call) plus array reads. Everything else -- push!, pop!, sizehint!, Vector{...} --
# falls outside, which is what keeps a stack pop, whose two evaluations return DIFFERENT values,
# from ever being treated as a repeated expression.
emit_cse_candidate(e, ctx, in_index) = e isa Expr &&
    ((e.head == :call && !isempty(e.args) && e.args[1] isa Symbol && e.args[1] in ctx.pure) ||
     (e.head == :ref && !isempty(e.args) && e.args[1] isa Symbol)) &&
    emit_cse_bindable(e, ctx, in_index)

# An integer-valued expression is deliberately never a candidate, even when it repeats. An index
# formula like `i_x - 1` is where cgen_/jgen_ read the thread variable to decide which loops can be
# offloaded and whether a split write needs an atomic; naming it hides the variable from that
# analysis, and doing so pushed four mpnn loops back onto the host for the sake of one integer
# subtraction. Every temporary this pass creates is float-valued, so a `__cse_*` read counts as
# touching a float -- an hvp body goes through the pass twice and the second pass has to recognize
# the first pass's names.
function emit_cse_bindable(e, ctx, in_index)
    emit_cse_touches_float(e, ctx) && return true
    # Inside an array subscript nothing is bindable. A subscript is Int-valued, and its text is what
    # cgen_/jgen_ pattern-match to decide thread mapping and atomics. Rewriting any part of one -- it
    # was `2n_node_feat` inside `msg_input[in_off + 2n_node_feat + k]` -- stops cgen_ offloading
    # mpnn's whole outer edge loop and leaves it launching six inner kernels from the host instead.
    #
    # This is the ONLY restriction integer expressions need. An earlier version also refused any
    # expression mentioning a loop variable, on the theory that naming `i_x - 1` hides the thread
    # variable. That theory was wrong: with subscripts left alone, allowing loop-variable-carrying
    # expressions keeps validate_offload byte-identical to the pre-CSE run, and refusing them cost
    # 4.4 percentage points of the saving.
    in_index && return false
    return true
end

function emit_cse_touches_float(e, ctx)
    for v in emit_cse_reads!(e, Set{Symbol}())
        (v in ctx.floats || emit_cse_is_float_temp(v)) && return true
    end
    return false
end

# Four prefixes, along two axes that both matter downstream.
#
# Float vs integer: a float temporary that is bound before doubling must be given a second-order
# shadow, and an integer one must not -- `__icse_0 = div(n - 1, 1) + 1` has no tangent. Splitting
# them is what lets validate_cse.jl state the shadow requirement as a rule rather than a guess.
#
# Before vs after doubling: an hvp body goes through this pass twice, and the second run must not
# adopt, prune, or renumber the first run's names.
# `__oldb_*` belongs here too: agen_backward_assign's captured incoming adjoint is float-valued, and
# an expression whose only non-constant read is one of them would otherwise be classed as integer,
# take the `__icse_` prefix, and be denied the second-order shadow it needs.
emit_cse_is_float_temp(v) = v isa Symbol &&
    (startswith(string(v), "__cse_") || startswith(string(v), "__hcse_") ||
     startswith(string(v), "__oldb_"))

emit_cse_is_temp(v) = v isa Symbol &&
    (emit_cse_is_float_temp(v) || startswith(string(v), "__icse_") || startswith(string(v), "__ihcse_"))

# Every variable an expression reads: bare symbols, and the NAME of any array it indexes.
function emit_cse_reads!(e, acc)
    e isa Symbol && (push!(acc, e); return acc)
    e isa Expr || return acc
    e.head == :ref && e.args[1] isa Symbol && push!(acc, e.args[1])
    start = (e.head == :call || e.head == :ref) ? 2 : 1
    for a in e.args[start:end]
        emit_cse_reads!(a, acc)
    end
    return acc
end

# An occurrence's identity. Two textually identical occurrences share a key only when every
# variable they read still carries the same generation and no barrier has intervened, so a write
# between them (or a push!/pop! anywhere) forces a fresh key rather than a wrong reuse.
function emit_cse_key(e, gen, barrier)
    vs = sort(collect(emit_cse_reads!(e, Set{Symbol}())); by = string)
    return (e, barrier[], Tuple((v, get(gen, v, 0)) for v in vs))
end

emit_cse_bump!(gen, v::Symbol) = (gen[v] = get(gen, v, 0) + 1; nothing)

# Applies one statement's writes. An assignment invalidates its target; a push!/pop! anywhere in
# the statement invalidates that stack; anything that is not an assignment at all is a full
# barrier, since this pass does not model what it might touch.
function emit_cse_apply_writes!(s, gen, barrier)
    emit_cse_bump_stack_ops!(s, gen)
    if s isa Expr && s.head == :(=)
        lhs = s.args[1]
        lhs isa Symbol && return emit_cse_bump!(gen, lhs)
        lhs isa Expr && lhs.head == :ref && lhs.args[1] isa Symbol && return emit_cse_bump!(gen, lhs.args[1])
    end
    barrier[] += 1
    return nothing
end

function emit_cse_bump_stack_ops!(e, gen)
    if e isa Expr
        e.head == :call && length(e.args) >= 2 && e.args[1] in (:push!, :pop!) &&
            e.args[2] isa Symbol && emit_cse_bump!(gen, e.args[2])
        for a in e.args
            emit_cse_bump_stack_ops!(a, gen)
        end
    end
    return nothing
end

# Pass 1: count every candidate occurrence, keyed as above. Walks rvalue positions only -- an
# assignment's target is a store, though the index expressions inside it are ordinary reads.
function emit_cse_count!(e, gen, barrier, counts, ctx, in_index = false)
    if emit_cse_candidate(e, ctx, in_index)
        key = emit_cse_key(e, gen, barrier)
        counts[key] = get(counts, key, 0) + 1
    end
    if e isa Expr
        start = (e.head == :call || e.head == :ref) ? 2 : 1
        for a in e.args[start:end]
            emit_cse_count!(a, gen, barrier, counts, ctx, in_index || e.head == :ref)
        end
    end
    return nothing
end

function emit_cse_count_stmt!(s, gen, barrier, counts, ctx)
    if s isa Expr && s.head == :(=)
        emit_cse_count!(s.args[2], gen, barrier, counts, ctx)
        lhs = s.args[1]
        if lhs isa Expr && lhs.head == :ref
            for a in lhs.args[2:end]
                emit_cse_count!(a, gen, barrier, counts, ctx, true)
            end
        end
    else
        emit_cse_count!(s, gen, barrier, counts, ctx)
    end
    return nothing
end

# Pass 2: rewrite, replaying pass 1's generation bookkeeping exactly so the keys line up. Children
# are rewritten before their parent, but the parent's KEY is taken from the original expression --
# otherwise a parent whose child got named would key differently in the two passes and never match.
# Naming children first is also what puts `__cse_0 = -(ub[i_x])` ahead of the `__cse_1` that uses it.
function emit_cse_rewrite(e, gen, barrier, counts, names, pending, counter, ctx, in_index = false)
    emit_cse_candidate(e, ctx, in_index) ||
        return emit_cse_rewrite_args(e, gen, barrier, counts, names, pending, counter, ctx, in_index)
    key = emit_cse_key(e, gen, barrier)
    haskey(names, key) && return names[key]
    rebuilt = emit_cse_rewrite_args(e, gen, barrier, counts, names, pending, counter, ctx, in_index)
    get(counts, key, 0) > 1 || return rebuilt
    tmp = Symbol(emit_cse_touches_float(e, ctx) ? ctx.fprefix : ctx.iprefix, "t", counter[])
    counter[] += 1
    push!(pending, Expr(:(=), tmp, rebuilt))
    names[key] = tmp
    return tmp
end

function emit_cse_rewrite_args(e, gen, barrier, counts, names, pending, counter, ctx, in_index = false)
    e isa Expr || return e
    start = (e.head == :call || e.head == :ref) ? 2 : 1
    args = Any[e.args[1:(start - 1)]...]
    for a in e.args[start:end]
        push!(args, emit_cse_rewrite(a, gen, barrier, counts, names, pending, counter, ctx, in_index || e.head == :ref))
    end
    return Expr(e.head, args...)
end

function emit_cse_rewrite_stmt(s, gen, barrier, counts, names, pending, counter, ctx)
    if s isa Expr && s.head == :(=)
        rhs = emit_cse_rewrite(s.args[2], gen, barrier, counts, names, pending, counter, ctx)
        lhs = s.args[1]
        if lhs isa Expr && lhs.head == :ref
            lhs = emit_cse_rewrite_args(lhs, gen, barrier, counts, names, pending, counter, ctx, false)
        end
        return Expr(:(=), lhs, rhs)
    end
    return emit_cse_rewrite(s, gen, barrier, counts, names, pending, counter, ctx)
end

# Counts references to each `__cse_*` name, reading positions only.
function emit_cse_tally!(e, counts)
    if e isa Symbol
        emit_cse_is_temp(e) && (counts[e] = get(counts, e, 0) + 1)
    elseif e isa Expr
        if e.head == :(=)
            emit_cse_tally!(e.args[2], counts)
            e.args[1] isa Expr && for a in e.args[1].args[2:end]; emit_cse_tally!(a, counts); end
        else
            start = (e.head == :call || e.head == :ref) ? 2 : 1
            for a in e.args[start:end]; emit_cse_tally!(a, counts); end
        end
    end
    return nothing
end

# Only this pass's own prefix is pruned. An hvp body goes through the pass a second time after
# doubling, and the `__cse_*` names and their `__cse_*d` shadows from the first pass are ordinary
# variables to that second run -- folding one of them away is value-preserving but dissolves the
# structure validate_cse.jl checks the shadow invariant against.
#
# A temporary loses uses to a LARGER one bound around it: `1.0 / dx` is named because it occurs
# three times, then all three occurrences vanish into the single `(1.0 / dx) * -(ub[i_x])` that
# replaces them. What is left with one use, or none, is folded back into that use and dropped, to a
# fixpoint. Without this the pass would trade an operation for an assignment and emit a chain of
# one-use names -- more lines than it started with, and no faster.
function emit_cse_prune(stmts, prefixes)
    while true
        counts = Dict{Symbol,Int}()
        for s in stmts
            emit_cse_tally!(s, counts)
        end
        out = Any[]
        subst = Dict{Symbol,Any}()
        removed = false
        for s in stmts
            s2 = isempty(subst) ? s : agen_substitute_vars(s, subst)
            if s2 isa Expr && s2.head == :(=) && s2.args[1] isa Symbol &&
               any(startswith(string(s2.args[1]), p) for p in prefixes) && get(counts, s2.args[1], 0) <= 1
                subst[s2.args[1]] = s2.args[2]
                removed = true
                continue
            end
            push!(out, s2)
        end
        stmts = out
        removed || return stmts
    end
end

function emit_cse_run(run, counter, ctx)
    isempty(run) && return Any[]
    provisional = Ref(0)
    counts = Dict{Any,Int}()
    gen = Dict{Symbol,Int}()
    barrier = Ref(0)
    for s in run
        emit_cse_count_stmt!(s, gen, barrier, counts, ctx)
        emit_cse_apply_writes!(s, gen, barrier)
    end
    out = Any[]
    names = Dict{Any,Symbol}()
    gen = Dict{Symbol,Int}()
    barrier = Ref(0)
    for s in run
        pending = Any[]
        rewritten = emit_cse_rewrite_stmt(s, gen, barrier, counts, names, pending, provisional, ctx)
        append!(out, pending)
        push!(out, rewritten)
        emit_cse_apply_writes!(s, gen, barrier)
    end
    return emit_cse_renumber(emit_cse_prune(out, ctx.prefixes), counter, ctx.prefixes)
end

# Pruning leaves holes in the numbering, since a name that was bound and then folded back still
# consumed one. Survivors are renumbered in definition order after the fixpoint, so what reaches the
# file runs 0, 1, 2, ... -- this output is meant to be read.
function emit_cse_renumber(stmts, counter, prefixes)
    subst = Dict{Symbol,Any}()
    for s in stmts
        if s isa Expr && s.head == :(=) && s.args[1] isa Symbol
            for p in prefixes
                startswith(string(s.args[1]), p * "t") || continue
                subst[s.args[1]] = Symbol(p, counter[])
                counter[] += 1
                break
            end
        end
    end
    isempty(subst) && return stmts
    return Any[agen_substitute_vars(s, subst) for s in stmts]
end

# `counter` is threaded so every temporary in one generated function has a distinct name, which is
# what lets a body be run through this pass more than once (hvp_ does, before and after doubling).
function emit_cse_block(stmts, counter, ctx)
    out = Any[]
    run = Any[]
    for s in stmts
        s isa LineNumberNode && continue
        if s isa Expr && s.head == :for
            append!(out, emit_cse_run(run, counter, ctx)); empty!(run)
            push!(out, Expr(:for, s.args[1], Expr(:block, emit_cse_block(s.args[2].args, counter, ctx)...)))
        elseif s isa Expr && s.head == :while
            # Without this clause a whole `while` joins the straight-line run,
            # and CSE hoists subexpressions out of it, past the loop
            # variables of any `for` inside. That produced
            # `__cse_0 = u[i_x]` above the loop that binds i_x.
            append!(out, emit_cse_run(run, counter, ctx)); empty!(run)
            push!(out, Expr(:while, s.args[1], Expr(:block, emit_cse_block(s.args[2].args, counter, ctx)...)))
        elseif s isa Expr && s.head == :if
            append!(out, emit_cse_run(run, counter, ctx)); empty!(run)
            args = Any[s.args[1]]
            for a in s.args[2:end]
                push!(args, a isa Expr && a.head == :block ? Expr(:block, emit_cse_block(a.args, counter, ctx)...) :
                            first(emit_cse_block(Any[a], counter, ctx)))
            end
            push!(out, Expr(:if, args...))
        else
            push!(run, s)
        end
    end
    append!(out, emit_cse_run(run, counter, ctx))
    return out
end

emit_cse_stmts(stmts, counter, floats; prefix = "__cse_", int_prefix = "__icse_") =
    emit_cse_block(stmts, counter, (pure = Set(keys(der_table())), floats = floats,
                                    fprefix = prefix, iprefix = int_prefix, prefixes = (prefix, int_prefix)))

# Rewrites a whole `function f(args...) ... end` Expr in place of its body.


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
        elseif stmt.kind == :while
            # The condition carries no activity: it is never differentiated,
            # and a `while` has no bound variable. Only the body can.
            #
            # Without this clause activity never entered a `while` body at
            # all. Every local the body assigned stayed passive, so tgen_
            # emitted the tangent of its DEFINITION and then dropped it at
            # every USE, which silently zeroed the derivative rather than
            # failing.
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

# ---- integer state the reverse sweep must restore (:ivalue sites) ----
#
# agen_backward_body re-derives an integer local in the reverse sweep by re-running its assignment, which is only
# sound for a variable in no `unsafe` chain (see agen_unsafe_int_vars). An `unsafe` integer is skipped there
# entirely, so a float statement that reads it -- as an array index, say -- would see an undefined variable or
# the value of the LAST forward iteration. These integers get the treatment a float scalar already gets: the old
# value is pushed before each write and popped at the write's own reverse position.
#
# Only an integer READ by an active float statement (rhs or lhs index) needs it. Conditions are replayed from
# the branch and count stacks, and a bound already on a :tripcount stack is restored from there.
# Integer arrays that are read and then written again get the same treatment: see snap_itbr_arrays.

# the reads one statement makes by itself, children excluded
function snap_itbr_stmt_reads!(st, kinds, active_map, reassigned, acc, iarr = Set{Symbol}())
    if st.kind == :assign
        var = st.lhs isa Symbol ? st.lhs : st.lhs.args[1]
        if kinds[var] in (:scalar_float, :array_float) && get(active_map, var, false)
            agen_collect_expr_vars!(st.rhs, acc)
            if st.lhs isa Expr
                for a in st.lhs.args[2:end]
                    agen_collect_expr_vars!(a, acc)
                end
            end
        elseif st.lhs isa Expr && var in iarr
            # the reverse sweep pops the old element back into a[idx], so the index is evaluated there
            for a in st.lhs.args[2:end]
                agen_collect_expr_vars!(a, acc)
            end
        end
    elseif st.kind == :for
        bounds = Set{Symbol}()
        for e in (st.lo, st.hi, st.step)
            agen_collect_expr_vars!(e, bounds)
        end
        union!(acc, setdiff(bounds, reassigned))
    end
    return nothing
end

function snap_itbr_collect_reads!(body, kinds, active_map, reassigned, acc, iarr = Set{Symbol}())
    for st in body
        snap_itbr_stmt_reads!(st, kinds, active_map, reassigned, acc, iarr)
        if st.kind == :for || st.kind == :while
            snap_itbr_collect_reads!(st.body, kinds, active_map, reassigned, acc, iarr)
        elseif st.kind == :if
            snap_itbr_collect_reads!(st.then, kinds, active_map, reassigned, acc, iarr)
            snap_itbr_collect_reads!(st.els, kinds, active_map, reassigned, acc, iarr)
        end
    end
    return nothing
end

function snap_itbr_writes_in(st, v)
    if st.kind == :assign
        return (st.lhs isa Symbol ? st.lhs : st.lhs.args[1]) == v
    elseif st.kind == :for || st.kind == :while
        return any(t -> snap_itbr_writes_in(t, v), st.body)
    elseif st.kind == :if
        return any(t -> snap_itbr_writes_in(t, v), st.then) || any(t -> snap_itbr_writes_in(t, v), st.els)
    end
    return false
end

# Does the :whilecarry restore of this `while` body already hand `v` the right value at every active read inside
# it? The replay starts each iteration from the value `v` held when the iteration began, so that holds exactly
# when `v` is carried, every write to it is a direct statement of the body, and no active read follows the first
# write. A counter bumped last (`k = k + 1`) is the usual case.
function snap_itbr_start_covered(body, kinds, active_map, reassigned, v, iarr = Set{Symbol}())
    v in snap_while_carried_ints(body, kinds) || return false
    first_write = 0
    for (i, st) in enumerate(body)
        if st.kind == :assign && (st.lhs isa Symbol ? st.lhs : st.lhs.args[1]) == v
            first_write == 0 && (first_write = i)
        elseif snap_itbr_writes_in(st, v)
            return false
        end
    end
    first_write == 0 && return false
    for st in body[first_write + 1:end]
        reads = Set{Symbol}()
        snap_itbr_collect_reads!([st], kinds, active_map, reassigned, reads, iarr)
        v in reads && return false
    end
    return true
end

# true when every active read of `v` lies inside a `while` that covers it (`cov` is the enclosing state)
function snap_itbr_reads_covered(body, kinds, active_map, reassigned, v, cov, iarr = Set{Symbol}())
    for st in body
        reads = Set{Symbol}()
        snap_itbr_stmt_reads!(st, kinds, active_map, reassigned, reads, iarr)
        v in reads && !cov && return false
        if st.kind == :while
            inner = cov || snap_itbr_start_covered(st.body, kinds, active_map, reassigned, v, iarr)
            snap_itbr_reads_covered(st.body, kinds, active_map, reassigned, v, inner, iarr) || return false
        elseif st.kind == :for
            snap_itbr_reads_covered(st.body, kinds, active_map, reassigned, v, cov, iarr) || return false
        elseif st.kind == :if
            snap_itbr_reads_covered(st.then, kinds, active_map, reassigned, v, cov, iarr) || return false
            snap_itbr_reads_covered(st.els, kinds, active_map, reassigned, v, cov, iarr) || return false
        end
    end
    return true
end

function snap_itbr_count_reads(body, kinds, active_map, reassigned, v, iarr = Set{Symbol}())
    n = 0
    for st in body
        reads = Set{Symbol}()
        snap_itbr_stmt_reads!(st, kinds, active_map, reassigned, reads, iarr)
        v in reads && (n = n + 1)
        if st.kind == :for || st.kind == :while
            n = n + snap_itbr_count_reads(st.body, kinds, active_map, reassigned, v, iarr)
        elseif st.kind == :if
            n = n + snap_itbr_count_reads(st.then, kinds, active_map, reassigned, v, iarr)
            n = n + snap_itbr_count_reads(st.els, kinds, active_map, reassigned, v, iarr)
        end
    end
    return n
end

# A reassigned loop bound sits on a :tripcount stack and is popped just before the loop is reversed. That
# restores `v` for every read positioned before the loop in the same body -- and inside it -- provided nothing
# writes `v` first. richardson_substep reads `nsub` in `h = dt / nsub` and then loops to `nsub`: covered. `total`
# is the number of statements anywhere in the kernel that read `v`; the loop must account for all of them.
function snap_itbr_tripcount_covered(body, kinds, active_map, reassigned, v, total, iarr = Set{Symbol}())
    for (i, st) in enumerate(body)
        if st.kind == :for
            bounds = Set{Symbol}()
            for e in (st.lo, st.hi, st.step)
                agen_collect_expr_vars!(e, bounds)
            end
            if v in bounds && v in reassigned &&
               !any(t -> snap_itbr_writes_in(t, v), body[1:i - 1]) && !any(t -> snap_itbr_writes_in(t, v), st.body) &&
               snap_itbr_count_reads(body[1:i], kinds, active_map, reassigned, v, iarr) == total
                return true
            end
            snap_itbr_tripcount_covered(st.body, kinds, active_map, reassigned, v, total, iarr) && return true
        elseif st.kind == :while
            snap_itbr_tripcount_covered(st.body, kinds, active_map, reassigned, v, total, iarr) && return true
        elseif st.kind == :if
            snap_itbr_tripcount_covered(st.then, kinds, active_map, reassigned, v, total, iarr) && return true
            snap_itbr_tripcount_covered(st.els, kinds, active_map, reassigned, v, total, iarr) && return true
        end
    end
    return false
end

# The integer scalars to restore. A scalar needs the restore when it is `unsafe`, read by an active float
# statement, and written either inside a loop or at more than one site -- anything else holds one value for the
# whole run, which the last-value read already sees. Two cases keep their existing mechanism: a loop bound on a
# :tripcount stack (snap_itbr_tripcount_covered), and a counter a `while` already restores per iteration (snap_itbr_start_covered).
function snap_itbr_scalars(kernel, active_map, unsafe, iarr = Set{Symbol}())
    kinds = kernel.sig.kinds
    isempty(unsafe) && return Set{Symbol}()
    reassigned = agen_collect_reassigned_all(kernel.body)
    counts = agen_count_assign_sites(kernel.body)
    reads = Set{Symbol}()
    snap_itbr_collect_reads!(kernel.body, kinds, active_map, reassigned, reads, iarr)
    return Set{Symbol}(v for v in unsafe
                       if kinds[v] == :scalar_int && v in reads &&
                          (v in reassigned || get(counts, v, 0) >= 2) &&
                          !snap_itbr_reads_covered(kernel.body, kinds, active_map, reassigned, v, false, iarr) &&
                          !snap_itbr_tripcount_covered(kernel.body, kinds, active_map, reassigned, v,
                                                       snap_itbr_count_reads(kernel.body, kinds, active_map, reassigned, v, iarr), iarr))
end

# ---- integer arrays ----
# An integer array that is read and then written again holds, at the reverse sweep, the value of its LAST write.
# Three readers see the stale value: an integer scalar recomputed from it (`j = idx[i]`), a loop bound that names it
# (`for m = 1:cnt[1]`), and an active float statement that uses it as a number. The old value of every write to such
# an array is pushed before the write and popped at the write's reverse position, as for a restored scalar. The parser
# already refuses `x[idx[k]]`, so these are the only readers.
#
# Events are numbered in program order. A write W may follow a read R when W is numbered at or after R, or when the two
# share a loop (the next pass writes what this pass read). Reads inside a write to the same array are not readers: the
# statement is never replayed. Reads already covered by a restored scalar copy (`covered`) are left to that scalar.
function snap_itbr_array_events!(body, kinds, active_map, covered, loops, pos, reads, writes)
    for st in body
        pos[] += 1
        if st.kind == :assign
            lv = st.lhs isa Symbol ? st.lhs : st.lhs.args[1]
            lk = get(kinds, lv, :none)
            replayed = lk in (:scalar_int, :array_int) ||
                       (lk in (:scalar_float, :array_float) && get(active_map, lv, false))
            if replayed && !(st.lhs isa Symbol && lv in covered)
                vs = Set{Symbol}()
                agen_collect_expr_vars!(st.rhs, vs)
                if st.lhs isa Expr
                    for a in st.lhs.args[2:end]
                        agen_collect_expr_vars!(a, vs)
                    end
                end
                for v in vs
                    get(kinds, v, :none) == :array_int && v != lv &&
                        push!(get!(reads, v, Any[]), (pos[], copy(loops)))
                end
            end
            st.lhs isa Expr && lk == :array_int && push!(get!(writes, lv, Any[]), (pos[], copy(loops)))
        elseif st.kind == :for
            bounds = Set{Symbol}()
            for e in (st.lo, st.hi, st.step)
                agen_collect_expr_vars!(e, bounds)
            end
            for v in bounds
                get(kinds, v, :none) == :array_int && push!(get!(reads, v, Any[]), (pos[], copy(loops)))
            end
            push!(loops, pos[])
            snap_itbr_array_events!(st.body, kinds, active_map, covered, loops, pos, reads, writes)
            pop!(loops)
        elseif st.kind == :while
            push!(loops, pos[])
            snap_itbr_array_events!(st.body, kinds, active_map, covered, loops, pos, reads, writes)
            pop!(loops)
        elseif st.kind == :if
            snap_itbr_array_events!(st.then, kinds, active_map, covered, loops, pos, reads, writes)
            snap_itbr_array_events!(st.els, kinds, active_map, covered, loops, pos, reads, writes)
        end
    end
    return nothing
end

function snap_itbr_arrays(kernel, active_map, covered)
    reads = Dict{Symbol,Any}()
    writes = Dict{Symbol,Any}()
    snap_itbr_array_events!(kernel.body, kernel.sig.kinds, active_map, covered, Int[], Ref(0), reads, writes)
    out = Set{Symbol}()
    for (a, ws) in writes
        for (rp, rl) in get(reads, a, Any[]), (wp, wl) in ws
            if wp >= rp || !isempty(intersect(rl, wl))
                push!(out, a)
            end
        end
    end
    return out
end

# The integers to restore: the scalars (snap_itbr_scalars), plus the arrays whose later writes would be seen by an
# earlier read. The arrays seed `unsafe` again, since a scalar copy of one can no longer be recomputed at the head
# of the reverse body, and that can add scalars.
function snap_itbr_vars(kernel, active_map)
    base = snap_itbr_scalars(kernel, active_map, agen_unsafe_int_vars(kernel))
    arrays = snap_itbr_arrays(kernel, active_map, base)
    isempty(arrays) && return base
    return union(arrays, snap_itbr_scalars(kernel, active_map, agen_unsafe_int_vars(kernel, arrays), arrays))
end

function snap_plan(kernel, active_map; site_needed = nothing)
    reassigned = snap_collect_reassigned_all(kernel.body)
    value_needed = snap_value_needed_vars(kernel)
    assign_counts = snap_count_assign_sites(kernel.body)
    sites = NamedTuple[]
    counter = Ref(0)
    itbr = snap_itbr_vars(kernel, active_map)
    snap_walk!(kernel.body, active_map, kernel.sig.kinds, reassigned, value_needed, sites, counter,
               false, assign_counts, kernel.body, site_needed, itbr)
# A block-boundary push owns its stack outright instead of borrowing one an :assign site happened to
# create. The boundary kill in snap_fwd_walk! retires exactly those sites, and
# agen_block_boundary_vars skips a var with no (:value, var) stack -- so without this the push and its
# matching pop would vanish silently rather than raise. A stack with no remaining reference is pruned
# later by agen_finalize_stacks.
    for var in sort(collect(snap_boundary_snapshot_vars(kernel)); by = string)
        any(s -> s.kind == :value && s.array == var, sites) && continue
        counter[] = counter[] + 1
        push!(sites, (kind = :value, array = var, at = counter[]))
    end
    # one Int64 stack per restored integer, after every other site: see snap_itbr_vars
    for var in sort(collect(itbr); by = string)
        counter[] = counter[] + 1
        push!(sites, (kind = :ivalue, array = var, at = counter[]))
    end
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

# Integer scalars a while body assigns directly. Nested `for` bodies are
# skipped: their own index is regenerated by the reversed header, and a
# scalar they assign belongs to that loop rather than to this one. A nested
# `while` gets its own sites when snap_walk! reaches it.
# ---- carried across iterations ------------------------------------
#
# A value is CARRIED by a loop body when the body reads it BEFORE writing
# it, and also writes it somewhere. Both halves matter and both have been
# got wrong more than once.
#
# Without read-before-write, a temporary written and then read inside ONE
# iteration counts as carried. Example: `un = ...; u[i] = un` makes `un` look
# carried.
#
# Without the intersection with what the body writes, a read-only parameter
# counts as carried however early it is read. This is the opposite mistake.
#
# This helper is shared. snap_while_carried_ints calls it. The caller filters
# the result by kind.
function carry_carried_vars(body::Vector{NamedTuple})
    out = Set{Symbol}()
    carry_walk!(body, out, Set{Symbol}())
    written = Set{Symbol}()
    parse_collect_written!(body, written)
    return intersect(out, written)
end

function carry_walk!(body::Vector{NamedTuple}, out::Set{Symbol}, defined::Set{Symbol})
    for stmt in body
        if stmt.kind == :assign
            rd = Set{Symbol}()
            parse_collect_expr_vars!(stmt.rhs, rd)
            stmt.lhs isa Expr && for a in stmt.lhs.args[2:end]
                parse_collect_expr_vars!(a, rd)
            end
            for v in rd; v in defined || push!(out, v); end
            stmt.lhs isa Symbol && push!(defined, stmt.lhs)
        elseif stmt.kind == :for
            carry_walk!(stmt.body, out, union(defined, Set([stmt.var])))
        elseif stmt.kind == :while
            rd = Set{Symbol}(); parse_collect_expr_vars!(stmt.cond, rd)
            for v in rd; v in defined || push!(out, v); end
            carry_walk!(stmt.body, out, copy(defined))
        elseif stmt.kind == :if
            rd = Set{Symbol}(); parse_collect_expr_vars!(stmt.cond, rd)
            for v in rd; v in defined || push!(out, v); end
            carry_walk!(stmt.then, out, copy(defined))
            carry_walk!(stmt.els, out, copy(defined))
        end
    end
    return nothing
end

function snap_while_carried_ints(body, kinds, itbr = Set{Symbol}())
    return Set(v for v in carry_carried_vars(body)
               if get(kinds, v, :none) == :scalar_int && !(v in itbr))
end

# the integers that own an :ivalue stack, read back from the stack map so that agen_ never recomputes the decision
agen_itbr_vars(stacks) = Set{Symbol}(v for (k, v) in keys(stacks) if k == :ivalue)

# A scalar is CARRIED only if the body reads it before assigning it. One the
# body defines first is fresh each iteration, so its earlier value is not
# needed and pushing it reads an undefined variable: in a nested pair, the
# outer body's `i_k = 1` made the outer loop push `i_k` before anything had
# set it.
#
# The read-before-write test also keeps the two loops of a nested pair off
# each other's stack. The inner `i_k` is carried and the outer one is not,
# so only one loop pushes it.

function snap_collect_value_needed!(body, acc)
    for stmt in body
        if stmt.kind == :assign
            snap_var_value_needed!(stmt.rhs, acc, false)
        elseif stmt.kind == :for
            snap_collect_value_needed!(stmt.body, acc)
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
function snap_walk!(body, active_map, kinds, reassigned, value_needed, sites, counter, in_loop, assign_counts, full_body, site_needed = nothing, itbr = Set{Symbol}())
    for idx in eachindex(body)
        stmt = body[idx]
        if stmt.kind == :assign
            snap_check_assign!(stmt, active_map, kinds, value_needed, sites, counter, in_loop, assign_counts, full_body, body, idx, site_needed)
        elseif stmt.kind == :for
            snap_check_tripcount!(stmt, reassigned, sites, counter)
            snap_walk!(stmt.body, active_map, kinds, reassigned, value_needed, sites, counter,
                       true, assign_counts, full_body, site_needed, itbr)
        elseif stmt.kind == :while
            counter[] = counter[] + 1
            push!(sites, (kind = :whilecount, array = :count, at = counter[]))
            # A `while` has no loop variable, so the backward sweep cannot
            # rebuild a carried scalar at iteration k the way it rebuilds a
            # `for` index. Active floats already get a :value site through
            # the ordinary gate. The INTEGERS do not: snap_check_assign!
            # gates on activity, and a loop counter is inactive. They get
            # their own site here, on their own Int64 stack, pushed inside
            # the body and popped inside the replay. Without this the replay
            # reads the value the counter held when the loop ENDED,
            # identically on every iteration.
            for cv in sort(collect(snap_while_carried_ints(stmt.body, kinds, itbr)))
                counter[] = counter[] + 1
                push!(sites, (kind = :whilecarry, array = cv, at = counter[]))
            end
            # in_loop = true: a while body re-executes, so a write inside it
            # needs its site exactly as one inside a `for` does.
            snap_walk!(stmt.body, active_map, kinds, reassigned, value_needed, sites, counter,
                       true, assign_counts, full_body, site_needed, itbr)
        elseif stmt.kind == :if
            counter[] = counter[] + 1
            push!(sites, (kind = :branch, array = :cond, at = counter[]))
            snap_walk!(stmt.then, active_map, kinds, reassigned, value_needed, sites, counter, in_loop, assign_counts, full_body, site_needed, itbr)
            snap_walk!(stmt.els, active_map, kinds, reassigned, value_needed, sites, counter, in_loop, assign_counts, full_body, site_needed, itbr)
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

# ---- block-boundary snapshot bookkeeping ----
# Everything below describes the extra end-of-body push agen_forward_body emits (see
# agen_block_boundary_vars), duplicated on the snap_ side for the same purity reason as every
# other snap_/agen_ pair here. Two consumers: snap_plan gives each such var a :value site of
# its own, and snap_fwd_walk! kills it from `seen` at the end of the body it covers.

# agen_exempt_vars' twin: vars whose sole kernel-wide write qualifies for the iteration-
# independent elision, so no site -- and hence no boundary push -- is emitted for them.
function snap_exempt_vars(kernel, value_needed)
    assign_counts = snap_count_assign_sites(kernel.body)
    exempt = Set{Symbol}()
    snap_collect_exempt_vars!(kernel.body, value_needed, assign_counts, kernel.body, false, exempt)
    return exempt
end

function snap_collect_exempt_vars!(body, value_needed, assign_counts, full_body, in_loop, exempt)
    for stmt in body
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            self_ref = snap_count_var_refs(stmt.rhs, var) > 0
            if !self_ref && var in value_needed && !in_loop && get(assign_counts, var, 0) == 1 &&
               !snap_read_before(full_body, stmt, var)
                push!(exempt, var)
            end
        elseif stmt.kind == :for
            snap_collect_exempt_vars!(stmt.body, value_needed, assign_counts, full_body, true, exempt)
        elseif stmt.kind == :while
            snap_collect_exempt_vars!(stmt.body, value_needed, assign_counts, full_body, true, exempt)
        elseif stmt.kind == :if
            snap_collect_exempt_vars!(stmt.then, value_needed, assign_counts, full_body, in_loop, exempt)
            snap_collect_exempt_vars!(stmt.els, value_needed, assign_counts, full_body, in_loop, exempt)
        end
    end
    return nothing
end

# The float scalars a boundary push can cover at all: agen_block_boundary_vars' own filters
# minus its per-body structural test. Computed once per kernel and threaded through the walk
# so the per-body test stays a cheap structural one.
function snap_boundary_stack_vars(kernel)
    value_needed = snap_value_needed_vars(kernel)
    exempt = snap_exempt_vars(kernel, value_needed)
    return Set(v for v in value_needed if get(kernel.sig.kinds, v, :none) == :scalar_float && !(v in exempt))
end

# Vars this body's own boundary push is GUARANTEED to store: written inside a nested :for/:if,
# which is what puts it on agen_block_boundary_vars' candidate list, AND written by a top-level
# statement of this same body. The second condition is what makes the guarantee hold under
# fuse_ii_loops: agen_ii_covered_write_check can never call a top-level write covered, so no
# ii_plan retires this push. Without it the kill below could rest on a push already fused away.
function snap_boundary_kill_vars(body, bvars)
    isempty(bvars) && return Set{Symbol}()
    nested = Set{Symbol}()
    top = Set{Symbol}()
    for stmt in body
        if stmt.kind == :assign
            stmt.lhs isa Symbol && push!(top, stmt.lhs)
        elseif stmt.kind == :for
            union!(nested, snap_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :while
            union!(nested, snap_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :if
            union!(nested, snap_collect_reassigned(stmt.then, true))
            union!(nested, snap_collect_reassigned(stmt.els, true))
        end
    end
    return intersect(nested, top, bvars)
end

# Every var some body's boundary push stores, anywhere in the kernel. Deliberately NOT
# restricted to snap_boundary_kill_vars' guaranteed subset: a stack is needed whenever the
# push CAN be emitted, while the kill needs it to be certain.
function snap_boundary_snapshot_vars(kernel)
    out = Set{Symbol}()
    snap_boundary_walk!(kernel.body, snap_boundary_stack_vars(kernel), out)
    return out
end

function snap_boundary_walk!(body, bvars, out)
    for stmt in body
        if stmt.kind == :for
            union!(out, intersect(snap_collect_reassigned(stmt.body, true), bvars))
            snap_boundary_walk!(stmt.body, bvars, out)
        elseif stmt.kind == :while
            union!(out, intersect(snap_collect_reassigned(stmt.body, true), bvars))
            snap_boundary_walk!(stmt.body, bvars, out)
        elseif stmt.kind == :if
            union!(out, intersect(snap_collect_reassigned(stmt.then, true), bvars))
            union!(out, intersect(snap_collect_reassigned(stmt.els, true), bvars))
            snap_boundary_walk!(stmt.then, bvars, out)
            snap_boundary_walk!(stmt.els, bvars, out)
        end
    end
    return nothing
end

# ---- site-level (forward-seen) TBR ----
# `snap_value_needed_vars` decides whole-variable necessity; `snap_fwd_walk!` refines to a per-write
# decision: w needs its pre-write value iff (a) w self-references, or (b) a nonlinear read of var
# occurred strictly before w in forward order. This is the production path, reached via snap_plan's
# site_needed kwarg; stade_site_level_tbr_check asserts it stays a subset of the whole-variable set.
function snap_fwd_walk!(body, seen, active_map, decisions, bvars = Set{Symbol}())
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
            seen_then = snap_fwd_walk!(stmt.then, copy(seen), active_map, decisions, bvars)
            seen_els  = snap_fwd_walk!(stmt.els,  copy(seen), active_map, decisions, bvars)
            seen = union(seen_then, seen_els)
        elseif stmt.kind == :for
            seen = snap_fwd_walk_loop!(stmt.body, seen, active_map, decisions, bvars)
        elseif stmt.kind == :while
            seen = snap_fwd_walk_loop!(stmt.body, seen, active_map, decisions, bvars)
        end
    end
    # This body's block-boundary push re-establishes each of these vars at the head of its own reverse body, so every
    # read inside `body` is served by that pop, never a site outside it. Dropping them here stops the loop back edge
    # (snap_fwd_walk_loop!) demanding a redundant push at the first write of a body that already has a boundary snapshot
    # -- the cell-loop `aux = 0.0` shape, whose push makes the loop look loop-carried to cgen_. A read before that write,
    # in the same body, re-adds the var; so does a read in an enclosing body, restored by this body's own first site.
    kill = snap_boundary_kill_vars(body, bvars)
    isempty(kill) && return seen
    return setdiff(seen, kill)
end

# Loop forward fixed point: a read near the top of a loop body is, for every iteration but
# the first, preceded by the previous iteration's later reads -- so a write near the top can
# need a snapshot due to a read near the bottom, one iteration back. `seen` only grows
# across passes, so this terminates; a final pass re-walks with the converged set so
# `decisions` reflects the fixed point.
function snap_fwd_walk_loop!(body, in_seen, active_map, decisions, bvars = Set{Symbol}())
    seen = copy(in_seen)
    while true
        scratch = Dict{Any,Bool}()
        out_seen = snap_fwd_walk!(body, union(seen, in_seen), active_map, scratch, bvars)
        out_seen == seen && break
        seen = out_seen
    end
    final_in = union(seen, in_seen)
    return snap_fwd_walk!(body, final_in, active_map, decisions, bvars)
end

# public entry point: per-site (not per-variable) TBR decisions for every :assign in
# `kernel`, keyed by agen_site_key(body, idx). Consumed by snap_plan's site_needed and,
# via stade_site_level_tbr_check, by agen_needs_snapshot.
function snap_value_needed_sites(kernel)
    decisions = Dict{Any,Bool}()
    snap_fwd_walk!(kernel.body, Set{Symbol}(), act_analyze(kernel), decisions, snap_boundary_stack_vars(kernel))
    return decisions
end

# key -> var for every :assign site, keyed identically to snap_value_needed_sites (same
# agen_site_key over the same body objects). Lets a per-site decision be traced back to
# the variable it concerns, e.g. for the subset check in stade_site_level_tbr_check.
function snap_site_vars(kernel)
    vars = Dict{Any,Symbol}()
    snap_site_vars_walk!(kernel.body, vars)
    return vars
end

function snap_site_vars_walk!(body, vars)
    for idx in eachindex(body)
        stmt = body[idx]
        if stmt.kind == :assign
            vars[agen_site_key(body, idx)] = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
        elseif stmt.kind == :for
            snap_site_vars_walk!(stmt.body, vars)
        elseif stmt.kind == :while
            snap_site_vars_walk!(stmt.body, vars)
        elseif stmt.kind == :if
            snap_site_vars_walk!(stmt.then, vars)
            snap_site_vars_walk!(stmt.els, vars)
        end
    end
    return nothing
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
                body = lin_build_body(stmt.body, active_map))
    elseif stmt.kind == :while
        # The condition is not linearized. It selects how many times the
        # body runs, and the trip count is replayed from a recorded value
        # rather than differentiated.
        return (kind = :while, cond = stmt.cond,
                body = lin_build_body(stmt.body, active_map))
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
    body_exprs = tgen_body(lin_plan, kernel.sig.kinds)
    reassigned = tgen_reassigned_scalar_args(kernel)
    push!(body_exprs, emit_return_scalars([tgen_shadow(v) for v in reassigned]))
    return Expr(:function, Expr(:call, fname, fargs...), Expr(:block, emit_cse_stmts(body_exprs, Ref(0), tgen_cse_float_names(kernel.sig))...))
end

# Every float-valued name a tangent body can mention: each float variable and its shadow. Anything
# absent is Int64, which emit_cse_stmts refuses to name.
function tgen_cse_float_names(sig)
    names = Set{Symbol}()
    for (v, k) in sig.kinds
        k in (:scalar_float, :array_float) || continue
        push!(names, v)
        push!(names, tgen_shadow(v))
    end
    return names
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

function tgen_body(plan, kinds)
    exprs = Any[]
    for stmt in plan
        if stmt.kind == :assign
            # an integer array has no shadow to reset: its name is not even a parameter of the tangent function
            int_array = stmt.lhs isa Expr && get(kinds, stmt.lhs.args[1], :none) == :array_int
            int_array || push!(exprs, Expr(:(=), tgen_shadow(stmt.lhs), tgen_tangent_expr(stmt.tree)))
            push!(exprs, Expr(:(=), stmt.lhs, stmt.tree.expr))
        elseif stmt.kind == :for
            inner = tgen_body(stmt.body, kinds)
            push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
        elseif stmt.kind == :while
            # Forward mode runs alongside the primal, so the tangent needs no
            # trip count: the same condition over the same values selects the
            # same iterations. Only the reverse mode has to record anything.
            push!(exprs, emit_while(stmt.cond, tgen_body(stmt.body, kinds)))
        elseif stmt.kind == :if
            then_exprs = tgen_body(stmt.then, kinds)
            els_exprs = tgen_body(stmt.els, kinds)
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
        elseif stmt.kind == :while
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

function agen_emit(kernel, lin_plan, snapshot_plan; keep_push_pop::Bool = true, push_pop = nothing, ii_plan = nothing, remat = nothing)
    active_map = act_analyze(kernel)
    layout = nothing
    value_needed = exempt = stacks = nothing
    if !keep_push_pop
        agen_refuse_while_sized(kernel.body)
        agen_refuse_itbr_sized(snapshot_plan)
        # Tier B kernels (a ragged/data-dependent loop bound) are no longer refused: agen_layout resolves as
        # much as possible into closed-form ragged-block tables, using plain :stack semantics only for what a
        # block can't resolve. ii_plan/fuse_ii_loops is exercised with keep_push_pop=false against the full
        # corpus; the Phase 3 cleanup below prunes a fused var's now-unused stack in this mode too.
        value_needed = agen_value_needed_vars(kernel)
        reassigned = agen_collect_reassigned_all(kernel.body)
        exempt = agen_exempt_vars(kernel, value_needed)
        stacks = agen_stack_map(snapshot_plan)
        layout = agen_layout(kernel, kernel.sig.kinds, active_map, value_needed, reassigned, exempt, stacks; push_pop = push_pop)
    end
    (initstacks_expr, table_names, tot_names, val_names) = agen_init_emit(kernel, snapshot_plan; keep_push_pop = keep_push_pop, layout = layout,
                                      active_map = active_map, value_needed = value_needed, exempt = exempt, stacks = stacks, push_pop = push_pop)
    tier_b_extra_args = vcat(table_names, tot_names, val_names)
    adjoint_expr = agen_adjoint_emit(kernel, active_map, lin_plan, snapshot_plan; keep_push_pop = keep_push_pop, layout = layout, push_pop = push_pop,
                                      tier_b_extra_args = tier_b_extra_args, ii_plan = ii_plan, remat = remat)
    # fuse_ii_loops can leave a stack with zero remaining push!/pop!/indexed-ref references
    # anywhere in the generated body (every site that would have used it got fused away) --
    # checked from the actual generated code, never predicted ahead of time, since a var's
    # ii_plan coverage does NOT by itself guarantee its stack is fully unused. Runs under both
    # keep_push_pop values; agen_finalize_stacks detects use under either storage mode.
    (adjoint_expr, initstacks_expr) = agen_finalize_stacks(adjoint_expr, initstacks_expr, snapshot_plan, ii_plan; force = !keep_push_pop && (any(s -> agen_asnap_site_array(s, kernel.sig.kinds) !== nothing, snapshot_plan) || !isempty(agen_remat_cross_vars(remat))))
    return (adjoint = adjoint_expr, initstacks = initstacks_expr)
end

# Every float-valued name an adjoint or hvp body can mention: each float variable, its adjoint, both
# of their second-order shadows, and the stacks that hold floats. A :branch or :tripcount stack holds
# Int64 and is deliberately left out, so `n = tripcount_stack[i]` is never named.
function agen_cse_float_names(sig, sites)
    names = Set{Symbol}()
    for (v, k) in sig.kinds
        k in (:scalar_float, :array_float) || continue
        for nm in (v, agen_shadow(v))
            push!(names, nm)
            push!(names, tgen_shadow(nm))
        end
    end
    for s in sites
        s.kind in (:array, :value) || continue
        nm = agen_site_stack_name(s)
        push!(names, nm)
        push!(names, hvp_stack_shadow(nm))
    end
    return names
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
                            tier_b_extra_args::Vector{Symbol} = Symbol[], ii_plan = nothing, remat = nothing)
    fname = agen_fname(kernel.sig.name)
    stacks = agen_stack_map(sites)
    # `tier_b_extra_args` (Phase D) is the same table/total/value-table name list agen_init_emit
    # returned, appended in the same order so it lines up with initstacks_*'s return tuple,
    # keeping val_init_stacks' splat-the-whole-tuple convention working. Empty under
    # keep_push_pop=true or a kernel with no ragged block.
    fargs = vcat(agen_signature_args(kernel.sig), agen_stack_names(sites), tier_b_extra_args)

    value_needed = agen_value_needed_vars(kernel)
    reassigned = agen_collect_reassigned_all(kernel.body)
    unsafe = agen_unsafe_int_vars(kernel, agen_itbr_vars(stacks))
    exempt = agen_exempt_vars(kernel, value_needed)

    ectx = (keep_push_pop = keep_push_pop, loop_ctx = Any[], layout = layout, push_pop = push_pop, ii_plan = ii_plan, remat = remat)

    body = Any[]
    append!(body, agen_local_primal_inits(kernel, active_map, agen_itbr_vars(stacks)))
    append!(body, agen_local_shadow_inits(kernel, active_map))
    append!(body, agen_forward_body(kernel.body, kernel.sig.kinds, active_map, value_needed, reassigned, stacks, exempt; ectx = ectx, lin_body = lin_plan, unsafe = unsafe))
    append!(body, agen_backward_body(lin_plan, kernel.body, kernel.sig.kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = ectx))

    scalar_args = [a for a in kernel.sig.args if kernel.sig.kinds[a] == :scalar_float]
    push!(body, emit_return_scalars([agen_shadow(a) for a in scalar_args]))

    return Expr(:function, Expr(:call, fname, fargs...), Expr(:block, emit_cse_stmts(body, Ref(0), agen_cse_float_names(kernel.sig, sites))...))
end

# Int64 variables reassigned at more than one :assign site are evolving state depending on the full
# control-flow history -- a Julia `for` loop assigning an outer variable leaks past it, so recomputing
# from a fixed start is wrong. Anything depending on such a variable is excluded too. The fix: never
# hoist/recompute this set; loop bounds are already restored via :tripcount. The set is seeded by
# self-reference, not mere multiplicity, so a fresh per-loop variable stays hoistable.
function agen_unsafe_int_vars(kernel, itbr_arrays = Set{Symbol}())
    kinds = kernel.sig.kinds
    unsafe = Set{Symbol}(a for a in itbr_arrays if get(kinds, a, :none) == :array_int)
    agen_seed_unsafe_self_ref!(kernel.body, kinds, unsafe)
    agen_seed_unsafe_multi_site!(kernel.body, kinds, unsafe)
    changed = true
    while changed
        changed = agen_propagate_unsafe!(kernel.body, kinds, unsafe)
    end
    return unsafe
end

# The reverse body re-runs a loop body's integer assignments up front, in forward order, so a scalar that the body
# assigns at two or more sites ends the re-run with its last value. A read that sits before the last assignment, in
# program order, would see the wrong one. Such a scalar is unsafe, and the restore analysis decides what it needs. A scalar
# assigned once per body, or at several sites that nothing reads in between, stays safe, and so does the same name used in
# two different loops (each loop body is its own scope). Positions are the pre-order index of the statement; a nested
# loop's reads count in the scope that encloses it, its writes belong to the nested scope.
function agen_multi_site_walk!(body, kinds, sites, reads, pos, nested)
    for st in body
        pos[] += 1
        if st.kind == :assign
            rd = Set{Symbol}()
            agen_collect_expr_vars!(st.rhs, rd)
            st.lhs isa Expr && for ix in st.lhs.args[2:end]
                agen_collect_expr_vars!(ix, rd)
            end
            for r in rd
                push!(get!(reads, r, Int[]), pos[])
            end
            if !nested && st.lhs isa Symbol && get(kinds, st.lhs, :none) == :scalar_int
                push!(get!(sites, st.lhs, Int[]), pos[])
            end
        elseif st.kind == :for
            rd = Set{Symbol}()
            for e in (st.lo, st.hi, st.step)
                agen_collect_expr_vars!(e, rd)
            end
            for r in rd
                push!(get!(reads, r, Int[]), pos[])
            end
            agen_multi_site_walk!(st.body, kinds, sites, reads, pos, true)
        elseif st.kind == :while
            rd = Set{Symbol}()
            agen_collect_expr_vars!(st.cond, rd)
            for r in rd
                push!(get!(reads, r, Int[]), pos[])
            end
            agen_multi_site_walk!(st.body, kinds, sites, reads, pos, true)
        elseif st.kind == :if
            rd = Set{Symbol}()
            agen_collect_expr_vars!(st.cond, rd)
            for r in rd
                push!(get!(reads, r, Int[]), pos[])
            end
            agen_multi_site_walk!(st.then, kinds, sites, reads, pos, nested)
            agen_multi_site_walk!(st.els, kinds, sites, reads, pos, nested)
        end
    end
    return nothing
end

function agen_seed_unsafe_multi_site!(body, kinds, unsafe)
    sites = Dict{Symbol,Vector{Int}}()
    reads = Dict{Symbol,Vector{Int}}()
    agen_multi_site_walk!(body, kinds, sites, reads, Ref(0), false)
    for (v, ws) in sites
        length(ws) >= 2 || continue
        last_w = maximum(ws)
        any(p -> p < last_w, get(reads, v, Int[])) && push!(unsafe, v)
    end
    for st in body
        if st.kind == :for || st.kind == :while
            agen_seed_unsafe_multi_site!(st.body, kinds, unsafe)
        elseif st.kind == :if
            agen_seed_unsafe_multi_site!(st.then, kinds, unsafe)
            agen_seed_unsafe_multi_site!(st.els, kinds, unsafe)
        end
    end
    return nothing
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
        elseif stmt.kind == :while
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
                # the index of an array write is re-run with it, so an unsafe integer in the index taints the array too
                stmt.lhs isa Expr && for ix in stmt.lhs.args[2:end]
                    agen_collect_expr_vars!(ix, refs)
                end
                if any(r in unsafe for r in refs)
                    push!(unsafe, var)
                    changed = true
                end
            end
        elseif stmt.kind == :for
            changed = agen_propagate_unsafe!(stmt.body, kinds, unsafe) || changed
        elseif stmt.kind == :while
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
# Every :whilecount site shares one stack, as :branch and :tripcount do. It
# is kept separate from :tripcount_stack on purpose: the two are pushed at
# different moments, a trip count before its loop and a while count after
# it, so one shared stack would interleave them and a wrong count would be
# hard to attribute. See section 8.1 of the plan.
# The counter a `while` forward sweep increments. Named from the site key so
# the backward sweep can never pick a different one, and so two `while`
# statements in one body cannot collide.
agen_whilecount_var(body, idx) =
    Symbol("__wcount_", string(hash(agen_site_key(body, idx)), base = 16)[1:6])

# The replay loop's own index. It indexes nothing and exists only to run the
# body the recorded number of times, but it still needs a name no primal
# statement can shadow.
agen_whilecount_iter_var(body, idx) =
    Symbol("__witer_", string(hash(agen_site_key(body, idx)), base = 16)[1:6])

function agen_site_stack_name(site)
    site.kind in (:array, :value) && return Symbol(string(site.array) * "_stack")
    site.kind == :branch && return :branch_stack
    site.kind == :whilecount && return :whilecount_stack
    # one stack per carried variable, not one shared: the first version of
    # this will get a value wrong, and a shared stack makes that unattributable
    site.kind == :whilecarry && return Symbol("__wcarry_", site.array, "_stack")
    site.kind == :ivalue && return Symbol("__ival_", site.array, "_stack")
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
# Section 3 of the while plan. keep_push_pop = false gives every stack a
# closed-form size so nothing allocates inside the kernel. A `while` states
# no trip count, so its count and carry stacks have no size formula and the
# mode cannot be honoured. Refused with the alternative named, rather than
# failing later inside the sizing pass with a message about a stack.
#
# This confines a `while` to the CPU path, since the GPU stages require
# this mode. That is a real limit of the feature and not an oversight.
function agen_refuse_while_sized(body)
    for stmt in body
        if stmt.kind == :while
            error("stade_adjoint: `keep_push_pop = false` needs a closed-form size for " *
                  "every stack, and a `while` loop has no trip count until it has run. " *
                  "Use keep_push_pop = true, or write a counted loop with an early exit " *
                  "(`for i = 1:n_max ... end` with a guard), which is supported and offloads")
        elseif stmt.kind == :for
            agen_refuse_while_sized(stmt.body)
        elseif stmt.kind == :if
            agen_refuse_while_sized(stmt.then); agen_refuse_while_sized(stmt.els)
        end
    end
    return nothing
end

# An :ivalue stack has no closed-form size: how often a restored integer is written depends on the run. Refused with
# the alternative named, as a `while` is.
function agen_refuse_itbr_sized(sites)
    names = sort(collect(string(s.array) for s in sites if s.kind == :ivalue))
    isempty(names) && return nothing
    error("stade_adjoint: `keep_push_pop = false` needs a closed-form size for every stack, and the integer " *
          join(("`" * n * "`" for n in names), ", ") * " must be restored in the reverse sweep: " *
          "an active statement reads it after an iteration-dependent update. Use keep_push_pop = true")
end

function agen_refuse_while(body)
    for stmt in body
        if stmt.kind == :while
            error("stade_adjoint: reverse mode over a `while` loop is not finished " *
                  "(phase 4 of the while plan). The forward count is correct, but the " *
                  "scalars the body carries are not yet snapshotted, so the gradient " *
                  "would be silently wrong. `stade_tangent` works on this kernel today")
        elseif stmt.kind == :for
            agen_refuse_while(stmt.body)
        elseif stmt.kind == :if
            agen_refuse_while(stmt.then); agen_refuse_while(stmt.els)
        end
    end
    return nothing
end

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

# The arrays that some statement of the kernel writes.
function agen_written_arrays!(body, acc)
    for st in body
        if st.kind == :assign
            st.lhs isa Expr && push!(acc, st.lhs.args[1])
        elseif st.kind == :for || st.kind == :while
            agen_written_arrays!(st.body, acc)
        elseif st.kind == :if
            agen_written_arrays!(st.then, acc)
            agen_written_arrays!(st.els, acc)
        end
    end
    return acc
end

function agen_init_derived_stmts(kernel, wanted, already_defined)
    defs = Dict{Symbol,Any}()
    counts = agen_count_assign_sites(kernel.body)
    # a scalar read from an array can be rebuilt at init only if the kernel never writes that array: the sizing function
    # sees the array as it is on entry, and a size taken from it would be too small, or too large, for the loop that runs
    written = agen_written_arrays!(kernel.body, Set{Symbol}())
    for stmt in kernel.body
        stmt.kind == :assign || continue
        stmt.lhs isa Symbol || continue
        get(counts, stmt.lhs, 0) == 1 || continue
        rd = Set{Symbol}()
        agen_collect_expr_vars!(stmt.rhs, rd)
        isempty(intersect(rd, written)) || continue
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
    # a stack whose every site was removed (an array snapshot took them all) has no layout entry and no size
    keep_push_pop || (names = [nm for nm in names if haskey(layout.sizes, nm) || haskey(layout.block_totals, nm) || nm in layout.tainted_stacks])
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
        # a captured size term is declared first, so the replay can set it from inside a loop or an `if`
        for c in layout.captures
            pushfirst!(table_stmts, Expr(:(=), c.sym, 0))
            push!(tot_names, c.sym)
        end
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
    # array snapshots: the stack size needs length(A), so A joins the parameters and its length is returned after the stacks
    snap_arrays = sort(unique(Symbol[A for A in (agen_asnap_site_array(s, kernel.sig.kinds) for s in sites) if A !== nothing]); by = string)
    snap_arrays = [A for A in snap_arrays if agen_site_stack_name(agen_asnap_site(A)) in names]
    len_stmts = Any[Expr(:(=), agen_asnap_len(A), Expr(:call, :length, A)) for A in snap_arrays]
    isempty(snap_arrays) || (fargs = sort(union(fargs, snap_arrays); by = string))
    val_names = vcat(val_names, Symbol[agen_asnap_len(A) for A in snap_arrays])
    body = vcat(len_stmts, derived_stmts, sizing_stmts, table_stmts, alloc_stmts)
    keep_push_pop || agen_refuse_unsized(kernel, fargs, body)
    push!(body, emit_return_scalars(vcat(names, table_names, tot_names, val_names)))
    return (Expr(:function, Expr(:call, fname, fargs...), Expr(:block, body...)), table_names, tot_names, val_names)
end

# The names an `initstacks_*` body reads that nothing in it defines: not a parameter, not assigned in the body, not a loop
# variable, not a Base name. Flow-insensitive on purpose: it catches a name that is never defined, which is what an
# `UndefVarError` at the first call comes from.
function agen_init_free_names(fargs, body)
    defined = Set{Symbol}(fargs)
    reads = Set{Symbol}()
    function walk(e, as_read = true)
        if e isa Symbol
            as_read && push!(reads, e)
        elseif e isa Expr
            if e.head == :(=) || e.head in (:+=, :-=, :*=)
                lhs = e.args[1]
                if lhs isa Symbol
                    push!(defined, lhs)
                    e.head == :(=) || push!(reads, lhs)
                else
                    walk(lhs)
                end
                walk(e.args[2])
            elseif e.head == :for
                hdr = e.args[1]
                hdr isa Expr && hdr.head == :(=) && hdr.args[1] isa Symbol && push!(defined, hdr.args[1])
                hdr isa Expr && walk(hdr.args[2])
                walk(e.args[2])
            elseif e.head == :curly
                # a type, never a variable
            elseif e.head == :call
                e.args[1] isa Symbol || walk(e.args[1])
                for a in e.args[2:end]
                    walk(a)
                end
            else
                for a in e.args
                    walk(a)
                end
            end
        end
        return nothing
    end
    for st in body
        walk(st)
    end
    return sort([v for v in reads if !(v in defined) && !isdefined(Base, v) && v != :undef && v != :nothing]; by = string)
end

# `keep_push_pop = false` sizes every stack from the kernel's arguments, before the kernel runs. A loop bound that is read
# from an array the kernel writes (`c = cnt[1]` after `cnt[1] = ...`), or from a scalar assigned at several sites, is a value
# the sizing pass cannot know: the array on entry is not the array when the loop runs. The sizing function would reference an
# undefined name. It is refused here, with the alternative named.
function agen_refuse_unsized(kernel, fargs, body)
    free = agen_init_free_names(fargs, body)
    isempty(free) && return nothing
    defs = Dict{Symbol,Any}()
    function find(stmts)
        for st in stmts
            if st.kind == :assign && st.lhs isa Symbol
                haskey(defs, st.lhs) || (defs[st.lhs] = st.rhs)
            elseif st.kind == :for || st.kind == :while
                find(st.body)
            elseif st.kind == :if
                find(st.then); find(st.els)
            end
        end
    end
    find(kernel.body)
    what = String[]
    for v in free
        push!(what, haskey(defs, v) ? "`$(v)` (= `$(defs[v])`)" : "`$(v)`")
    end
    error("stade_adjoint: `keep_push_pop = false` cannot size the stacks of `$(kernel.sig.name)`: the size depends on " *
          join(what, ", ") * ", which cannot be rebuilt before the kernel runs. The sizing function rebuilds a scalar only from " *
          "the arguments and from arrays that the kernel never writes, and only if the scalar is assigned once. Pass the loop " *
          "bound as an integer argument, or use keep_push_pop = true")
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
        elseif stmt.kind == :while
            union!(reassigned, agen_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :if
            union!(reassigned, agen_collect_reassigned(stmt.then, in_loop))
            union!(reassigned, agen_collect_reassigned(stmt.els, in_loop))
        end
    end
    return reassigned
end

# A loop bound held in a scalar that is assigned at two or more sites (`c = n`, a loop, `c = m`, a loop) is reassigned in the
# sense that matters here, even when no assignment sits in a loop: the reverse sweep reaches the first loop with the value
# the last assignment left. Such a bound goes on a :tripcount stack like one reassigned inside a loop. Shared by the snap_ and
# agen_ copies, so the two cannot disagree.
function agen_collect_for_bound_vars!(body, acc)
    for st in body
        if st.kind == :for
            union!(acc, agen_for_bound_vars(st))
            agen_collect_for_bound_vars!(st.body, acc)
        elseif st.kind == :while
            agen_collect_for_bound_vars!(st.body, acc)
        elseif st.kind == :if
            agen_collect_for_bound_vars!(st.then, acc)
            agen_collect_for_bound_vars!(st.els, acc)
        end
    end
    return acc
end

function agen_multisite_bound_vars(body)
    counts = agen_count_assign_sites(body)
    bounds = agen_collect_for_bound_vars!(body, Set{Symbol}())
    return Set{Symbol}(v for v in bounds if get(counts, v, 0) >= 2)
end

agen_collect_reassigned_all(body) = union(agen_collect_reassigned(body), agen_multisite_bound_vars(body))
snap_collect_reassigned_all(body) = union(snap_collect_reassigned(body), agen_multisite_bound_vars(body))

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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
function agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks; ii_plan = nothing, cross = nothing)
    cand = agen_nested_write_vars(body, kinds)
    return sort(collect(v for v in cand if v in value_needed && !(v in exempt) && haskey(stacks, (:value, v)) &&
                          !(ii_plan !== nothing && agen_ii_covered_write_check(body, v, ii_plan, false)) &&
                          !(cross !== nothing && v in cross) &&
                          !agen_boundary_push_redundant(body, v, kinds, value_needed, exempt)); by = string)
end

# The scalars agen_remat_cross_apply! made loop-local. Every read of one sits in a loop body after that body's own write and the reverse sweep
# rebuilds the end-of-iteration value at the head of each reverse iteration, so no read observes the value a body boundary push would restore.
# The emission walks drop their boundary push and pop (`cross` of agen_block_boundary_vars); the layout walks keep sizing them, and
# agen_finalize_stacks prunes the stack once nothing references it.
function agen_remat_cross_vars(remat)
    remat === nothing && return Set{Symbol}()
    return Set{Symbol}(r.args[1] for r in values(remat) if r isa Expr && r.head == :cross)
end

# This body's own boundary push is redundant when the body doesn't write `var` at its own top level AND some nested body is guaranteed to push it
# -- the outer push reads `var` after a loop cgen_ may have split onto the device, storing a stale host value (confirmed live). The GUARANTEED
# condition doesn't make dropping safe, since the inner pushes already form a complete chain; it bounds how aggressive the collapse gets, as
# dropping more moves the value need into per-iteration pushes that cgen_contains_stackop then refuses to split, lowering offload coverage. Keyed
# on agen_boundary_kill_vars' condition so the layout/emit walks and the fwd-walk kill can never disagree.
function agen_boundary_push_redundant(body, var, kinds, value_needed, exempt)
    agen_has_top_level_write(body, var) && return false
    return agen_boundary_guaranteed_below(body, var, kinds, value_needed, exempt)
end

agen_has_top_level_write(body, var) =
    any(st -> st.kind == :assign && st.lhs isa Symbol && st.lhs == var, body)

agen_boundary_guaranteed_here(body, var, kinds, value_needed, exempt) =
    var in value_needed && !(var in exempt) && get(kinds, var, :none) == :scalar_float &&
    var in agen_nested_write_vars(body, kinds) && agen_has_top_level_write(body, var)

function agen_boundary_guaranteed_below(body, var, kinds, value_needed, exempt)
    for st in body
        if st.kind == :for
            agen_boundary_guaranteed_here(st.body, var, kinds, value_needed, exempt) && return true
            agen_boundary_guaranteed_below(st.body, var, kinds, value_needed, exempt) && return true
        elseif st.kind == :while
            agen_boundary_guaranteed_here(st.body, var, kinds, value_needed, exempt) && return true
            agen_boundary_guaranteed_below(st.body, var, kinds, value_needed, exempt) && return true
        elseif st.kind == :if
            for b in (st.then, st.els)
                agen_boundary_guaranteed_here(b, var, kinds, value_needed, exempt) && return true
                agen_boundary_guaranteed_below(b, var, kinds, value_needed, exempt) && return true
            end
        end
    end
    return false
end

# The float scalars a boundary push can cover at all: the filters above minus the per-body
# structural test. See snap_boundary_stack_vars -- identical logic, duplicated for the same
# purity-rule reason as every other agen_/snap_ pair in this file.
function agen_boundary_stack_vars(kernel)
    value_needed = agen_value_needed_vars(kernel)
    exempt = agen_exempt_vars(kernel, value_needed)
    return Set(v for v in value_needed if get(kernel.sig.kinds, v, :none) == :scalar_float && !(v in exempt))
end

# Vars this body's own boundary push is GUARANTEED to store: a nested write, which puts it on
# agen_block_boundary_vars' candidate list, AND a top-level write of this same body, which
# agen_ii_covered_write_check can never call covered -- so no ii_plan retires this push. See
# snap_boundary_kill_vars.
function agen_boundary_kill_vars(body, bvars)
    isempty(bvars) && return Set{Symbol}()
    nested = Set{Symbol}()
    top = Set{Symbol}()
    for stmt in body
        if stmt.kind == :assign
            stmt.lhs isa Symbol && push!(top, stmt.lhs)
        elseif stmt.kind == :for
            union!(nested, agen_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :while
            union!(nested, agen_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :if
            union!(nested, agen_collect_reassigned(stmt.then, true))
            union!(nested, agen_collect_reassigned(stmt.els, true))
        end
    end
    return intersect(nested, top, bvars)
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
        elseif stmt.kind == :while
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
# every agen_/snap_ pair in this file. stade_site_level_tbr_check asserts exact Dict
# equality against snap_value_needed_sites on every call -- forward push and backward pop
# must decide identically at every site or push/pop counts desync.
function agen_fwd_walk!(body, seen, active_map, decisions, bvars = Set{Symbol}())
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
            seen_then = agen_fwd_walk!(stmt.then, copy(seen), active_map, decisions, bvars)
            seen_els  = agen_fwd_walk!(stmt.els,  copy(seen), active_map, decisions, bvars)
            seen = union(seen_then, seen_els)
        elseif stmt.kind == :for
            seen = agen_fwd_walk_loop!(stmt.body, seen, active_map, decisions, bvars)
        elseif stmt.kind == :while
            seen = agen_fwd_walk_loop!(stmt.body, seen, active_map, decisions, bvars)
        end
    end
    # end-of-body block-boundary kill -- see snap_fwd_walk!'s comment for the full argument
    kill = agen_boundary_kill_vars(body, bvars)
    isempty(kill) && return seen
    return setdiff(seen, kill)
end

function agen_fwd_walk_loop!(body, in_seen, active_map, decisions, bvars = Set{Symbol}())
    seen = copy(in_seen)
    while true
        scratch = Dict{Any,Bool}()
        out_seen = agen_fwd_walk!(body, union(seen, in_seen), active_map, scratch, bvars)
        out_seen == seen && break
        seen = out_seen
    end
    final_in = union(seen, in_seen)
    return agen_fwd_walk!(body, final_in, active_map, decisions, bvars)
end

function agen_value_needed_sites(kernel)
    decisions = Dict{Any,Bool}()
    agen_fwd_walk!(kernel.body, Set{Symbol}(), act_analyze(kernel), decisions, agen_boundary_stack_vars(kernel))
    return decisions
end


# bound-variables of a :for statement that are reassigned somewhere
# else in the kernel -- the same set snap_plan's :tripcount sites are
# keyed on. Sorted by name: all :tripcount sites share one stack, so the push order and the
# (reversed) pop order must be the same list every time it is asked for.
function agen_tripcount_bound_vars(stmt, reassigned)
    bound_vars = agen_for_bound_vars(stmt)
    return sort!([bv for bv in bound_vars if bv in reassigned]; by = string)
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

# Where a reversed loop must START. `hi` is the last iterate only when the
# step divides the span, which for |step| = 1 is always. For `1:2:i_n`
# with i_n even, the forward loop visits 1, 3, ... i_n-1 while `i_n:-2:1`
# visits i_n, i_n-2, ... 2 -- disjoint sets, and a silently wrong
# gradient. Caught by red_strided at 1.33 relative error on the CPU
# oracles, before any GPU was involved.
#
# fld and not div: div truncates toward zero, so an empty forward loop
# (`1:2:0`) would give last = lo and the reverse would run once. fld
# floors, giving last = lo - step, which leaves the reverse empty. A loop
# that may not run has not run, in reverse.
#
# |step| == 1 keeps the plain `hi`, so every adjoint STADE already emits
# is unchanged and only the strided case moves.
function agen_reverse_start(lo, step, hi)
    step isa Number && abs(step) == 1 && return hi
    return Expr(:call, :+, lo,
                Expr(:call, :*, step,
                     Expr(:call, :fld, Expr(:call, :-, hi, lo), step)))
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
# with the same traversal agen_forward_body's pushes use, keyed structurally, never by a running count. Tier A: closed-form
# trip counts, single offset/size formulas. Tier B: a loop whose bound is reassigned in an ancestor sequential loop taints its stack, falling back to growable push!/pop! regardless of keep_push_pop.

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

# A loop whose bound has fallen below its start runs zero times, but div(hi-lo,step)+1 goes negative as soon as
# hi<=lo-2. Inside a size SUM a negative term shrinks the allocation rather than contributing nothing, undersizing
# the stack -- mg_vcycle at num_levels>=4 hit exactly this (nc=-1). Clamped here, the one place trip counts are
# summed into a size. agen_stride/cgen_kernel_def's bounds check keep the raw form: a stride only evaluates for a
# site that ran (every factor >=1), and a negative bounds check already makes every thread return.
agen_size_trip_count(lo, step, hi) = Expr(:call, :max, 0, cgen_trip_count(lo, step, hi))

# product of trip counts of every frame in loop_ctx -- one
# occurrence's own local multiplicity (§5's term), or 1 for a
# non-loop site
agen_local_multiplicity(loop_ctx) = agen_prod_exprs(Any[agen_size_trip_count(f.lo, f.step, f.hi) for f in loop_ctx])

# Which loops of the primal kernel does the GPU stage run as a kernel (one thread per iteration)? Runs the GPU stage's own
# host-body walk on the primal and reads its decision per loop, keyed by the loop's body object. The forward and the backward sweep
# walk the same body objects, so both read the same verdict. It is a guess about the adjoint: the adjoint's loops can fail a gate
# the primal's pass. A wrong guess costs speed only, since the layout stays a bijection. Every loop gets an entry; a loop the walk
# never decides (inside a kernel, or after a failure of the walk) is sequential.
function agen_loop_par_map(kernel)
    par = Dict{UInt,Bool}()
    agen_loop_par_default!(par, kernel.body)
    try
        gk = cgen_from_kernel(kernel)
        decisions = NamedTuple[]
        cgen_body(gk.body, Expr[], gk.name, cgen_backend_cuda(), Set{Symbol}(), Set{Symbol}(gk.args);
                  outer_defs = cgen_scalar_def_map(gk.body), decisions = decisions)
        for d in decisions
            haskey(par, d.body_id) && (par[d.body_id] = d.verdict !== :host)
        end
    catch
        agen_loop_par_default!(par, kernel.body)
    end
    return par
end

function agen_loop_par_default!(par::Dict{UInt,Bool}, body)
    for stmt in body
        if stmt.kind == :for
            par[objectid(stmt.body)] = false
            agen_loop_par_default!(par, stmt.body)
        elseif stmt.kind == :if
            agen_loop_par_default!(par, stmt.then)
            agen_loop_par_default!(par, stmt.els)
        elseif stmt.kind == :while
            agen_loop_par_default!(par, stmt.body)
        end
    end
    return par
end

# A loop_ctx frame, with the layout's verdict on whether the loop is the GPU thread. Stack mode has no layout and no positions.
# `primal` is the kernel's own loop statement when `stmt` is its linearization node (the backward sweep): the mark is keyed on
# the primal body object, which the forward and the backward sweep share.
function agen_frame(stmt, layout, primal = stmt)
    par = false
    if layout !== nothing
        haskey(layout.par, objectid(primal.body)) || error("agen_frame: loop over `$(stmt.var)` has no parallel mark in the layout")
        par = layout.par[objectid(primal.body)]
    end
    return (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step, par = par)
end

# 1-based flat position within one occurrence's own local block, from the current
# enclosing loop nest (outermost first) -- degenerates to the literal 1 for a non-loop site.
# The first frame that is the GPU thread (`par`) is the fastest-varying digit, so adjacent threads touch adjacent slots.
# The other frames follow it row-major in nest order. With no such frame the whole order is row-major, innermost fastest.
# Bijection onto 1:multiplicity either way, so sizes, offsets and the hoisted-index checks do not depend on the order.
# Verified against a hand-derived nested-loop offset formula (a single global +1, not one per level).
function agen_local_position(loop_ctx)
    isempty(loop_ctx) && return 1
    s = findfirst(f -> f.par, loop_ctx)
    if s === nothing
        terms = Any[agen_mul_exprs(agen_pos0(loop_ctx[i]), agen_stride(loop_ctx, i)) for i in eachindex(loop_ctx)]
        return agen_add_exprs(agen_sum_exprs(terms), 1)
    end
    thread = loop_ctx[s]
    rest = Any[loop_ctx[i] for i in eachindex(loop_ctx) if i != s]
    inner = Any[agen_mul_exprs(agen_pos0(rest[i]), agen_stride(rest, i)) for i in eachindex(rest)]
    thread_trip = cgen_trip_count(thread.lo, thread.step, thread.hi)
    terms = Any[agen_pos0(thread)]
    isempty(inner) || push!(terms, agen_mul_exprs(agen_sum_exprs(inner), thread_trip))
    return agen_add_exprs(agen_sum_exprs(terms), 1)
end

# ---- Tier B detection ------------------------------------------------
# A loop's bound-determining symbol is ever an assignment target inside any ancestor loop (a multigrid solver's
# ragged level-size halving sequence is the confirmed instance). The ancestor need not carry a value between
# iterations: `w = m0 + i` in an iteration-independent outer loop varies the inner trip count too, and gating on that
# made STADE size a stack from a body-local variable. Returns the offending bound var, or `nothing` if fully Tier A.
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
            inner_seq = union(seq_reassigned, agen_collect_reassigned(stmt.body, true))
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
            inner_seq = union(seq_reassigned, agen_collect_reassigned(stmt.body, true))
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
# A 'ragged block' = an ancestor loop AL whose body contains a ragged descendant it alone governs. Returns `nothing` if `stmt` isn't a
# genuine AL -- raggedness of the descendant is the whole test. AL's body is laid out via agen_indexed_layout reused as the AL-scoped
# sub-engine, seeded with the incoming reassignments (not AL's own top-level ones), so only genuine AL-within-AL raggedness taints.
# Returns `(header, local_offsets, local_sizes, ineligible_stacks)`, scoped to AL's own frame.
function agen_ragged_block(stmt, kinds, active_map, value_needed, reassigned, exempt, stacks, seq_reassigned; push_pop = nothing)
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
    # Only a kernel with ragged blocks has a replay of its scalars (agen_tier_b_kernel_skeleton) to capture a size term in.
    captures = Any[]
    if !isempty(blocks)
        offsets = Dict{Any,Any}(); current = Dict{Symbol,Any}(); blocks = Dict{Any,Any}(); block_of = Dict{Any,Int}()
        block_touched = Set{Symbol}(); ineligible = Set{Symbol}(); block_counter = Ref(0)
        agen_layout_walk_top!(kernel.body, kinds, active_map, value_needed, reassigned, exempt, stacks, Any[], offsets, current, blocks, block_of, block_touched, ineligible, Set{Symbol}(), block_counter, push_pop, captures)
    end
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
    # a snapshot stack counts iterations, and every iteration holds a whole copy of its array
    for ((kind, arr), stack) in stacks
        A = agen_asnap_site_array((kind = kind, array = arr), kinds)
        (A === nothing || !haskey(sizes, stack)) && continue
        sizes[stack] = agen_mul_exprs(sizes[stack], agen_asnap_len(A))
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
            derived_vars = derived, captures = captures, par = agen_loop_par_map(kernel))
end

function agen_layout_walk_top!(body, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, offsets, current, blocks, block_of, block_touched, ineligible, seq_reassigned, block_counter, push_pop, cap = nothing, later = Set{Symbol}())
    # with `cap` on, `later_at[idx]` holds every scalar assigned from statement idx on (at any depth) plus whatever the ancestors assign after this body
    later_at = fill(later, length(body) + 1)
    if cap !== nothing
        for i in length(body):-1:1
            later_at[i] = union(later_at[i + 1], agen_collect_reassigned(body[i:i], true))
        end
    end
    for (idx, stmt) in enumerate(body)
        lat = later_at[idx]
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            site_source = push_pop === nothing ? value_needed : push_pop
            if kinds[var] in (:scalar_float, :array_float) && get(active_map, var, false) &&
               agen_needs_snapshot(stmt.lhs, stmt.rhs, var, site_source, agen_site_key(body, idx)) && !(var in exempt)
                nm = agen_site_stack_name((kind = agen_snapshot_kind(stmt.lhs), array = var, at = 0))
                agen_layout_static_record!(offsets, current, nm, loop_ctx, agen_site_key(body, idx), cap, lat)
            end
        elseif stmt.kind == :for
            for bv in agen_tripcount_bound_vars(stmt, reassigned)
                agen_layout_static_record!(offsets, current, :tripcount_stack, loop_ctx, agen_site_key(body, idx, bv), cap, lat)
            end
            blk = agen_ragged_block(stmt, kinds, active_map, value_needed, reassigned, exempt, stacks, seq_reassigned; push_pop = push_pop)
            if blk === nothing
                inner_seq = union(seq_reassigned, agen_collect_reassigned(stmt.body, true))
                push!(loop_ctx, (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step))
                agen_layout_walk_top!(stmt.body, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, offsets, current, blocks, block_of, block_touched, ineligible, inner_seq, block_counter, push_pop, cap,
                                      cap === nothing ? later : union(later_at[idx + 1], agen_collect_reassigned(stmt.body, true)))
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
            agen_layout_static_record!(offsets, current, :branch_stack, loop_ctx, agen_site_key(body, idx), cap, lat)
            agen_layout_walk_top!(stmt.then, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, offsets, current, blocks, block_of, block_touched, ineligible, seq_reassigned, block_counter, push_pop, cap, later_at[idx + 1])
            agen_layout_walk_top!(stmt.els, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, offsets, current, blocks, block_of, block_touched, ineligible, seq_reassigned, block_counter, push_pop, cap, later_at[idx + 1])
        end
    end
    for var in agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks)
        agen_layout_static_record!(offsets, current, stacks[(:value, var)], loop_ctx, agen_site_key(body, 0, var), cap, later)
    end
    for A in agen_asnap_boundary_arrays(body, kinds, push_pop)
        agen_layout_static_record!(offsets, current, stacks[(:array, agen_asnap_site(A).array)], loop_ctx, agen_site_key(body, 0, A), cap, later)
    end
    return nothing
end

# exactly agen_layout_record!'s own base/advance split, just without
# the occ_mult/key_order two-pass deferral -- `current[stack]` IS the
# running sum, updated in place as we go, so a ragged block's own
# entry (agen_layout in the caller above) can read the right "so far"
# value out of it directly instead of needing a second folding pass.
function agen_layout_static_record!(offsets, current, stack_name, loop_ctx, key, cap = nothing, later = nothing)
    base = get(current, stack_name, 0)
    offsets[key] = (stack_name, base)
    mult = agen_local_multiplicity(loop_ctx)
    if cap !== nothing
        # A size term reading a scalar that is assigned again later would be read stale by every later offset and by the
        # allocation. The sizing pass of initstacks_* replays the scalars, so it captures the term where the site stands.
        vs = Set{Symbol}()
        agen_collect_expr_vars!(mult, vs)
        if !isempty(intersect(vs, later))
            sym = Symbol("__mult_", length(cap) + 1)
            push!(cap, (sym = sym, expr = mult, key = key))
            mult = sym
        end
    end
    current[stack_name] = agen_add_exprs(base, mult)
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
    reassigned = agen_collect_reassigned_all(kernel.body)
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

# Program-order events of one statement, as `(:consume, vars)` for a loop header and `(:assign, var)` for a scalar assignment, descending into
# loops and both branches of an `if`. One pass over each body: the repeat of a loop body is not unrolled.
function agen_tier_b_events!(st, out)
    if st.kind == :assign
        st.lhs isa Symbol && push!(out, (:assign, st.lhs))
    elseif st.kind == :for
        vs = Set{Symbol}()
        agen_collect_expr_vars!(st.lo, vs); agen_collect_expr_vars!(st.hi, vs); agen_collect_expr_vars!(st.step, vs)
        push!(out, (:consume, vs))
        for t in st.body
            agen_tier_b_events!(t, out)
        end
    elseif st.kind == :while
        vs = Set{Symbol}()
        agen_collect_expr_vars!(st.cond, vs)
        push!(out, (:consume, vs))
        for t in st.body
            agen_tier_b_events!(t, out)
        end
    elseif st.kind == :if
        for t in st.then
            agen_tier_b_events!(t, out)
        end
        for t in st.els
            agen_tier_b_events!(t, out)
        end
    end
    return out
end

# Index of the top-level statement of a ragged block's body before which the value tables and totals must be written, or `nothing`
# when they belong after the whole body. That statement is the first one holding a RETIRING assignment: a write, at any depth, to a
# size-relevant scalar that a loop header (in an earlier statement, or earlier inside the same one) has already read. The assignment
# belongs to the next iteration, so the tables must see the value as it stood before it.
# The tables hold one value per iteration, so the shape is refused when a relevant scalar is read by a loop header AFTER an assignment
# to it in the stream that starts at the split statement: that read would need the new value and the table holds the old one.
function agen_tier_b_split_point(body, relevant::Set{Symbol})
    consumed = Set{Symbol}()
    for (i, st) in enumerate(body)
        local_c = copy(consumed)
        retire = false
        for e in agen_tier_b_events!(st, Any[])
            if e[1] == :consume
                union!(local_c, e[2])
            elseif e[2] in relevant && e[2] in local_c
                retire = true
            end
        end
        if retire
            assigned = Set{Symbol}()
            for t in body[i:end], e in agen_tier_b_events!(t, Any[])
                if e[1] == :assign
                    e[2] in relevant && push!(assigned, e[2])
                elseif !isempty(intersect(e[2], assigned))
                    error("agen: a ragged block reads the bound `", first(sort(collect(intersect(e[2], assigned)); by = string)),
                          "` in a loop header after it is reassigned within the same iteration; keep_push_pop = false cannot size its stacks. ",
                          "Use keep_push_pop = true, or give the second use its own variable.")
                end
            end
            return i
        end
        consumed = local_c
    end
    return nothing
end

# `blk` is one entry of agen_layout(...).blocks. For every stack this block touches, in one shared loop over AL's header:
# allocates a `prefix_<stack>_<block_id>` table and `__tot_<stack>_<block_id>`, writes the pre-iteration total at
# `agen_pos0(header)+1` (Phase C's read uses the same index, guaranteeing agreement), then updates the total. Also builds a
# per-iteration `val_<var>_<block_id>` table for every block-local scalar a sizing formula depends on, so agen_site_index can
# resolve it by lookup instead of an unsafe bare reference.
function agen_tier_b_block_stmts(block_id, blk, kinds)
    header = blk.header
    # clamped: this table is indexed by agen_pos0(header) + 1, so its length must be the
    # header's ACTUAL trip count -- a negative would be a Vector length, not just a bad sum
    tripcount_expr = agen_size_trip_count(header.lo, header.step, header.hi)
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
    split_at = agen_tier_b_split_point(blk.body, relevant)
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
function agen_tier_b_kernel_skeleton(body, kinds, layout, done = Set{Symbol}())
    out = Any[]
    caps_here(i) = [c for c in layout.captures if c.key[1] == objectid(body) && c.key[2] == i && !(c.sym in done)]
    for (idx, stmt) in enumerate(body)
        for c in caps_here(idx)
            push!(out, Expr(:(=), c.sym, c.expr))
            push!(done, c.sym)
        end
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
                # A captured term inside this loop nest goes in front of the outermost loop when nothing in the nest changes
                # what it reads: the replay loop then holds no live-out scalar and still offloads.
                ids = Set{UInt}(); lvars = Set{Symbol}([stmt.var])
                agen_subtree_info!(stmt.body, ids, lvars)
                moved = agen_collect_reassigned(stmt.body, true)
                for c in layout.captures
                    (c.sym in done || !(c.key[1] in ids)) && continue
                    vs = Set{Symbol}()
                    agen_collect_expr_vars!(c.expr, vs)
                    isempty(intersect(vs, moved)) && isempty(intersect(vs, lvars)) || continue
                    push!(out, Expr(:(=), c.sym, c.expr))
                    push!(done, c.sym)
                end
                inner = agen_tier_b_kernel_skeleton(stmt.body, kinds, layout, done)
                push!(out, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
            end
        elseif stmt.kind == :if
            then_inner = agen_tier_b_kernel_skeleton(stmt.then, kinds, layout, done)
            els_inner = agen_tier_b_kernel_skeleton(stmt.els, kinds, layout, done)
            push!(out, emit_if(stmt.cond, then_inner, els_inner))
        end
    end
    for c in caps_here(0)
        push!(out, Expr(:(=), c.sym, c.expr))
        push!(done, c.sym)
    end
    return out
end

# identity of every statement list below `body`, and every loop variable declared there
function agen_subtree_info!(body, ids, lvars)
    push!(ids, objectid(body))
    for st in body
        if st.kind == :for || st.kind == :while
            st.kind == :for && push!(lvars, st.var)
            agen_subtree_info!(st.body, ids, lvars)
        elseif st.kind == :if
            agen_subtree_info!(st.then, ids, lvars)
            agen_subtree_info!(st.els, ids, lvars)
        end
    end
    return nothing
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
agen_ectx_stack() = (keep_push_pop = true, loop_ctx = Any[], layout = nothing, push_pop = nothing, ii_plan = nothing, remat = nothing)

# Resolves which value_needed-like info the per-statement push/pop gate should consult: the
# site-level Dict (ectx.push_pop) when one was computed, else the ordinary whole-variable
# Set (`value_needed`) -- exactly today's behavior when the flag is off. Never affects
# agen_block_boundary_vars/agen_exempt_vars, which stay on the
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

# Reads a computed stack index into its own scalar, on its own line, and returns that scalar.
# Every :indexed site goes through here. A Tier B index embeds a prefix/val table lookup, which
# inline would be the a[b[i]] indirect indexing parse_check_no_indirect_indexing forbids -- STADE's
# own front end rejecting its own output. A Tier A index carries no lookup but is still a
# multi-term position formula, and the same one-name-per-site form is what a hand-written
# skill-stade kernel would use for either. `__idx_<stack>[_<block>]_<position in exprs>` is unique
# within one statement list; a sibling list reusing the name is an ordinary reassignment, always
# read by the very next statement. An already-atomic index (a bare symbol, or a non-loop site's
# literal 1) is returned as is -- naming it would add a line that says nothing.
function agen_hoist_index!(exprs, stack, block_id, index)
    (index isa Symbol || index isa Number) && return index
    tmp = block_id === nothing ? Symbol("__idx_", string(stack), "_", length(exprs)) :
                                 Symbol("__idx_", string(stack), "_", block_id, "_", length(exprs))
    push!(exprs, Expr(:(=), tmp, index))
    return tmp
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
        # j = 1:n`) -- so it needs the identical substitution,
        # not just `local_offset` above.
        local_position = agen_substitute_vars(agen_local_position(Any[merge(f, (par = false,)) for f in inner_ctx]), subst)
        # `blk.base[stack]` is the offset this whole block starts at (0, or a prior block's
        # total, for a stack touched by more than one block). The prefix table is only relative
        # to within this block alone, so omitting `base` would make later blocks silently
        # overlap earlier ones' index range. `base` is always a plain symbol/kernel-arg
        # expression, never a block-local scalar, so it needs no substitution.
        full_index = agen_add_exprs(agen_add_exprs(agen_add_exprs(get(blk.base, stack, 0), Expr(:ref, table_name, table_idx)), local_offset), local_position)
        return agen_hoist_index!(exprs, stack, block_id, full_index)
    end
    (stack, offset) = entry
    return agen_hoist_index!(exprs, stack, nothing, agen_add_exprs(offset, agen_local_position(ectx.loop_ctx)))
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
# statement list: agen_site_index pushes the hoisted index assignment onto it before returning,
# landing ahead of whatever statement the caller builds from this call's return value.
function agen_emit_pop(stack_name::Symbol, ectx, key, exprs)
    agen_use_stack_push(ectx, stack_name) && return Expr(:call, :pop!, stack_name)
    return Expr(:ref, stack_name, agen_site_index(exprs, ectx, key))
end

# ---- Phase 3 cleanup: drop stack args left unused by fusion ----
# A var covered by an ii_plan site doesn't always mean its stack becomes fully unused -- agen_block_boundary_vars can
# still need it. A blanket drop would be a real bug, so the safe approach generates the adjoint body first, then
# drops only what provably has zero remaining stack references in the actual output. Covers both storage modes:
# keep_push_pop=true via push!/pop!, keep_push_pop=false via an indexed ref, nm[idx].
function agen_collect_used_stacks!(expr, used)
    if expr isa Expr
        if expr.head == :call && length(expr.args) >= 2 && expr.args[1] in (:push!, :pop!) && expr.args[2] isa Symbol
            push!(used, expr.args[2])
        elseif expr.head == :ref && !isempty(expr.args) && expr.args[1] isa Symbol
            push!(used, expr.args[1])
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

# Shared by agen_emit and stade_hvp: both need identical post-fusion cleanup -- scan the generated
# function for real stack use (agen_used_stack_names, covering both push!/pop! and indexed-ref
# access), then drop whatever agen_stack_names(sites) lists but the scan never found. `is_hvp`
# additionally drops hvp_emit's own orphaned `<stack>_d` shadow allocs, since hvp_emit builds one for
# every site regardless of fusion. A single shared function keeps the two call sites from drifting.
function agen_finalize_stacks(fn_expr, initstacks_expr, snapshot_plan, ii_plan; is_hvp::Bool = false, force::Bool = false)
    ii_plan === nothing && !force && return (fn_expr, initstacks_expr)
    used = agen_used_stack_names(fn_expr)
    unused_stacks = Set(nm for nm in agen_stack_names(snapshot_plan) if !(nm in used))
    isempty(unused_stacks) && return (fn_expr, initstacks_expr)
    fn_expr = agen_drop_unused_stack_args(fn_expr, unused_stacks)
    is_hvp && (fn_expr = hvp_drop_unused_shadow_stack_allocs(fn_expr, unused_stacks))
    initstacks_expr = agen_drop_unused_stack_allocs(initstacks_expr, unused_stacks)
    return (fn_expr, initstacks_expr)
end

# ---- ii_* Phase 3 codegen: :independent fusion only ----
# Builds one un-reversed loop fusing the primal recompute of stmt.body with its backward differentiation, for `:independent`
# sites only (`:reduction`/`:mixed` use separate handling, since fusing reads an undertotaled reduction too early). Fused vars
# are excluded via a locally-scoped value_needed Set, not kernel-wide. `agen_ii_override_ectx` is required: site-level TBR
# consults `ectx.push_pop`, so excluding a var from value_needed alone leaves an unmatched push.
# ---- rematerialized restores (keep_push_pop=false) ----
# A pre-write snapshot of scalar `v` saves the value of the nearest earlier write `v = rhs_d` in the same block. When that
# rhs can be re-evaluated at the backward position and gives the same value, the restore is `v = rhs_d` and the slot goes.
# The rhs may use: literals, scalars the kernel never assigns, enclosing loop vars, ints defined once at the top level of this
# or an enclosing block (and nowhere nested in it) that the sweep recomputes up front, and loads of arrays with no write that
# can execute after the load. No recursion into :if/:while bodies: agen_branch_pushed mirrors the forward gate there.
function agen_stmt_assigned(stmt)
    acc = Set{Symbol}()
    agen_stmt_assigned!(stmt, acc)
    return acc
end

function agen_stmt_assigned!(stmt, acc)
    if stmt.kind == :assign
        stmt.lhs isa Symbol && push!(acc, stmt.lhs)
    elseif stmt.kind == :for || stmt.kind == :while
        stmt.kind == :for && push!(acc, stmt.var)
        foreach(st -> agen_stmt_assigned!(st, acc), stmt.body)
    elseif stmt.kind == :if
        foreach(st -> agen_stmt_assigned!(st, acc), stmt.then)
        foreach(st -> agen_stmt_assigned!(st, acc), stmt.els)
    end
    return nothing
end

# Lexical position of every statement (`seq`, pre-order), the loops enclosing it, and every write to each array.
function agen_remat_index!(body, chain, seq, pos, loops, writes, assigned_syms)
    for (idx, stmt) in enumerate(body)
        seq[] += 1
        pos[agen_site_key(body, idx)] = seq[]
        loops[agen_site_key(body, idx)] = chain
        if stmt.kind == :assign
            if stmt.lhs isa Symbol
                push!(assigned_syms, stmt.lhs)
            else
                push!(get!(writes, stmt.lhs.args[1], Any[]), (seq[], chain))
            end
        elseif stmt.kind == :for || stmt.kind == :while
            stmt.kind == :for && push!(assigned_syms, stmt.var)
            agen_remat_index!(stmt.body, vcat(chain, seq[]), seq, pos, loops, writes, assigned_syms)
        elseif stmt.kind == :if
            agen_remat_index!(stmt.then, chain, seq, pos, loops, writes, assigned_syms)
            agen_remat_index!(stmt.els, chain, seq, pos, loops, writes, assigned_syms)
        end
    end
    return nothing
end

function agen_remat_pure(e, kinds, assigned_syms, writes, avail, d_seq, d_chain)
    e isa Number && return true
    if e isa Symbol
        e in avail && return true
        return !(e in assigned_syms) && haskey(kinds, e) && kinds[e] in (:scalar_int, :scalar_float)
    end
    e isa Expr || return false
    if e.head == :call
        e.args[1] in (:+, :-, :*, :/, :div, :mod, :÷, :%) || return false
        return all(a -> agen_remat_pure(a, kinds, assigned_syms, writes, avail, d_seq, d_chain), e.args[2:end])
    elseif e.head == :ref
        a = e.args[1]
        (a isa Symbol && haskey(kinds, a) && kinds[a] in (:array_float, :array_int)) || return false
        # a write that runs after the load (later in program order, or anywhere in a loop enclosing it) can leave the array
        # different at the backward position than it was at the load
        for (wseq, wchain) in get(writes, a, Any[])
            (wseq > d_seq || !isempty(intersect(wchain, d_chain))) && return false
        end
        return all(x -> agen_remat_pure(x, kinds, assigned_syms, writes, avail, d_seq, d_chain), e.args[2:end])
    end
    return false
end

function agen_remat_walk!(body, kinds, sites, assigned_syms, writes, pos, loops, unsafe, active_map, exempt, avail0, plan, cross, is_loop)
    avail = copy(avail0)
    last_def = Dict{Symbol,Any}()   # float scalar -> rhs of its last top-level write when that rhs is rematerializable
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            stmt.lhs isa Symbol || continue
            v = stmt.lhs
            if kinds[v] == :scalar_int
                # recomputed up front by this block's backward sweep: stable only if this is its one write in the block
                n_here = count(st -> st.kind == :assign && st.lhs === v, body)
                nested = any(st -> st.kind != :assign && v in agen_stmt_assigned(st), body)
                (n_here == 1 && !nested && !(v in unsafe)) && push!(avail, v)
            elseif kinds[v] == :scalar_float
                key = agen_site_key(body, idx)
                if haskey(last_def, v) && get(sites, key, false) === true && get(active_map, v, false) && !(v in exempt)
                    plan[key] = last_def[v]
                end
                if agen_remat_pure(stmt.rhs, kinds, assigned_syms, writes, avail, pos[key], loops[key])
                    last_def[v] = stmt.rhs
                else
                    delete!(last_def, v)
                end
            end
        elseif stmt.kind == :for
            for w in agen_stmt_assigned(stmt)
                delete!(last_def, w)
            end
            agen_remat_walk!(stmt.body, kinds, sites, assigned_syms, writes, pos, loops, unsafe, active_map, exempt, union(avail, Set([stmt.var])), plan, cross, true)
        elseif stmt.kind == :if || stmt.kind == :while
            for w in agen_stmt_assigned(stmt)
                delete!(last_def, w)
            end
        end
    end
    is_loop && agen_remat_cross_record!(body, kinds, sites, active_map, exempt, last_def, cross)
    return nothing
end

# Cross-iteration restores. The first write W1 of a float scalar in a loop body saves the value the previous iteration (or the code before
# the loop) left behind, so that whoever read that value gets it back in the reverse sweep. For a LOOP-LOCAL scalar nobody did: every read
# of it sits in a loop body after that body's own first write, so no read observes a value from another iteration, another loop or the
# entry. Its only observable values live inside one iteration, and those come from the end-of-iteration value -- the pure rhs of the body's
# last top-level write (`last_def`), rebuilt at the head of each reverse iteration -- and the block-local restores above. W1 saves nothing.
# All-or-nothing per scalar: a loop that kept its first save would rely on the value a transformed neighbour no longer restores.
function agen_remat_cross_record!(body, kinds, sites, active_map, exempt, last_def, cross)
    seen = Set{Symbol}()
    for (idx, st) in enumerate(body)
        (st.kind == :assign && st.lhs isa Symbol && get(kinds, st.lhs, :none) == :scalar_float && !(st.lhs in seen)) || continue
        v = st.lhs
        push!(seen, v)
        key1 = agen_site_key(body, idx)
        (get(sites, key1, false) === true && get(active_map, v, false) && !(v in exempt)) || continue
        rec = haskey(last_def, v) ? (agen_site_key(body, 0), key1, last_def[v]) : (nothing, key1, nothing)
        push!(get!(cross, v, Any[]), rec)
    end
    return nothing
end

function agen_remat_cross_apply!(plan, cross, kernel)
    isempty(cross) && return nothing
    bad = Set{Symbol}()
    defined = Set{Symbol}()
    agen_remat_cross_scan!(kernel.body, false, Set{Symbol}(), kernel.sig.kinds, bad, defined)
    for v in sort!(collect(keys(cross)); by = string)
        (v in defined && !(v in bad) && all(r -> r[1] !== nothing, cross[v])) || continue
        for (head_key, key1, rhs) in cross[v]
            plan[key1] = Expr(:cross, v)
            push!(get!(plan, head_key, Expr(:block)).args, Expr(:(=), v, rhs))
        end
    end
    return nothing
end

# Is the scalar loop-local? Collects into `bad` every float scalar that is written anywhere but the top level of a loop body (a branch-
# local value would lean on restores the transformed loops no longer perform), read where no enclosing body has yet written it, or written
# by a statement nested in a body that also writes it at its top level; `defined` gets every scalar written at the top level of some loop
# body. A body that writes the scalar itself is the only one that covers its reads: an enclosing body's earlier write does not, since the
# first iteration would read it and the later ones the previous iteration's value.
function agen_remat_cross_scan!(body, in_loop, covered, kinds, bad, defined)
    isfloat(v) = v isa Symbol && get(kinds, v, :none) == :scalar_float
    first_write = Dict{Symbol,Int}()
    for (idx, st) in enumerate(body)
        if st.kind == :assign && isfloat(st.lhs)
            in_loop ? get!(first_write, st.lhs, idx) : push!(bad, st.lhs)
        end
    end
    now = covered
    read_here(e) = for v in cgen_expr_symbols(e); isfloat(v) && !(v in now) && push!(bad, v); end
    for (idx, st) in enumerate(body)
        now = Set{Symbol}(v for v in covered if !haskey(first_write, v))
        union!(now, (v for (v, i1) in first_write if i1 < idx))
        if st.kind == :assign
            read_here(st.rhs)
            st.lhs isa Expr && read_here(st.lhs)
        else
            for v in agen_stmt_assigned(st)
                isfloat(v) && haskey(first_write, v) && push!(bad, v)
            end
            if st.kind == :for
                read_here(st.lo); read_here(st.hi); read_here(st.step)
                agen_remat_cross_scan!(st.body, true, now, kinds, bad, defined)
            elseif st.kind == :if
                read_here(st.cond)
                agen_remat_cross_scan!(st.then, false, now, kinds, bad, defined)
                agen_remat_cross_scan!(st.els, false, now, kinds, bad, defined)
            elseif st.kind == :while
                read_here(st.cond)
                agen_remat_cross_scan!(st.body, false, now, kinds, bad, defined)
            end
        end
    end
    union!(defined, keys(first_write))
    return nothing
end

function agen_remat_plan(kernel, active_map, sites)
    kinds = kernel.sig.kinds
    pos = Dict{Any,Int}()
    loops = Dict{Any,Vector{Int}}()
    writes = Dict{Symbol,Any}()
    assigned_syms = Set{Symbol}()
    agen_remat_index!(kernel.body, Int[], Ref(0), pos, loops, writes, assigned_syms)
    unsafe = agen_unsafe_int_vars(kernel, snap_itbr_vars(kernel, active_map))
    exempt = agen_exempt_vars(kernel, agen_value_needed_vars(kernel))
    plan = Dict{Any,Any}()
    cross = Dict{Symbol,Vector{Any}}()
    agen_remat_walk!(kernel.body, kinds, sites, assigned_syms, writes, pos, loops, unsafe, active_map, exempt, Set{Symbol}(), plan, cross, false)
    agen_remat_cross_apply!(plan, cross, kernel)
    return plan
end

# Remat sites leave the site-level push/pop map, so layout, stack sizing and every emission gate see them as never pushed.
function agen_remat_apply(kernel, active_map, sites)
    plan = agen_remat_plan(kernel, active_map, sites)
    isempty(plan) && return (sites, nothing)
    out = copy(sites)
    for k in keys(plan)
        k[2] == 0 || (out[k] = false)
    end
    return (out, plan)
end

# ---- array snapshot (keep_push_pop=false only) ----
# A loop `L` whose body `B` writes a float array `A` normally pushes the old value at every write site, one slot per executed write.
# When every derivative read of `A` inside `B` follows every write to `A` in `B`, those pushes carry nothing the reverse sweep reads:
# a final read sees the end-of-body value, and the end-of-body value is what memory holds at the start of the reverse of `B`.
# One copy of the whole of `A` at the start of `B`, restored after the reverse of `B`, replaces them. The copy needs no fact about
# which elements `B` writes, so an indirect index costs nothing. Refusing is always safe: a refused loop keeps its per-write pushes.
# Guards (see the plan, section 4): G1 float array argument, G2 :for loop, G3 a removable write site, G4 final-read order,
# G5 a read in a condition counts, G6 outermost loop only, G7 no write inside an ii-classified loop, G9 Tier A loop nest, G10 B sweeps A at run time, G13 no write index follows the loop variable, G14 a pushed scatter site exists.
function agen_asnap_written_arrays!(body, out::Set{Symbol}, kinds)
    for st in body
        if st.kind == :assign
            st.lhs isa Expr && st.lhs.head == :ref && get(kinds, st.lhs.args[1], :none) == :array_float && push!(out, st.lhs.args[1])
        elseif st.kind == :for || st.kind == :while
            agen_asnap_written_arrays!(st.body, out, kinds)
        elseif st.kind == :if
            agen_asnap_written_arrays!(st.then, out, kinds)
            agen_asnap_written_arrays!(st.els, out, kinds)
        end
    end
    return out
end

# how many times the array name `A` appears in `expr`, as the array of a ref or as a bare symbol
function agen_asnap_count_refs(expr, A::Symbol)
    expr isa Symbol && return expr == A ? 1 : 0
    expr isa Expr || return 0
    total = 0
    for a in expr.args[(expr.head == :call ? 2 : 1):end]
        total += agen_asnap_count_refs(a, A)
    end
    return total
end

# Does `st` read `A` anywhere except as the self term of a pure accumulation whose other terms do not mention `A`?
function agen_asnap_has_bad_read(st, A::Symbol)
    if st.kind == :assign
        lhs_base = st.lhs isa Symbol ? st.lhs : st.lhs.args[1]
        n_rhs = agen_asnap_count_refs(st.rhs, A)
        lhs_idx = st.lhs isa Expr ? sum(agen_asnap_count_refs(a, A) for a in st.lhs.args[2:end]; init = 0) : 0
        lhs_idx > 0 && return true
        lhs_base == A || return n_rhs > 0
        n_rhs == 0 && return false
        return !(n_rhs == 1 && agen_is_pure_accumulation(st.lhs, st.rhs, A))
    elseif st.kind == :for
        any(e -> agen_asnap_count_refs(e, A) > 0, (st.lo, st.hi, st.step)) && return true
        return any(s -> agen_asnap_has_bad_read(s, A), st.body)
    elseif st.kind == :if
        agen_asnap_count_refs(st.cond, A) > 0 && return true
        return any(s -> agen_asnap_has_bad_read(s, A), st.then) || any(s -> agen_asnap_has_bad_read(s, A), st.els)
    end
    return true   # :while and anything unknown
end

# Does `st` mention `A` at all, as a read or a write?
function agen_asnap_touches(st, A::Symbol)
    w = Set{Symbol}()
    agen_asnap_written_arrays!([st], w, Dict{Symbol,Symbol}(A => :array_float))
    A in w && return true
    return agen_asnap_reads_any(st, A)
end

function agen_asnap_reads_any(st, A::Symbol)
    if st.kind == :assign
        return agen_asnap_count_refs(st.rhs, A) > 0 || (st.lhs isa Expr && any(a -> agen_asnap_count_refs(a, A) > 0, st.lhs.args[2:end]))
    elseif st.kind == :for
        return any(e -> agen_asnap_count_refs(e, A) > 0, (st.lo, st.hi, st.step)) || any(s -> agen_asnap_reads_any(s, A), st.body)
    elseif st.kind == :if
        return agen_asnap_count_refs(st.cond, A) > 0 || any(s -> agen_asnap_reads_any(s, A), st.then) || any(s -> agen_asnap_reads_any(s, A), st.els)
    end
    return true
end

# every write site (assign statement) to `A` inside `body`, at any depth, as agen_site_key keys
function agen_asnap_write_keys!(body, A::Symbol, out::Vector{Any})
    for (idx, st) in enumerate(body)
        if st.kind == :assign
            st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == A && push!(out, agen_site_key(body, idx))
        elseif st.kind == :for || st.kind == :while
            agen_asnap_write_keys!(st.body, A, out)
        elseif st.kind == :if
            agen_asnap_write_keys!(st.then, A, out)
            agen_asnap_write_keys!(st.els, A, out)
        end
    end
    return out
end

# G14 (benefit): the snapshot exists to let a scatter loop offload. A scatter is a read-modify-write of `A` through an index that an integer array
# supplies, which two threads may share. Names that carry such an index: every scalar assigned from an integer array or from another such scalar.
function agen_asnap_indirect_vars(B, kinds)
    D = Set{Symbol}()
    changed = true
    while changed
        changed = agen_asnap_indirect_walk!(B, D, kinds) 
    end
    return D
end

function agen_asnap_indirect_walk!(body, D, kinds)
    grew = false
    for st in body
        if st.kind == :assign
            if st.lhs isa Symbol && !(st.lhs in D)
                vs = Set{Symbol}()
                agen_collect_expr_vars!(st.rhs, vs)
                if any(v -> v in D || get(kinds, v, :none) == :array_int, vs)
                    push!(D, st.lhs)
                    grew = true
                end
            end
        elseif st.kind == :for || st.kind == :while
            agen_asnap_indirect_walk!(st.body, D, kinds) && (grew = true)
        elseif st.kind == :if
            agen_asnap_indirect_walk!(st.then, D, kinds) && (grew = true)
            agen_asnap_indirect_walk!(st.els, D, kinds) && (grew = true)
        end
    end
    return grew
end

# keys of the scatter writes to `A` in `body`
function agen_asnap_scatter_keys!(body, A::Symbol, D, kinds, out::Vector{Any})
    for (idx, st) in enumerate(body)
        if st.kind == :assign
            if st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == A && agen_asnap_count_refs(st.rhs, A) > 0
                vs = Set{Symbol}()
                for a in st.lhs.args[2:end]
                    agen_collect_expr_vars!(a, vs)
                end
                any(v -> v in D || get(kinds, v, :none) == :array_int, vs) && push!(out, agen_site_key(body, idx))
            end
        elseif st.kind == :for || st.kind == :while
            agen_asnap_scatter_keys!(st.body, A, D, kinds, out)
        elseif st.kind == :if
            agen_asnap_scatter_keys!(st.then, A, D, kinds, out)
            agen_asnap_scatter_keys!(st.els, A, D, kinds, out)
        end
    end
    return out
end

# is a write to `A` anywhere inside a loop that ii_plan classifies? (guard G7)
function agen_asnap_write_in_ii_loop(body, A::Symbol, ii_plan, kinds)
    ii_plan === nothing && return false
    for (idx, st) in enumerate(body)
        if st.kind == :for
            w = agen_asnap_written_arrays!(st.body, Set{Symbol}(), kinds)
            A in w && get(ii_plan, agen_site_key(body, idx), nothing) !== nothing && return true
            agen_asnap_write_in_ii_loop(st.body, A, ii_plan, kinds) && return true
        elseif st.kind == :if
            (agen_asnap_write_in_ii_loop(st.then, A, ii_plan, kinds) || agen_asnap_write_in_ii_loop(st.els, A, ii_plan, kinds)) && return true
        elseif st.kind == :while
            return true
        end
    end
    return false
end

# G10 (cost): the copy costs length(A) per iteration of L, so it pays only when B itself sweeps. Require a write site to A under an inner
# loop of B whose bounds are not both literals. A loop with literal bounds (the cell-to-node loop of three) writes a few slots per iteration.
function agen_asnap_swept_write(body, A::Symbol, under_runtime::Bool)
    for st in body
        if st.kind == :assign
            under_runtime && st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == A && return true
        elseif st.kind == :for
            rt = under_runtime || !(st.lo isa Number && st.hi isa Number && st.step isa Number)
            agen_asnap_swept_write(st.body, A, rt) && return true
        elseif st.kind == :if
            (agen_asnap_swept_write(st.then, A, under_runtime) || agen_asnap_swept_write(st.els, A, under_runtime)) && return true
        end
    end
    return false
end

# G13 (cost): a write whose index follows the loop variable touches a different part of A in every iteration (a per-edge scratch slice, a
# per-step row), so one whole-array copy per iteration costs length(A) where the writes cost one slot each. The names that follow the loop
# variable are the loop variable, every scalar assigned from them, every inner loop variable whose bounds use them, and every integer array
# written in the body (conservative).
function agen_asnap_follow_vars(B, var::Symbol, kinds)
    T = Set{Symbol}([var])
    changed = true
    while changed
        changed = false
        agen_asnap_follow_walk!(B, T, kinds) && (changed = true)
    end
    return T
end

function agen_asnap_follow_walk!(body, T, kinds)
    grew = false
    add!(v) = v in T ? false : (push!(T, v); true)
    mentions(e) = (vs = Set{Symbol}(); agen_collect_expr_vars!(e, vs); !isempty(intersect(vs, T)))
    for st in body
        if st.kind == :assign
            if st.lhs isa Symbol
                mentions(st.rhs) && add!(st.lhs) && (grew = true)
            elseif get(kinds, st.lhs.args[1], :none) == :array_int
                add!(st.lhs.args[1]) && (grew = true)
            end
        elseif st.kind == :for
            (mentions(st.lo) || mentions(st.hi) || mentions(st.step)) && add!(st.var) && (grew = true)
            agen_asnap_follow_walk!(st.body, T, kinds) && (grew = true)
        elseif st.kind == :if
            agen_asnap_follow_walk!(st.then, T, kinds) && (grew = true)
            agen_asnap_follow_walk!(st.els, T, kinds) && (grew = true)
        elseif st.kind == :while
            agen_asnap_follow_walk!(st.body, T, kinds) && (grew = true)
        end
    end
    return grew
end

# true when no write to `A` in `body` indexes with a name in `T`
function agen_asnap_index_free(body, A::Symbol, T)
    for st in body
        if st.kind == :assign
            if st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == A
                idx_vars = Set{Symbol}()
                for a in st.lhs.args[2:end]
                    agen_collect_expr_vars!(a, idx_vars)
                end
                isempty(intersect(idx_vars, T)) || return false
            end
        elseif st.kind == :for || st.kind == :while
            agen_asnap_index_free(st.body, A, T) || return false
        elseif st.kind == :if
            (agen_asnap_index_free(st.then, A, T) && agen_asnap_index_free(st.els, A, T)) || return false
        end
    end
    return true
end

# G4: every top-level statement of `B` that touches `A` after the last top-level statement that writes it may read it freely. Any read at
# or before that statement must be the self term of a pure accumulation.
function agen_asnap_final_read_ok(B, A::Symbol, kinds)
    last_w = 0
    for (j, st) in enumerate(B)
        w = Set{Symbol}()
        agen_asnap_written_arrays!([st], w, kinds)
        A in w && (last_w = j)
    end
    last_w == 0 && return false
    for j in 1:last_w
        agen_asnap_has_bad_read(B[j], A) && return false
    end
    return true
end

# Pseudo-site of an array snapshot: stack `<A>_snap_stack`. The name is refused when the kernel already has a variable called `<A>_snap`.
agen_asnap_site(A::Symbol) = (kind = :array, array = Symbol(string(A) * "_snap"), at = 0)
agen_asnap_len(A::Symbol) = Symbol("__len_", A)

# The array a pseudo-site snapshots, or `nothing` for any other site. A real variable `<X>_snap` makes the planner refuse X, so a site
# whose array name is not a kernel variable is never ambiguous.
function agen_asnap_site_array(site, kinds)
    site.kind == :array || return nothing
    str = string(site.array)
    endswith(str, "_snap") || return nothing
    haskey(kinds, site.array) && return nothing
    A = Symbol(chop(str; tail = 5))
    return get(kinds, A, :none) == :array_float ? A : nothing
end

# Arrays whose snapshot starts at the head of `body`, sorted by name.
function agen_asnap_boundary_arrays(body, kinds, push_pop)
    push_pop === nothing && return Symbol[]
    return sort([A for (A, k) in kinds if k == :array_float && get(push_pop, agen_site_key(body, 0, A), false) === true]; by = string)
end

# Returns `(sites, plan)`. `sites` has every write site to a snapshotted array removed, plus one `agen_site_key(B, 0, A) => true` entry
# per snapshot, which the layout and both sweeps read. `plan` is `nothing` when no loop qualifies, else `(bodies, arrays)`.
function agen_asnap_apply(kernel, active_map, sites; ii_plan = nothing)
    (out, plan) = agen_asnap_plan(kernel, active_map, sites; ii_plan = ii_plan)
    isempty(plan.arrays) && return (sites, nothing)
    return (out, plan)
end

function agen_asnap_sites(plan)
    return [agen_asnap_site(A) for A in plan.arrays]
end

# Returns `(sites, plan)` as agen_asnap_apply does, but always a plan. `plan.bodies` maps agen_site_key(B, 0) to the arrays snapshotted
# at the head of B.
function agen_asnap_plan(kernel, active_map, sites; ii_plan = nothing)
    kinds = kernel.sig.kinds
    bodies = Dict{Any,Vector{Symbol}}()
    out_sites = copy(sites)
    agen_asnap_walk!(kernel.body, kinds, Set(kernel.sig.args), active_map, sites, out_sites, bodies, ii_plan, Set{Symbol}(), false, false)
    arrays = sort(unique(vcat(Symbol[], values(bodies)...)); by = string)
    return (out_sites, (bodies = bodies, arrays = arrays))
end

# `in_block`: L or an ancestor is a Tier B ragged-block ancestor, whose layout uses block-local offsets (G9).
# `in_ii`: L or an ancestor is classified by ii_plan, and its body is emitted by a different path (G7).
function agen_asnap_walk!(body, kinds, args, active_map, sites, out_sites, bodies, ii_plan, handled::Set{Symbol}, in_block::Bool, in_ii::Bool)
    for (idx, st) in enumerate(body)
        if st.kind == :for
            ii_here = in_ii || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), nothing) !== nothing)
            block_here = in_block || agen_tier_b_walk(st.body, agen_collect_reassigned(st.body, true)) !== nothing
            newly = Symbol[]
            if !block_here && !ii_here
                B = st.body
                follow = agen_asnap_follow_vars(B, st.var, kinds)
                indirect = agen_asnap_indirect_vars(B, kinds)
                for A in sort(collect(agen_asnap_written_arrays!(B, Set{Symbol}(), kinds)); by = string)
                    (A in args && A ∉ handled && get(active_map, A, false)) || continue
                    (haskey(kinds, Symbol(string(A) * "_snap")) || haskey(kinds, agen_asnap_len(A))) && continue
                    wkeys = agen_asnap_write_keys!(B, A, Any[])
                    any(k -> get(sites, k, false) === true, wkeys) || continue
                    any(k -> get(sites, k, false) === true, agen_asnap_scatter_keys!(B, A, indirect, kinds, Any[])) || continue
                    agen_asnap_write_in_ii_loop(B, A, ii_plan, kinds) && continue
                    agen_asnap_final_read_ok(B, A, kinds) || continue
                    agen_asnap_swept_write(B, A, false) || continue
                    agen_asnap_index_free(B, A, follow) || continue
                    push!(newly, A)
                    for k in wkeys
                        out_sites[k] = false
                    end
                    out_sites[agen_site_key(B, 0, A)] = true
                end
                isempty(newly) || (bodies[agen_site_key(B, 0)] = newly)
            end
            agen_asnap_walk!(st.body, kinds, args, active_map, sites, out_sites, bodies, ii_plan, union(handled, Set(newly)), block_here, ii_here)
        elseif st.kind == :if
            agen_asnap_walk!(st.then, kinds, args, active_map, sites, out_sites, bodies, ii_plan, handled, in_block, in_ii)
            agen_asnap_walk!(st.els, kinds, args, active_map, sites, out_sites, bodies, ii_plan, handled, in_block, in_ii)
        end
    end
    return nothing
end

# Forward head and reverse tail of a snapshot. The copy is a plain loop over the linear index of A, whatever its rank. Slot n of the
# stack holds iteration n, so the base is (position - 1) * length(A).
function agen_asnap_save(body, kinds, ectx)
    exprs = Any[]
    ectx.keep_push_pop && return exprs
    for A in agen_asnap_boundary_arrays(body, kinds, ectx.push_pop)
        stack = agen_site_stack_name(agen_asnap_site(A))
        len = agen_asnap_len(A)
        base = Symbol("__sbase_", A)
        k = Symbol("__sk_", A)
        idx = agen_site_index(exprs, ectx, agen_site_key(body, 0, A))
        push!(exprs, Expr(:(=), base, Expr(:call, :*, Expr(:call, :-, idx, 1), len)))
        push!(exprs, emit_forloop(k, 1, len, 1, Any[Expr(:(=), Expr(:ref, stack, Expr(:call, :+, base, k)), Expr(:ref, A, k))]))
    end
    return exprs
end

function agen_asnap_restore(body, kinds, ectx)
    exprs = Any[]
    ectx.keep_push_pop && return exprs
    for A in agen_asnap_boundary_arrays(body, kinds, ectx.push_pop)
        stack = agen_site_stack_name(agen_asnap_site(A))
        len = agen_asnap_len(A)
        base = Symbol("__sbase_", A)
        k = Symbol("__sk_", A)
        idx = agen_site_index(exprs, ectx, agen_site_key(body, 0, A))
        push!(exprs, Expr(:(=), base, Expr(:call, :*, Expr(:call, :-, idx, 1), len)))
        push!(exprs, emit_forloop(k, 1, len, 1, Any[Expr(:(=), Expr(:ref, A, k), Expr(:ref, stack, Expr(:call, :+, base, k)))]))
    end
    return exprs
end

function agen_ii_force_no_snapshot!(body, vn_local, override)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            var in vn_local && (override[agen_site_key(body, idx)] = false)
        elseif stmt.kind == :for
            agen_ii_force_no_snapshot!(stmt.body, vn_local, override)
        elseif stmt.kind == :while
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
    return (keep_push_pop = ectx.keep_push_pop, loop_ctx = ectx.loop_ctx, layout = ectx.layout, push_pop = override, ii_plan = ectx.ii_plan, remat = ectx.remat)
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
    exprs = agen_asnap_save(body, kinds, ectx)
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
            # a restored integer: the old value goes onto its stack before every write, as a float's does
            haskey(stacks, (:ivalue, var)) &&
                agen_emit_push!(exprs, stacks[(:ivalue, var)], stmt.lhs, ectx, agen_site_key(body, idx))
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
                push!(ectx.loop_ctx, agen_frame(stmt, ectx.layout))
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
                push!(ectx.loop_ctx, agen_frame(stmt, ectx.layout))
                inner = agen_forward_body(stmt.body, kinds, active_map, loop_value_needed, reassigned, stacks, exempt;
                                           ectx = loop_ectx,
                                           lin_body = lin_body === nothing ? nothing : lin_body[idx].body,
                                           unsafe = unsafe)
                pop!(ectx.loop_ctx)
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, inner))
            end
        elseif stmt.kind == :while
            # The whole feature in one place. A `while` states no trip count,
            # so the forward sweep counts its own iterations and pushes the
            # total; the backward sweep pops it and replays the body that
            # many times as a plain `for`.
            #
            # The counter is per site, so two sequential loops and a nested
            # pair never share one. The push is AFTER the loop, because the
            # total is not known until it ends. That is the reason
            # :whilecount does not share :tripcount's stack, whose pushes
            # happen before their loops.
            cnt = agen_whilecount_var(body, idx)
            push!(exprs, Expr(:(=), cnt, 0))
            inner = Any[]
            # Pushed at the TOP of the body, before the body updates them, so
            # the value recorded for an iteration is the one that iteration
            # actually used. Pushing after the update would record the NEXT
            # iteration's value.
            carried = sort(collect(snap_while_carried_ints(stmt.body, kinds, agen_itbr_vars(stacks))))
            for cv in carried
                agen_emit_push!(inner, stacks[(:whilecarry, cv)], cv, ectx, agen_site_key(body, idx, cv))
            end
            append!(inner, agen_forward_body(stmt.body, kinds, active_map, value_needed, reassigned, stacks, exempt;
                                       ectx = ectx,
                                       lin_body = lin_body === nothing ? nothing : lin_body[idx].body,
                                       unsafe = unsafe))
            push!(inner, Expr(:(=), cnt, Expr(:call, :+, cnt, 1)))
            push!(exprs, emit_while(stmt.cond, inner))
            agen_emit_push!(exprs, stacks[(:whilecount, :count)], cnt, ectx, agen_site_key(body, idx))
        elseif stmt.kind == :if
            nm = stacks[(:branch, :cond)]
            key = agen_site_key(body, idx)
            # Built as a fresh list, not the previous vcat one-liner, so
            # agen_emit_push! has this branch's own statement list to hoist a
            # ragged index assignment into, ahead of the branch-flag push --
            # hoisting into the outer `exprs` would be wrong, since it isn't
            # guarded by `stmt.cond`.
            # Under keep_push_pop = true the flag goes AFTER the arm's body. The backward sweep pops this flag first, then
            # replays the arm, so a flag pushed by a nested `if` has to sit below the outer one on the shared stack. A
            # flag pushed at the top of the arm came out in the wrong order for every `if` inside an `if`. Indexed mode
            # addresses each flag by its own key, so its emission stays as it was.
            then_exprs = Any[]
            els_exprs = Any[]
            if ectx.keep_push_pop
                append!(then_exprs, agen_forward_body(stmt.then, kinds, active_map, value_needed, reassigned, stacks, exempt;
                                                        ectx = ectx,
                                                        lin_body = lin_body === nothing ? nothing : lin_body[idx].then,
                                                        unsafe = unsafe))
                agen_emit_push!(then_exprs, nm, 1, ectx, key)
                append!(els_exprs, agen_forward_body(stmt.els, kinds, active_map, value_needed, reassigned, stacks, exempt;
                                                       ectx = ectx,
                                                       lin_body = lin_body === nothing ? nothing : lin_body[idx].els,
                                                       unsafe = unsafe))
                agen_emit_push!(els_exprs, nm, 0, ectx, key)
            else
                agen_emit_push!(then_exprs, nm, 1, ectx, key)
                append!(then_exprs, agen_forward_body(stmt.then, kinds, active_map, value_needed, reassigned, stacks, exempt;
                                                        ectx = ectx,
                                                        lin_body = lin_body === nothing ? nothing : lin_body[idx].then,
                                                        unsafe = unsafe))
                agen_emit_push!(els_exprs, nm, 0, ectx, key)
                append!(els_exprs, agen_forward_body(stmt.els, kinds, active_map, value_needed, reassigned, stacks, exempt;
                                                       ectx = ectx,
                                                       lin_body = lin_body === nothing ? nothing : lin_body[idx].els,
                                                       unsafe = unsafe))
            end
            push!(exprs, emit_if(stmt.cond, then_exprs, els_exprs))
        end
    end
    # block-boundary restoration (see agen_block_boundary_vars above):
    # snapshot each such var's value once here, at the end of `body`,
    # capturing whatever this ENCLOSING iteration's own nested writes
    # left it at -- restored symmetrically at the start of this same
    # body's own backward processing in agen_backward_body.
    for var in agen_block_boundary_vars(body, kinds, value_needed, exempt, stacks; ii_plan = ectx.ii_plan, cross = agen_remat_cross_vars(ectx.remat))
        agen_emit_push!(exprs, stacks[(:value, var)], var, ectx, agen_site_key(body, 0, var))
    end
    return exprs
end

# an active lhs is always scalar_float or array_float -- :ref means
# the array kind (:array), a bare Symbol means the scalar kind
# (:value), matching how snap_plan classified the same write
agen_snapshot_kind(lhs) = lhs isa Symbol ? :value : :array

# ---- backward sweep (walks lin_plan, whose :for/:if fields mirror
#      the primal's own exactly -- only :assign carries a built tree) -

function agen_backward_body(plan, primal_body, kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = agen_ectx_stack())
    exprs = Any[]
    # Block-boundary restoration: restore each such var here, at the very start of this
    # body's own backward processing, from the matching push agen_forward_body emitted at
    # the end of this same body. Must run before anything else below, since this is what
    # makes this body's own nested-loop-written value visible again instead of a later
    # sibling iteration's.
    for var in agen_block_boundary_vars(primal_body, kinds, value_needed, exempt, stacks; ii_plan = ectx.ii_plan, cross = agen_remat_cross_vars(ectx.remat))
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
    # end-of-iteration values of the scalars whose first-write save was dropped -- see agen_remat_cross!
    head = ectx.remat === nothing ? nothing : get(ectx.remat, agen_site_key(primal_body, 0), nothing)
    head === nothing || append!(exprs, head.args)
    for idx in length(plan):-1:1
        stmt = plan[idx]
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            if haskey(stacks, (:ivalue, var))
                # reverse position of a restored integer's write: the value it held before the write
                push!(exprs, Expr(:(=), stmt.lhs, agen_emit_pop(stacks[(:ivalue, var)], ectx, agen_site_key(primal_body, idx), exprs)))
                continue
            end
            kinds[var] in (:scalar_int, :array_int) && continue   # hoisted above, or unsafe (skipped entirely)
            append!(exprs, agen_backward_assign(stmt, kinds, active_map, value_needed, reassigned, stacks, exempt; ectx = ectx, key = agen_site_key(primal_body, idx)))
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
                push!(ectx.loop_ctx, agen_frame(stmt, ectx.layout, primal_body[idx]))
                inner = agen_backward_body(stmt.body, primal_body[idx].body, kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = ectx)
                pop!(ectx.loop_ctx)
                # Every backward loop reverses. A loop that carries a value from one iteration to
                # the next must reverse to be correct, and STADE cannot see that coupling when it
                # runs through an array with no snapshot. An independent loop computes the same
                # adjoint either direction, so reversing it costs nothing. Over-reversal is always
                # right, under-reversal a silent wrong gradient -- not worth an analysis.
                loop_expr = emit_forloop(stmt.var, agen_reverse_start(stmt.lo, stmt.step, stmt.hi), stmt.lo, agen_negate_step(stmt.step), inner)
                # LIFO: the forward sweep pushed the bounds in sorted order, so pop them in reverse
                for bv in reverse(agen_tripcount_bound_vars(stmt, reassigned))
                    push!(exprs, Expr(:(=), bv, agen_emit_pop(stacks[(:tripcount, bv)], ectx, agen_site_key(primal_body, idx, bv), exprs)))
                end
                push!(exprs, loop_expr)
            end
        elseif stmt.kind == :while
            # Pop the count the forward sweep pushed, then replay the body
            # that many times as a plain `for`. The condition is never
            # re-evaluated: by now the values it read have been overwritten
            # by the reverse walk, and the number of iterations is a
            # recorded fact rather than something to recompute.
            #
            # The loop runs 1:n rather than n:-1:1 only because the counter
            # is not an index. The BODY is still differentiated in reverse
            # by agen_backward_body, which is what the reversal has to
            # achieve.
            cnt = agen_whilecount_var(primal_body, idx)
            push!(exprs, Expr(:(=), cnt,
                              agen_emit_pop(stacks[(:whilecount, :count)], ectx,
                                            agen_site_key(primal_body, idx), exprs)))
            inner = Any[]
            # LIFO: the replay runs the iterations in reverse, so each one
            # pops the value its own forward iteration pushed. Popped BEFORE
            # the body's adjoint, which reads them as indices.
            # primal_body, not stmt.body: stmt comes from the lin plan, whose
            # assign nodes carry .tree rather than .rhs
            for cv in sort(collect(snap_while_carried_ints(primal_body[idx].body, kinds, agen_itbr_vars(stacks))))
                push!(inner, Expr(:(=), cv,
                                  agen_emit_pop(stacks[(:whilecarry, cv)], ectx,
                                                agen_site_key(primal_body, idx, cv), inner)))
            end
            append!(inner, agen_backward_body(stmt.body, primal_body[idx].body, kinds, active_map, unsafe,
                                        value_needed, reassigned, stacks, exempt; ectx = ectx))
            push!(exprs, emit_forloop(agen_whilecount_iter_var(primal_body, idx), 1, cnt, 1, inner))
        elseif stmt.kind == :if
            then_exprs = agen_backward_body(stmt.then, primal_body[idx].then, kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = ectx)
            els_exprs = agen_backward_body(stmt.els, primal_body[idx].els, kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = ectx)
            nm = stacks[(:branch, :cond)]
            push!(exprs, Expr(:(=), :__branch, agen_emit_pop(nm, ectx, agen_site_key(primal_body, idx), exprs)))
            push!(exprs, emit_if(Expr(:call, :(==), :__branch, 1), then_exprs, els_exprs))
        end
    end
    append!(exprs, agen_asnap_restore(primal_body, kinds, ectx))
    return exprs
end

function agen_backward_assign(stmt, kinds, active_map, value_needed, reassigned, stacks, exempt; ectx = agen_ectx_stack(), key = nothing)
    var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
    is_accum = agen_is_pure_accumulation(stmt.lhs, stmt.tree.expr, var)
    exprs = Any[]
    # int-kinded lhs can't own a shadow, or a pop, at all
    if kinds[var] in (:scalar_float, :array_float)
        # Restore this write's overwritten old value whenever active_map[var] does, matching
        # agen_forward_body's push gate -- not stmt.active, since a write can destroy a value
        # another statement's nonlinear derivative still needs even when this write's own rhs is a
        # plain inactive literal. `exempt` mirrors the forward sweep's own skips -- no
        # push happened for those, so there's nothing to pop.
        remat_rhs = (ectx.remat === nothing || key === nothing) ? nothing : get(ectx.remat, key, nothing)
        if remat_rhs !== nothing
            # rematerialized: the old value is an earlier pure load in this block, recomputed instead of popped
            (remat_rhs isa Expr && remat_rhs.head == :cross) || push!(exprs, Expr(:(=), stmt.lhs, remat_rhs))
        elseif get(active_map, var, false) && agen_needs_snapshot(stmt.lhs, stmt.tree.expr, var, agen_push_pop_source(value_needed, ectx), key) && !(var in exempt)
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
                # Read the incoming adjoint into a scalar, consume the output slot, THEN distribute.
                # In that order a leaf reading the lhs's own slot needs no special case: it
                # accumulates onto the zero this just wrote, giving `lhsb = partial * seed`, which is
                # what the self_terms path used to build by hand. It also stays correct when the leaf
                # only aliases the lhs at RUNTIME -- `y[i_r] = y[i_k] * x[i_r]` with a gathered `i_k`
                # -- which the syntactic self_expr match could never see, and which produced a
                # silently wrong adjoint whenever i_k == i_r.
                seed = Symbol("__oldb_", length(exprs))
                push!(exprs, Expr(:(=), seed, lhsb))
                push!(exprs, Expr(:(=), lhsb, 0.0))
                agen_distribute!(stmt.tree, seed, exprs)
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
# whose slot differs from `skip_expr`. `skip_expr` is honored only against a direct child of the node it's passed to -- the
# top-level slot a pure accumulation's lhs occupies.
#
# There used to be a second exemption here, `self_expr`/`self_terms`, which diverted a leaf reading the statement's own lhs
# into a list the caller summed into `lhsb` by hand. It matched syntactically, so it saw `y[i] = y[i] * x` but not
# `y[i_r] = y[i_k] * x` with a gathered `i_k`, and the latter got the ordinary accumulate followed by `yb[i_r] = 0.0`, which
# discarded the contribution whenever the two aliased at runtime. agen_backward_assign now zeroes the slot BEFORE
# distributing, so a self-reading leaf accumulates onto that zero and needs no exemption at all.
function agen_distribute!(node, seed, exprs; skip_expr = nothing)
    node.active || return nothing
    if node.kind == :leaf
        skip_expr !== nothing && node.expr == skip_expr && return nothing
        target = agen_shadow(node.expr)
        push!(exprs, Expr(:(=), target, Expr(:call, :+, target, seed)))
        return nothing
    end
    contributions = der_adjoint_generic(node.op, node.args, seed)
    for (child, contrib) in zip(node.children, contributions)
        skip_expr !== nothing && child.expr == skip_expr && continue
        child.active || continue
        agen_distribute!(child, contrib, exprs)
    end
    return nothing
end

# ---- local shadow initialization -----------------------------------

# Every local (non-argument) scalar needs to already exist before the forward sweep runs --
# normally its first primal assignment establishes that, but a local flagged for
# snapshotting can get pushed (reading its pre-overwrite value) as early as the first loop
# iteration, before any primal assignment. Without this, that first push would be an
# UndefVarError. Harmless for locals that don't need it.
function agen_local_primal_inits(kernel, active_map, itbr = Set{Symbol}())
    arg_set = Set(kernel.sig.args)
    exprs = Any[]
    for v in sort(collect(keys(kernel.sig.kinds)))
        if !(v in arg_set) && kernel.sig.kinds[v] == :scalar_float && active_map[v]
            push!(exprs, Expr(:(=), v, 0.0))
        end
        # a restored integer is pushed before its first write too, so it must exist by then
        if !(v in arg_set) && v in itbr
            push!(exprs, Expr(:(=), v, 0))
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
                   tier_b_extra_args::Vector{Symbol} = Symbol[], ii_plan = nothing, remat = nothing)
    sig = kernel.sig
    stacks = agen_stack_map(sites)
    value_needed = agen_value_needed_vars(kernel)
    reassigned = agen_collect_reassigned_all(kernel.body)
    unsafe = agen_unsafe_int_vars(kernel, agen_itbr_vars(stacks))
    exempt = agen_exempt_vars(kernel, value_needed)

    ectx = (keep_push_pop = keep_push_pop, loop_ctx = Any[], layout = layout, push_pop = push_pop, ii_plan = ii_plan, remat = remat)

    cse_counter = Ref(0)
    cse_floats = agen_cse_float_names(sig, sites)
    fwd = emit_cse_stmts(agen_forward_body(kernel.body, sig.kinds, active_map, value_needed, reassigned, stacks, exempt; ectx = ectx, lin_body = lin_plan, unsafe = unsafe), cse_counter, cse_floats)
    bwd = emit_cse_stmts(agen_backward_body(lin_plan, kernel.body, sig.kinds, active_map, unsafe, value_needed, reassigned, stacks, exempt; ectx = ectx), cse_counter, cse_floats)

    shadow_of = hvp_shadow_map(kernel, sites)
    hvp_register_temp_shadows!(shadow_of, fwd)
    hvp_register_temp_shadows!(shadow_of, bwd)

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
    append!(body, agen_local_primal_inits(kernel, active_map, agen_itbr_vars(stacks)))
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

    return Expr(:function, Expr(:call, fname, fargs...), Expr(:block, emit_cse_stmts(body, Ref(0), cse_floats; prefix = "__hcse_", int_prefix = "__ihcse_")...))
end

# Temporaries invented after shape inference -- emit_cse_stmts' bound subexpressions and
# agen_backward_assign's captured incoming adjoints -- are locals kernel.sig.kinds has never heard
# of, so hvp_shadow_map cannot see them. Left unregistered, hvp_tangent_expr
# resolves each one to the literal 0.0 and hvp_double_stmt emits no shadow line, silently dropping
# every second-order term that flows through a bound subexpression -- the corpus catches this on
# some kernels but not all. Registered in program order, so a temporary defined from an earlier one
# sees it. A temporary whose own tangent is 0.0 (a pure index computation) needs no shadow, and
# giving it one would only emit a dead `__cse_Nd = 0.0`.
function hvp_register_temp_shadows!(shadow_of, stmts)
    for e in stmts
        if e isa Expr && e.head == :(=) && e.args[1] isa Symbol &&
           (startswith(string(e.args[1]), "__cse_") || startswith(string(e.args[1]), "__oldb_"))
            any(haskey(shadow_of, v) for v in emit_cse_reads!(e.args[2], Set{Symbol}())) || continue
            hvp_tangent_expr(e.args[2], shadow_of) == 0.0 && continue
            shadow_of[e.args[1]] = tgen_shadow(e.args[1])
        elseif e isa Expr
            hvp_register_temp_shadows!(shadow_of, e.args)
        end
    end
    return nothing
end

hvp_fname(name::Symbol) = Symbol(string(name) * "_hv")
hvp_stack_shadow(stack::Symbol) = Symbol(string(stack) * "_d")

# Mirrors agen_drop_unused_stack_allocs, but for hvp_emit's OWN body: hvp_shadow_stack_inits
# unconditionally allocates a `<stack>_d` shadow for every site, even one whose primal stack
# turns out unused in this hvp_expr (fusion eliminated its only push/pop/ref). Once `nm` is
# confirmed unused, its `nm_d` shadow alloc is dead too -- nothing else in hvp_expr can
# reference it, since hvp_double_body only doubles statements that already exist.
function hvp_drop_unused_shadow_stack_allocs(hvp_expr, unused)
    shadow_unused = Set(hvp_stack_shadow(nm) for nm in unused)
    body_block = hvp_expr.args[2]
    new_stmts = Any[]
    for stmt in body_block.args
        if stmt isa Expr && stmt.head == :(=) && stmt.args[1] isa Symbol && stmt.args[1] in shadow_unused
            continue
        elseif stmt isa Expr && stmt.head == :call && stmt.args[1] == :sizehint! && stmt.args[2] isa Symbol && stmt.args[2] in shadow_unused
            continue
        else
            push!(new_stmts, stmt)
        end
    end
    return Expr(:function, hvp_expr.args[1], Expr(:block, new_stmts...))
end

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
    elseif e.head == :while
        # The adjoint's forward sweep replays the primal, so a `while`
        # survives into generated code even though the backward sweep is a
        # plain `for`. The condition is not doubled: it reads primal values
        # and selects iterations, and a shadow of it would mean nothing.
        inner = hvp_double_body(e.args[2].args, shadow_of)
        return Any[Expr(:while, e.args[1], Expr(:block, inner...))]
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
# CUDA codegen: turns a validated kernel, or a STADE-generated function, into a host launcher plus one `@cuda` device kernel per data-parallel loop
# -- proved from the loop's own structure via cgen_reduction_only_loop, never declared by the source. Ingests via cgen_from_kernel (plain skill-
# stade) or cgen_parse_generated (STADE's own vocabulary). Stack safety: a loop with push!/pop! anywhere is never split, since LIFO order can't
# survive concurrent threads. Race safety: a split write is atomic-free only if the thread var occurs in its index, else CUDA.@atomic.

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
    elseif stmt.head == :while
        # The adjoint's forward sweep replays the primal, so a `while`
        # reaches this parser even though the backward sweep is a plain
        # `for`. cgen_body keeps it on the host: no launch can size itself
        # around a loop whose trip count is unknown until it has run.
        length(stmt.args) == 2 && stmt.args[2] isa Expr && stmt.args[2].head == :block ||
            error("cgen_parse_generated: malformed `while` `$(stmt)`")
        return (kind = :while, cond = stmt.args[1],
                body = cgen_parse_generated_statements(parse_strip_lines(stmt.args[2])))
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
    if cgen_is_pop_call(rhs) || cgen_is_stack_alloc(rhs) || cgen_is_length_call(rhs)
        return (kind = :assign, lhs = lhs, rhs = rhs)
    end
    parse_check_expr(rhs, false)
    return (kind = :assign, lhs = lhs, rhs = rhs)
end

cgen_is_pop_call(rhs) = rhs isa Expr && rhs.head == :call && length(rhs.args) == 2 && rhs.args[1] == :pop!

# `__len_A = length(A)`: initstacks_* of an array snapshot reads the length of an array argument once, to size the snapshot stack
cgen_is_length_call(rhs) = rhs isa Expr && rhs.head == :call && length(rhs.args) == 2 && rhs.args[1] == :length && rhs.args[2] isa Symbol

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
    body_stmts = [s for s in stmt.args[2].args if !(s isa LineNumberNode)]
    body = cgen_parse_generated_statements(body_stmts)
    return (kind = :for, var = var, lo = lo, hi = hi, step = step, body = body)
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
            cgen_collect_scalar_reductions!(stmt.body, names)
        end
    end
    return nothing
end

# ---- reduction-eligibility check ------------------------------------
# A loop that carries a value between iterations is normally kept host-side, but is split when its only cross-iteration coupling is a commutative
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
        elseif stmt.kind == :while
            cgen_collect_all_assigned!(stmt.body, names)
        end
    end
end

# ---- array-privacy proof: is a written array confined to a private slice? ----
# The pairwise write/read-index check below treats every access to an array as one undifferentiated set, so it can't tell disjoint per-
# sub-loop slices assembled into a whole apart from a genuine value recurrence. This adds a narrower, additive proof for exactly that
# safe shape, scoped per sub-loop: an array is accepted only when every access decomposes into an outer-invariant base plus an inner
# loop's additive index, matching one of two provably-safe patterns. Anything else falls back to the pairwise check, unchanged.

# Canonicalizes an additive/subtractive expression into a sign-tagged multiset of leaf terms --
# `+`/`-` flatten at any depth, and a literal-integer multiplier expands into repeated leaves (`2*n`
# becomes two copies of `n`), so two differently-shaped sums can be compared on equal footing once
# definitions are substituted in. Bounded to a sane literal range so a stray large constant can't blow
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

# Collects every occurrence of `arr` within one top-level statement of the candidate loop, at any nesting depth, tagged
# with the full chain of enclosing loop vars paired with their own `.hi` (never just the name -- sibling sub-loops
# routinely reuse the same loop-variable name with different trip counts, so a name-keyed table would silently collide).
# Bails (`nothing`) the moment `arr` is touched inside an `:if`, or a for-loop whose bounds aren't a literal `1:hi` step-1
# range, since the additive stripping below assumes that shape.
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
            # `1:hi` and its mirror `hi:-1:1` touch the identical index set, so the additive
            # stripping below is the same for both. Only the ascending form used to be accepted,
            # which refused every backward sweep -- an adjoint reverses each loop, so no
            # sub-loop of a backward-sweep body is ever ascending.
            up = stmt.lo isa Integer && stmt.lo == 1 && stmt.step isa Integer && stmt.step == 1
            down = stmt.hi isa Integer && stmt.hi == 1 && stmt.step isa Integer && stmt.step == -1
            (up || down) || return nothing
            cgen_deep_array_occurrences!(stmt.body, arr, vcat(chain, [(stmt.var, up ? stmt.hi : stmt.lo)]), out) === nothing && return nothing
        elseif stmt.kind == :while
            # A while contributes no index variable, so the chain of
            # (loopvar, bound) pairs to strip is unchanged.
            cgen_deep_array_occurrences!(stmt.body, arr, chain, out) === nothing && return nothing
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

# The proof itself. Groups every occurrence of `arr` by which top-level statement it came from, requiring each group's write/read indices
# to agree internally (after stripping any local sub-loop variable). Accepts the array when no group is a pure reader (each a self-
# contained accumulation, so cross-group base coincidence is cgen_device_assign's job), or when every group shares the same (base,trip),
# or a cumulative-assembly pattern where one reading group's range provably matches a running sum of write-only groups' trips from the
# same base. Anything uncovered returns `nothing`, falling back to the pairwise check.
function cgen_elide_snapshot_saves(stmt, arr::Symbol)
    if stmt.kind == :for
        return (kind = :for, var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step,
                body = cgen_elide_snapshot_saves_body(stmt.body, arr))
    elseif stmt.kind == :while
        return (kind = :while, cond = stmt.cond,
                body = cgen_elide_snapshot_saves_body(stmt.body, arr))
    elseif stmt.kind == :if
        return (kind = :if, cond = stmt.cond,
                then = cgen_elide_snapshot_saves_body(stmt.then, arr),
                els = cgen_elide_snapshot_saves_body(stmt.els, arr))
    end
    return stmt
end

function cgen_elide_snapshot_saves_body(body::Vector{NamedTuple}, arr::Symbol)
    out = NamedTuple[]
    for (i, st) in enumerate(body)
        if cgen_is_snapshot_save(st, arr) && cgen_snapshot_save_dead(body, i, arr, st.rhs.args[2])
            push!(out, (kind = :assign, lhs = st.lhs, rhs = 0.0))
        else
            push!(out, cgen_elide_snapshot_saves(st, arr))
        end
    end
    return out
end

# The save's read contributes nothing to `arr`'s access region when the same element is overwritten later in this
# statement list with nothing in between touching `arr`: it's dead on arrival, and the covering write is already an
# occurrence of its own. An adjacency test instead of this scan is correct for an adjoint and wrong for an HVP, where
# hvp_double_stmt interleaves each statement's shadow with its primal, putting the overwrite two statements after its
# own save -- mpnn is the witness: its adjoint offloads every loop, its HVP left two on the host.
function cgen_snapshot_save_dead(body::Vector{NamedTuple}, i::Int, arr::Symbol, idx)
    for j in (i + 1):length(body)
        cgen_overwrites_same_element(body[j], arr, idx) && return true
        cgen_stmt_touches_array(body[j], arr) && return false
    end
    return false
end

cgen_stmt_touches_array(st, arr::Symbol) = begin
    writes = Dict{Any,Vector{Any}}(); reads = Dict{Any,Vector{Any}}()
    cgen_collect_array_accesses!(NamedTuple[st], writes, reads)
    haskey(writes, arr) || haskey(reads, arr)
end

cgen_is_snapshot_save(st, arr::Symbol) =
    st.kind == :assign &&
    st.lhs isa Expr && st.lhs.head == :ref && length(st.lhs.args) == 2 &&
    st.lhs.args[1] isa Symbol && st.lhs.args[1] != arr &&
    st.rhs isa Expr && st.rhs.head == :ref && length(st.rhs.args) == 2 && st.rhs.args[1] == arr

cgen_overwrites_same_element(nxt, arr::Symbol, idx) =
    nxt.kind == :assign &&
    nxt.lhs isa Expr && nxt.lhs.head == :ref && length(nxt.lhs.args) == 2 &&
    nxt.lhs.args[1] == arr && nxt.lhs.args[2] == idx

function cgen_array_private_to_loop(body::Vector{NamedTuple}, arr::Symbol, defs::Dict{Symbol,Any})
    occ = Tuple{Symbol,Any,Any,Int}[]   # (kind, base, trip_or_nothing, group_id)
    for (gi, stmt0) in enumerate(body)
        stmt = cgen_elide_snapshot_saves(stmt0, arr)
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
    # The tiling argument below is about regions, not program order. A backward sweep is the
    # forward sweep mirrored, so the consumer runs first and the assembly groups after, in
    # reverse order -- requiring writers-before-reader refused every such loop. Order must
    # still be uniform: writers on both sides of the reader tile nothing.
    writers_after = !isempty(writers) && all(w -> w[4] > R[4], writers)
    (writers_after || all(w -> w[4] < R[4], writers)) || return nothing
    sort!(writers, by = w -> w[4])
    writers_after && reverse!(writers)
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

# ---- two multi-dimensional relaxations of the pairwise fallback ----------
# Both exist because the additive-decomposition proof above reads a 1-D linear index only
# (cgen_deep_array_occurrences! bails on any other rank), so a kernel written with genuine 2-D
# fields gets no proof at all and falls straight to the pairwise check.

# Every access to this array pins one fixed dimension to the candidate loop's own variable, so
# iteration i touches only slice [.., i, ..] and two iterations never overlap. This is the multi-
# dimensional counterpart of an index that reduces to `base + i`: `skx[i_loc, i_cell]` inside `for
# i_cell` is private for the same reason as `x[off + k]`. Sound because a split loop runs its whole
# body in one thread, so only cross-iteration overlap has to be ruled out.
function cgen_array_sliced_by_loopvar(widxs, ridxs, loopvar::Symbol)
    idxs = vcat(collect(widxs), collect(ridxs))
    isempty(idxs) && return false
    n = length(first(idxs))
    n >= 2 && all(v -> length(v) == n, idxs) || return false
    return any(d -> all(v -> v[d] === loopvar, idxs), 1:n)
end

# Two index vectors that disagree at a position where both are integer literals address
# different memory whatever the rest holds -- `cb[1, i_node]` and `cb[2, i_node]` are separate
# components of one vector field and can never alias. Without this the pairwise check reads a
# component mismatch as a value recurrence and refuses the loop.
function cgen_indices_cannot_alias(a, b)
    length(a) == length(b) || return false
    return any(i -> a[i] isa Integer && b[i] isa Integer && a[i] != b[i], eachindex(a))
end

# The loop-split gates share three parts: the scalar part below, a per-array predicate, and the loop summary that feeds both.
# cgen_reduction_only_loop composes the scalar part with the strict array predicate. It is also the proof that snap_ii_classify and
# agen_ii_classify use to reorder a loop, so what it accepts must not change with the GPU stage.
function cgen_loop_scalar_synth(body::Vector{NamedTuple}, known_consts::Dict{Symbol,Any})
    local_names = cgen_locally_assigned_scalars(body)
# A locally-assigned scalar that fails the plain use-before-def walk below may still be safe if it provably converges to
# the same literal on every control-flow path, PROVIDED that constant matches the value known coming into the loop
# (known_consts, from a top-level literal assignment immediately preceding this loop). By induction every iteration then
# starts at that value, so injecting it as the kernel body's first statement reproduces sequential semantics. Without a
# matching known_consts entry, the loop stays refused.
    synth = Dict{Symbol,Any}()
    for v in local_names
        haskey(known_consts, v) || continue
        c = cgen_loop_convergent_constant(body, v, get(known_consts, v, nothing))
        c !== nothing && c == known_consts[v] && (synth[v] = c)
    end
    cgen_use_before_def(body, local_names, synth)[1] || return nothing
    return synth
end

# ---- save-only scalars: a local whose only read before its own definition is a stack save ----
# `X_stack[i] = v` stores v's current value for the backward sweep. In a tile loop the first save of an iteration stores the value
# v held on ENTRY, which a split thread does not have (the sequential loop passed it from the previous iteration). That slot is
# dead exactly when nothing reads v's pre-definition value, which takes four structural facts, all checked below:
#   1. every other read of v inside the loop is preceded by an assignment (cgen_use_before_def with save_ok);
#   2. the loop body ends by pushing v's final value after its last write, so its reverse iteration starts from its own slot
#      and never from the restoration chain the entry saves feed;
#   3. no statement after the loop reads v before redefining it (ii_escapes_nested with save_ok);
#   4. every non-save read of v anywhere in the program sits inside a loop that writes v and is covered there, so the value
#      the reverse sweep restores past the loop is never read.
# Fact 2 is what separates a per-tile loop (split) from the tap loop inside it (never split): a tap iteration's entry save
# carries the previous tap's value, which the reverse of that previous tap reads.
# `<var>_stack`, and `<var>_stack_d` for the stack shadow an HVP adds beside it (hvp_stack_shadow)
cgen_is_stack_name(a) = a isa Symbol && (endswith(string(a), "_stack") || endswith(string(a), "_stack_d"))
cgen_is_save_stmt(stmt) = stmt.kind == :assign && stmt.rhs isa Symbol && stmt.lhs isa Expr && stmt.lhs.head == :ref &&
    cgen_is_stack_name(stmt.lhs.args[1])

function cgen_ends_with_push(body::Vector{NamedTuple}, v::Symbol)
    lw = findlast(st -> cgen_writes_var([st], v), body)
    lw === nothing && return false
    sv = string(v)
    stacks = Symbol[Symbol(sv * "_stack")]
    endswith(sv, "d") && push!(stacks, Symbol(chop(sv) * "_stack_d"))
    return any(j -> cgen_is_save_stmt(body[j]) && body[j].rhs === v && body[j].lhs.args[1] in stacks, lw+1:length(body))
end

# Fact 4. `in_writer` is true inside a loop that writes v, where coverage was already proven for the outermost such loop.
function cgen_saveonly_closed(body::Vector{NamedTuple}, v::Symbol, in_writer::Bool = false)
    reads_v(e) = v in cgen_read_vars(e)
    for st in body
        if st.kind == :assign
            cgen_is_save_stmt(st) && continue
            if !in_writer
                reads_v(st.rhs) && return false
                st.lhs isa Expr && reads_v(st.lhs.args[2:end]) && return false
            end
        elseif st.kind == :if
            !in_writer && reads_v(st.cond) && return false
            cgen_saveonly_closed(st.then, v, in_writer) || return false
            cgen_saveonly_closed(st.els, v, in_writer) || return false
        elseif st.kind == :for || st.kind == :while
            if st.kind == :for
                !in_writer && (reads_v(st.lo) || reads_v(st.hi) || reads_v(st.step)) && return false
            else
                !in_writer && reads_v(st.cond) && return false
            end
            if !in_writer && cgen_writes_var(st.body, v)
                cgen_use_before_def(st.body, Set{Symbol}([v]), Dict{Symbol,Any}(); save_ok = true)[1] || return false
            elseif !cgen_saveonly_closed(st.body, v, in_writer)
                return false
            end
        else
            return false
        end
    end
    return true
end

# The scalars that need the relaxation, or nothing when the loop is refused anyway. `synth0` is the convergent-constant seed.
function cgen_saveonly_synth(body::Vector{NamedTuple}, known_consts::Dict{Symbol,Any}, root_body, protect)
    root_body === nothing && return nothing
    local_names = cgen_locally_assigned_scalars(body)
    synth = Dict{Symbol,Any}()
    for v in local_names
        haskey(known_consts, v) || continue
        c = cgen_loop_convergent_constant(body, v, get(known_consts, v, nothing))
        c !== nothing && c == known_consts[v] && (synth[v] = c)
    end
    cgen_use_before_def(body, local_names, synth; save_ok = true)[1] || return nothing
    saveonly = Symbol[v for v in sort(collect(local_names); by = string) if !haskey(synth, v) &&
        !cgen_use_before_def(body, Set{Symbol}([v]), synth)[1]]
    isempty(saveonly) && return nothing
    for v in saveonly
        v in protect && return nothing
        cgen_ends_with_push(body, v) || return nothing
        cgen_saveonly_closed(root_body, v) || return nothing
        synth[v] = 0.0
    end
    return (synth = synth, saveonly = Set{Symbol}(saveonly))
end

function cgen_loop_array_summary(body::Vector{NamedTuple}, outer_defs::Dict{Symbol,Any})
    writes = Dict{Any,Vector{Any}}()
    reads = Dict{Any,Vector{Any}}()
    cgen_collect_array_accesses!(body, writes, reads)
    return (writes = writes, reads = reads, defs = merge(outer_defs, cgen_scalar_def_map(body)))
end

# Deliberately not scoped to index expressions mentioning the loop's own variable: a sweep-style recurrence can carry its cross-iteration
# coupling through an inner loop's index, with no mention of the outer variable at all. A structural mismatch between a write and read index
# of the same array disqualifies the loop -- UNLESS cgen_array_private_to_loop can prove that array's whole access pattern is confined to a
# private, non-overlapping per-outer-iteration region. Only tried for a written array; a purely-read array never needs proving, since
# concurrent reads are always safe.
function cgen_array_ok_strict(body::Vector{NamedTuple}, arr, widxs, ridxs, loopvar::Symbol, defs::Dict{Symbol,Any})
    cgen_array_private_to_loop(body, arr, defs) === true && return true
    cgen_array_sliced_by_loopvar(widxs, ridxs, loopvar) && return true
    for w in widxs, r in ridxs
        w == r && continue
        cgen_indices_cannot_alias(w, r) && continue
        return false
    end
    return true
end

function cgen_reduction_only_loop(body::Vector{NamedTuple}, loopvar::Symbol, known_consts::Dict{Symbol,Any} = Dict{Symbol,Any}(), outer_defs::Dict{Symbol,Any} = Dict{Symbol,Any}())
    synth = cgen_loop_scalar_synth(body, known_consts)
    synth === nothing && return nothing
    summary = cgen_loop_array_summary(body, outer_defs)
    for (arr, widxs) in summary.writes
        ridxs = get(summary.reads, arr, Any[])
        cgen_array_ok_strict(body, arr, widxs, ridxs, loopvar, summary.defs) || return nothing
    end
    return synth
end

# ---- idiom classifiers and the GPU-split gate ------------------------------
# Accumulate-only: every write is `A[e] = A[e] + ...` and A occurs nowhere else in the loop, so iterations commute.
# Consume-only: every occurrence is a read `old = A[e]` closed by a write `A[e] = c` with nothing touching A in between.
cgen_expr_reads_array(e, arr::Symbol) = (r = Dict{Any,Vector{Any}}(); cgen_collect_refs!(e, r); haskey(r, arr))

function cgen_ref_index_reads_array(ref, arr::Symbol)
    return ref isa Expr && ref.head == :ref && any(a -> cgen_expr_reads_array(a, arr), ref.args[2:end])
end

function cgen_array_accumulate_only(body::Vector{NamedTuple}, arr::Symbol)
    nwrites = Ref(0)
    return cgen_accumulate_only_walk(body, arr, nwrites) && nwrites[] > 0
end

function cgen_accumulate_only_walk(body::Vector{NamedTuple}, arr::Symbol, nwrites::Base.RefValue{Int})
    for st in body
        if st.kind == :assign
            if st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == arr
                cgen_ref_index_reads_array(st.lhs, arr) && return false
                terms = cgen_flatten_sum(st.rhs)
                self_idx = findfirst(t -> t == st.lhs, terms)
                self_idx === nothing && return false
                any(j -> j != self_idx && cgen_expr_reads_array(terms[j], arr), eachindex(terms)) && return false
                nwrites[] += 1
            else
                (cgen_ref_index_reads_array(st.lhs, arr) || cgen_expr_reads_array(st.rhs, arr)) && return false
            end
        elseif st.kind == :for
            any(e -> cgen_expr_reads_array(e, arr), (st.lo, st.hi, st.step)) && return false
            cgen_accumulate_only_walk(st.body, arr, nwrites) || return false
        elseif st.kind == :if
            cgen_expr_reads_array(st.cond, arr) && return false
            cgen_accumulate_only_walk(st.then, arr, nwrites) || return false
            cgen_accumulate_only_walk(st.els, arr, nwrites) || return false
        else
            return false
        end
    end
    return true
end

function cgen_array_consume_only(body::Vector{NamedTuple}, arr::Symbol)
    nwrites = Ref(0)
    return cgen_consume_only_walk(body, arr, nwrites) && nwrites[] > 0
end

# One block at a time. At most one read is open. Any other touch of `arr` while a read is open refuses the loop, so an HVP
# twin statement between the read and its write is fine and an intervening access to `arr` is not.
function cgen_consume_only_walk(body::Vector{NamedTuple}, arr::Symbol, nwrites::Base.RefValue{Int})
    pending = nothing
    for st in body
        if st.kind == :assign
            lhs_is_arr = st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == arr
            rhs_is_arr = st.rhs isa Expr && st.rhs.head == :ref && st.rhs.args[1] == arr
            if st.lhs isa Symbol && rhs_is_arr
                pending === nothing || return false
                cgen_ref_index_reads_array(st.rhs, arr) && return false
                pending = st.rhs.args[2:end]
            elseif lhs_is_arr
                pending === nothing && return false
                st.lhs.args[2:end] == pending || return false
                (cgen_ref_index_reads_array(st.lhs, arr) || cgen_expr_reads_array(st.rhs, arr)) && return false
                pending = nothing
                nwrites[] += 1
            else
                (cgen_ref_index_reads_array(st.lhs, arr) || cgen_expr_reads_array(st.rhs, arr)) && return false
            end
        elseif st.kind == :for
            any(e -> cgen_expr_reads_array(e, arr), (st.lo, st.hi, st.step)) && return false
            if cgen_stmt_touches_array(st, arr)
                pending === nothing || return false
                cgen_consume_only_walk(st.body, arr, nwrites) || return false
            end
        elseif st.kind == :if
            cgen_expr_reads_array(st.cond, arr) && return false
            if cgen_stmt_touches_array(st, arr)
                pending === nothing || return false
                cgen_consume_only_walk(st.then, arr, nwrites) || return false
                cgen_consume_only_walk(st.els, arr, nwrites) || return false
            end
        else
            return false
        end
    end
    return pending === nothing
end

cgen_array_ok_idiom(body::Vector{NamedTuple}, arr::Symbol) =
    cgen_array_accumulate_only(body, arr) || cgen_array_consume_only(body, arr)

# ---- plain-write overlap proof and CSE-accumulate folding (GPU split gate) ----
# Two plain writes to one array from two DIFFERENT threads are a lost-update or last-writer race with no atomic form. The gate refuses a
# loop only when that overlap is PROVEN, from indices affine in the loop variable with offsets that differ by a multiple of the stride
# (`v[i]` and `v[i+1]`). An index it cannot analyse stays trusted, as the plain-write contract says: mpnn writes three slices of one
# buffer whose disjointness is not provable from the indices, and refusing on "unknown" would drop it.
#
# The difference `kb - ka` of two offsets when it is a literal integer, else nothing. Shared symbolic terms cancel.
function cgen_offset_difference(ka, kb)
    terms = cgen_flatten_additive_terms(Expr(:call, :-, kb, ka))
    net = Dict{String,Int}()
    lit = 0
    for (sg, t) in terms
        if t isa Integer
            lit += sg * t
        else
            net[string(t)] = get(net, string(t), 0) + sg
        end
    end
    all(v -> v == 0, values(net)) || return nothing
    return lit
end

# Can two writes with these index vectors hit one element from two DIFFERENT threads, and is that provable? Every dimension must be
# analysable: either thread-independent and equal, or affine in `v` with the same non-zero stride, and all strided dimensions must agree
# on one thread distance. Anything else answers false (unknown or provably apart).
function cgen_writes_overlap_proven(a::Vector, b::Vector, v::Symbol, varying::Set{Symbol})
    length(a) == length(b) || return false
    shift = nothing
    for d in eachindex(a)
        pa = cgen_index_affine(a[d], v)
        pb = cgen_index_affine(b[d], v)
        (pa === nothing || pb === nothing) && return false
        (cgen_affine_invariant(pa[2], varying) && cgen_affine_invariant(pb[2], varying)) || return false
        if pa[1] == 0 && pb[1] == 0
            cgen_offset_difference(pa[2], pb[2]) == 0 || return false
        elseif pa[1] == pb[1]
            diff = cgen_offset_difference(pa[2], pb[2])
            (diff === nothing || diff == 0 || diff % pa[1] != 0) && return false
            sh = diff // pa[1]
            shift === nothing ? (shift = sh) : (shift == sh || return false)
        else
            return false
        end
    end
    return shift !== nothing
end

# Arrays with two distinct write index vectors, at least one of them not an additive self-reference (an additive pair is repaired by an
# atomic), that provably address one element from two threads. Same-body let-bindings are folded into the indices first.
function cgen_overlap_proven_arrays(body::Vector{NamedTuple}, thread_var::Symbol)
    varying = cgen_all_assigned_scalars(body)
    cgen_collect_loop_vars!(body, varying)
    writes = Dict{Symbol,Vector{Any}}()
    reads = Dict{Symbol,Vector{Any}}()
    cgen_collect_subst_accesses!(body, Dict{Symbol,Any}(), writes, reads)
    proven = Set{Symbol}()
    for (arr, sites) in writes
        for p in eachindex(sites), q in (p + 1):lastindex(sites)
            sites[p][1] == sites[q][1] && continue
            (sites[p][2] && sites[q][2]) && continue
            if cgen_writes_overlap_proven(sites[p][1], sites[q][1], thread_var, varying)
                push!(proven, arr)
                break
            end
        end
    end
    return proven
end

cgen_count_sym_uses(e, s::Symbol) = e isa Symbol ? Int(e === s) : e isa Expr ? sum(a -> cgen_count_sym_uses(a, s), e.args; init = 0) : 0

function cgen_count_sym_uses_body(body::Vector{NamedTuple}, s::Symbol)
    n = 0
    for st in body
        if st.kind == :assign
            n += cgen_count_sym_uses(st.rhs, s) + (st.lhs isa Expr ? cgen_count_sym_uses(st.lhs, s) : 0)
        elseif st.kind == :if
            n += cgen_count_sym_uses(st.cond, s) + cgen_count_sym_uses_body(st.then, s) + cgen_count_sym_uses_body(st.els, s)
        elseif st.kind == :for
            n += cgen_count_sym_uses(st.lo, s) + cgen_count_sym_uses(st.hi, s) + cgen_count_sym_uses(st.step, s) + cgen_count_sym_uses_body(st.body, s)
        elseif st.kind == :while
            n += cgen_count_sym_uses(st.cond, s) + cgen_count_sym_uses_body(st.body, s)
        end
    end
    return n
end

# Does any statement in `body` write array `arr` or assign a scalar in `syms`?
function cgen_body_touches(body::Vector{NamedTuple}, arr::Symbol, syms::Set{Symbol})
    for st in body
        if st.kind == :assign
            (st.lhs isa Symbol && st.lhs in syms) && return true
            (st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] === arr) && return true
        elseif st.kind == :if
            (cgen_body_touches(st.then, arr, syms) || cgen_body_touches(st.els, arr, syms)) && return true
        elseif st.kind == :for
            (st.var in syms || cgen_body_touches(st.body, arr, syms)) && return true
        elseif st.kind == :while
            cgen_body_touches(st.body, arr, syms) && return true
        end
    end
    return false
end

function cgen_expr_symbols(e, out::Set{Symbol} = Set{Symbol}())
    e isa Symbol && push!(out, e)
    e isa Expr && foreach(a -> cgen_expr_symbols(a, out), e.args)
    return out
end

# A CSE'd read-modify-write `t = A[e]; ...; A[e] = t + x` (t used nowhere else) is the accumulation `A[e] = A[e] + x` with the read
# hoisted. The additive test and the atomic rewrite only see the second form, so the first stays a plain write: ttgc's forward
# re-execution scatters into mup that way, and two cells sharing a node lose an update on a real device. The fold deletes the hoisted
# read and puts the array element back as the sum's own term. Applied to a split loop's body only, never to code that stays on the host.
function cgen_fold_accumulates(body::Vector{NamedTuple}, root::Vector{NamedTuple} = body)
    out = NamedTuple[]
    skip = Set{Int}()
    subst = Dict{Int,Tuple{Symbol,Any}}()
    for (i, st) in enumerate(body)
        i in skip && continue
        if st.kind == :assign && st.lhs isa Symbol && st.rhs isa Expr && st.rhs.head == :ref && st.rhs.args[1] isa Symbol &&
           !cgen_expr_has_ref(Expr(:tuple, st.rhs.args[2:end]...)) && cgen_count_sym_uses(st.rhs, st.lhs) == 0 &&
           cgen_count_sym_uses_body(root, st.lhs) == 1
            s, ref = st.lhs, st.rhs
            watch = cgen_expr_symbols(Expr(:tuple, ref.args[2:end]...))
            push!(watch, s)
            for j in (i + 1):length(body)
                w = body[j]
                if w.kind == :assign && w.lhs == ref && !(w.rhs isa Number)
                    terms = cgen_flatten_sum(w.rhs)
                    k = findfirst(t -> t === s, terms)
                    k === nothing && break
                    new_rhs = cgen_subst_expr(w.rhs, Dict{Symbol,Any}(s => ref))
                    push!(skip, i)
                    subst[j] = (s, new_rhs)
                    break
                end
                cgen_body_touches(NamedTuple[w], ref.args[1], watch) && break
            end
        end
    end
    for (i, st) in enumerate(body)
        i in skip && continue
        if haskey(subst, i)
            push!(out, merge(st, (rhs = subst[i][2],)))
        elseif st.kind == :if
            push!(out, merge(st, (then = cgen_fold_accumulates(st.then, root), els = cgen_fold_accumulates(st.els, root))))
        elseif st.kind == :for || st.kind == :while
            push!(out, merge(st, (body = cgen_fold_accumulates(st.body, root),)))
        else
            push!(out, st)
        end
    end
    return out
end

# A read-modify-write that no atomic add repairs, through an index that gathers from an array (so two threads can reach one element):
# `t = A[e]; push t; A[e] = t + x` (the old value is also saved, so the read cannot be folded away), or `A[e] = f(A[e], ..)` with f
# not a sum. Two threads on one element lose an update or save a value that depends on thread order. ttgc's forward re-execution
# scatters into mup this way. So does its undo, `A[e] = A_stack[k]`, whose writers carry different saved values. Only a gathered index
# counts: an index computed with div/mod is the flattened-nest idiom and stays trusted, and any other plain replacement through a
# gathered index stays trusted too, since a permutation-style scatter is the common case. `gathered` is the set of scalars whose
# value comes from an array read, directly or through another such scalar.
function cgen_rmw_hazard_arrays(body::Vector{NamedTuple})
    acc = Set{Symbol}()
    other = Set{Symbol}()
    hazard = cgen_rmw_walk!(body, Dict{Symbol,Any}(), Set{Symbol}(), acc, other)
    # an array accumulated through a gathered index has said its index is not unique, so a plain write to it through a gathered
    # index in the same loop races with the other threads' adds: the reverse of `old = m[n]; m[n] = old + x` zeroes mb[n] and re-adds
    union!(hazard, intersect(acc, other))
    return hazard
end

function cgen_rmw_walk!(body::Vector{NamedTuple}, held::Dict{Symbol,Any}, gathered::Set{Symbol}, acc::Set{Symbol}, other::Set{Symbol})
    hazard = Set{Symbol}()
    gathers(idx) = cgen_expr_has_ref(Expr(:tuple, idx...)) || any(x -> x in gathered, cgen_expr_symbols(Expr(:tuple, idx...)))
    for st in body
        if st.kind == :assign
            if st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] isa Symbol
                arr = st.lhs.args[1]
                idx = st.lhs.args[2:end]
                gathered_idx = gathers(idx)
                terms = cgen_flatten_sum(st.rhs)
                self_idx = findfirst(t -> t == st.lhs, terms)
                via_scalar = any(t -> get(held, t, nothing) == st.lhs, cgen_expr_symbols(st.rhs))
                direct = any(r -> r == st.lhs, cgen_collect_ref_exprs(st.rhs))
                restore = st.rhs isa Expr && st.rhs.head == :ref && cgen_is_stack_name(st.rhs.args[1])
                if gathered_idx
                    if self_idx !== nothing
                        push!(acc, arr)
                        o = cgen_sum_excluding(terms, self_idx)
                        (cgen_expr_reads_array(o, arr) || via_scalar) && push!(hazard, arr)
                    else
                        push!(other, arr)
                        (restore || via_scalar || direct) && push!(hazard, arr)
                    end
                end
                for (k, r) in collect(held)
                    r.args[1] === arr && delete!(held, k)
                end
            elseif st.lhs isa Symbol
                if st.rhs isa Expr && st.rhs.head == :ref && st.rhs.args[1] isa Symbol && !cgen_expr_has_ref(Expr(:tuple, st.rhs.args[2:end]...))
                    held[st.lhs] = st.rhs
                else
                    delete!(held, st.lhs)
                end
                if cgen_expr_has_ref(st.rhs) || any(x -> x in gathered, cgen_expr_symbols(st.rhs))
                    push!(gathered, st.lhs)
                else
                    delete!(gathered, st.lhs)
                end
            end
        elseif st.kind == :if
            union!(hazard, cgen_rmw_walk!(st.then, copy(held), copy(gathered), acc, other))
            union!(hazard, cgen_rmw_walk!(st.els, copy(held), copy(gathered), acc, other))
        elseif st.kind == :for
            union!(hazard, cgen_rmw_walk!(st.body, copy(held), copy(gathered), acc, other))
        end
    end
    return hazard
end

function cgen_collect_ref_exprs(e, out::Vector{Any} = Any[])
    if e isa Expr
        e.head == :ref && push!(out, e)
        foreach(a -> cgen_collect_ref_exprs(a, out), e.args)
    end
    return out
end

# Gate for the GPU split. The strict predicate runs first, so every loop that cgen_reduction_only_loop accepts is accepted here
# with the same synth. The idiom predicate only relaxes the array check. A refusal names the smallest failing array.
# `idiom` is true when an accepted loop needed the idiom predicate for at least one array.
function cgen_split_loop_gate(body::Vector{NamedTuple}, loopvar::Symbol, known_consts::Dict{Symbol,Any} = Dict{Symbol,Any}(), outer_defs::Dict{Symbol,Any} = Dict{Symbol,Any}(); root_body = nothing, protect = Set{Symbol}())
    body = cgen_fold_accumulates(body)
    synth = cgen_loop_scalar_synth(body, known_consts)
    saveonly = Set{Symbol}()
    if synth === nothing
        so = cgen_saveonly_synth(body, known_consts, root_body, protect)
        so === nothing || ((synth, saveonly) = (so.synth, so.saveonly))
    end
    synth === nothing && return (synth = nothing, reason = :scalar, array = nothing, idiom = false, saveonly = saveonly)
    summary = cgen_loop_array_summary(body, outer_defs)
    failing = Symbol[]
    idiom = false
    for (arr, widxs) in summary.writes
        ridxs = get(summary.reads, arr, Any[])
        cgen_array_ok_strict(body, arr, widxs, ridxs, loopvar, summary.defs) && continue
        if arr isa Symbol && cgen_array_ok_idiom(body, arr)
            idiom = true
            continue
        end
        push!(failing, arr isa Symbol ? arr : :unknown)
    end
    isempty(failing) || return (synth = nothing, reason = :array, array = minimum(failing), idiom = false, saveonly = saveonly)
    overlap = cgen_overlap_proven_arrays(body, loopvar)
    isempty(overlap) || return (synth = nothing, reason = :overlap, array = minimum(overlap), idiom = false, saveonly = saveonly)
    rmw = cgen_rmw_hazard_arrays(body)
    isempty(rmw) || return (synth = nothing, reason = :rmw, array = minimum(rmw), idiom = false, saveonly = saveonly)
    return (synth = synth, reason = :ok, array = nothing, idiom = idiom, saveonly = saveonly)
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
        elseif stmt.kind == :while
            cgen_var_assigned_anywhere(stmt.body, var) && return true
        end
    end
    return false
end

# The value `var` provably holds at the end of one traversal of body, if it's the same literal on
# every control-flow path, else `nothing`. Walks in program order threading a running state:
# :unchanged (not touched), :unknown (touched but not provably a single literal), or a Number
# (every touch since the last branch point agrees). `entry` is the value `var` is known to hold
# coming into this body, mattering only for a nested loop that might not run.
function cgen_loop_convergent_constant(body::Vector{NamedTuple}, var::Symbol, entry = nothing)
    state = cgen_terminal_value_walk(body, var, :unchanged, entry)
    state === :unchanged && return entry
    return state isa Number ? state : nothing
end

function cgen_terminal_value_walk(body::Vector{NamedTuple}, var::Symbol, state, entry = nothing)
    for stmt in body
        if stmt.kind == :assign && stmt.lhs === var
            state = stmt.rhs isa Number ? stmt.rhs : :unknown
        elseif stmt.kind == :if
            then_state = cgen_terminal_value_walk(stmt.then, var, state, entry)
            els_state = cgen_terminal_value_walk(stmt.els, var, state, entry)
            state = if then_state isa Number && els_state isa Number && then_state == els_state
                then_state
            elseif then_state == :unchanged && els_state == :unchanged
                :unchanged
            else
                :unknown
            end
        elseif stmt.kind == :for
            if cgen_var_assigned_anywhere(stmt.body, var)
                # A var touched inside a nested loop isn't automatically :unknown -- if that nested loop's own body provably converges `var` to
                # the same literal on every path too, that literal is what `var` holds after it finishes. Needed once this proof runs on loops
                # whose reset lives one level deeper than the reduction. But only when the nested loop is GUARANTEED to run: with a runtime
                # bound there are two outcomes, and the claim survives only if they agree -- taking `inner` unconditionally made entry_empty's
                # GPU adjoint wrong by 1.43 while its CPU adjoint was correct.
                held = state === :unchanged ? entry : state
                inner = cgen_loop_convergent_constant(stmt.body, var, held)
                state = if cgen_loop_runs_once(stmt)
                    inner === nothing ? :unknown : inner
                elseif inner !== nothing && held isa Number && held == inner
                    inner
                else
                    :unknown
                end
            end
        elseif stmt.kind == :while
            if cgen_var_assigned_anywhere(stmt.body, var)
                # A var touched inside a nested loop isn't automatically :unknown -- if that nested loop's own body provably converges `var` to
                # the same literal on every path too, that literal is what `var` holds after it finishes. Needed once this proof runs on loops
                # whose reset lives one level deeper than the reduction. But only when the nested loop is GUARANTEED to run: with a runtime
                # bound there are two outcomes, and the claim survives only if they agree -- taking `inner` unconditionally made entry_empty's
                # GPU adjoint wrong by 1.43 while its CPU adjoint was correct.
                held = state === :unchanged ? entry : state
                inner = cgen_loop_convergent_constant(stmt.body, var, held)
                state = if cgen_loop_runs_once(stmt)
                    inner === nothing ? :unknown : inner
                elseif inner !== nothing && held isa Number && held == inner
                    inner
                else
                    :unknown
                end
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
function cgen_use_before_def(body::Vector{NamedTuple}, local_names::Set{Symbol}, synth::Dict{Symbol,Any}, defined_in::Set{Symbol} = Set{Symbol}(); save_ok::Bool = false)
    defined = union(defined_in, Set{Symbol}(keys(synth)))
    flagged(vars) = any(v -> v in local_names && !(v in defined), vars)
    for stmt in body
        if stmt.kind == :assign
            # save_ok: a bare `X_stack[i] = v` copies v into storage and computes nothing from it
            (save_ok && cgen_is_save_stmt(stmt)) || flagged(cgen_read_vars(stmt.rhs)) && return (false, defined)
            if stmt.lhs isa Expr && stmt.lhs.head == :ref
                flagged(cgen_read_vars(stmt.lhs.args[2:end])) && return (false, defined)
            elseif stmt.lhs isa Symbol
                push!(defined, stmt.lhs)
            end
        elseif stmt.kind == :if
            flagged(cgen_read_vars(stmt.cond)) && return (false, defined)
            ok_then, defined_then = cgen_use_before_def(stmt.then, local_names, synth, defined; save_ok)
            ok_then || return (false, defined)
            ok_els, defined_els = cgen_use_before_def(stmt.els, local_names, synth, defined; save_ok)
            ok_els || return (false, defined)
            defined = intersect(defined_then, defined_els)
        elseif stmt.kind == :for
            flagged(cgen_read_vars(stmt.lo)) && return (false, defined)
            flagged(cgen_read_vars(stmt.hi)) && return (false, defined)
            flagged(cgen_read_vars(stmt.step)) && return (false, defined)
            # the loop's own iteration variable is bound by its header,
            # hence always "defined" throughout its body -- not a read-
            # before-def case at all, unlike a genuine local scalar.
            ok, _ = cgen_use_before_def(stmt.body, local_names, synth, union(defined, Set([stmt.var])); save_ok)
            ok || return (false, defined)
        elseif stmt.kind == :while
            # no loop variable to add to the defined set
            ok, _ = cgen_use_before_def(stmt.body, local_names, synth, defined; save_ok)
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
            cgen_collect_vars!(stmt.body, vars)
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
# precision_locked, precision_lock_reason, warp_atomics, sync_free_reduce). Only the launch macro, launch keywords, thread-index intrinsic, atomic/allowscalar modules, and preamble differ
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
        warp_atomics = true,
        sync_free_reduce = true,
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
        warp_atomics = false,
        sync_free_reduce = false,
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
        warp_atomics = false,
        sync_free_reduce = false,
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

# ---- idiomatic scalar-reduction detection ----
# Recognizes the narrow shape `target = target +/- f(arr_1[loopvar], ..., free scalars)` as the whole and only statement in a loop body
# already proven safe to split -- a bare dot-product-style `loss[1]=loss[1]+u[i]*v[i]`, not a case with an :if picking the term (a second
# statement, so this declines, falling back to today's atomic-kernel codegen). Returns `(target, op, arrs, term)` or `nothing`; the caller
# turns `(arrs, term, loopvar)` into a replacement expression, which differs by target.
function cgen_idiomatic_scalar_reduction(body::Vector{NamedTuple}, loopvar::Symbol)
    length(body) == 1 || return nothing
    stmt = body[1]
    stmt.kind == :assign || return nothing
    target = stmt.lhs
    # An array element, or a plain local scalar. The array-element form
    # writes a device buffer from the host and needs an allowscalar
    # wrapper; the scalar form needs none, because mapreduce and sum both
    # return a host value. Only the second reaches a LOCAL accumulator,
    # which cgen_body's safe_scope test otherwise rejects for not being a
    # function argument.
    (target isa Symbol || (target isa Expr && target.head == :ref)) || return nothing
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

# `sum(abs2, ·)` and `mapreduce` dispatch on the array TYPE of their argument and reach GPUArrays.jl's
# generic machinery, so one call works on every backend with no backend-conditional codegen. Both come
# from `using LinearAlgebra`/Base, present in every GPU backend's preamble.
#
# A `dot(a, b)` special case used to sit here, on the stated grounds that a vendor cuBLAS/rocBLAS
# method would be picked at runtime. That is not what happens on the reference cluster: every cuBLAS
# Level-1 entry point raises `DivideError: integer division error`, measured on a Tesla V100 for `dot`
# and `norm`, for Float32 and Float64, at length 16 and 4096. `gemv` on the same device is fine, so
# cuBLAS is present and only its Level-1 bindings fail. Nothing in the test suite could see it:
# validate_corpus runs on the CPU and validate_offload is codegen only, so the reduction path
# emitted code that failed at runtime for as long as the branch existed.
#
# The generic form costs nothing to prefer. On the same device, dotprod's adjoint reduces in 72.6 us
# through `mapreduce` against 132.5 us through the atomic kernel it replaces, both exact. A vendor
# path that cannot be measured is not a fast path.
function cgen_idiomatic_reduction_value(arrs::Vector{Symbol}, term, loopvar::Symbol, lo, step, hi; dims::Bool = false)
    rng = cgen_reduction_range(lo, step, hi)
    src = [Expr(:call, :view, a, rng) for a in arrs]
    # `init` is not decoration. A loop that may not run has not run, and
    # its reduction must contribute zero. Julia infers a neutral element
    # for a multi-collection mapreduce but NOT for a single collection
    # under an anonymous closure: `mapreduce(x->x*x, +, view(a, 1:0))`
    # raises "reducing over an empty collection is not allowed". So a
    # kernel whose only reduction reads one array, with a term that is not
    # exactly `a[i]^2`, failed at runtime whenever its trip count reached
    # zero. validate_zerotrip draws integers from [0, 2] but runs on the
    # CPU, where this path is never emitted, so nothing saw it.
    #
    # `zero(eltype(·))` resolves at run time, so no precision assumption
    # is baked in, and it changes no non-empty result because the operator
    # is always `+`.
    init = Expr(:kw, :init, Expr(:call, :zero, Expr(:call, :eltype, src[1])))
    # dims = true keeps the result on the device as a one-element array, so no host synchronization is needed
    kws = dims ? Any[init, Expr(:kw, :dims, 1)] : Any[init]
    if length(arrs) == 1 && term == Expr(:call, :^, Expr(:ref, arrs[1], loopvar), 2)
        return Expr(:call, :sum, kws..., :abs2, src[1])
    else
        closure_args = [Symbol(:__mr_, i) for i in eachindex(arrs)]
        subst = Dict(arrs[i] => closure_args[i] for i in eachindex(arrs))
        closure_body = cgen_substitute_indexed_refs(term, subst, loopvar)
        return Expr(:call, :mapreduce, kws...,
                    Expr(:->, Expr(:tuple, closure_args...), closure_body), :+, src...)
    end
end

# The reduction reads exactly the elements the loop visited, and no more.
# Without this the emitted call reduced the WHOLE array whatever the loop
# bounds were: `for i_x = 2:i_n` became `mapreduce(*, +, u, v)`, which
# silently folds in index 1. Measured on a Tesla V100 at length 8, the
# loop gives 70.0 and the whole-array form 72.0.
#
# A view is used rather than a bounds restriction because STADE never
# derives array extents, so `1:i_n` is not provably the whole of `u`
# either. Views are exact for any range and were confirmed on device,
# including a strided one: view(u, 1:2:8) reduced to 32.0, matching the
# host.
cgen_reduction_range(lo, step, hi) =
    step == 1 ? Expr(:call, :(:), lo, hi) : Expr(:call, :(:), lo, step, hi)

# JACC: no confirmed BLAS-level acceleration to special-case, so every matched shape goes through the
# same `JACC.@parallel_reduce range=N f(args...)` primitive, replacing the `@parallel_for` +
# `Atomix.@atomic` kernel it would otherwise get.
#
# JACC's convention is a closure whose first parameter is the POSITION in `1:range`, not the loop's own
# value. The two coincide only for a loop that starts at 1 with step 1. The comment that used to sit here
# claimed they always coincide, and the closure therefore left `arr[loopvar]` as written -- so every
# reduction whose loop had another lower bound, a stride, or a descending direction silently summed
# elements `1 .. trip_count` instead of the ones the loop actually visited. Measured on a Tesla V100 with
# reduction_threshold = 0: `2:i_n` at n = 100 returned 90.974668 against CUDA's exact 90.616806, and
# `3:3:i_n` was out by 45%. A wrong value, never an error. Nothing saw it because the reduce branch needs
# n_iter >= CGEN_REDUCTION_THRESHOLD_DEFAULT and corpus integers are drawn from 3..5 -- the same
# invisibility that hid the cross-thread write-overlap race.
#
# The position is therefore mapped back to the loop's value, `lo + (k - 1) * step`, and substituted into
# the term's array indices. cgen_idiomatic_scalar_reduction has already proven that every :ref in `term`
# is indexed by exactly `loopvar` and that `loopvar` appears nowhere else, so that rewrite is total. A
# negative step needs no special case: `i_n:-1:3` maps k = 1 to i_n and counts down, which is the order
# the loop ran in.
#
# `lo == 1 && step == 1` keeps the original closure verbatim, because the mapping is the identity there
# and dotprod, matvec_loss and every `1:n` loss wrapper must stay byte-identical.
function jgen_idiomatic_reduction_value(arrs::Vector{Symbol}, term, loopvar::Symbol, lo, step, n_iter)
    if lo == 1 && step == 1
        closure = Expr(:->, Expr(:tuple, loopvar, arrs...), term)
    else
        k = :__jgen_k
        shift = Expr(:call, :-, k, 1)
        pos = step == 1 ? Expr(:call, :+, lo, shift) :
                          Expr(:call, :+, lo, Expr(:call, :*, shift, step))
        closure = Expr(:->, Expr(:tuple, k, arrs...), jgen_substitute_loop_index(term, loopvar, pos))
    end
    return Expr(:macrocall, Expr(:., :JACC, QuoteNode(Symbol("@parallel_reduce"))), nothing,
                Expr(:(=), :range, n_iter), Expr(:call, closure, arrs...))
end

# Rewrites `arr[loopvar]` to `arr[idx]` everywhere in `e`. Narrower than substituting the bare symbol on
# purpose: cgen_idiomatic_scalar_reduction accepts a term only when loopvar occurs nowhere except as a
# sole :ref index, so any other thing wearing that name is not this loop's variable.
function jgen_substitute_loop_index(e, loopvar::Symbol, idx)
    e isa Expr || return e
    if e.head == :ref && length(e.args) == 2 && e.args[2] === loopvar
        return Expr(:ref, e.args[1], idx)
    end
    return Expr(e.head, Any[jgen_substitute_loop_index(a, loopvar, idx) for a in e.args]...)
end

# ---- JACC idiomatic-reduction write-back ---
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

# Would splitting this loop actually produce a valid device kernel? cgen_reduction_only_loop's array test is coarse: it admits a write/read at the SAME
# index, right for a commutative accumulation but wrong for a non-additive read-modify-write like `y[i] = 0.5*y[i]+x[i]` that a repeating outer loop re-
# applies -- concurrent threads race, no atomic fixes it. cgen_device_assign already refuses this, but refusing THERE fails the whole conversion; asking
# here lets the loop stay on the host instead. This replaced cgen_prefer_inner_split, a preference that stranded every eligible outer loop with one
# child on the host.
function cgen_kernel_def_ok(build)
    try
        build()
        return true
    catch
        return false
    end
end

# ---- host-side body walk: splits device kernels off ----
# Host-side body walk: splits off one device kernel per eligible iteration-independent loop; anything left over runs as ordinary host-side
# Julia, wrapped in one allowscalar_macro block if a touched statement reads a device array (confirmed live-GPU crash otherwise).
# A splittable loop is additionally offered to cgen_idiomatic_scalar_reduction. outer_defs: scalar definitions from the
# kernel's top-level body, threaded through so the array-privacy proof can see a size relationship defined outside any candidate loop.

# Splitting a loop moves its assignments onto the DEVICE, leaving the host's copy of every local scalar behind. Any later HOST read of one
# then sees a stale value -- cgen_use_before_def only checks reads WITHIN the loop; this is the gate for reads after it. An adjoint shadow is
# exempt, provably: agen_backward_assign ends every consumed shadow with `vb = 0.0`, so device and host copies already agree (28 of 41 live-
# out sites). The rest are accumulators, stencil taps, and window bounds, where the values genuinely differ. Depends on the block-boundary
# collapse having already removed redundant outer pushes.
function cgen_liveout_is_zeroed(stmt, root_body, loop_vars, synth = Dict{Symbol,Any}(), saveonly = Set{Symbol}())
    locals = setdiff(cgen_locally_assigned_scalars(stmt.body), loop_vars)
    isempty(locals) && return true
    # a save-only scalar is seeded in `synth` but is not converged to a constant: a later read is a hazard unless it is a stack save
    so = intersect(locals, saveonly)
    isempty(so) || isempty(ii_escapes_nested(root_body, stmt.body, so; save_ok = true)) || return false
    locals = setdiff(locals, so)
    isempty(locals) && return true
    for v in ii_escapes_nested(root_body, stmt.body, locals)
        # A var in `synth` is not a hazard however the escape looks. synth[v] = c means the value coming into
        # the loop is already c and every path converges back to c, so the host copy still holds c after the
        # split -- exactly what the sequential loop leaves. mpnn's reverse node loop is this case: `sb` enters
        # at 0.0 and its last write is 0.0, but that write sits inside a runtime-bounded feature loop, so
        # cgen_last_assign_is_zero cannot prove it.
        haskey(synth, v) && continue
        cgen_last_assign_is_zero(stmt.body, v) || return false
    end
    return true
end

# Is the LAST assignment to `v` on every path through `body` the literal 0.0? A write only counts when GUARANTEED to run: at this body's own
# top level, in both arms of an `:if`, or inside a loop whose literal bounds prove at least one iteration. An unprovable nested write clears
# the flag rather than being ignored, since it might destroy a zero already established. entry_empty is why: its adjoint zeroes the shadow
# inside `for i_j = w:-1:1`, which runs zero times once the window retires, leaving `sb` holding a live accumulation -- caught on a live GPU
# at 1.43 relative error while the CPU adjoint was correct.
function cgen_last_assign_is_zero(body, v)
    found = false
    for st in body
        if st.kind == :assign && st.lhs isa Symbol && st.lhs == v
            found = st.rhs isa Number && st.rhs == 0.0
        elseif st.kind == :for
            if cgen_writes_var(st.body, v)
                found = cgen_loop_runs_once(st) && cgen_last_assign_is_zero(st.body, v)
            end
        elseif st.kind == :while
            if cgen_writes_var(st.body, v)
                found = cgen_loop_runs_once(st) && cgen_last_assign_is_zero(st.body, v)
            end
        elseif st.kind == :if
            tw = cgen_writes_var(st.then, v)
            ew = cgen_writes_var(st.els, v)
            if tw || ew
                found = tw && ew && cgen_last_assign_is_zero(st.then, v) &&
                        cgen_last_assign_is_zero(st.els, v)
            end
        end
    end
    return found
end

cgen_loop_runs_once(st) =
    st.lo isa Number && st.hi isa Number && st.step isa Number &&
    (st.step > 0 ? st.hi >= st.lo : st.hi <= st.lo)

function cgen_writes_var(body, v)
    for st in body
        if st.kind == :assign && st.lhs isa Symbol && st.lhs == v
            return true
        elseif st.kind == :for
            cgen_writes_var(st.body, v) && return true
        elseif st.kind == :while
            cgen_writes_var(st.body, v) && return true
        elseif st.kind == :if
            (cgen_writes_var(st.then, v) || cgen_writes_var(st.els, v)) && return true
        end
    end
    return false
end

# every loop's index variable, at any depth -- a `for` header rebinds these host-side, so they are
# never stale and must not be mistaken for live-out locals
function cgen_collect_loop_vars(body, acc)
    for st in body
        if st.kind == :for
            push!(acc, st.var); cgen_collect_loop_vars(st.body, acc)
        elseif st.kind == :while
            cgen_collect_loop_vars(st.body, acc)   # a while has no loop variable
        elseif st.kind == :if
            cgen_collect_loop_vars(st.then, acc); cgen_collect_loop_vars(st.els, acc)
        end
    end
    return acc
end

# One record per `for` loop that cgen_body or jgen_body visits. `verdict`: :kernel, :reduction (a kernel or a mapreduce lowering) or :host.
# `reason`: :ok, or the first gate that refused (:scalar, :array, :scope, :stackop, :kernel_def, :liveout).
# `via_idiom` marks a loop that was accepted only through the idiom predicate.
cgen_decision_record(owner::Symbol, stmt, depth::Int, verdict::Symbol, reason::Symbol, array, idiom::Bool) =
    (owner = owner, loop_var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step, depth = depth,
     verdict = verdict, reason = reason, array = reason in (:array, :overlap, :rmw) ? array : nothing, via_idiom = reason === :ok && idiom,
     body_id = objectid(stmt.body))

# Large-trip branch of an array-element reduction. A backend with sync_free_reduce adds the device-side reduction into the element with a broadcast on
# one-element views, which never copies a scalar to the host, so the host does not wait mid-sequence. This branch is still not capturable in a CUDA
# graph (measured: the dims reduction of CUDA.jl fails with invalid argument), so a captured call needs reduction_threshold = typemax(Int). The
# other backends keep the host form: a mapreduce returning a host scalar inside allowscalar.
cgen_sync_free_on(backend) = get(backend, :sync_free_reduce, false) === true

function cgen_reduce_into_element(backend, target, op, value, arrs, term, stmt)
    if cgen_sync_free_on(backend) && target isa Expr && target.head == :ref && length(target.args) == 2 && target.args[1] isa Symbol
        dvalue = cgen_idiomatic_reduction_value(arrs, term, stmt.var, stmt.lo, stmt.step, stmt.hi; dims = true)
        cell = Expr(:call, :view, target.args[1], Expr(:call, :(:), target.args[2], target.args[2]))
        return Expr(:(.=), cell, Expr(:., op, Expr(:tuple, cell, dvalue)))
    end
    return Expr(:macrocall, backend.allowscalar_macro, nothing, Expr(:block, Expr(:(=), target, Expr(:call, op, target, value))))
end

function cgen_body(body::Vector{NamedTuple}, kernels::Vector{Expr}, owner::Symbol, backend, reduce_vars::Set{Symbol}, fn_args::Set{Symbol}; reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT, outer_defs::Dict{Symbol,Any} = Dict{Symbol,Any}(), outer_known_consts::Dict{Symbol,Any} = Dict{Symbol,Any}(), root_body = body, decisions::Union{Nothing,Vector{NamedTuple}} = nothing, depth::Int = 0)
    exprs = Any[]
    # Seeded from the enclosing body's own known_consts at the point this body was reached (never written back) -- a scalar
    # reset at the tail of a non-split ancestor loop's body converges by induction to the same literal on every iteration, so
    # a literal known several nesting levels up remains valid here even though nothing in this body's own statement list re-
    # establishes it. Without this, cgen_reduction_only_loop misses a reduction scalar whose reset lives one level below its
    # true initializer.
    known_consts = copy(outer_known_consts)
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
        elseif stmt.kind == :while
            # A `while` stays on the host. No launch can size itself around a
            # loop whose trip count is unknown until it has run, and no thread
            # can be assigned an iteration. Loops INSIDE the body still offload
            # normally, which is why this recurses rather than refusing.
            #
            # A missing clause here would be silent: the statement would fall
            # through and vanish from the generated host function.
            flush_pending!()
            push!(exprs, emit_while(stmt.cond, cgen_body(stmt.body, kernels, owner, backend, reduce_vars, fn_args; reduction_threshold, outer_defs, outer_known_consts = known_consts, root_body, decisions, depth = depth + 1)))
        elseif stmt.kind == :if
            flush_pending!()
            cond = cgen_expr_has_ref(stmt.cond) ? Expr(:macrocall, backend.allowscalar_macro, nothing, stmt.cond) : stmt.cond
            push!(exprs, emit_if(cond, cgen_body(stmt.then, kernels, owner, backend, reduce_vars, fn_args; reduction_threshold, outer_defs, outer_known_consts = known_consts, root_body, decisions, depth = depth + 1), cgen_body(stmt.els, kernels, owner, backend, reduce_vars, fn_args; reduction_threshold, outer_defs, outer_known_consts = known_consts, root_body, decisions, depth = depth + 1)))
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
            gate = cgen_split_loop_gate(stmt.body, stmt.var, known_consts, outer_defs; root_body, protect = Set{Symbol}(fn_args))
            synth = gate.synth
            # A var cgen_scalar_reduction_vars calls reduction-worthy is only safe for cgen_emit's global box-once/unbox-once treatment when it's a
            # whole-function-lifetime entity (a top-level argument like `cb`). An internal scratch scalar can satisfy the same shape by accident when
            # an enclosing loop demotes for an unrelated reason, its true reset ending up a sibling statement one level up. Globally boxing it is
            # wrong two ways: the box precedes its own host-side reset (UndefVarError), and even fixed, one shared box accumulates across launches
            # meant to start fresh. Refusing to split is the same conservative choice as any unprovable case: slower, never wrong.
            loop_reduce_vars = synth === nothing ? Set{Symbol}() : cgen_scalar_reduction_vars(stmt.body)
            safe_scope = issubset(loop_reduce_vars, fn_args)
            # safe_scope guards the KERNEL-SPLITTING branch: a reduction
            # variable that is not a function argument was never boxed into
            # a device array, so no kernel can write it. The idiomatic
            # reduction writes nothing on the device -- `s = s + mapreduce(...)`
            # evaluates on the host and mapreduce returns a host value -- so
            # that gate does not apply to it.
            #
            # mlp1d is the case this reaches. Its output layer accumulates a
            # dot product into the local `s3`, leaving a host loop that did
            # two scalar device reads per iteration. Boxing `s3` instead
            # would work and is strictly more invasive: every reference
            # becomes `s3[1]`, it allocates per call, and it routes the loop
            # through the atomic kernel, measured at 132.5 us against
            # mapreduce's 72.6 us on a V100.
            local_red = nothing
            if !safe_scope && !cgen_contains_stackop(stmt.body)
                cand = cgen_idiomatic_scalar_reduction(stmt.body, stmt.var)
                cand !== nothing && cand[1] isa Symbol && (local_red = cand)
            end
            # `stage` names the first gate that refuses the loop, in the order the gates run
            stage = synth === nothing ? gate.reason : !safe_scope ? :scope : cgen_contains_stackop(stmt.body) ? :stackop : :ok
            # see cgen_kernel_def_ok -- the last gate, and the only one that asks the backend's own
            # codegen rather than a structural approximation of it
            stage === :ok && !cgen_kernel_def_ok(() -> cgen_kernel_def(stmt, owner, 0, cgen_free_vars(stmt, stmt.var), backend, loop_reduce_vars, synth)) && (stage = :kernel_def)
            # see cgen_liveout_is_zeroed -- refuses a split whose local scalars a later host
            # statement reads, which would read them stale
            stage === :ok && !cgen_liveout_is_zeroed(stmt, root_body, cgen_collect_loop_vars(root_body, Set{Symbol}()), synth === nothing ? Dict{Symbol,Any}() : synth, gate.saveonly) && (stage = :liveout)
            eligible = stage === :ok
            verdict = :host
            if eligible
                red = cgen_idiomatic_scalar_reduction(stmt.body, stmt.var)
                verdict = red !== nothing ? :reduction : :kernel
                if red !== nothing
                    target, op, arrs, term = red
                    value = cgen_idiomatic_reduction_value(arrs, term, stmt.var, stmt.lo, stmt.step, stmt.hi)
                    # Emit BOTH lowerings and pick at run time on the trip
                    # count. Neither wins everywhere, measured on a Tesla
                    # V100 with dotprod's adjoint:
                    #
                    #        n        atomic    mapreduce
                    #       16       8.66 us     53.73 us
                    #      256      10.46 us     58.37 us
                    #     4096     164.18 us     72.11 us
                    #  1048576    2506.31 us    150.12 us
                    #
                    # A tree reduction pays a fixed multi-pass cost that a
                    # small loop cannot amortise, while an atomic on ONE
                    # address serialises its contributing threads. The
                    # crossover sits between 256 and 4096, so the threshold
                    # is 1024. The trip count is already computed for the
                    # launch, so the test costs one host comparison.
                    #
                    # Only the array-element target gets this branch. A
                    # local accumulator has no atomic alternative: its loop
                    # would otherwise run on the host, one scalar device
                    # read per element, which mapreduce beats at every size.
                    idx = length(kernels) + 1
                    fargs = cgen_free_vars(stmt, stmt.var)
                    union!(reduce_vars, loop_reduce_vars)
                    push!(kernels, cgen_kernel_def(stmt, owner, idx, fargs, backend, loop_reduce_vars, synth))
                    atomic_branch = cgen_launch_expr(stmt, owner, idx, fargs, backend)
                    reduce_branch = cgen_reduce_into_element(backend, target, op, value, arrs, term, stmt)
                    flush_pending!()
                    push!(exprs, emit_if(Expr(:call, :<,
                                              cgen_trip_count(stmt.lo, stmt.step, stmt.hi),
                                              reduction_threshold),
                                         Any[atomic_branch], Any[reduce_branch]))
                else
                    idx = length(kernels) + 1
                    fargs = cgen_free_vars(stmt, stmt.var)
                    union!(reduce_vars, loop_reduce_vars)
                    push!(kernels, cgen_kernel_def(stmt, owner, idx, fargs, backend, loop_reduce_vars, synth))
                    push!(exprs, cgen_launch_expr(stmt, owner, idx, fargs, backend))
                end
            elseif local_red !== nothing
                verdict = :reduction
                lt, lop, larrs, lterm = local_red
                lvalue = cgen_idiomatic_reduction_value(larrs, lterm, stmt.var, stmt.lo, stmt.step, stmt.hi)
                # Same run-time test as the array-element case, against the
                # host loop rather than an atomic kernel. An earlier version
                # lowered this unconditionally, on the reasoning that a
                # device reduction must beat a host loop doing one scalar
                # read per element. That was one unrepeated measurement and
                # it is wrong at small trip counts: on a Tesla V100 with
                # n_h = 16, three consecutive best-of-5 runs put the host
                # loop ahead at 1600.9, 1599.1 and 1524.6 us against
                # mapreduce's 1820.5, 1872.0 and 1668.0. The full mlp1d
                # training took 178.0 s lowered against 86.6 s looped, with
                # bit-identical gradients.
                #
                # The reduction still wins by a wide margin at large trip
                # counts, so the choice is the trip count, not the shape.
                loop_branch = emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step,
                                           cgen_body(stmt.body, kernels, owner, backend, reduce_vars, fn_args; reduction_threshold, outer_defs,
                                                     outer_known_consts = known_consts, root_body, decisions, depth = depth + 1))
                # the target is a host scalar and every array is read through
                # a view, so this branch needs no allowscalar
                red_branch = Expr(:(=), lt, Expr(:call, lop, lt, lvalue))
                flush_pending!()
                push!(exprs, emit_if(Expr(:call, :<,
                                          cgen_trip_count(stmt.lo, stmt.step, stmt.hi),
                                          reduction_threshold),
                                     Any[loop_branch], Any[red_branch]))
            else
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, cgen_body(stmt.body, kernels, owner, backend, reduce_vars, fn_args; reduction_threshold, outer_defs, outer_known_consts = known_consts, root_body, decisions, depth = depth + 1)))
            end
            decisions === nothing || push!(decisions, cgen_decision_record(owner, stmt, depth, verdict, stage, gate.array, gate.idiom))
            delete!(known_consts, stmt.var)
            # Blind invalidation here would throw away entries that remain true: if this loop's own body
            # provably converges an assigned scalar to a literal by the walk cgen_reduction_only_loop
            # already trusts, that literal holds after the loop regardless of whether it split -- update
            # rather than erase, so a later sibling statement can still see it. Only truly unprovable
            # scalars get deleted.
            for v in cgen_all_assigned_scalars(stmt.body)
                c = cgen_loop_convergent_constant(stmt.body, v, get(known_consts, v, nothing))
                # The convergent constant only describes the value after the loop if the loop actually ran. A runtime bound
                # may give zero iterations, in which case `v` still holds what it held coming in -- unless that was already
                # the same constant. Recording it unconditionally lets cgen_reduction_only_loop synthesise `v = c` for a
                # LATER loop; every thread starts from c instead of the sequential value. ii_kill is the witness: the GPU
                # adjoint came out wrong by 0.98 while the CPU one was correct.
                if c === nothing || !(cgen_loop_runs_once(stmt) || get(known_consts, v, nothing) == c)
                    delete!(known_consts, v)
                else
                    known_consts[v] = c
                end
            end
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

# `div(hi - lo, step) + 1` is wrong for |step| > 1 when the loop is empty:
# `1:2:0` gives div(-1, 2) + 1 = 1, so one thread clears the `__tid >
# trip_count` guard and runs an iteration the loop never had. Measured on
# a V100, red_strided at n = 0 returned 1.0 from the atomic path against
# 0.0 from the reduction path, which is the correct answer.
#
# `div(hi - lo + step, step)` counts correctly for both signs, and max(0, .)
# covers a range that is empty by more than one step. fld would also work
# and is not used: this expression is re-ingested, where only plain names
# are allowed, and a primal scalar named `fl` has the shadow `fld`. `max`
# has no such collision, since a shadow is its primal plus b, d or bd.
#
# |step| == 1 keeps the original expression, so every file STADE already
# emits is unchanged and only the strided case moves.
# Below this many iterations an atomic kernel beats a tree reduction.
#
# The default is deliberately high, because the only measurement that
# reproduced says so. On an NVIDIA A30, best-of-5 after 30 warmup calls,
# dotprod's adjoint gave a smooth monotone curve crossing over near 24000:
#
#        n     atomic   mapreduce   speedup
#     1024    11.72 us    71.59 us     0.16
#     8192    35.70 us    71.91 us     0.50
#    16384    62.25 us    72.12 us     0.86
#    32768   117.09 us    73.38 us     1.60
#   262144   876.21 us    85.36 us    10.27
#
# The same benchmark on a Tesla V100 does NOT reproduce. Two identical
# runs gave 1160.74 us and 88.60 us for mapreduce at n = 262144, a 13x
# swing, while the atomic timings on those runs agreed within a few
# percent. Whatever causes that, a threshold cannot be fitted to it.
#
# The JACC path reaches its reductions through `JACC.@parallel_reduce`, a
# different primitive, and was measured separately on a Tesla V100:
#
#        n     atomic      reduce   speedup
#     1024   735.31 us  1043.96 us      0.70
#    16384   808.66 us   978.51 us      0.83
#    65536   780.64 us   726.00 us      1.08
#   262144  1592.85 us   641.37 us      2.48
#
# That crossover sits near 65000, against roughly 24000 for CUDA on the
# A30. One shared default is used rather than one per backend, because the
# two numbers bracket it and neither measurement is precise enough to
# justify splitting them.
#
# 32768 is therefore chosen to avoid a regression rather than to capture
# every win: on the clean CUDA data the atomic path is still ahead at 16384
# and behind at 32768. A user who has measured their own hardware can pass
# `reduction_threshold` to change it. Being slower than optimal is
# recoverable; being slower than the code STADE used to emit is not.
const CGEN_REDUCTION_THRESHOLD_DEFAULT = 32768

function cgen_trip_count(lo, step, hi)
    step isa Number && abs(step) == 1 &&
        return Expr(:call, :+, Expr(:call, :div, Expr(:call, :-, hi, lo), step), 1)
    return Expr(:call, :max, 0,
                Expr(:call, :div, Expr(:call, :+, Expr(:call, :-, hi, lo), step), step))
end

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
    fbody = cgen_fold_accumulates(stmt.body)
    append!(body, cgen_device_body(fbody, stmt.var, backend, reduce_vars, cgen_race_write_arrays(fbody, stmt.var)))
    push!(body, emit_return_nothing())
    return Expr(:function, Expr(:call, cgen_kernel_fname(owner, idx, backend), fargs...), Expr(:block, body...))
end

function cgen_launch_expr(stmt, owner::Symbol, idx::Int, fargs::Vector{Symbol}, backend)
    n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)
# A zero-trip loop is a legal no-op on the CPU, but a launch with zero blocks is a hard CUDA error, so the kernel would crash on the device -- a
# multigrid solver coarsened to zero points reaches exactly this. One block is launched instead, its threads failing the `__tid > trip_count` guard
# and returning. `Base.cld`, not bare `cld`: a shadow name is its primal name plus b/d/bd, so a primal scalar `cl` becomes `cld`, colliding here
# (mg_vcycle's HVP died live with 'objects of type Float64 are not callable'). Only the launch expression can be qualified, since re-ingestion
# allows plain names only; `mo`->`mod` and `fl`->`fld` remain live traps in kernel bodies.
    nblocks = Expr(:call, :max, 1, Expr(:call, Expr(:., :Base, QuoteNode(:cld)), n_iter, :nthread_per_block))
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

# ---- cross-thread write-overlap analysis ----------------------------------
# An index injective in the thread variable proves a thread never collides with ITSELF. It proves nothing about two
# DIFFERENT writes in the same body: `ub[i-1]`, `ub[i]` and `ub[i+1]` are each injective, yet thread i and thread i+2
# both land on `ub[i+1]`, and a plain read-modify-write there loses updates. Measured on a V100 as a wrong, run-to-run-
# varying `ub` from stencil_loss's and advection's adjoints at n >= 256 -- invisible below one warp, where the hardware's
# lockstep serialises the updates for free. Pairing a site with ITSELF is what catches the companion shape `a[i+j]` under
# an inner serial loop over j, where one thread's later iteration collides with another thread's earlier one.
#
# Every write to an array whose accumulation sites can overlap becomes an atomic accumulation.
#
# Two deliberate limits on scope. Only ADDITIVE sites are paired: `+=` is what an atomic repairs, and a lost-update race
# is what the measured defect was. A plain assignment racing another plain assignment is a different hazard with no
# atomic form, and refusing the kernel for it rejects correct code here -- mpnn writes three disjoint slices of one
# concatenated buffer at `(e-1)*n_in_msg + k`, `+ n_node_feat + k` and `+ 2n_node_feat + k`, whose disjointness needs the
# symbolic stride's relation to the inner loop's range and is not provable from the indices alone. That hazard stays
# where it already sat, with cgen_device_assign's existing non-additive tolerance. Second, only pairs of DIFFERENT index
# expressions: whether ONE expression collides with itself across threads is what thread_invariant and
# cgen_expr_injective_ok already decide, and they answer it with this same atomic rewrite.
function cgen_race_write_arrays(body::Vector{NamedTuple}, thread_var::Symbol)
    varying = cgen_all_assigned_scalars(body)
    cgen_collect_loop_vars!(body, varying)
    writes = Dict{Symbol,Vector{Any}}()
    reads = Dict{Symbol,Vector{Any}}()
    cgen_collect_subst_accesses!(body, Dict{Symbol,Any}(), writes, reads)
    racy = Set{Symbol}()
    for (arr, all_sites) in writes
        sites = Any[s[1] for s in all_sites if s[2]]
        for p in eachindex(sites), q in (p + 1):lastindex(sites)
            sites[p] == sites[q] && continue
            if cgen_writes_may_overlap(sites[p], sites[q], thread_var, varying)
                push!(racy, arr)
                break
            end
        end
        arr in racy || any(site -> cgen_write_self_overlaps(site, thread_var, varying), sites) && push!(racy, arr)
    end
    return racy
end

# One additive site against ITSELF. cgen_expr_injective_ok trusts an index that is `t + k` with `k` thread-independent, but an inner serial
# loop makes `k` move: thread `t` at one iteration lands on the element thread `t + 1` writes at the previous one (`xb[t + ci - 1]`, the
# adjoint of a convolution). Only POSITIVE evidence counts, so every shape the old gate trusted stays trusted: a dimension that is not
# affine in the thread variable (a symbolic stride, a gather) leaves the verdict to cgen_expr_injective_ok, and a dimension whose offset is
# invariant proves the threads disjoint. What is left is an integer multiple of the thread variable plus an offset that varies.
function cgen_write_self_overlaps(site::Vector, thread_var::Symbol, varying::Set{Symbol})
    moving = false
    for d in eachindex(site)
        p = cgen_index_affine(site[d], thread_var)
        p === nothing && return false
        p[1] == 0 && continue
        cgen_affine_invariant(p[2], varying) && return false
        cgen_varying_scaled_by_integers(p[2], varying) || return false
        moving = true
    end
    return moving
end

# True iff every varying symbol of `k` enters only through sums and differences, scaled at most by integer literals. A symbolic stride on
# the loop variable (`(ci - 1) * hw`, a channel-major layout whose threads cover one slice each) says nothing about overlap either way, so
# it stays with cgen_expr_injective_ok.
function cgen_varying_scaled_by_integers(k, varying::Set{Symbol})
    cgen_affine_invariant(k, varying) && return true
    k isa Symbol && return true
    k isa Expr && k.head == :call || return false
    if k.args[1] in (:+, :-)
        return all(a -> cgen_varying_scaled_by_integers(a, varying), k.args[2:end])
    elseif k.args[1] === :*
        moving = [f for f in k.args[2:end] if !cgen_affine_invariant(f, varying)]
        return length(moving) == 1 && all(f -> f isa Integer || f === moving[1], k.args[2:end]) && cgen_varying_scaled_by_integers(moving[1], varying)
    end
    return false
end

# Can two index vectors address one element from two DIFFERENT threads? Disjointness only has to be proven in ONE
# dimension, since a collision needs every dimension to agree -- that is what keeps `a[i, j]` and `a[i, j+1]` apart
# across threads however the inner index moves.
function cgen_writes_may_overlap(a::Vector, b::Vector, thread_var::Symbol, varying::Set{Symbol})
    length(a) == length(b) || return true
    for d in eachindex(a)
        pa = cgen_index_affine(a[d], thread_var)
        pb = cgen_index_affine(b[d], thread_var)
        (pa === nothing || pb === nothing) && continue
        (cgen_affine_invariant(pa[2], varying) && cgen_affine_invariant(pb[2], varying)) || continue
        pa[1] == pb[1] || continue
        # identical offsets: `c*t1 + k == c*t2 + k` forces t1 == t2, so no CROSS-thread collision
        pa[1] != 0 && pa[2] == pb[2] && return false
        ka, kb = pa[2], pb[2]
        (ka isa Integer && kb isa Integer) || continue
        # `c*t1 + ka == c*t2 + kb` needs c to divide the offset difference; c == 0 leaves two fixed addresses
        pa[1] == 0 ? (ka != kb && return false) : ((kb - ka) % pa[1] != 0 && return false)
    end
    return true
end

# `expr` written as `c * v + k` with `k` free of `v`, or nothing when it isn't affine in `v`. div/mod/rem and any array
# read are deliberately absent, so they fall through to nothing and the caller assumes overlap.
function cgen_index_affine(expr, v::Symbol)
    expr isa Symbol && return expr === v ? (1, 0) : (0, expr)
    expr isa Number && return expr isa Integer ? (0, expr) : nothing
    expr isa Expr || return nothing
    if expr.head == :call && expr.args[1] === :+ && length(expr.args) >= 2
        c, k = 0, 0
        for a in expr.args[2:end]
            p = cgen_index_affine(a, v)
            p === nothing && return nothing
            c += p[1]
            k = cgen_affine_add(k, p[2])
        end
        return (c, k)
    elseif expr.head == :call && expr.args[1] === :- && length(expr.args) == 3
        a = cgen_index_affine(expr.args[2], v)
        b = cgen_index_affine(expr.args[3], v)
        (a === nothing || b === nothing) && return nothing
        return (a[1] - b[1], cgen_affine_add(a[2], cgen_affine_neg(b[2])))
    elseif expr.head == :call && expr.args[1] === :- && length(expr.args) == 2
        a = cgen_index_affine(expr.args[2], v)
        a === nothing && return nothing
        return (-a[1], cgen_affine_neg(a[2]))
    elseif expr.head == :call && expr.args[1] === :* && length(expr.args) == 3
        a = cgen_index_affine(expr.args[2], v)
        b = cgen_index_affine(expr.args[3], v)
        (a === nothing || b === nothing) && return nothing
        a[1] == 0 && a[2] isa Integer && return (a[2] * b[1], cgen_affine_scale(a[2], b[2]))
        b[1] == 0 && b[2] isa Integer && return (b[2] * a[1], cgen_affine_scale(b[2], a[2]))
        a[1] == 0 && b[1] == 0 && return (0, Expr(:call, :*, a[2], b[2]))
        return nothing
    end
    return nothing
end

cgen_affine_add(a, b) =
    a isa Integer && b isa Integer ? a + b :
    a isa Integer && a == 0 ? b :
    b isa Integer && b == 0 ? a : Expr(:call, :+, a, b)
cgen_affine_neg(a) = a isa Integer ? -a : Expr(:call, :-, a)
cgen_affine_scale(s::Integer, a) = a isa Integer ? s * a : s == 1 ? a : Expr(:call, :*, s, a)

# Does `k` hold one value across every thread and every inner-loop iteration? Any symbol a loop or an assignment can
# move, and any array read, says no.
function cgen_affine_invariant(k, varying::Set{Symbol})
    k isa Integer && return true
    k isa Symbol && return !(k in varying)
    k isa Expr || return false
    k.head == :ref && return false
    return all(a -> cgen_affine_invariant(a, varying), k.args)
end

function cgen_collect_loop_vars!(body::Vector{NamedTuple}, vars::Set{Symbol})
    for stmt in body
        if stmt.kind == :for
            push!(vars, stmt.var)
            cgen_collect_loop_vars!(stmt.body, vars)
        elseif stmt.kind == :while
            cgen_collect_loop_vars!(stmt.body, vars)
        elseif stmt.kind == :if
            cgen_collect_loop_vars!(stmt.then, vars)
            cgen_collect_loop_vars!(stmt.els, vars)
        end
    end
    return nothing
end

# Array accesses with same-body scalar let-bindings folded in, so `b = i + 1; a[b]` is compared as `a[i + 1]` rather
# than as an opaque name. A binding is dropped the moment it stops being a straight-line fact: reassigned in a branch
# or a loop it outlives, array-derived, self-referential, or expanded past a size bound. A dropped name stays in
# `varying`, so the index reading it is simply never proven disjoint. Writes carry whether they are an additive
# self-reference, since only those have an atomic rewrite available.
function cgen_collect_subst_accesses!(body::Vector{NamedTuple}, subst::Dict{Symbol,Any}, writes::Dict{Symbol,Vector{Any}}, reads::Dict{Symbol,Vector{Any}})
    for stmt in body
        if stmt.kind == :assign
            if stmt.lhs isa Expr && stmt.lhs.head == :ref && stmt.lhs.args[1] isa Symbol
                idxs = Any[cgen_subst_expr(a, subst) for a in stmt.lhs.args[2:end]]
                additive = findfirst(t -> t == stmt.lhs, cgen_flatten_sum(stmt.rhs)) !== nothing
                push!(get!(writes, stmt.lhs.args[1], Any[]), (idxs, additive))
                for a in stmt.lhs.args[2:end]
                    cgen_collect_subst_refs!(a, subst, reads, nothing, Ref(false))
                end
                cgen_collect_subst_refs!(stmt.rhs, subst, reads, stmt.lhs, Ref(false))
            else
                cgen_collect_subst_refs!(stmt.rhs, subst, reads, nothing, Ref(false))
            end
            if stmt.lhs isa Symbol
                if cgen_expr_has_ref(stmt.rhs) || cgen_expr_contains(stmt.rhs, stmt.lhs)
                    delete!(subst, stmt.lhs)
                else
                    e = cgen_subst_expr(stmt.rhs, subst)
                    cgen_expr_node_count(e) <= 64 ? (subst[stmt.lhs] = e) : delete!(subst, stmt.lhs)
                end
            end
        elseif stmt.kind == :if
            cgen_collect_subst_refs!(stmt.cond, subst, reads, nothing, Ref(false))
            cgen_collect_subst_accesses!(stmt.then, copy(subst), writes, reads)
            cgen_collect_subst_accesses!(stmt.els, copy(subst), writes, reads)
            for v in cgen_all_assigned_scalars(stmt.then)
                delete!(subst, v)
            end
            for v in cgen_all_assigned_scalars(stmt.els)
                delete!(subst, v)
            end
        elseif stmt.kind == :for || stmt.kind == :while
            # A binding assigned anywhere in the body is cleared BEFORE the walk too: the body re-runs, so the value
            # standing at the body's first statement is the previous iteration's, not the one carried in from outside.
            inner = copy(subst)
            for v in cgen_all_assigned_scalars(stmt.body)
                delete!(inner, v)
                delete!(subst, v)
            end
            cgen_collect_subst_accesses!(stmt.body, inner, writes, reads)
        end
    end
    return nothing
end

function cgen_collect_subst_refs!(e, subst::Dict{Symbol,Any}, reads::Dict{Symbol,Vector{Any}}, skip, skipped::Base.RefValue{Bool})
    e isa Expr || return nothing
    if e.head == :ref && e.args[1] isa Symbol
        if skip !== nothing && !skipped[] && e == skip
            skipped[] = true
        else
            push!(get!(reads, e.args[1], Any[]), Any[cgen_subst_expr(a, subst) for a in e.args[2:end]])
        end
    end
    for a in e.args
        cgen_collect_subst_refs!(a, subst, reads, skip, skipped)
    end
    return nothing
end

function cgen_subst_expr(e, subst::Dict{Symbol,Any})
    e isa Symbol && return get(subst, e, e)
    e isa Expr || return e
    return Expr(e.head, Any[cgen_subst_expr(a, subst) for a in e.args]...)
end

cgen_expr_node_count(e) = e isa Expr ? 1 + sum(a -> cgen_expr_node_count(a), e.args; init = 0) : 1

# device-side body walk -- never sees :stackpush or a pop!-rhs assign,
# since cgen_body only reaches here for a loop cgen_contains_stackop
# already confirmed is clean at every depth
function cgen_device_body(body::Vector{NamedTuple}, thread_var::Symbol, backend, reduce_vars::Set{Symbol}, race_arrays::Set{Symbol} = Set{Symbol}(), thread_dep::Set{Symbol} = Set([thread_var]), injective_dep::Set{Symbol} = Set([thread_var]))
    exprs = Any[]
    for stmt in body
        if stmt.kind == :assign
            push!(exprs, cgen_device_assign(stmt, thread_var, thread_dep, injective_dep, backend, reduce_vars, race_arrays))
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
            push!(exprs, emit_if(stmt.cond, cgen_device_body(stmt.then, thread_var, backend, reduce_vars, race_arrays, copy(thread_dep), copy(injective_dep)), cgen_device_body(stmt.els, thread_var, backend, reduce_vars, race_arrays, copy(thread_dep), copy(injective_dep))))
        elseif stmt.kind == :for
            push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, cgen_device_body(stmt.body, thread_var, backend, reduce_vars, race_arrays, copy(thread_dep), copy(injective_dep))))
        elseif stmt.kind == :while
            # A while has no trip count, so no launch can size itself around
            # it and no thread can be assigned an iteration. cgen_body must
            # have kept the enclosing loop on the host before reaching here.
            error("cgen_device_body: a `while` cannot run inside a device kernel")
        end
    end
    return exprs
end

# ---- warp-aggregated atomics (CUDA) ----
# An atomic add whose address is the same for every lane of a warp (a weight-gradient accumulation, a boxed scalar reduction) serializes 32 lanes on one
# address. The helper sums the 32 values with shuffles and lets lane 0 do one atomic. It stays correct for any address: it aggregates only when all 32
# lanes are active and agree on the index, and issues the plain atomic otherwise. The summation order changes, so results differ from the plain atomic
# in the last bits; stade_gpu(...; warp_atomics = false) restores the plain atomic exactly.
cgen_warp_add_name() = :__stade_wa_add!
cgen_warp_atomics_on(backend) = get(backend, :warp_atomics, false) === true

# `backend.atomic_macro arr[idx] += val`, or a call of the warp helper for a single-index ref of a named array on a backend that has it
function cgen_atomic_add(backend, target, other)
    if cgen_warp_atomics_on(backend) && target isa Expr && target.head == :ref && length(target.args) == 2 && target.args[1] isa Symbol
        return Expr(:call, cgen_warp_add_name(), target.args[1], target.args[2], other)
    end
    return Expr(:macrocall, backend.atomic_macro, nothing, Expr(:(+=), target, other))
end

function cgen_expr_calls(expr, name::Symbol)
    expr isa Expr || return false
    expr.head == :call && !isempty(expr.args) && expr.args[1] === name && return true
    return any(a -> cgen_expr_calls(a, name), expr.args)
end

# Definitions the helper calls need, in file order. Shuffles move 32 bits at a time, so a Float64 or an index travels as two UInt32 halves. Lane ids
# assume the one-dimensional blocks cgen_launch_expr emits. The lane is 1-based for shfl_sync.
function cgen_warp_helper_exprs()
    src = """
    @inline function __stade_wa_xor(m::UInt32, u::UInt64, k::Int32)
        lo = shfl_xor_sync(m, UInt32(u & 0x00000000ffffffff), k)
        hi = shfl_xor_sync(m, UInt32(u >> 32), k)
        return (UInt64(hi) << 32) | UInt64(lo)
    end
    @inline function __stade_wa_lane1(m::UInt32, u::UInt64)
        lo = shfl_sync(m, UInt32(u & 0x00000000ffffffff), Int32(1))
        hi = shfl_sync(m, UInt32(u >> 32), Int32(1))
        return (UInt64(hi) << 32) | UInt64(lo)
    end
    @inline function __stade_wa_add!(arr, idx::Integer, val::Float64)
        m = active_mask()
        if m == 0xffffffff
            i = Int64(idx)
            if vote_all_sync(m, i == reinterpret(Int64, __stade_wa_lane1(m, reinterpret(UInt64, i))))
                v = val
                v += reinterpret(Float64, __stade_wa_xor(m, reinterpret(UInt64, v), Int32(16)))
                v += reinterpret(Float64, __stade_wa_xor(m, reinterpret(UInt64, v), Int32(8)))
                v += reinterpret(Float64, __stade_wa_xor(m, reinterpret(UInt64, v), Int32(4)))
                v += reinterpret(Float64, __stade_wa_xor(m, reinterpret(UInt64, v), Int32(2)))
                v += reinterpret(Float64, __stade_wa_xor(m, reinterpret(UInt64, v), Int32(1)))
                if ((threadIdx().x - 1) & 31) == 0
                    CUDA.@atomic arr[idx] += v
                end
                return nothing
            end
        end
        CUDA.@atomic arr[idx] += val
        return nothing
    end
    @inline function __stade_wa_add!(arr, idx::Integer, val::Float32)
        m = active_mask()
        if m == 0xffffffff
            i = Int64(idx)
            if vote_all_sync(m, i == reinterpret(Int64, __stade_wa_lane1(m, reinterpret(UInt64, i))))
                v = val
                v += shfl_xor_sync(m, v, Int32(16))
                v += shfl_xor_sync(m, v, Int32(8))
                v += shfl_xor_sync(m, v, Int32(4))
                v += shfl_xor_sync(m, v, Int32(2))
                v += shfl_xor_sync(m, v, Int32(1))
                if ((threadIdx().x - 1) & 31) == 0
                    CUDA.@atomic arr[idx] += v
                end
                return nothing
            end
        end
        CUDA.@atomic arr[idx] += val
        return nothing
    end
    @inline function __stade_wa_add!(arr, idx, val)
        CUDA.@atomic arr[idx] += val
        return nothing
    end
    """
    return Expr[cgen_strip_macro_lines(Base.remove_linenums!(a)) for a in Meta.parseall(src).args if a isa Expr]
end

# Base.remove_linenums! leaves the line slot of a :macrocall, which would print as a `#= none:1 =#` marker
function cgen_strip_macro_lines(expr)
    expr isa Expr || return expr
    args = Any[cgen_strip_macro_lines(a) for a in expr.args]
    expr.head == :macrocall && length(args) >= 2 && args[2] isa LineNumberNode && (args[2] = nothing)
    return Expr(expr.head, args...)
end

# A write races across threads unless the enclosing device loop's thread-mapped variable occurs in the write's index, or through a chain of let-
# bindings (`thread_dep`) -- a real occurs-check. Any thread-invariant-indexed write is a race regardless of pattern: an accumulation is fixable
# with an atomic +=, a plain replacement is not, so it's refused outright. A thread-DEPENDENT index isn't automatically race-free either: a gather-
# derived index is thread-dependent but not injective, and gets the same atomic treatment as the invariant case; a non-additive write through it is
# left untouched, since an atomic can't fix a replacement race either way.
#
# An index that passes both tests is still only proven unique PER THREAD. `race_arrays` carries the separate cross-write
# verdict from cgen_race_write_arrays, forcing the atomic rewrite for every write to an array whose accumulation sites
# can land on one element from two threads.
function cgen_device_assign(stmt, thread_var::Symbol, thread_dep::Set{Symbol}, injective_dep::Set{Symbol}, backend, reduce_vars::Set{Symbol}, race_arrays::Set{Symbol} = Set{Symbol}())
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
        return cgen_atomic_add(backend, Expr(:ref, stmt.lhs, 1), other)
    end
    if stmt.lhs isa Expr && stmt.lhs.head == :ref
        overlapping_writes = stmt.lhs.args[1] isa Symbol && stmt.lhs.args[1] in race_arrays
        thread_invariant = !cgen_expr_contains_any(stmt.lhs.args[2:end], thread_dep)
        needs_atomic_check = thread_invariant || overlapping_writes || !cgen_expr_injective_ok(stmt.lhs.args[2:end], injective_dep, thread_dep)
        if needs_atomic_check
            terms = cgen_flatten_sum(stmt.rhs)
            self_idx = findfirst(t -> t == stmt.lhs, terms)
            if self_idx !== nothing
                other = cgen_sum_excluding(terms, self_idx)
                return cgen_atomic_add(backend, stmt.lhs, other)
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

function cgen_emit(gk, backend; reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT)
    kernels = Expr[]
    reduce_vars = Set{Symbol}()
    fn_args = Set{Symbol}(gk.args)
    outer_defs = cgen_scalar_def_map(gk.body)
    decisions = NamedTuple[]
    host_body = cgen_body(gk.body, kernels, gk.name, backend, reduce_vars, fn_args; reduction_threshold, outer_defs, decisions)
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
    helpers = any(k -> cgen_expr_calls(k, cgen_warp_add_name()), kernels) ? cgen_warp_helper_exprs() : Expr[]
    return (host = host, kernels = kernels, decisions = decisions, helpers = helpers)
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

# A matched loop is replaced by one `JACC.@parallel_reduce` call
# instead of a synthesized per-element atomic kernel. Writing its plain host-scalar result back into `target` (a
# device-array-backed ref) via a bare host assignment is a host setindex! against a device array; unlike
# CUDA/AMDGPU/Metal, JACC has no allowscalar escape hatch (confirmed live: 'Scalar indexing is disallowed'). Fix:
# accumulate on-device via a synthesized range=1 kernel instead.
function jgen_body(body::Vector{NamedTuple}, kernels::Vector{Expr}, owner::Symbol, reduce_vars::Set{Symbol}, fn_args::Set{Symbol}, allowscalar_macro; reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT, outer_defs::Dict{Symbol,Any} = Dict{Symbol,Any}(), outer_known_consts::Dict{Symbol,Any} = Dict{Symbol,Any}(), root_body = body, decisions::Union{Nothing,Vector{NamedTuple}} = nothing, depth::Int = 0)
    exprs = Any[]
    # See cgen_body's matching comment: seeded from the enclosing body's own known_consts so a
    # reduction scalar reset at the tail of a non-split ancestor loop's body stays provably known
    # this many nesting levels down, not just when the resetting literal is an immediate sibling.
    known_consts = copy(outer_known_consts)
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
        elseif stmt.kind == :while
            # A `while` stays on the host. No launch can size itself around a
            # loop whose trip count is unknown until it has run, and no thread
            # can be assigned an iteration. Loops INSIDE the body still offload
            # normally, which is why this recurses rather than refusing.
            #
            # A missing clause here would be silent: the statement would fall
            # through and vanish from the generated host function.
            flush_pending!()
            push!(exprs, emit_while(stmt.cond, jgen_body(stmt.body, kernels, owner, reduce_vars, fn_args, allowscalar_macro; reduction_threshold, outer_defs, outer_known_consts = known_consts, root_body, decisions, depth = depth + 1)))
        elseif stmt.kind == :if
            flush_pending!()
            # A host-side branch condition that reads a device array element needs the same @allowscalar
            # wrapping cgen_body gives it. jgen_body wrapped the `pending` assignments but not the
            # condition itself, so an entry branch keyed on an array element died with 'Scalar indexing
            # is disallowed' on JACC while the CUDA path, which wraps the condition, ran fine. Found by
            # the first full-corpus live GPU run.
            cond = cgen_expr_has_ref(stmt.cond) && allowscalar_macro !== nothing ?
                   Expr(:macrocall, allowscalar_macro, nothing, stmt.cond) : stmt.cond
            push!(exprs, emit_if(cond, jgen_body(stmt.then, kernels, owner, reduce_vars, fn_args, allowscalar_macro; reduction_threshold, outer_defs, outer_known_consts = known_consts, root_body, decisions, depth = depth + 1), jgen_body(stmt.els, kernels, owner, reduce_vars, fn_args, allowscalar_macro; reduction_threshold, outer_defs, outer_known_consts = known_consts, root_body, decisions, depth = depth + 1)))
            for v in cgen_all_assigned_scalars(vcat(stmt.then, stmt.els))
                delete!(known_consts, v)
            end
        elseif stmt.kind == :for
            flush_pending!()
            gate = cgen_split_loop_gate(stmt.body, stmt.var, known_consts, outer_defs; root_body, protect = Set{Symbol}(fn_args))
            synth = gate.synth
            loop_reduce_vars = synth === nothing ? Set{Symbol}() : cgen_scalar_reduction_vars(stmt.body)
            safe_scope = issubset(loop_reduce_vars, fn_args)
            # `stage` names the first gate that refuses the loop, in the order the gates run
            stage = synth === nothing ? gate.reason : !safe_scope ? :scope : cgen_contains_stackop(stmt.body) ? :stackop : :ok
            # see cgen_kernel_def_ok -- the last gate, and the only one that asks the backend's own
            # codegen rather than a structural approximation of it
            stage === :ok && !cgen_kernel_def_ok(() -> jgen_kernel_def(stmt, owner, 0, cgen_free_vars(stmt, stmt.var), loop_reduce_vars, synth)) && (stage = :kernel_def)
            # Same live-out refusal cgen_body applies -- jgen_body had no equivalent, and no root_body
            # to apply it against. A split loop's scalars live on the device; a later host statement
            # reading one gets the stale host copy instead. ii_kill is the witness: cgen keeps that loop
            # on the host, jgen offloaded it and the host's v stayed 0.0 -- silent wrong gradients,
            # measured at max_rel_err 0.85 (adjoint) and 1.23 (HVP) live while the CUDA path was exact.
            stage === :ok && !cgen_liveout_is_zeroed(stmt, root_body, cgen_collect_loop_vars(root_body, Set{Symbol}()), synth === nothing ? Dict{Symbol,Any}() : synth, gate.saveonly) && (stage = :liveout)
            eligible = stage === :ok
            verdict = :host
            if eligible
                red = cgen_idiomatic_scalar_reduction(stmt.body, stmt.var)
                verdict = red !== nothing ? :reduction : :kernel
                if red !== nothing
                    target, op, arrs, term = red
                    n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)
                    value = jgen_idiomatic_reduction_value(arrs, term, stmt.var, stmt.lo, stmt.step, n_iter)
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
                    # NOT boxed via JACC.array here, despite that being jgen_emit's reduce_vars convention: a live GPU
                    # diagnostic confirmed `JACC.@parallel_reduce` already returns a device-resident `CuArray{Float64,1}`
                    # directly, not a host Float64. Wrapping it again via `JACC.array([redvar])` (an earlier fix) tried to
                    # build a CuArray whose element type is itself a CuArray -- illegal, confirmed live. `redvar` is
                    # passed straight through as the kernel argument; the kernel indexes `[1]` on it on-device.
                    push!(kernels, jgen_reduction_writeback_kernel(owner, widx, target, wfargs))
                    # Same run-time trip-count test cgen_body applies. JACC
                    # reaches its reductions through JACC.@parallel_reduce
                    # rather than mapreduce, so the crossover is its own
                    # measurement, but the shape of the problem is identical:
                    # a tree reduction pays a fixed cost a short loop cannot
                    # amortise, while an atomic on one address serialises its
                    # threads. Both kernels are emitted and only the launch
                    # is chosen.
                    #
                    # Without this = false lowered every
                    # JACC reduction unconditionally at any trip count. That
                    # is the state the CUDA path was in at v0.2.5, measured
                    # there as a regression below the threshold.
                    # The same `> 0` guard jgen_launch_expr puts on a @parallel_for, for the same reason:
                    # JACC computes its grid from `range` and a zero range is not a no-op but an error
                    # ("Grid dimensions CuDim3(0x0, 0x1, 0x1) are not positive", measured on a V100 for all
                    # four of lo/step/direction witness kernels). The atomic branch was already guarded,
                    # and the `n_iter < reduction_threshold` test above sends a zero trip count there at any
                    # positive threshold -- which is why this was latent rather than visible. At
                    # reduction_threshold = 0 the test is `0 < 0`, false, and the reduce branch ran empty.
                    # CUDA has no matching hazard: its mapreduce carries `init`, so it returns zero.
                    red_exprs = Any[emit_if(Expr(:call, :>, n_iter, 0),
                                            Any[Expr(:(=), redvar, signed_value),
                                                jgen_reduction_writeback_launch(owner, widx, wfargs, redvar)],
                                            Any[])]
                    aidx = length(kernels) + 1
                    afargs = cgen_free_vars(stmt, stmt.var)
                    union!(reduce_vars, loop_reduce_vars)
                    push!(kernels, jgen_kernel_def(stmt, owner, aidx, afargs, loop_reduce_vars, synth))
                    push!(exprs, emit_if(Expr(:call, :<, n_iter, reduction_threshold),
                                         Any[jgen_launch_expr(stmt, owner, aidx, afargs)],
                                         red_exprs))
                else
                    idx = length(kernels) + 1
                    fargs = cgen_free_vars(stmt, stmt.var)
                    union!(reduce_vars, loop_reduce_vars)
                    push!(kernels, jgen_kernel_def(stmt, owner, idx, fargs, loop_reduce_vars, synth))
                    push!(exprs, jgen_launch_expr(stmt, owner, idx, fargs))
                end
            else
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, jgen_body(stmt.body, kernels, owner, reduce_vars, fn_args, allowscalar_macro; reduction_threshold, outer_defs, outer_known_consts = known_consts, root_body, decisions, depth = depth + 1)))
            end
            decisions === nothing || push!(decisions, cgen_decision_record(owner, stmt, depth, verdict, stage, gate.array, gate.idiom))
            delete!(known_consts, stmt.var)
            # See cgen_body's matching comment: update via cgen_loop_convergent_constant rather
            # than blindly deleting -- a scalar that provably converges to a literal by this loop's
            # own tail stays known for statements (or further sibling loops) that follow.
            for v in cgen_all_assigned_scalars(stmt.body)
                c = cgen_loop_convergent_constant(stmt.body, v, get(known_consts, v, nothing))
                # The convergent constant only describes the value after the loop if the loop actually ran. A runtime bound
                # may give zero iterations, in which case `v` still holds what it held coming in -- unless that was already
                # the same constant. Recording it unconditionally lets cgen_reduction_only_loop synthesise `v = c` for a
                # LATER loop; every thread starts from c instead of the sequential value. ii_kill is the witness: the GPU
                # adjoint came out wrong by 0.98 while the CPU one was correct.
                if c === nothing || !(cgen_loop_runs_once(stmt) || get(known_consts, v, nothing) == c)
                    delete!(known_consts, v)
                else
                    known_consts[v] = c
                end
            end
        end
    end
    flush_pending!()
    return exprs
end

# JACC hands the loop index in directly as the split-off function's first parameter -- no
# thread-index intrinsic to bind, unlike cgen_kernel_def, since JACC.@parallel_for range=N
# already guarantees the index range. cgen_loopvar_from_tid still does the affine lo/step
# remapping, same as for cgen_.
# A kernel with more free variables than this takes them packed in one tuple. JACC splats its varargs into the kernel, and past Julia's
# MAX_TUPLE_SPLAT the compiler emits a dynamic jl_f__apply_iterate that no GPU can run (ttgc's 42-argument HVP kernel, measured on a V100).
const JGEN_MAX_SPLAT_ARGS = 32

function jgen_kernel_def(stmt, owner::Symbol, idx::Int, fargs::Vector{Symbol}, reduce_vars::Set{Symbol}, synth::Dict{Symbol,Any} = Dict{Symbol,Any}())
    jidx = :__jacc_i
    packed = length(fargs) > JGEN_MAX_SPLAT_ARGS
    # A packed kernel takes its free variables as one tuple and unpacks them first, so JACC's splat sees a single argument.
    body = Any[Expr(:(=), v, Expr(:ref, :__jargs, k)) for (k, v) in enumerate(fargs) if packed]
    push!(body, Expr(:(=), stmt.var, cgen_loopvar_from_tid(stmt.lo, stmt.step, jidx)))
    # see cgen_kernel_def's matching comment: cgen_reduction_only_loop's
    # convergent-constant proof injects each synthesized initializer
    # here, before the loop's own body runs.
    for (v, c) in synth
        push!(body, Expr(:(=), v, c))
    end
    fbody = cgen_fold_accumulates(stmt.body)
    append!(body, jgen_device_body(fbody, stmt.var, reduce_vars, cgen_race_write_arrays(fbody, stmt.var)))
    push!(body, emit_return_nothing())
    return Expr(:function, Expr(:call, jgen_kernel_fname(owner, idx), jidx, (packed ? Any[:__jargs] : fargs)...), Expr(:block, body...))
end

# JACC.jl v1.x: JACC.@parallel_for range=N f(args...) -- a macro with a `range=` keyword, not a plain call; the kernel
# function's own signature is unchanged, only the launch call's shape moved. A zero-trip loop is a legal CPU no-op, but
# `JACC.@parallel_for range=0` raises DivideError on the CUDA backend, since JACC computes its block count with `cld` against
# the range -- confirmed twice on real hardware. cgen_launch_expr survives the same case with one block whose threads fail the
# device kernel's own guard, but JACC's plain-indexed-function model has no such guard, so the launch itself must be skipped.
function jgen_launch_expr(stmt, owner::Symbol, idx::Int, fargs::Vector{Symbol})
    n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)
    packed = length(fargs) > JGEN_MAX_SPLAT_ARGS
    tname = Symbol("__jargs_", owner, "_", idx)
    launch = Expr(:macrocall, Expr(:., :JACC, QuoteNode(Symbol("@parallel_for"))), nothing,
                  Expr(:(=), :range, n_iter),
                  Expr(:call, jgen_kernel_fname(owner, idx), (packed ? Any[tname] : fargs)...))
    # The tuple is built in its own host statement, never inside the launch macro's argument list.
    return Expr(:if, Expr(:call, :>, n_iter, 0), Expr(:block, (packed ? Any[Expr(:(=), tname, Expr(:tuple, fargs...))] : Any[])..., launch))
end

function jgen_device_body(body::Vector{NamedTuple}, thread_var::Symbol, reduce_vars::Set{Symbol}, race_arrays::Set{Symbol} = Set{Symbol}(), thread_dep::Set{Symbol} = Set([thread_var]), injective_dep::Set{Symbol} = Set([thread_var]))
    exprs = Any[]
    for stmt in body
        if stmt.kind == :assign
            push!(exprs, jgen_device_assign(stmt, thread_var, thread_dep, injective_dep, reduce_vars, race_arrays))
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
            push!(exprs, emit_if(stmt.cond, jgen_device_body(stmt.then, thread_var, reduce_vars, race_arrays, copy(thread_dep), copy(injective_dep)), jgen_device_body(stmt.els, thread_var, reduce_vars, race_arrays, copy(thread_dep), copy(injective_dep))))
        elseif stmt.kind == :for
            push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, jgen_device_body(stmt.body, thread_var, reduce_vars, race_arrays, copy(thread_dep), copy(injective_dep))))
        elseif stmt.kind == :while
            # A while has no trip count, so no launch can size itself around
            # it and no thread can be assigned an iteration. jgen_body must
            # have kept the enclosing loop on the host before reaching here.
            error("jgen_device_body: a `while` cannot run inside a device kernel")
        end
    end
    return exprs
end

# Identical decision to cgen_device_assign, reusing cgen_'s occurs-check, injectivity-check, and sum-flattening helpers --
# only the atomic macro's target module is fixed. The scalar-reduction branch mirrors cgen_device_assign's exactly:
# reduce_vars was boxed into a 1-element JACC.array by jgen_emit, so `cb = cb + other` becomes an atomic add against index
# 1. The array-ref branch now also mirrors the thread-invariant refusal and injective_dep distinction -- previously this
# used a bare, non-transitive thread_var check with no refusal path for a genuine non-additive race.
function jgen_device_assign(stmt, thread_var::Symbol, thread_dep::Set{Symbol}, injective_dep::Set{Symbol}, reduce_vars::Set{Symbol}, race_arrays::Set{Symbol} = Set{Symbol}())
    if stmt.lhs isa Symbol && stmt.lhs in reduce_vars
        terms = cgen_flatten_sum(stmt.rhs)
        self_idx = findfirst(t -> t == stmt.lhs, terms)
        self_idx === nothing && error("jgen_device_assign: `$(stmt.lhs)` was classified as a scalar reduction target (see cgen_scalar_reduction_vars) but `$(stmt.lhs) = $(stmt.rhs)` doesn't self-reference -- this should be unreachable")
        other = cgen_sum_excluding(terms, self_idx)
        return Expr(:macrocall, Expr(:., :Atomix, QuoteNode(Symbol("@atomic"))), nothing,
                    Expr(:(+=), Expr(:ref, stmt.lhs, 1), other))
    end
    if stmt.lhs isa Expr && stmt.lhs.head == :ref
        overlapping_writes = stmt.lhs.args[1] isa Symbol && stmt.lhs.args[1] in race_arrays
        thread_invariant = !cgen_expr_contains_any(stmt.lhs.args[2:end], thread_dep)
        needs_atomic_check = thread_invariant || overlapping_writes || !cgen_expr_injective_ok(stmt.lhs.args[2:end], injective_dep, thread_dep)
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

function jgen_emit(gk; reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT, allowscalar_macro = jgen_default_allowscalar_macro())
    kernels = Expr[]
    reduce_vars = Set{Symbol}()
    fn_args = Set{Symbol}(gk.args)
    outer_defs = cgen_scalar_def_map(gk.body)
    decisions = NamedTuple[]
    host_body = jgen_body(gk.body, kernels, gk.name, reduce_vars, fn_args, allowscalar_macro; reduction_threshold, outer_defs, decisions)
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
    return (host = host, kernels = kernels, decisions = decisions)
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
                                scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
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
# A scalar_int arg is often an iteration count -- more iterations means more chances for per-step amplification to blow up. Narrowing the
# range's upper end after a divergence, rather than redrawing, targets that directly. Allowed to fall to 1 (int_lo=0 from the caller is still
# honored, deliberately drawing zero-trip loops -- see validate_zerotrip.jl). Six separate analyses used to treat a loop that may not run as
# having run, producing silently wrong gradients, until a kernel was hand-written to retire a bound past zero; a draw carrying no signal
# passing trivially is now a rejection criterion below, not an assumption here.
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
# A candidate whose observed output is identically zero exercises nothing: every derivative check
# compares 0 against 0 and passes whatever the generated code does. That is the real hazard behind
# the old "trivially pass" worry, and it bites hardest at small int args, where a kernel can
# degenerate to running no iterations at all. Rejecting here is what makes int_lo = 0 usable: the
# draws that keep some loops running survive, the ones that flatten the whole kernel are redrawn.
                if maximum(abs.(y0); init = 0.0) == 0.0
                    error("candidate baseline's observed primal output is identically zero -- " *
                          "nothing would be exercised, so every derivative check would pass vacuously")
                end
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


# ==================== bgen_* (data-parallel epilogue) =============
# Every rank runs the unmodified single-sample kernel on its own
# samples. This stage derives which buffers must be summed across ranks
# and which must not, then generates the reset and collective functions
# that surround the call. It reads the primal only: the role table and
# the companion names follow from the primal signature and the frozen
# argument layout, so no generated derivative file is an input.
#
# The user declares one fact. Which read-only arrays change from one
# sample to the next is a property of the caller's outer loop, not of
# the kernel: in `a[i] * u[i]` the two factors are symmetric, so no
# analysis separates a parameter from an input feature.

# Array-valued float arguments the kernel never writes. Only these can
# be named in `per_sample`; everything else already has a derived role.
function bgen_readonly_float_args(kernel)
    written = Set{Symbol}()
    bgen_collect_written!(kernel.body, written)
    return [a for a in kernel.sig.args
            if kernel.sig.kinds[a] == :array_float && !(a in written)]
end

# Union collector: an unexecuted loop can only leave names out, never
# add a false one, so no runs-once proof is needed here.
function bgen_collect_written!(body::Vector{NamedTuple}, out::Set{Symbol})
    for st in body
        if st.kind == :assign
            lhs = st.lhs
            lhs isa Symbol && push!(out, lhs)
            lhs isa Expr && lhs.head == :ref && push!(out, lhs.args[1])
        elseif st.kind == :for
            bgen_collect_written!(st.body, out)
        elseif st.kind == :while
            bgen_collect_written!(st.body, out)
        elseif st.kind == :if
            bgen_collect_written!(st.then, out)
            bgen_collect_written!(st.els, out)
        end
    end
    return out
end

# `per_sample` and `reduced` are user-facing channels, not classes. A name
# in `per_sample` resolves to class :per_sample when the kernel does not
# write it, and to :scratch when it does. Both get no reset and no
# collective, and only the read-only one can be checked for differing
# across ranks, so keeping them one class would overload it.
function bgen_check_declaration(kernel, per_sample::Vector{Symbol},
                                reduced::Vector{Symbol} = Symbol[])
    args = Set(kernel.sig.args)
    written = Set{Symbol}()
    bgen_collect_written!(kernel.body, written)
    _, ambiguous = bgen_reduced_outputs(kernel)
    amb = Set(ambiguous)

    for (kw, names) in (("per_sample", per_sample), ("reduced", reduced))
        not_args = [p for p in names if !(p in args)]
        isempty(not_args) || error("bgen_check_declaration: $kw names `" *
            join(not_args, "`, `") * "`, which " * (length(not_args) == 1 ? "is" : "are") *
            " not an argument of $(kernel.sig.name)")
        not_float_array = [p for p in names if kernel.sig.kinds[p] != :array_float]
        isempty(not_float_array) || error("bgen_check_declaration: $kw names `" *
            join(not_float_array, "`, `") * "`, which is not a float array argument")
        dups = [p for p in unique(names) if count(==(p), names) > 1]
        isempty(dups) || error("bgen_check_declaration: $kw names `" *
            join(dups, "`, `") * "` more than once")
    end

    both = [p for p in per_sample if p in reduced]
    isempty(both) || error("bgen_check_declaration: `" * join(both, "`, `") *
        "` appears in per_sample and in reduced. An array is one or the other")

    # A written array may be named only when its shape is genuinely
    # ambiguous. A plain overwrite already has one derived role, so naming
    # it would give it two.
    bad_ps = [p for p in per_sample if p in written && !(p in amb)]
    isempty(bad_ps) || error("bgen_check_declaration: per_sample names `" *
        join(bad_ps, "`, `") * "`, which the kernel writes by plain overwrite. " *
        "That array already takes a derived role, so naming it here would give it two")
    bad_rd = [p for p in reduced if !(p in amb)]
    isempty(bad_rd) || error("bgen_check_declaration: reduced names `" *
        join(bad_rd, "`, `") * "`, whose role is already derived. `reduced` " *
        "resolves an array whose writes are all additive at an index that varies " *
        "with an enclosing loop, and no other")

    # Every ambiguous array must be resolved. Leaving one out is the error
    # this refusal exists to catch: both readings run without complaint and
    # only one is right.
    unresolved = [a for a in ambiguous if !(a in per_sample) && !(a in reduced)]
    isempty(unresolved) || error("bgen_check_declaration: `" *
        join(unresolved, "`, `") * "` takes only additive writes at an index that " *
        "varies with an enclosing loop. Two readings fit that shape and the kernel " *
        "does not say which. Name it in `per_sample` if it holds one sample's own " *
        "result, which needs no communication. Name it in `reduced` if it " *
        "accumulates across samples, which is summed across ranks")
    return nothing
end

# A `:reduced` output accumulates across samples, so the epilogue sums it
# across ranks. Two conditions, both necessary.
#
# Every write must add the array to itself. A plain overwrite means the
# array carries no value in from the caller, so it is scratch. `agg` in
# mpnn is excluded by this: the kernel zeroes it first.
#
# The index must not contain an enclosing loop variable. `loss[1]` hits
# one slot on every iteration and is a reduction. `v[i_x]` hits a
# different slot each time, which is an elementwise accumulate into
# caller-zeroed storage and holds a per-sample result. The two have the
# same write shape and differ only here, so an index test is the only
# thing separating a quantity to sum from one to keep local.
function bgen_reduced_outputs(kernel)
    acc = Dict{Symbol,Symbol}()   # array => :reduced | :ambiguous | :scratch
    bgen_classify_writes!(kernel.body, Set{Symbol}(), acc)
    return sort([a for (a, v) in acc if v == :reduced]),
           sort([a for (a, v) in acc if v == :ambiguous])
end

# `dep` holds every symbol whose value can vary with an enclosing loop
# index. It is seeded with the loop variables and grows through scalar
# assignments, because a flattened nest reaches its indices through
# block-local scalars: matvec_loss writes v[i_i] with
# i_i = div(idx - 1, i_n) + 1, so a test against the loop variable alone
# sees a loop-invariant index and calls a per-sample accumulator a
# reduction.
#
# The set only grows. A symbol is never removed on reassignment, so the
# analysis over-approximates dependence and errs toward :ambiguous, which
# refuses. That is the safe direction: a refusal is read by a human, and
# a wrong :reduced sums a per-sample result across ranks in silence.
function bgen_classify_writes!(body::Vector{NamedTuple}, dep::Set{Symbol},
                               acc::Dict{Symbol,Symbol})
    for st in body
        if st.kind == :assign
            lhs = st.lhs
            if lhs isa Expr && lhs.head == :ref
                arr = lhs.args[1]
                verdict = if !bgen_adds_to_itself(lhs, st.rhs)
                    :scratch
                elseif bgen_expr_uses_any(lhs.args[2:end], dep)
                    :ambiguous
                else
                    :reduced
                end
                # :scratch wins over everything: one plain overwrite means
                # the array does not carry a value in from the caller.
                prev = get(acc, arr, nothing)
                acc[arr] = (prev === :scratch || verdict === :scratch) ? :scratch :
                           (prev === :ambiguous || verdict === :ambiguous) ? :ambiguous : :reduced
            elseif lhs isa Symbol && bgen_expr_uses_any(Any[st.rhs], dep)
                push!(dep, lhs)
            end
        elseif st.kind == :for
            inner = copy(dep)
            push!(inner, st.var)
            bgen_classify_writes!(st.body, inner, acc)
        elseif st.kind == :while
            bgen_classify_writes!(st.body, inner, acc)
        elseif st.kind == :if
            bgen_classify_writes!(st.then, copy(dep), acc)
            bgen_classify_writes!(st.els, copy(dep), acc)
        end
    end
    return acc
end

# True when `rhs` is a sum with `lhs` as one of its terms.
function bgen_adds_to_itself(lhs, rhs)
    terms = bgen_flatten_sum(rhs)
    return any(t -> t == lhs, terms)
end

function bgen_flatten_sum(expr)
    if expr isa Expr && expr.head == :call && expr.args[1] == :+
        out = Any[]
        for a in expr.args[2:end]
            append!(out, bgen_flatten_sum(a))
        end
        return out
    end
    return Any[expr]
end

function bgen_expr_uses_any(exprs, names)
    isempty(names) && return false
    for e in exprs
        bgen_expr_uses_any_one(e, names) && return true
    end
    return false
end

function bgen_expr_uses_any_one(expr, names)
    expr isa Symbol && return expr in names
    if expr isa Expr
        for a in expr.args
            bgen_expr_uses_any_one(a, names) && return true
        end
    end
    return false
end

# Companion names for one primal argument, in every mode. The layout is
# frozen: agen_signature_args interleaves the adjoint after each float
# argument, and hvp appends tgen_shadow(a) and tgen_shadow(agen_shadow(a))
# for each. Building the names from those helpers keeps this stage
# correct if the shadow spelling ever changes.
function bgen_companion_map(sig, mode::Symbol)
    mode in (:tangent, :adjoint, :hvp) ||
        error("bgen_companion_map: mode must be :tangent, :adjoint or :hvp, got $mode")
    out = Dict{Symbol,NamedTuple}()
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        tangent = mode in (:tangent, :hvp) ? tgen_shadow(a) : nothing
        adjoint = mode in (:adjoint, :hvp) ? agen_shadow(a) : nothing
        adjtan  = mode == :hvp ? tgen_shadow(agen_shadow(a)) : nothing
        out[a] = (primal = a, tangent = tangent, adjoint = adjoint, adjoint_of_tangent = adjtan)
    end
    return out
end

# The role of every buffer, from two axes. The class of the primal
# argument, and whether the buffer is an input (primal- or
# tangent-valued) or an output (adjoint-valued).
#
#   class         input buffer     adjoint-valued buffer
#   :shared       :shared          :accum
#   :reduced      :reduced         :seed
#   :per_sample   :per_sample      :local
#   :scratch      :scratch         :local
#
# :shared and :reduced swap columns, which is the duality this design
# rests on: the adjoint of a broadcast is a sum, and the adjoint of a sum
# is a broadcast.
function bgen_roles(kernel, per_sample::Vector{Symbol}, mode::Symbol;
                   reduced::Vector{Symbol} = Symbol[])
    bgen_check_declaration(kernel, per_sample, reduced)
    derived_reduced, _ = bgen_reduced_outputs(kernel)
    readonly = bgen_readonly_float_args(kernel)
    written = Set{Symbol}()
    bgen_collect_written!(kernel.body, written)
    all_reduced = union(Set(derived_reduced), Set(reduced))
    class = Dict{Symbol,Symbol}()
    for a in kernel.sig.args
        kernel.sig.kinds[a] in (:scalar_float, :array_float) || continue
        # a declared per_sample name that the kernel writes is a per-rank
        # output, which is class :scratch; only a read-only one is an input
        # the caller fills and can be checked for differing across ranks
        class[a] = (a in per_sample && !(a in written)) ? :per_sample :
                   (a in per_sample)                    ? :scratch :
                   a in all_reduced                     ? :reduced :
                   a in readonly                        ? :shared : :scratch
    end
    companions = bgen_companion_map(kernel.sig, mode)
    roles = Dict{Symbol,Symbol}()
    for (a, c) in class
        cm = companions[a]
        input_role = c
        # A scalar float argument's adjoint is RETURNED by value, not
        # accumulated into a buffer. It can be neither fill!ed nor
        # Allreduce!d in place, so it takes its own role and stays out of
        # every emitted function. The header table lists it, because a
        # buffer omitted in silence is the failure mode this design spends
        # its whole guard budget avoiding.
        out_role = kernel.sig.kinds[a] == :scalar_float ? :returned :
                   c == :shared ? :accum : c == :reduced ? :seed : :local
        roles[cm.primal] = input_role
        cm.tangent === nothing || (roles[cm.tangent] = input_role)
        cm.adjoint === nothing || (roles[cm.adjoint] = out_role)
        cm.adjoint_of_tangent === nothing || (roles[cm.adjoint_of_tangent] = out_role)
    end
    return roles
end

# Buffers carrying one role, in signature order. The epilogue needs a
# deterministic argument order, and the roles Dict has none.
function bgen_role_args(kernel, roles::Dict{Symbol,Symbol}, mode::Symbol, want::Symbol;
                        arrays_only::Bool = false)
    layout = mode == :tangent ? tgen_signature_args(kernel.sig) :
             mode == :adjoint ? agen_signature_args(kernel.sig) :
             bgen_hvp_signature_args(kernel.sig)
    out = [b for b in layout if get(roles, b, nothing) === want]
    arrays_only || return out
    scalars = bgen_scalar_float_buffers(kernel.sig, mode)
    return [b for b in out if !(b in scalars)]
end

# Every buffer that is a scalar rather than an array, in this mode's
# layout. fill! and Allreduce! both need an array.
function bgen_scalar_float_buffers(sig, mode::Symbol)
    out = Set{Symbol}()
    for a in sig.args
        sig.kinds[a] == :scalar_float || continue
        cm = bgen_companion_map(sig, mode)[a]
        push!(out, cm.primal)
        cm.tangent === nothing || push!(out, cm.tangent)
        cm.adjoint === nothing || push!(out, cm.adjoint)
        cm.adjoint_of_tangent === nothing || push!(out, cm.adjoint_of_tangent)
    end
    return out
end

function bgen_hvp_signature_args(sig)
    seed_args = Symbol[]
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        push!(seed_args, tgen_shadow(a))
        push!(seed_args, tgen_shadow(agen_shadow(a)))
    end
    return vcat(agen_signature_args(sig), seed_args)
end

# ---- phase 5: an optimization, not a requirement ------------------
# bgen_reset_local! already makes a stale :local buffer unrepresentable.
# This proves the write unnecessary for a buffer the sweep leaves at
# zero, so the epilogue can skip one fill!.
#
# The recursion is deliberately the same shape as
# cgen_last_assign_is_zero, including `found = cgen_loop_runs_once(st) &&
# ...`. That shape is where all six instances of the zero-trip bug lived,
# and copying the corrected one is safer than inventing a new one. A loop
# whose bounds are not literal proves nothing, and an unprovable write
# also destroys a fact established earlier, so `found` is assigned rather
# than or-ed.
#
# Coverage is the obligation the scalar version does not have. An array
# needs every element the sweep touched to be zero, not merely one write
# of 0.0 somewhere. This proves that only when EVERY write to the array
# is a zeroing write, which removes the need to compare index ranges: if
# nothing ever writes a non-zero, no touched element can hold one.
function bgen_array_left_zeroed(body::Vector{NamedTuple}, arr::Symbol)
    bgen_all_writes_are_zero(body, arr) || return false
    return bgen_zero_write_reached(body, arr)
end

function bgen_all_writes_are_zero(body::Vector{NamedTuple}, arr::Symbol)
    for st in body
        if st.kind == :assign && st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == arr
            (st.rhs isa Number && st.rhs == 0.0) || return false
        elseif st.kind == :assign && st.lhs isa Symbol && st.lhs == arr
            return false
        elseif st.kind == :for
            bgen_all_writes_are_zero(st.body, arr) || return false
        elseif st.kind == :while
            bgen_all_writes_are_zero(st.body, arr) || return false
        elseif st.kind == :if
            bgen_all_writes_are_zero(st.then, arr) || return false
            bgen_all_writes_are_zero(st.els, arr) || return false
        end
    end
    return true
end

# `loopvars` carries the enclosing loop variables. A zeroing write proves
# coverage only when its index IS one of them: vb[3] = 0.0 zeroes one
# element, not the array, and an earlier version of this function read it
# as a whole-array proof. An index that merely mentions a loop variable
# does not count either, since vb[i + 1] leaves an element behind.
function bgen_zero_write_reached(body::Vector{NamedTuple}, arr::Symbol,
                                 loopvars::Vector{Symbol} = Symbol[])
    found = false
    for st in body
        if st.kind == :assign && st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == arr
            idx = st.lhs.args[2:end]
            found = st.rhs isa Number && st.rhs == 0.0 &&
                    length(idx) == 1 && idx[1] isa Symbol && idx[1] in loopvars
        elseif st.kind == :for
            if bgen_writes_array(st.body, arr)
                found = cgen_loop_runs_once(st) &&
                        bgen_zero_write_reached(st.body, arr, vcat(loopvars, st.var))
            end
        elseif st.kind == :while
            # A while has no literal bounds, so cgen_loop_runs_once can never
            # prove it runs. Rule 11: a loop that may not run has not run.
            if bgen_writes_array(st.body, arr)
                found = false
            end
        elseif st.kind == :if
            tw = bgen_writes_array(st.then, arr)
            ew = bgen_writes_array(st.els, arr)
            if tw || ew
                found = tw && ew && bgen_zero_write_reached(st.then, arr, loopvars) &&
                        bgen_zero_write_reached(st.els, arr, loopvars)
            end
        end
    end
    return found
end

function bgen_writes_array(body::Vector{NamedTuple}, arr::Symbol)
    for st in body
        if st.kind == :assign && st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == arr
            return true
        elseif st.kind == :for
            bgen_writes_array(st.body, arr) && return true
        elseif st.kind == :while
            bgen_writes_array(st.body, arr) && return true
        elseif st.kind == :if
            (bgen_writes_array(st.then, arr) || bgen_writes_array(st.els, arr)) && return true
        end
    end
    return false
end

# ---- emission ----------------------------------------------------
# The epilogue never names the function it wraps. It operates on buffers
# only, so one generated file serves the CPU, CUDA and JACC targets.

bgen_fname(name::Symbol, mode::Symbol, suffix::String) =
    Symbol(string(name) * bgen_mode_suffix(mode) * suffix)

bgen_mode_suffix(mode::Symbol) =
    mode == :tangent ? "_d" : mode == :adjoint ? "_b" : "_hv"

# `fill!` and not a loop: the buffer may be a host Array or a device
# array, and `fill!` is defined for both.
function bgen_fill_fn(fname::Symbol, bufs::Vector{Symbol}, doc::String)
    body = Any[]
    for b in bufs
        push!(body, Expr(:call, :fill!, b, 0.0))
    end
    push!(body, Expr(:return, :nothing))
    return (doc = doc, expr = Expr(:function, Expr(:call, fname, bufs...), Expr(:block, body...)))
end

function bgen_reset_accum_expr(kernel, roles, mode)
    bufs = bgen_role_args(kernel, roles, mode, :accum; arrays_only = true)
    return bgen_fill_fn(bgen_fname(kernel.sig.name, mode, "_reset_accum!"), bufs,
        "Zero the :accum buffers. Call once per step, on every rank. The " *
        "Allreduce sums each rank's contribution, so a buffer entering the step " *
        "non-zero is counted once per rank.")
end

function bgen_reset_reduced_expr(kernel, roles, mode)
    bufs = bgen_role_args(kernel, roles, mode, :reduced; arrays_only = true)
    return bgen_fill_fn(bgen_fname(kernel.sig.name, mode, "_reset_reduced!"), bufs,
        "Zero the :reduced buffers. Separate from the :accum reset because the " *
        "two roles may take two reset periods; their communication is the same.")
end

function bgen_reset_local_expr(kernel, roles, mode)
    bufs = bgen_role_args(kernel, roles, mode, :local; arrays_only = true)
    return bgen_fill_fn(bgen_fname(kernel.sig.name, mode, "_reset_local!"), bufs,
        "Zero the :local buffers. Call before each local sample, or one sample's " *
        "intermediate adjoints seed the next one's reverse sweep. Only " *
        "adjoint-valued buffers appear here: a tangent-valued companion is an " *
        "input the caller loads.")
end

# One collective per role, not one for both. The two roles can take two
# periods: a training loop that accumulates gradients over several
# batches before one optimizer step sums :accum once per window, while a
# per-batch quantity such as a loss is summed every batch.
#
# A single collective forces a caller who wants both to sum the gradients
# every batch and again at the window boundary, which multiplies them by
# the window length and raises nothing. Two collectives make the period
# an explicit choice at the call site.
function bgen_allreduce_accum_expr(kernel, roles, mode)
    bufs = bgen_role_args(kernel, roles, mode, :accum; arrays_only = true)
    fname = bgen_fname(kernel.sig.name, mode, "_allreduce_accum!")
    args = vcat(bufs, [:n_local, :comm])
    body = Any[]
    for b in bufs
        push!(body, Expr(:call, Expr(:., :MPI, QuoteNode(Symbol("Allreduce!"))), b, :+, :comm))
    end
    # returned so a mean objective divides by the same number on every rank
    push!(body, Expr(:return, Expr(:call, Expr(:., :MPI, QuoteNode(:Allreduce)), :n_local, :+, :comm)))
    return (doc = "Sum the :accum buffers across ranks, once per accumulation " *
                  "window. Pair it with reset_accum!, which must run at the same " *
                  "boundary. Buffers pass to MPI untouched, so a CUDA-aware build " *
                  "keeps them on device. Returns the global sample count of the " *
                  "window, so a mean objective divides by the same number on every rank.",
            expr = Expr(:function, Expr(:call, fname, args...), Expr(:block, body...)))
end

function bgen_allreduce_reduced_expr(kernel, roles, mode)
    bufs = bgen_role_args(kernel, roles, mode, :reduced; arrays_only = true)
    fname = bgen_fname(kernel.sig.name, mode, "_allreduce_reduced!")
    args = vcat(bufs, [:comm])
    body = Any[]
    for b in bufs
        push!(body, Expr(:call, Expr(:., :MPI, QuoteNode(Symbol("Allreduce!"))), b, :+, :comm))
    end
    push!(body, Expr(:return, :nothing))
    return (doc = "Sum the :reduced buffers across ranks. Pair it with " *
                  "reset_reduced!, which must run at the same boundary. That " *
                  "boundary need not be the one the :accum buffers use.",
            expr = Expr(:function, Expr(:call, fname, args...), Expr(:block, body...)))
end

# Two checksums per buffer, compared by min against max across ranks.
# Agreement is evidence and not proof: this finds a mis-declared array or
# a failed broadcast, not a crafted collision.
function bgen_check_expr(kernel, roles, mode)
    bufs = vcat(bgen_role_args(kernel, roles, mode, :shared; arrays_only = true),
                bgen_role_args(kernel, roles, mode, :seed; arrays_only = true))
    fname = bgen_fname(kernel.sig.name, mode, "_check_replicas")
    args = vcat(bufs, [:comm])
    pairs = Any[Expr(:tuple, string(b), b) for b in bufs]
    body = Any[
        Expr(:(=), :bad, Expr(:ref, :String)),
        Expr(:for, Expr(:(=), Expr(:tuple, :nm, :arr), Expr(:tuple, pairs...)),
            Expr(:block,
                Expr(:(=), :c1, Expr(:call, :sum, :arr)),
                Expr(:(=), :c2, Expr(:call, :sum, :abs, :arr)),
                Expr(:(=), :agree, Expr(:&&,
                    Expr(:call, :(==), Expr(:call, Expr(:., :MPI, QuoteNode(:Allreduce)), :c1, :min, :comm),
                                        Expr(:call, Expr(:., :MPI, QuoteNode(:Allreduce)), :c1, :max, :comm)),
                    Expr(:call, :(==), Expr(:call, Expr(:., :MPI, QuoteNode(:Allreduce)), :c2, :min, :comm),
                                        Expr(:call, Expr(:., :MPI, QuoteNode(:Allreduce)), :c2, :max, :comm)))),
                Expr(:||, :agree, Expr(:call, :push!, :bad, :nm)))),
        Expr(:||, Expr(:call, :isempty, :bad),
            Expr(:call, :error, Expr(:call, :*,
                string(fname) * ": `",
                Expr(:call, :join, :bad, "`, `"),
                "` differ across ranks. A :shared or :seed buffer must hold " *
                "identical values on every rank. The usual cause is a per-sample " *
                "array left out of per_sample"))),
        Expr(:return, :nothing)]
    return (doc = "Check that the :shared and :seed buffers match on every rank. " *
                  "Call after the first sample is loaded, on the first step. " *
                  "Uninitialised buffers match everywhere, so an earlier call " *
                  "reports nothing.",
            expr = Expr(:function, Expr(:call, fname, args...), Expr(:block, body...)))
end

function bgen_preamble(kernel, roles, per_sample, reduced, mode)
    layout = mode == :tangent ? tgen_signature_args(kernel.sig) :
             mode == :adjoint ? agen_signature_args(kernel.sig) :
             bgen_hvp_signature_args(kernel.sig)
    treat = Dict(:returned => "none. Returned by value; sum it yourself.",
                 :shared => "none. Must match on every rank.",
                 :seed => "none. Must match on every rank.",
                 :accum => "Allreduce!(+), zeroed per step",
                 :reduced => "Allreduce!(+), zeroed per step",
                 :local => "none. Zeroed before each sample.",
                 :per_sample => "none. The caller loads it.",
                 :scratch => "none. The kernel overwrites it.")
    lines = ["# Generated by STADE.jl -- do not edit.",
             "#",
             "# Data-parallel epilogue for $(bgen_fname(kernel.sig.name, mode, "")). Every rank runs the",
             "# unmodified single-sample kernel on its own samples. These functions",
             "# zero the buffers between calls and sum what must be summed.",
             "#",
             "#   mode       = :$(mode)",
             "#   per_sample = $(per_sample)",
             "#   reduced    = $(reduced)",
             "#",
             "# ---- derived role table ----"]
    for b in layout
        haskey(roles, b) || continue
        push!(lines, "#   " * rpad(string(b), 12) * rpad(":" * string(roles[b]), 13) * treat[roles[b]])
    end
    push!(lines, "#")
    push!(lines, "# Read the role table, not the numbers. A wrong communication plan")
    push!(lines, "# still returns plausible values on every rank.")
    return join(lines, "\n") * "\n\nusing MPI\n"
end

# Six functions. Each keeps its one-line contract as a comment above it. The
# generated file is read by whoever writes the training loop, and a role
# table with no statement of what to call when is half a plan.
function stade_batch_file(in_path::String, out_path::String;
                          per_sample::Vector{Symbol}, mode::Symbol,
                          reduced::Vector{Symbol} = Symbol[])
    primal = io_read_corpus_entry(in_path)
    r = stade_batch(primal; per_sample = per_sample, mode = mode, reduced = reduced)
    parts = String[]
    for f in (r.reset_accum, r.reset_reduced, r.reset_local,
              r.allreduce_accum, r.allreduce_reduced, r.check)
        push!(parts, "# " * f.doc * "\n" * io_expr_to_source(f.expr))
    end
    open(out_path, "w") do io
        write(io, r.preamble * "\n" * join(parts, "\n\n") * "\n")
    end
    return r.roles
end

function stade_batch(primal::Expr; per_sample::Vector{Symbol}, mode::Symbol,
                     reduced::Vector{Symbol} = Symbol[])
    kernel = parse_kernel(primal)
    roles = bgen_roles(kernel, per_sample, mode; reduced = reduced)
    return (reset_accum   = bgen_reset_accum_expr(kernel, roles, mode),
            reset_reduced = bgen_reset_reduced_expr(kernel, roles, mode),
            reset_local   = bgen_reset_local_expr(kernel, roles, mode),
            allreduce_accum   = bgen_allreduce_accum_expr(kernel, roles, mode),
            allreduce_reduced = bgen_allreduce_reduced_expr(kernel, roles, mode),
            check         = bgen_check_expr(kernel, roles, mode),
            preamble      = bgen_preamble(kernel, roles, per_sample, reduced, mode),
            roles         = roles)
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
    if length(kernels) == 1
        # Nothing to inline, but the call graph is still built: it is the only
        # place a self-call is caught, and skipping it let one escape to
        # parse_kernel to be misreported as an unsupported statement form.
        inl_build_call_graph(kernels)
        return kernels[entry_name]
    end
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

# GPU parity gets its own baseline file rather than sharing the CPU oracles'. `.yaml` is
# gitignored, so whichever script ran last owns the shared name -- and validate_zerotrip.jl
# deliberately draws integers from [0, 2], leaving every extent at or near zero. A GPU run
# inheriting that draw executes no device code and reports a pass. Separating the namespaces
# keeps a GPU result independent of test ordering.
io_gpu_yaml_path(in_path::String) = splitext(in_path)[1] * ".gpu.yaml"

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

# Computes the site-level TBR decision from both independently duplicated implementations (snap_* and agen_*) and
# asserts they agree exactly before returning either -- the permanent guard, so a future silent divergence fails
# loudly here. Also asserts every site-level 'needed' site is backed by the coarser whole-variable value_needed set
# fuse_ii_loops relies on -- site-level must only refine it, never diverge. Returns (snap_sites, agen_sites), each
# stage's own duplicate as its sole input. Always run -- no longer opt-in.
function stade_site_level_tbr_check(kernel)
    snap_sites = snap_value_needed_sites(kernel)
    agen_sites = agen_value_needed_sites(kernel)
    @assert keys(snap_sites) == keys(agen_sites) "site_level_tbr: snap_* and agen_* site-level analyses produced different site keys for `$(kernel.sig.name)` -- this should never happen; the two stages walked the kernel differently somewhere"
    @assert snap_sites == agen_sites "site_level_tbr: snap_* and agen_* site-level analyses DISAGREE for `$(kernel.sig.name)` -- refusing to generate code that could desync push/pop counts. The two independently-duplicated implementations have drifted; see skill-stade's site-level TBR rollout notes."
    value_needed = snap_value_needed_vars(kernel)
    site_vars = snap_site_vars(kernel)
    for (key, needed) in snap_sites
        needed || continue
        @assert site_vars[key] in value_needed "site_level_tbr: site-level analysis marked a write to `$(site_vars[key])` needed in `$(kernel.sig.name)`, but the whole-variable analysis does not -- site-level must be a subset"
    end
    return snap_sites, agen_sites
end


# ---- dead loop-entry scalars ----
# A scalar assigned only inside a nested block of `body`, written there before anything reads it, carries a value from the previous iteration that nothing uses --
# the adjoint's pre-write snapshot then reads that value, making the loop look loop-carried to cgen_use_before_def and blocking the split. ttgc's second cell nest
# is the case in point: `vere` is assigned every i_loc iteration before use, but without a reset the cell loop stays on the host. Writing `var = 0.0` at the top
# of `body` states that kill. Suppressing the snapshot instead does NOT work: cgen_use_before_def refuses regardless.
function norm_dead_entry_vars(body, kinds, value_needed)
    locals = agen_collect_reassigned(body, true)
    return sort(collect(v for v in intersect(locals, value_needed)
                        if get(kinds, v, :none) == :scalar_float &&
                           !agen_has_top_level_write(body, v) &&
                           norm_first_touch(body, v) === :write); by = string)
end

# :read, :write or :none -- what happens to `v` FIRST along every path through `body`. A write
# only counts when it is GUARANTEED: both arms of an `:if`, or a loop whose literal bounds prove
# at least one iteration. Anything weaker and the entry value can survive to a later read, so the
# conservative answer is to keep scanning rather than claim a kill.
function norm_first_touch(body, v)
    for st in body
        if st.kind == :assign
            reads = Set{Symbol}()
            agen_var_value_needed!(st.rhs, reads, false)
            if st.lhs isa Expr
                for a in st.lhs.args[2:end]
                    agen_var_value_needed!(a, reads, true)
                end
            end
            v in reads && return :read
            st.lhs isa Symbol && st.lhs == v && agen_count_var_refs(st.rhs, v) == 0 && return :write
        elseif st.kind == :for
            t = norm_first_touch(st.body, v)
            t === :read && return :read
            if t === :write && st.lo isa Number && st.hi isa Number && st.step isa Number
                (st.step > 0 ? st.hi >= st.lo : st.hi <= st.lo) && return :write
            end
        elseif st.kind == :while
            t = norm_first_touch(st.body, v)
        elseif st.kind == :if
            reads = Set{Symbol}()
            agen_var_value_needed!(st.cond, reads, true)
            v in reads && return :read
            tt = norm_first_touch(st.then, v)
            te = norm_first_touch(st.els, v)
            (tt === :read || te === :read) && return :read
            (tt === :write && te === :write) && return :write
        end
    end
    return :none
end

# Rewrites `kernel` with those resets inserted, innermost bodies included. Applied once, before
# any analysis runs, so every stage downstream -- TBR, snapshot layout, lin_plan, cgen_ -- sees
# one consistent kernel. The finite-difference oracles still evaluate the ORIGINAL primal, so a
# reset inserted where the entry value was in fact live shows up as a gradient mismatch.
function norm_insert_dead_entry_resets(kernel)
    kinds = kernel.sig.kinds
    value_needed = agen_value_needed_vars(kernel)
    return merge(kernel, (body = norm_reset_body(kernel.body, kinds, value_needed),))
end

function norm_reset_body(body, kinds, value_needed)
    out = NamedTuple[]
    for st in body
        if st.kind == :for
            inner = norm_reset_body(st.body, kinds, value_needed)
            for v in reverse(norm_dead_entry_vars(st.body, kinds, value_needed))
                pushfirst!(inner, (kind = :assign, lhs = v, rhs = 0.0))
            end
            push!(out, merge(st, (body = inner,)))
        elseif st.kind == :while
            inner = norm_reset_body(st.body, kinds, value_needed)
            for v in reverse(norm_dead_entry_vars(st.body, kinds, value_needed))
                pushfirst!(inner, (kind = :assign, lhs = v, rhs = 0.0))
            end
            push!(out, merge(st, (body = inner,)))
        elseif st.kind == :if
            push!(out, merge(st, (then = norm_reset_body(st.then, kinds, value_needed),
                                  els = norm_reset_body(st.els, kinds, value_needed))))
        else
            push!(out, st)
        end
    end
    return out
end

function stade_adjoint(expr::Expr; independents::Union{Vector{Symbol},Nothing}=nothing,
                        dependents::Union{Vector{Symbol},Nothing}=nothing,
                        keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    kernel = norm_insert_dead_entry_resets(kernel)   # see norm_dead_entry_vars
    active_map = act_analyze(kernel)
    snap_sites, agen_sites = stade_site_level_tbr_check(kernel)
    snapshot_plan = snap_plan(kernel, active_map; site_needed = snap_sites)
    keep_push_pop || agen_refuse_itbr_sized(snapshot_plan)
    lin_plan = lin_build(kernel, active_map)
    ii_plan = fuse_ii_loops ? stade_ii_plan_check(kernel) : nothing
    (agen_sites, remat) = keep_push_pop ? (agen_sites, nothing) : agen_remat_apply(kernel, active_map, agen_sites)
    (agen_sites, asnap) = keep_push_pop ? (agen_sites, nothing) : agen_asnap_apply(kernel, active_map, agen_sites; ii_plan = ii_plan)
    asnap === nothing || (snapshot_plan = vcat(snapshot_plan, agen_asnap_sites(asnap)))
    return agen_emit(kernel, lin_plan, snapshot_plan; keep_push_pop = keep_push_pop, push_pop = agen_sites, ii_plan = ii_plan, remat = remat)
end

function stade_hvp(expr::Expr; independents::Union{Vector{Symbol},Nothing}=nothing,
                    dependents::Union{Vector{Symbol},Nothing}=nothing,
                    keep_push_pop::Bool=true, fuse_ii_loops::Bool=false)
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    kernel = norm_insert_dead_entry_resets(kernel)   # see norm_dead_entry_vars
    active_map = act_analyze(kernel)
    snap_sites, agen_sites = stade_site_level_tbr_check(kernel)
    snapshot_plan = snap_plan(kernel, active_map; site_needed = snap_sites)
    if !keep_push_pop
        # the refusals stade_adjoint reaches inside agen_emit, made here before the layout pass can fail on a stack with no size
        agen_refuse_while_sized(kernel.body)
        agen_refuse_itbr_sized(snapshot_plan)
    end
    lin_plan = lin_build(kernel, active_map)
    ii_plan = fuse_ii_loops ? stade_ii_plan_check(kernel) : nothing
    (agen_sites, remat) = keep_push_pop ? (agen_sites, nothing) : agen_remat_apply(kernel, active_map, agen_sites)
    (agen_sites, asnap) = keep_push_pop ? (agen_sites, nothing) : agen_asnap_apply(kernel, active_map, agen_sites; ii_plan = ii_plan)
    asnap === nothing || (snapshot_plan = vcat(snapshot_plan, agen_asnap_sites(asnap)))
    layout = nothing
    value_needed = exempt = stacks = nothing
    if !keep_push_pop
        # Tier B: see the matching comment in agen_emit above.
        value_needed = agen_value_needed_vars(kernel)
        reassigned = agen_collect_reassigned_all(kernel.body)
        exempt = agen_exempt_vars(kernel, value_needed)
        stacks = agen_stack_map(snapshot_plan)
        layout = agen_layout(kernel, kernel.sig.kinds, active_map, value_needed, reassigned, exempt, stacks; push_pop = agen_sites)
    end
    (initstacks_expr, table_names, tot_names, val_names) = agen_init_emit(kernel, snapshot_plan; keep_push_pop = keep_push_pop, layout = layout,
                                      active_map = active_map, value_needed = value_needed, exempt = exempt, stacks = stacks, push_pop = agen_sites)
    tier_b_extra_args = vcat(table_names, tot_names, val_names)
    hvp_expr = hvp_emit(kernel, active_map, lin_plan, snapshot_plan; keep_push_pop = keep_push_pop, layout = layout, push_pop = agen_sites,
                         tier_b_extra_args = tier_b_extra_args, ii_plan = ii_plan, remat = remat)
# Mirrors agen_emit's own post-hoc cleanup, sharing agen_finalize_stacks so the two passes can't drift out of sync. Required for consistency:
# stade_adjoint's own initstacks_* already drops a stack once fusion leaves it unused, and validation code sharing initstacks_* across both
# adjoint and hvp calls needs hvp_expr's signature to match. hvp_expr's fwd/bwd reuse the same agen_forward_body/agen_backward_body calls as
# the adjoint path, landing on the same unused set. is_hvp=true additionally drops each dropped stack's orphaned `<stack>_d` shadow alloc.
    (hvp_expr, initstacks_expr) = agen_finalize_stacks(hvp_expr, initstacks_expr, snapshot_plan, ii_plan; is_hvp = true,
                                                         force = !keep_push_pop && (any(s -> agen_asnap_site_array(s, kernel.sig.kinds) !== nothing, snapshot_plan) || !isempty(agen_remat_cross_vars(remat))))
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
function stade_gpu(expr::Expr, backend; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT,
                   warp_atomics::Bool = true, sync_free_reduce::Bool = true)
    warp_atomics || (backend = merge(backend, (warp_atomics = false,)))
    sync_free_reduce || (backend = merge(backend, (sync_free_reduce = false,)))
    p = precision === nothing ? backend.default_precision : precision
    if backend.precision_locked && p !== backend.default_precision
        error("stade_gpu: backend `$(backend.kernel_tag)` only supports precision=$(backend.default_precision) (got $(p)) -- $(backend.precision_lock_reason)")
    end
    plan = cgen_emit(cgen_ingest(expr), backend; reduction_threshold)
    p === Float64 && return (host = plan.host, kernels = plan.kernels, helpers = plan.helpers)
    return (host = cgen_convert_precision(plan.host, p),
            kernels = Expr[cgen_convert_precision(k, p) for k in plan.kernels], helpers = plan.helpers)
end

stade_cuda(expr::Expr; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT, warp_atomics::Bool = true,
           sync_free_reduce::Bool = true) =
    stade_gpu(expr, cgen_backend_cuda(); precision, reduction_threshold, warp_atomics, sync_free_reduce)
stade_amdgpu(expr::Expr; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT) =
    stade_gpu(expr, cgen_backend_amdgpu(); precision, reduction_threshold)
stade_metal(expr::Expr; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT) =
    stade_gpu(expr, cgen_backend_metal(); precision, reduction_threshold)

# Path in, path out. Reads every function def in in_path and writes one file: every device kernel first, then every host
# function in original order. precision applies uniformly to every function converted in this call; for per-function
# control, call stade_gpu directly on each def. The input file is only ever read, never rewritten. A pure scalar
# reduction emits BOTH an atomic-accumulate kernel and a mapreduce call, and picks between them at run time on the
# loop's own trip count; reduction_threshold sets where that switch happens.
function stade_gpu_file(in_path::String, out_path::String, backend; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT,
                        warp_atomics::Bool = true, sync_free_reduce::Bool = true)
    defs = io_read_kernel_bundle(in_path)
    kernels = Expr[]
    hosts = Expr[]
    helpers = Expr[]
    for expr in defs
        plan = stade_gpu(expr, backend; precision, reduction_threshold, warp_atomics, sync_free_reduce)
        append!(kernels, plan.kernels)
        push!(hosts, plan.host)
        isempty(helpers) && append!(helpers, plan.helpers)
    end
    io_write_gpu_file(out_path, vcat(helpers, kernels, hosts); preamble = backend.preamble)
    return out_path
end

stade_cuda_file(in_path::String, out_path::String; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT, warp_atomics::Bool = true,
                sync_free_reduce::Bool = true) =
    stade_gpu_file(in_path, out_path, cgen_backend_cuda(); precision, reduction_threshold, warp_atomics, sync_free_reduce)
stade_amdgpu_file(in_path::String, out_path::String; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT) =
    stade_gpu_file(in_path, out_path, cgen_backend_amdgpu(); precision, reduction_threshold)
stade_metal_file(in_path::String, out_path::String; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT) =
    stade_gpu_file(in_path, out_path, cgen_backend_metal(); precision, reduction_threshold)

# JACC has no gpu_backend value at all -- there's only one JACC target from cgen_/jgen_'s point
# of view, since which vendor it runs on is chosen later, outside this call. precision has no
# locked default here for the same reason: Float64 unless the caller asks otherwise.
# On JACC a matched reduction becomes one
# JACC.@parallel_reduce call instead of @parallel_for + Atomix.@atomic.
function stade_jacc(expr::Expr; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT)
    plan = jgen_emit(cgen_ingest(expr); reduction_threshold)
    p = precision === nothing ? Float64 : precision
    p === Float64 && return (host = plan.host, kernels = plan.kernels)
    return (host = cgen_convert_precision(plan.host, p),
            kernels = Expr[cgen_convert_precision(k, p) for k in plan.kernels])
end

function stade_jacc_file(in_path::String, out_path::String; precision::Union{Nothing, Type{<:AbstractFloat}} = nothing, reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT)
    defs = io_read_kernel_bundle(in_path)
    kernels = Expr[]
    hosts = Expr[]
    for expr in defs
        plan = stade_jacc(expr; precision, reduction_threshold)
        append!(kernels, plan.kernels)
        push!(hosts, plan.host)
    end
    io_write_gpu_file(out_path, vcat(kernels, hosts); preamble = jgen_preamble())
    return out_path
end

# Per-loop offload decisions for one input file, one record per `for` loop (see cgen_decision_record). `backend` is a gpu_backend
# value, or `nothing` for the JACC path. The report runs the same walk as stade_gpu_file and writes no file.
function stade_gpu_report_file(in_path::String, backend = cgen_backend_cuda(); reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT)
    records = NamedTuple[]
    for expr in io_read_kernel_bundle(in_path)
        plan = backend === nothing ? jgen_emit(cgen_ingest(expr); reduction_threshold) : cgen_emit(cgen_ingest(expr), backend; reduction_threshold)
        append!(records, plan.decisions)
    end
    return records
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
                                      scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
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
                                       scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
                                       grow_max::Int = 512,
                                       self_check::Bool = true,
                                       divisible_by::Union{Dict{Symbol,Int},Nothing} = nothing)
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = val_generate_baseline(kernel, primal_expr; scale = scale, int_lo = int_lo, int_hi = int_hi,
                                      grow_max = grow_max,
                                      self_check = self_check, divisible_by = divisible_by)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_write_baseline_yaml(yp, kernel.sig.name, baseline.int_args, baseline.values)
    return yp
end

function stade_validate_tangent_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                      scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
                                      trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3,
                                      self_check::Bool = true, keep_push_pop::Bool = true, fuse_ii_loops::Bool = false)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    return stade_validate_from_baseline(:tangent, in_path, yp; trials = trials, epsilon = epsilon, rtol = rtol,
                                         keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
end

function stade_validate_adjoint_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                      scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
                                      trials::Int = 10, epsilon::Float64 = 1e-6, rtol::Float64 = 1e-3,
                                      self_check::Bool = true, keep_push_pop::Bool = true, fuse_ii_loops::Bool = false)
    yp = yaml_path === nothing ? io_default_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    return stade_validate_from_baseline(:adjoint, in_path, yp; trials = trials, epsilon = epsilon, rtol = rtol,
                                         keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
end

function stade_validate_hvp_file(in_path::String; yaml_path::Union{String,Nothing} = nothing,
                                  scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
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
                                              scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
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

# Positional call-argument NAMES (not values) for one device's copies, in kernel.sig.args order with each float-kind
# arg expanded to its (value, shadow) pair -- mirrors val_adjoint_call_args's ordering exactly (array_int gets no
# shadow; scalar_int is a bare name, shared verbatim between cpu/gpu). `mode = :hvp` appends one tangent pair per
# float arg after the whole adjoint list, mirroring val_hvp_call_args's second pass -- the layout is hvp_-specific
# and can't be derived from sig.kinds alone.
function val_gpu_call_args(sig, device::Symbol; mode::Symbol = :adjoint)
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
    if mode == :hvp
        for a in sig.args
            sig.kinds[a] in (:scalar_float, :array_float) || continue
            push!(parts, "$(a)_tan_$(device)", "$(a)_shtan_$(device)")
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

# The verdict on the BigFloat truth, as script text. One verdict for the whole run, and two ways to pass it.
# 1. The worst GPU error over the trials is at most `truth_factor` times the worst CPU error over the same trials. Judging every trial against its
#    own CPU error failed a transformer HVP on 3.7e-12 against a worst CPU error of 7.7e-13, deterministically, while other draws passed.
# 2. The worst GPU error is at most `truth_ceiling`. The device rounds differently from the host (its libm, fused operations, the order of an atomic
#    sum), so its error is not a multiple of the host's. Over 64 V100 runs (8 draws of four large kernels, adjoint and HVP, 192 trials) the per-trial ratio ran
#    from 0.01 to 87, and the worst-over-trials ratio reached 71 on one transformer adjoint (GPU 6.3e-12 against a CPU error of 9e-14). 63 of the 64
#    runs pass by (1); that one passes by (2), with a margin of 16. The largest GPU error seen was 1.5e-10, in a run whose CPU error was 4.5e-11:
#    it passes by (1), with a margin of 6, and it is above this ceiling, so (1) is not redundant.
#    A gap above both is no longer rounding. The stack comparison (`stack_max_rel_err`) stays exact and is reported beside the verdict.
# `rtol` is the floor of both. A trial whose BigFloat run threw is judged by the GPU-vs-CPU gap against `rtol`, as before.
function val_truth_verdict_lines(rtol, truth_factor, truth_ceiling = 1.0e-10)
    return String[
        "tr_tol = max($(rtol), $(truth_factor) * tr_cpu_worst, $(truth_ceiling))",
        "if truth_trials > 0",
        "  global __all_ok = __all_ok && tr_gpu_worst <= tr_tol",
        "end",
    ]
end

# Assembles the full script described above. `seeds` is a Vector of per-trial seed Dicts (same shape as `values`) -- `values`
# itself (the primal baseline) is shared across trials but freshly deepcopy'd every trial on both devices, since the call under
# test mutates its float arrays in place. `mode = :hvp` swaps the function under test for the HVP and consumes two further per-
# trial seed Vectors: `tan_seeds` (the Hessian's direction) and `shtan_seeds` (the reverse seed's tangent), both ignored for
# `:adjoint`.
function val_gpu_parity_script(kernel, int_args::Dict, values::Dict, seeds::Vector,
                                gen_out, gpu_init, gpu_gen, stack_arg_names::Vector{Symbol},
                                backend; rtol::Float64 = 2.5e-14, mode::Symbol = :adjoint,
                                tan_seeds::Vector = Any[], shtan_seeds::Vector = Any[],
                                truth::Bool = true, truth_factor::Float64 = 20.0, truth_bits::Int = 192,
                                truth_ceiling::Float64 = 1.0e-10)
    mode in (:adjoint, :hvp) || error("val_gpu_parity_script: mode must be :adjoint or :hvp, got $(mode)")
    sig = kernel.sig
    cpu_gen_expr = mode == :hvp ? gen_out.hvp : gen_out.adjoint
    cpu_init_name = val_def_fn_name(gen_out.initstacks)
    cpu_gen_name  = val_def_fn_name(cpu_gen_expr)
    gpu_init_name = val_def_fn_name(gpu_init.host)
    gpu_gen_name  = val_def_fn_name(gpu_gen.host)
    arrtype = backend.arrtype

    lines = String[]
    isempty(backend.preamble) || push!(lines, backend.preamble)
    push!(lines, "using JSON3", "")

    push!(lines, io_expr_to_source(gen_out.initstacks))
    push!(lines, io_expr_to_source(cpu_gen_expr))
    init_helpers = get(gpu_init, :helpers, Expr[])
    for h in vcat(init_helpers, isempty(init_helpers) ? get(gpu_gen, :helpers, Expr[]) : Expr[])
        push!(lines, io_expr_to_source(h))
    end
    push!(lines, io_expr_to_source(gpu_init.host))
    for k in gpu_init.kernels
        push!(lines, io_expr_to_source(k))
    end
    push!(lines, io_expr_to_source(gpu_gen.host))
    for k in gpu_gen.kernels
        push!(lines, io_expr_to_source(k))
    end

    push!(lines, "function _val_init_stacks_local(fn, extra_args)\n    r = fn(extra_args...)\n    r === nothing && return ()\n    t = r isa Tuple ? r : (r,)\n    for s in t\n        (s isa AbstractArray && eltype(s) <: AbstractFloat) && fill!(s, 0.0)\n    end\n    return t\nend\n")
    push!(lines, "_relerr_scalar(a, b) = abs(a - b) / max(abs(a), abs(b), 1.0)\n" *
# `init = 0.0` is load-bearing, not defensive: the baseline generator gates a draw
# only on initstacks_* running, so an all-zero integer draw (every extent 0) is
# dimensionally coherent and gets accepted, producing zero-length stacks and
# zero-length float arrays. `maximum` over those throws rather than comparing
# nothing, which is what an empty comparison should do.
                 "_relerr_arr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1.0); init = 0.0)\n")
# Compares the SNAPSHOT STACKS, not just the arguments. A stack slot written from a stale host scalar (a split loop assigned it on the device,
# leaving the host copy behind) shows up here and nowhere else, since gradients stay correct whenever that slot's restore happens to be dead
# -- confirmed live for cellscatter (gradients to 5.5e-14, stacks off by 1.0) and raggedii. Reported rather than folded into `ok`: a nonzero
# value means the kernel passes for a reason nothing enforces, a finding not a failure. Non-array tuple entries are skipped, since a Tier B
# initstacks_* returns prefix tables and totals alongside the stacks. Float stacks are zero-filled before the run (_val_init_stacks_local): a slot
# that nothing writes held whatever the allocator left, so host and device differed there from run to run (a fused ttgc HVP read 1.0 in one run
# and 5e-16 in the next, and so did the v0.4.7 code). Now an unwritten slot is 0.0 on both sides and only a written slot can differ.
    push!(lines, "function _val_stack_relerr(cpu_stacks, gpu_stacks)\n" *
                 "    worst = 0.0\n    n = 0\n" *
                 "    for (c, g) in zip(cpu_stacks, gpu_stacks)\n" *
                 "        (c isa AbstractArray && g isa AbstractArray) || continue\n" *
                 "        eltype(c) <: AbstractFloat || continue\n" *
                 "        a = Array(g)\n" *
                 "        length(c) == length(a) || continue\n" *
                 "        n += 1\n        worst = max(worst, _relerr_arr(c, a))\n" *
                 "    end\n    return (worst, n)\nend\n")

    for a in sig.args
        sig.kinds[a] == :scalar_int || continue
        push!(lines, "$(a) = $(int_args[a])")
    end

    float_or_intarr_args = [a for a in sig.args if sig.kinds[a] in (:scalar_float, :array_float, :array_int)]
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    array_float_args = [a for a in sig.args if sig.kinds[a] == :array_float]
    array_int_args = [a for a in sig.args if sig.kinds[a] == :array_int]
    float_args = [a for a in sig.args if sig.kinds[a] in (:scalar_float, :array_float)]

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
    if mode == :hvp
        for a in float_args
            tan_list = join([val_julia_literal(sd[a]) for sd in tan_seeds], ", ")
            sht_list = join([val_julia_literal(sd[a]) for sd in shtan_seeds], ", ")
            push!(lines, "__tan_$(a) = [$(tan_list)]")
            push!(lines, "__shtan_$(a) = [$(sht_list)]")
        end
    end

    stack_extra_cpu = join([val_gpu_stack_extra_ref(n, int_args, :cpu) for n in stack_arg_names], ", ")
    stack_extra_gpu = join([val_gpu_stack_extra_ref(n, int_args, :gpu) for n in stack_arg_names], ", ")
    cpu_call = "$(cpu_gen_name)($(join(val_gpu_call_args(sig, :cpu; mode), ", ")), stacks_cpu...)"
    gpu_call = "$(gpu_gen_name)($(join(val_gpu_call_args(sig, :gpu; mode), ", ")), stacks_gpu...)"
# Each scalar_float arg returns ONE value from an adjoint and TWO from an HVP
# (ab, abd), so emit_return_scalars' bare-value-vs-tuple rule only ever yields a
# bare value in the adjoint's single-scalar case -- an HVP with one scalar arg
# still returns a genuine 2-tuple. Same rule val_call_hv relies on.
    n_ret = mode == :hvp ? 2 * length(scalar_args) : length(scalar_args)
    ret_tuple(v) = isempty(scalar_args) ? "()" : (n_ret == 1 ? "($(v),)" : v)

# A run in which the kernel wrote nothing is not a pass. Array LENGTH is the wrong thing to
# check: an all-zero integer draw leaves every array at its baseline length while every loop
# bound collapses to zero trips, so the arrays compare equal and `ok` is reported for a run that
# executed no device code -- measured live at elements_compared=416, max_rel_err=0.0. What has to
# be counted is elements the CPU reference actually CHANGED.
# BigFloat truth. The CPU Float64 result is itself off the exact value, by up to 1e-10 on a large HVP, so a GPU-vs-CPU difference measures two
# roundings and passes or fails by draw. The same CPU code is run in BigFloat on the same inputs, the GPU result is compared to that, and a trial
# passes by `val_truth_verdict_lines`: the worst GPU error over the trials against `truth_factor` times the worst CPU error, or an absolute ceiling
# (never below the atomic-count `rtol`). A kernel whose BigFloat run throws falls back to the GPU-vs-CPU comparison with `rtol`.
    stems = String[]
    for a in array_float_args
        append!(stems, ["$(a)_val", "$(a)_sh"])
        mode == :hvp && append!(stems, ["$(a)_tan", "$(a)_shtan"])
    end
    if truth
        push!(lines, "setprecision(BigFloat, $(truth_bits))")
        push!(lines, "_relerr_big(t, a) = (m = 0.0; for i in eachindex(t); d = Float64(abs(t[i] - a[i])) / max(Float64(abs(t[i])), abs(Float64(a[i])), 1.0); d > m && (m = d); end; m)")
        init_src = io_expr_to_source(gen_out.initstacks)
        push!(lines, replace(replace(init_src, "function $(cpu_init_name)(" => "function $(cpu_init_name)_big("), "Float64" => "BigFloat"))
    end
    push!(lines, "__all_ok = true", "tr_cpu_worst = 0.0", "tr_gpu_worst = 0.0", "truth_trials = 0", "tr_pairs = Vector{Float64}[]")
    push!(lines, "max_rel_err = 0.0", "stack_max_rel_err = 0.0", "stacks_compared = 0",
                 "elements_compared = 0", "elements_changed = 0")
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
    if mode == :hvp
        for a in array_float_args
            push!(lines, "  $(a)_tan_cpu = deepcopy(__tan_$(a)[__trial]); $(a)_shtan_cpu = deepcopy(__shtan_$(a)[__trial])")
            push!(lines, "  $(a)_tan_gpu = $(arrtype)(deepcopy(__tan_$(a)[__trial])); $(a)_shtan_gpu = $(arrtype)(deepcopy(__shtan_$(a)[__trial]))")
        end
        for a in scalar_args
            push!(lines, "  $(a)_tan_cpu = __tan_$(a)[__trial]; $(a)_shtan_cpu = __shtan_$(a)[__trial]")
            push!(lines, "  $(a)_tan_gpu = __tan_$(a)[__trial]; $(a)_shtan_gpu = __shtan_$(a)[__trial]")
        end
    end
    push!(lines, "  stacks_cpu = _val_init_stacks_local($(cpu_init_name), Any[$(stack_extra_cpu)])")
    push!(lines, "  stacks_gpu = _val_init_stacks_local($(gpu_init_name), Any[$(stack_extra_gpu)])")
    for a in array_float_args
        push!(lines, "  __pre_$(a)_val = deepcopy($(a)_val_cpu); __pre_$(a)_sh = deepcopy($(a)_sh_cpu)")
        mode == :hvp && push!(lines, "  __pre_$(a)_tan = deepcopy($(a)_tan_cpu); __pre_$(a)_shtan = deepcopy($(a)_shtan_cpu)")
    end
    push!(lines, "  ret_cpu = $(cpu_call)")
    if truth
        push!(lines, "  __big = try")
        for a in array_float_args
            push!(lines, "    $(a)_val_big = BigFloat.(__pre_$(a)_val); $(a)_sh_big = BigFloat.(__pre_$(a)_sh)")
            mode == :hvp && push!(lines, "    $(a)_tan_big = BigFloat.(__pre_$(a)_tan); $(a)_shtan_big = BigFloat.(__pre_$(a)_shtan)")
        end
        for a in array_int_args
            push!(lines, "    $(a)_val_big = deepcopy($(a)_val_cpu)")
        end
        for a in scalar_args
            push!(lines, "    $(a)_val_big = BigFloat($(a)_val_cpu); $(a)_sh_big = BigFloat($(a)_sh_cpu)")
            mode == :hvp && push!(lines, "    $(a)_tan_big = BigFloat($(a)_tan_cpu); $(a)_shtan_big = BigFloat($(a)_shtan_cpu)")
        end
        push!(lines, "    stacks_big = _val_init_stacks_local($(cpu_init_name)_big, Any[$(stack_extra_cpu)])")
        push!(lines, "    ret_big = $(replace(cpu_call, "_cpu" => "_big"))")
        push!(lines, "    Dict{String,Any}(" * join(["\"$(st)\" => $(st)_big" for st in stems], ", ") * (isempty(stems) ? "" : ", ") * "\"__ret\" => ret_big)")
        push!(lines, "  catch", "    nothing", "  end")
    else
        push!(lines, "  __big = nothing")
    end
    push!(lines, "  ret_gpu = $(gpu_call)")
    push!(lines, "  ret_cpu_t = $(ret_tuple("ret_cpu"))")
    push!(lines, "  ret_gpu_t = $(ret_tuple("ret_gpu"))")
    push!(lines, "  __trc = 0.0; __trg = 0.0")
    if truth
        push!(lines, "  if __big !== nothing")
        for st in stems
            push!(lines, "    __trc = max(__trc, _relerr_big(__big[\"$(st)\"], $(st)_cpu)); __trg = max(__trg, _relerr_big(__big[\"$(st)\"], Array($(st)_gpu)))")
        end
        if n_ret > 0
            push!(lines, "    ret_big_t = $(ret_tuple("__big[\"__ret\"]"))")
            for i in 1:n_ret
                push!(lines, "    __trc = max(__trc, _relerr_scalar(Float64(ret_big_t[$(i)]), Float64(ret_cpu_t[$(i)]))); __trg = max(__trg, _relerr_scalar(Float64(ret_big_t[$(i)]), Float64(ret_gpu_t[$(i)])))")
            end
        end
        push!(lines, "  end")
    end
    push!(lines, "  local_max = 0.0")
    for a in array_float_args
        push!(lines, "  global elements_compared += length($(a)_val_cpu) + length($(a)_sh_cpu)")
        push!(lines, "  global elements_changed += count(!=(0.0), $(a)_val_cpu .- __pre_$(a)_val) + count(!=(0.0), $(a)_sh_cpu .- __pre_$(a)_sh)")
        push!(lines, "  local_max = max(local_max, _relerr_arr($(a)_sh_cpu, Array($(a)_sh_gpu)))")
        push!(lines, "  local_max = max(local_max, _relerr_arr($(a)_val_cpu, Array($(a)_val_gpu)))")
        if mode == :hvp
            push!(lines, "  global elements_compared += length($(a)_tan_cpu) + length($(a)_shtan_cpu)")
            push!(lines, "  global elements_changed += count(!=(0.0), $(a)_tan_cpu .- __pre_$(a)_tan) + count(!=(0.0), $(a)_shtan_cpu .- __pre_$(a)_shtan)")
            push!(lines, "  local_max = max(local_max, _relerr_arr($(a)_tan_cpu, Array($(a)_tan_gpu)))")
            push!(lines, "  local_max = max(local_max, _relerr_arr($(a)_shtan_cpu, Array($(a)_shtan_gpu)))")
        end
    end
    for a in array_int_args
        push!(lines, "  local_max = max(local_max, _relerr_arr(Float64.($(a)_val_cpu), Float64.(Array($(a)_val_gpu))))")
    end
    for i in 1:n_ret
        push!(lines, "  global elements_compared += 1")
        push!(lines, "  local_max = max(local_max, _relerr_scalar(ret_cpu_t[$(i)], ret_gpu_t[$(i)]))")
    end
    push!(lines, "  (__sworst, __sn) = _val_stack_relerr(stacks_cpu, stacks_gpu)")
    push!(lines, "  global stack_max_rel_err = max(stack_max_rel_err, __sworst)")
    push!(lines, "  global stacks_compared += __sn")
    push!(lines, "  global max_rel_err = max(max_rel_err, local_max)")
    push!(lines, "  if __big === nothing")
    push!(lines, "    global __all_ok = __all_ok && local_max <= $(rtol)")
    push!(lines, "  else")
    push!(lines, "    global tr_cpu_worst = max(tr_cpu_worst, __trc); global tr_gpu_worst = max(tr_gpu_worst, __trg); global truth_trials += 1")
    push!(lines, "    push!(tr_pairs, [__trc, __trg])")
    push!(lines, "  end")
    push!(lines, "end")
    append!(lines, val_truth_verdict_lines(rtol, truth_factor, truth_ceiling))
    push!(lines, "println(JSON3.write(Dict(\"ok\" => __all_ok && elements_changed > 0, \"mode\" => \"$(mode)\", \"max_rel_err\" => max_rel_err, \"truth_trials\" => truth_trials, \"tr_cpu\" => tr_cpu_worst, \"tr_gpu\" => tr_gpu_worst, \"tr_tol\" => tr_tol, \"tr_trials\" => tr_pairs, \"stack_max_rel_err\" => stack_max_rel_err, \"stacks_compared\" => stacks_compared, \"elements_compared\" => elements_compared, \"elements_changed\" => elements_changed, \"vacuous\" => elements_changed == 0, \"device\" => $(val_gpu_device_name_expr(backend)))))")

    return join(lines, "\n") * "\n"
end

# Counts every `@atomic ...` site (backend.atomic_macro) anywhere within
# a generated device kernel Expr, at any nesting depth -- used to scale
# stade_validate_gpu_file's default tolerance (see its own comment).
function val_count_atomic_sites(expr, atomic_macro)
    expr isa Expr || return 0
    n = (expr.head == :macrocall && !isempty(expr.args) && expr.args[1] == atomic_macro) ? 1 : 0
    n += (expr.head == :call && !isempty(expr.args) && expr.args[1] === cgen_warp_add_name()) ? 1 : 0
    for a in expr.args
        n += val_count_atomic_sites(a, atomic_macro)
    end
    return n
end

function stade_validate_gpu_file(in_path::String, out_path::String, backend;
                                  yaml_path::Union{String,Nothing} = nothing,
                                  scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
                                  trials::Int = 3, rtol::Union{Float64,Nothing} = nothing,
                                  self_check::Bool = true,
                                  keep_push_pop::Bool = false, fuse_ii_loops::Bool = false,
                                  mode::Symbol = :adjoint,
                                  reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT,
                                  warp_atomics::Bool = true, sync_free_reduce::Bool = true)
    keep_push_pop &&
        error("stade_validate_gpu_file: keep_push_pop=true has no GPU target -- :stack mode's push!/pop! is inherently host-only")
    mode in (:adjoint, :hvp) ||
        error("stade_validate_gpu_file: mode must be :adjoint or :hvp, got $(mode)")
    yp = yaml_path === nothing ? io_gpu_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = io_read_baseline_yaml(yp)
    val_coerce_int_arrays!(kernel, baseline.values)

    gen_out = mode == :hvp ? stade_hvp(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops) :
                             stade_adjoint(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops)
    gpu_init = stade_gpu(gen_out.initstacks, backend; warp_atomics = warp_atomics, sync_free_reduce = sync_free_reduce)
    # reduction_threshold reaches the GPU parity oracle. Without it the choice had
    # no test that ran on a device: validate_corpus is CPU and
    # validate_offload counts loops. That is how a `dot` call no cuBLAS on
    # this cluster can execute, and a whole-array lowering of a partial
    # range, both survived in emitted code.
    gpu_gen  = stade_gpu(mode == :hvp ? gen_out.hvp : gen_out.adjoint, backend; reduction_threshold = reduction_threshold, warp_atomics = warp_atomics, sync_free_reduce = sync_free_reduce)
    stack_arg_names = Symbol[a for a in gen_out.initstacks.args[1].args[2:end]]
    seeds = [val_random_values_like(kernel, baseline.values) for _ in 1:trials]
    # the HVP's two extra per-trial directions -- drawn the same way as the
    # reverse seed, independently of it and of each other
    tan_seeds   = mode == :hvp ? [val_random_values_like(kernel, baseline.values) for _ in 1:trials] : Any[]
    shtan_seeds = mode == :hvp ? [val_random_values_like(kernel, baseline.values) for _ in 1:trials] : Any[]

# IEEE float addition isn't associative, so an atomic accumulation sums its contributing threads in whatever order the
# scheduler runs them -- different from the CPU adjoint's sequential summation, though both compute the identical sum. A kernel
# with one or two atomic sites reproduces to within ~1e-16; unet's 35 sites accumulate that reordering noise repeatedly,
# measuring ~2.5e-13 -- correct, not a bug, just past a tolerance calibrated for the atomic-light case. Scaling linearly with
# atomic-site count is a conservative proxy, verified against unet's case. An explicit rtol always overrides this.
    resolved_rtol = rtol
    if resolved_rtol === nothing
        atomic_sites = sum(val_count_atomic_sites(k, backend.atomic_macro) for k in gpu_gen.kernels; init = 0)
        resolved_rtol = 2.5e-14 * max(1, atomic_sites)
    end

    script = val_gpu_parity_script(kernel, baseline.int_args, baseline.values, seeds,
                                    gen_out, gpu_init, gpu_gen, stack_arg_names, backend;
                                    rtol = resolved_rtol, mode, tan_seeds, shtan_seeds)
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
                                   scale::Float64 = 1.0, int_lo::Int = 3, int_hi::Int = 5,
                                   trials::Int = 3, rtol::Union{Float64,Nothing} = nothing,
                                   self_check::Bool = true,
                                   keep_push_pop::Bool = false, fuse_ii_loops::Bool = false,
                                   mode::Symbol = :adjoint,
                                   reduction_threshold::Int = CGEN_REDUCTION_THRESHOLD_DEFAULT)
    keep_push_pop &&
        error("stade_validate_jacc_file: keep_push_pop=true has no GPU target -- :stack mode's push!/pop! is inherently host-only")
    mode in (:adjoint, :hvp) ||
        error("stade_validate_jacc_file: mode must be :adjoint or :hvp, got $(mode)")
    yp = yaml_path === nothing ? io_gpu_yaml_path(in_path) : yaml_path
    io_path_exists(yp) || stade_generate_baseline_file(in_path; yaml_path = yp, scale = scale, int_lo = int_lo, int_hi = int_hi, self_check = self_check)
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = io_read_baseline_yaml(yp)
    val_coerce_int_arrays!(kernel, baseline.values)

    gen_out = mode == :hvp ? stade_hvp(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops) :
                             stade_adjoint(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops)
    backend = val_jacc_backend()
    jacc_init = stade_jacc(gen_out.initstacks)
    jacc_gen  = stade_jacc(mode == :hvp ? gen_out.hvp : gen_out.adjoint; reduction_threshold = reduction_threshold)
    stack_arg_names = Symbol[a for a in gen_out.initstacks.args[1].args[2:end]]
    seeds = [val_random_values_like(kernel, baseline.values) for _ in 1:trials]
    tan_seeds   = mode == :hvp ? [val_random_values_like(kernel, baseline.values) for _ in 1:trials] : Any[]
    shtan_seeds = mode == :hvp ? [val_random_values_like(kernel, baseline.values) for _ in 1:trials] : Any[]

    # same reasoning as stade_validate_gpu_file's own resolved_rtol --
    # Atomix.@atomic's cross-thread reordering is the same non-
    # associative-float-sum phenomenon regardless of which backend
    # emits the atomic.
    resolved_rtol = rtol
    if resolved_rtol === nothing
        atomic_sites = sum(val_count_atomic_sites(k, backend.atomic_macro) for k in jacc_gen.kernels; init = 0)
        resolved_rtol = 2.5e-14 * max(1, atomic_sites)
    end

    script = val_gpu_parity_script(kernel, baseline.int_args, baseline.values, seeds,
                                    gen_out, jacc_init, jacc_gen, stack_arg_names, backend;
                                    rtol = resolved_rtol, mode, tan_seeds, shtan_seeds)
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

# Value-needed variant of ii_expr_reads: counts only a read whose local partial actually requires
# the variable's own VALUE, using the same top-down linearity rule as snap_var_value_needed!. The
# canonical reduction write-back `out[i] = s` is linear in s and needs no restored primal, so it
# must NOT count as an escape for a kind whose adjoint stays at the backward position -- counting
# it would refuse every reduction loop there is. `out[i] = out[i] + s * s` does need it.
function ii_value_reads(expr, vars, acc, needed)
    if expr isa Expr && expr.head == :call
        op = expr.args[1]
        child = (op == :+ || op == :-) ? needed : true
        for a in expr.args[2:end]
            ii_value_reads(a, vars, acc, child)
        end
    elseif expr isa Expr && expr.head == :ref
        needed && (expr.args[1] in vars) && push!(acc, expr.args[1])
        for a in expr.args[2:end]
            ii_value_reads(a, vars, acc, needed)
        end
    elseif expr isa Symbol
        needed && (expr in vars) && push!(acc, expr)
    end
    return nothing
end

# Walks `body` in forward order, threading `alive` (vars still carrying `target`'s contribution) the way
# snap_fwd_walk!/agen_fwd_walk_loop! thread `seen` -- returns the updated alive-set, accumulating escapes into `escaped`. A var
# is removed from `alive` on a fresh (non-self-referencing) assignment; any read (linear or nonlinear) of a still-alive var is
# an escape. `:if` is conservative: a var survives only if it survives both arms. A write's own lhs index expressions are
# checked too -- an alive scalar used as an index elsewhere is still a genuine read.
function ii_kill_and_collect!(body, alive, escaped, value_only = false; save_ok = false)
    for stmt in body
        isempty(alive) && return alive
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            local_reads = Set{Symbol}()
            (save_ok && cgen_is_save_stmt(stmt)) ||
            (value_only ? ii_value_reads(stmt.rhs, alive, local_reads, false) :
                         ii_expr_reads(stmt.rhs, alive, local_reads))
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
            alive_then = ii_kill_and_collect!(stmt.then, copy(alive), escaped, value_only; save_ok)
            alive_els  = ii_kill_and_collect!(stmt.els,  copy(alive), escaped, value_only; save_ok)
            alive = intersect(alive_then, alive_els)
        elseif stmt.kind == :for
            # A kill inside a loop only counts when the loop is GUARANTEED to run -- a runtime bound may execute zero
            # times, leaving the incoming value for a later read. Reads are collected either way; only the kill is
            # conditional. The `:if` arm above is the same rule as an intersection: kill only when both branches kill.
            # Getting this wrong under-reports escapes, fusing a loop whose scalar is still live -- ii_kill is the
            # witness, at 0.39.
            inner = ii_kill_and_collect!(stmt.body, copy(alive), escaped, value_only; save_ok)
            cgen_loop_runs_once(stmt) && (alive = inner)
        elseif stmt.kind == :while
            # A kill inside a loop only counts when the loop is GUARANTEED to run -- a runtime bound may execute zero
            # times, leaving the incoming value for a later read. Reads are collected either way; only the kill is
            # conditional. The `:if` arm above is the same rule as an intersection: kill only when both branches kill.
            # Getting this wrong under-reports escapes, fusing a loop whose scalar is still live -- ii_kill is the
            # witness, at 0.39.
            inner = ii_kill_and_collect!(stmt.body, copy(alive), escaped, value_only; save_ok)
            cgen_loop_runs_once(stmt) && (alive = inner)
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
        elseif stmt.kind == :while
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
function ii_escapes_nested(kernel_body, target, vars, value_only = false; save_ok = false)
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
            ii_kill_and_collect!(vcat(after, before), copy(alive), escaped, value_only; save_ok)
            alive = ii_kill_and_collect!(after, alive, escaped, value_only; save_ok)
        else
            alive = ii_kill_and_collect!(after, alive, escaped, value_only; save_ok)
        end
    end
    return escaped
end


# Value-needed reads of `vars` that are a genuine hazard for a kind whose adjoint stays at the backward position.
# Same walk as ii_escapes_nested, value-needed only, EXCEPT at the innermost enclosing body: a var that body is
# guaranteed to block-boundary snapshot is restored at the head of every reverse iteration before any statements run,
# so a read there is already served -- exactly what makes a cell-scatter gather loop safe (`auxu = 0.0` at the top,
# gathered by one sub-loop, consumed by the next). Outer levels get no such exemption.
function ii_red_escapes(kernel_body, target, vars, bvars)
    path = ii_find_ancestor_path(kernel_body, target, false)
    path === nothing && return copy(vars)
    escaped = Set{Symbol}()
    alive = copy(vars)
    for i in eachindex(path)
        isempty(alive) && break
        (body, idx, repeating) = path[i]
        after = body[idx+1:end]
        here = Set{Symbol}()
        if repeating
            before = body[1:idx-1]
            ii_kill_and_collect!(vcat(after, before), copy(alive), here, true)
            alive = ii_kill_and_collect!(after, alive, here, true)
        else
            alive = ii_kill_and_collect!(after, alive, here, true)
        end
        i == 1 && setdiff!(here, snap_boundary_kill_vars(body, bvars))
        union!(escaped, here)
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
            # A nested loop that is ITSELF classified re-establishes its own scalars per inner iteration -- but only
            # for a read INSIDE that loop. A read at THIS level that precedes the loop is reversed in the fused body's
            # `bwd` half, running after the whole forward nest, so it sees the scalar's last inner-iteration value
            # instead of what it read going forward. A read that FOLLOWS the loop is safe for the same reason: going
            # forward it already read that last inner value.
            covered = plan !== nothing && haskey(plan, agen_site_key(body, idx))
            (!covered && ii_assigns_any(stmt.body, vars)) && return true
        elseif stmt.kind == :while
            # A nested loop that is ITSELF classified re-establishes its own scalars per inner iteration -- but only
            # for a read INSIDE that loop. A read at THIS level that precedes the loop is reversed in the fused body's
            # `bwd` half, running after the whole forward nest, so it sees the scalar's last inner-iteration value
            # instead of what it read going forward. A read that FOLLOWS the loop is safe for the same reason: going
            # forward it already read that last inner value.
            covered = plan !== nothing && haskey(plan, agen_site_key(body, idx))
            (!covered && ii_assigns_any(stmt.body, vars)) && return true
        elseif stmt.kind == :if
            (ii_fused_var_in_nested_for(stmt.then, vars, plan) ||
             ii_fused_var_in_nested_for(stmt.els, vars, plan)) && return true
        end
    end
    return false
end

# ---- reverse-direction escape: a read the loop is reversed BEFORE -------------------
# ii_escapes_nested asks whether the loop's own output reaches a later read. This asks the mirror question: does a read BEFORE the loop still need
# the value the loop overwrites? Such a read's adjoint runs AFTER the fused loop's backward code, and every fusing kind elides its fused scalars'
# snapshots, so nothing re-establishes the value in between. Walks each ancestor level's preceding siblings forward, accumulating vars read,
# dropping one on a fresh assignment (whose own snapshot site, outside the loop, isn't elided and restores the value itself).
function ii_pending_reads!(body, vars, pending, plan = nothing, limit = nothing)
    for (idx, stmt) in enumerate(body)
        # `limit` bounds this level to the statements BEFORE the candidate loop. Passed as an
        # index rather than a slice on purpose: agen_site_key is keyed on objectid(body), so a
        # slice would silently miss every plan entry and retire the exemption below.
        limit !== nothing && idx >= limit && break
        if stmt.kind == :assign
            reads = Set{Symbol}()
            ii_expr_reads(stmt.rhs, vars, reads)
            if stmt.lhs isa Expr
                for a in stmt.lhs.args[2:end]
                    ii_expr_reads(a, vars, reads)
                end
            end
            union!(pending, reads)
            if stmt.lhs isa Symbol && stmt.lhs in vars && agen_count_var_refs(stmt.rhs, stmt.lhs) == 0
                delete!(pending, stmt.lhs)
            end
        elseif stmt.kind == :for
            # A read inside a loop that is ITSELF classified is not this candidate's problem
            # for the scalars that loop assigns: its own fused body or backward-position
            # recompute rebuilds them for its own reads, so nothing downstream has to preserve
            # them. unet's two max-pool blocks are exactly this -- both name their taps a11/m1,
            # and without the exemption the second block inherits the first block's reads.
            own = plan !== nothing && haskey(plan, agen_site_key(body, idx)) ?
                      Set{Symbol}(v for v in vars if ii_assigns_any(stmt.body, Set([v]))) :
                      Set{Symbol}()
            # A nested loop otherwise contributes its reads but never kills: it may run zero
            # times, leaving the incoming value in place. Conservative in the safe direction,
            # since an un-killed var only adds refusals.
            ii_pending_reads!(stmt.body, setdiff(vars, own), pending, plan)
            setdiff!(pending, own)
        elseif stmt.kind == :while
            # A read inside a loop that is ITSELF classified is not this candidate's problem
            # for the scalars that loop assigns: its own fused body or backward-position
            # recompute rebuilds them for its own reads, so nothing downstream has to preserve
            # them. unet's two max-pool blocks are exactly this -- both name their taps a11/m1,
            # and without the exemption the second block inherits the first block's reads.
            own = plan !== nothing && haskey(plan, agen_site_key(body, idx)) ?
                      Set{Symbol}(v for v in vars if ii_assigns_any(stmt.body, Set([v]))) :
                      Set{Symbol}()
            # A nested loop otherwise contributes its reads but never kills: it may run zero
            # times, leaving the incoming value in place. Conservative in the safe direction,
            # since an un-killed var only adds refusals.
            ii_pending_reads!(stmt.body, setdiff(vars, own), pending, plan)
            setdiff!(pending, own)
        elseif stmt.kind == :if
            cond_reads = Set{Symbol}()
            ii_expr_reads(stmt.cond, vars, cond_reads)
            union!(pending, cond_reads)
            # a var stays pending unless BOTH arms re-establish it
            then_p = copy(pending); els_p = copy(pending)
            ii_pending_reads!(stmt.then, vars, then_p, plan)
            ii_pending_reads!(stmt.els, vars, els_p, plan)
            union!(pending, union(then_p, els_p))
        end
    end
    return nothing
end

function ii_read_reaches_from_before(kernel_body, loop_body, vars, plan = nothing)
    isempty(vars) && return Set{Symbol}()
    path = ii_find_ancestor_path(kernel_body, loop_body, false)
    path === nothing && return copy(vars)   # couldn't locate -- conservative
    # ii_find_ancestor_path returns innermost-first; walk it OUTERMOST-first and carry `pending`
    # inward, so a fresh assignment sitting between an outer level's read and this loop clears
    # it. `s = 0.0` immediately before a contraction loop is exactly that: the previous block's
    # read of the same scratch name is re-established before this loop ever runs, and checking
    # each level in isolation would refuse every block after the first.
    pending = Set{Symbol}()
    for (level_body, idx, _) in reverse(path)
        ii_pending_reads!(level_body, vars, pending, plan, idx)
    end
    return pending
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
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
        elseif stmt.kind == :while
            isempty(agen_tripcount_bound_vars(stmt, reassigned)) || return true
            ii_body_has_surviving_snapshot(stmt.body, kinds, active_map, sites, vn_local, reassigned) && return true
        elseif stmt.kind == :if
            return true   # agen_forward_body always pushes a branch flag
        end
    end
    return false
end

# A fused or recomputed local reaches the backward position holding the value of its LAST assignment in the body. A local assigned
# more than once per iteration with a read between two of its assignments (the nine taps of a tiled convolution each reload the same
# weight scalar) therefore feeds its early reads the wrong version. True when some variable of `vars` is read after its first
# assignment and before its last one, in program order. An accumulation `v = v + t` does not count as a read of v. A loop that reads it first and then assigns it once is a live-in, a
# different question, answered by ii_scalar_live_in.
function ii_versions_read(body::Vector{NamedTuple}, vars::Set{Symbol})
    first_w = Dict{Symbol,Int}()
    last_w = Dict{Symbol,Int}()
    nw = Dict{Symbol,Int}()
    reads = Dict{Symbol,Vector{Int}}()
    pos = Ref(0)
    note_reads!(e) = for v in cgen_expr_symbols(e)
        v in vars && push!(get!(reads, v, Int[]), pos[])
    end
    function walk!(b)
        for st in b
            pos[] += 1
            if st.kind == :assign
                # `v = v + t` reads v only to add to it: linear, so its adjoint needs no version of v
                if st.lhs isa Symbol && any(t -> t === st.lhs, cgen_flatten_sum(st.rhs)) &&
                   cgen_count_sym_uses(st.rhs, st.lhs) == 1
                    note_reads!(cgen_sum_excluding(cgen_flatten_sum(st.rhs), findfirst(t -> t === st.lhs, cgen_flatten_sum(st.rhs))))
                else
                    note_reads!(st.rhs)
                end
                st.lhs isa Expr && note_reads!(st.lhs)
                if st.lhs isa Symbol && st.lhs in vars
                    pos[] += 1
                    haskey(first_w, st.lhs) || (first_w[st.lhs] = pos[])
                    last_w[st.lhs] = pos[]
                    nw[st.lhs] = get(nw, st.lhs, 0) + 1
                end
            elseif st.kind == :if
                note_reads!(st.cond)
                walk!(st.then)
                walk!(st.els)
            elseif st.kind == :for
                note_reads!(Expr(:tuple, st.lo, st.hi, st.step))
                walk!(st.body)
            elseif st.kind == :while
                note_reads!(st.cond)
                walk!(st.body)
            end
        end
    end
    walk!(body)
    return Set(v for v in keys(nw) if nw[v] >= 2 && any(p -> first_w[v] < p < last_w[v], get(reads, v, Int[])))
end

function snap_ii_classify(stmt, kernel, value_needed, known_consts, active_map, plan = nothing)
    stmt.kind == :for || return :none
    # a loop that writes a restored integer (see snap_itbr_vars) pushes it on every iteration, which no fusing kind replays
    any(v -> snap_itbr_writes_in(stmt, v), snap_itbr_vars(kernel, active_map)) && return :none
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
    # see ii_versions_read -- a local reassigned between its own reads cannot be rebuilt from its last value
    isempty(ii_versions_read(stmt.body, setdiff(vn_local, redvars))) || return :none
    # see ii_read_reaches_from_before -- a read positioned before this loop is reversed AFTER
    # its backward code, and every fusing kind elides the fused scalars' snapshots, so the
    # value that read needs is gone. The forward-direction dual is the ii_escapes_nested gate
    # below; this one has no kind it is safe for, so it is checked for the whole vn_local set.
    isempty(ii_read_reaches_from_before(kernel.body, stmt.body, vn_local, plan)) || return :none
    # A loop whose own trip count is a reassigned scalar (`for i = 1:cur` with `cur` retired
    # each pass) carries a tripcount snapshot, and every fusing kind re-runs the header at the
    # backward position against whatever `cur` holds there rather than the value this iteration
    # used. Gate 2 above only inspects the loop's body, so this shape slipped through and
    # produced wrong gradients in both stack modes whenever fusion was on.
    isempty(agen_tripcount_bound_vars(stmt, snap_collect_reassigned_all(kernel.body))) || return :none
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
        # Same question OUTSIDE the body. vn_red is excluded from push, and every kind carrying it defers its
        # adjoint to the backward position -- so a read of the accumulator after this loop is reversed BEFORE
        # the loop's backward code, with nothing having restored the accumulator's primal. Only a value-needed
        # read matters: the canonical `out[i] = s` write-back is linear and safe, which is why this uses
        # ii_value_reads rather than the flat read test vn_ind gets below.
        isempty(ii_red_escapes(kernel.body, stmt.body, vn_red, snap_boundary_stack_vars(kernel))) || return :none
    end
    if !isempty(vn_ind)
        escaped = ii_escapes_nested(kernel.body, stmt.body, vn_ind)
        isempty(escaped) || return :none
    end
    # :independent is fused at the FORWARD position, so every adjoint it consumes must already exist there. The adjoint of an
    # active scalar this loop assigns and later code reads is seeded only by that code's backward sweep, which runs after the
    # forward position -- an accumulator is deferred to the backward position (:mixed) instead. Any other live-out local keeps
    # only its last iteration's value, so the un-reversed loops of every fusing kind would hand its adjoint to the first one.
    live_out = Set{Symbol}(v for v in union(local_names, redvars) if get(active_map, v, false))
    isempty(live_out) || (live_out = ii_escapes_nested(kernel.body, stmt.body, live_out))
    isempty(setdiff(live_out, redvars)) || return :none
    defer = !isempty(live_out)
    # only :reduction/:mixed run the body twice -- see
    # ii_body_has_surviving_snapshot.
    if (!isempty(vn_red) || has_escape || defer) &&
       (ii_body_has_surviving_snapshot(stmt.body, kernel.sig.kinds, active_map, snap_value_needed_sites(kernel),
                                        vn_local, snap_collect_reassigned_all(kernel.body)) ||
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
        return defer ? :mixed : :independent
    end
end

function agen_ii_classify(stmt, kernel, value_needed, known_consts, active_map, plan = nothing)
    stmt.kind == :for || return :none
    # see snap_ii_classify
    any(v -> snap_itbr_writes_in(stmt, v), snap_itbr_vars(kernel, active_map)) && return :none
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
    # see ii_versions_read -- a local reassigned between its own reads cannot be rebuilt from its last value
    isempty(ii_versions_read(stmt.body, setdiff(vn_local, redvars))) || return :none
    # see ii_read_reaches_from_before -- a read positioned before this loop is reversed AFTER
    # its backward code, and every fusing kind elides the fused scalars' snapshots, so the
    # value that read needs is gone. The forward-direction dual is the ii_escapes_nested gate
    # below; this one has no kind it is safe for, so it is checked for the whole vn_local set.
    isempty(ii_read_reaches_from_before(kernel.body, stmt.body, vn_local, plan)) || return :none
    # A loop whose own trip count is a reassigned scalar carries a tripcount snapshot, and every
    # fusing kind re-runs the header at the backward position against whatever the retired value
    # holds rather than what this iteration used. Gate 2 above only inspects the loop's body, so
    # this shape slipped through and produced wrong gradients in both stack modes whenever
    # fusion was on.
    isempty(agen_tripcount_bound_vars(stmt, agen_collect_reassigned_all(kernel.body))) || return :none
    vn_red = intersect(vn_local, redvars)
    vn_ind = setdiff(vn_local, redvars)
    if !isempty(vn_red)
        inside_nonlinear = Set{Symbol}()
        agen_collect_value_needed!(stmt.body, inside_nonlinear)
        isempty(intersect(vn_red, inside_nonlinear)) || return :none
        # Same question OUTSIDE the body. vn_red is excluded from push, and every kind carrying it defers its
        # adjoint to the backward position -- so a read of the accumulator after this loop is reversed BEFORE
        # the loop's backward code, with nothing having restored the accumulator's primal. Only a value-needed
        # read matters: the canonical `out[i] = s` write-back is linear and safe, which is why this uses
        # ii_value_reads rather than the flat read test vn_ind gets below.
        isempty(ii_red_escapes(kernel.body, stmt.body, vn_red, agen_boundary_stack_vars(kernel))) || return :none
    end
    if !isempty(vn_ind)
        escaped = ii_escapes_nested(kernel.body, stmt.body, vn_ind)
        isempty(escaped) || return :none
    end
    # :independent is fused at the FORWARD position, so every adjoint it consumes must already exist there. The adjoint of an
    # active scalar this loop assigns and later code reads is seeded only by that code's backward sweep, which runs after the
    # forward position -- an accumulator is deferred to the backward position (:mixed) instead. Any other live-out local keeps
    # only its last iteration's value, so the un-reversed loops of every fusing kind would hand its adjoint to the first one.
    live_out = Set{Symbol}(v for v in union(local_names, redvars) if get(active_map, v, false))
    isempty(live_out) || (live_out = ii_escapes_nested(kernel.body, stmt.body, live_out))
    isempty(setdiff(live_out, redvars)) || return :none
    defer = !isempty(live_out)
    if (!isempty(vn_red) || has_escape || defer) &&
       (ii_body_has_surviving_snapshot(stmt.body, kernel.sig.kinds, active_map, agen_value_needed_sites(kernel),
                                        vn_local, agen_collect_reassigned_all(kernel.body)) ||
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
        return defer ? :mixed : :independent
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