# ============================================================
# STADE.jl -- source-to-source AD for skill-jade-compliant Julia
# kernels. See skill-stade.md for the full house-style contract.
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
#   cgen_   GPU codegen   loop-nest transform (not a derivative): splits
#                         each iteration-independent (non-i_seq_) loop
#                         into a device kernel + launch call, for
#                         whichever GPU vendor a `gpu_backend` describes
#                         (CUDA/AMDGPU/Metal). Consumes a plain
#                         skill-jade kernel OR one of tgen_/agen_'s own
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
#     -- :while intentionally unsupported for now, see skill-stade.md
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



# ==================== inl_* ====================================
# Multi-kernel nested call graphs, via source-level inlining -- runs
# before parse_kernel ever sees anything. A nested call is always a
# bare statement `callee_name(arg1, arg2, ...)` on its own line
# (matches how skill-jade kernels return results: mutated array
# args, not values), with bare-symbol-only arguments. Splicing a
# callee's body into every one of its call sites removes the call
# boundary entirely, so parse_/shape_/act_/snap_/lin_/agen_ never
# see a multi-kernel graph at all -- only ever a single flat body.
# Recursion in the call graph is out of scope (hard error).

# inl_inline_calls(kernels) -> Dict{Symbol,Expr}
#   kernels :: Dict{Symbol,Expr}, one `function ... end` Expr per
#   kernel name, resolved from a corpus supplied up front (never
#   auto-discovered). Pure function of its input (rule 7) -- no
#   `rand`-based naming anywhere in this stage, see inl_rename_map.
#   Returns one fully-inlined Expr per kernel name (every kernel in
#   the input, not just call-graph roots), with every user-kernel
#   call anywhere in that kernel's call graph expanded away.
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
# bare :call statement is never legal skill-jade input on its own, so
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

# a caller's own confirmed kind map, computed ONCE up front from its
# *original* (pre-inlining) body with every bare-call statement
# simply dropped (recursively, at any nesting depth) -- a call site
# contributes no syntactic evidence of its own either way, so
# dropping it is exactly the same as never having seen it, and
# doesn't depend on inlining order or which call site is being
# checked. Reused for every call-site check in this caller.
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

# processes one (possibly nested) statement_list of `caller_name`,
# expanding every bare-call statement in it against `finalized`
# callees and returning the replacement statement vector.
# `confirmed` is caller_name's own confirmed kind map (see
# inl_confirmed_kinds), static for the whole caller; `counter` is the
# per-caller call-site counter shared across the whole recursive
# descent.
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

# kind-checks one call site against the callee's own declared
# signature before any substitution happens -- the one thing pure
# inlining would otherwise silently lose (a caller passing a
# scalar_int where the callee's signature says scalar_float would
# just get quietly merged in and reinterpreted). A `:scalar_float`
# confirmed kind is skipped: it's shape_infer's default for "no
# evidence found", which is exactly what a pure pass-through argument
# looks like (its only evidence lives in the callee, not yet
# inlined) -- flagging that as a mismatch would reject the single
# most common multi-level call-chain shape. `:array_float`/
# `:array_int`/`:scalar_int` are never defaults (shape_infer only
# ever reaches them from real local syntactic evidence), so those are
# always enforced.
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
# Raw Expr -> validated kernel. Enforces every skill-jade rule that
# is actually visible at the Expr level as a hard error: no keyword
# args/defaults (at the def or at any call site; type annotations are
# allowed but inert), only the four variable shapes (no
# Bool/String/tuple/range stored in a variable), no indirect indexing,
# no broadcasting, i_seq_ prefix discipline, the intrinsic whitelist,
# div-not-÷. Compound assignment (`+=`/`-=`/`*=`/`/=`/`^=`) is allowed
# and desugared to a plain assignment; `÷=`/`%=`/`\=`/`.=` remain
# errors (no registered operator rule, or broadcast). General
# snake_case is not enforced -- STADE identifies names by symbol
# identity only. Two skill-jade rules are pure *source-text* concerns
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

# ==================== shape_* =================================
# Infer each argument/local's kind syntactically -- no explicit
# manifest to cross-check against -- any type annotation a kernel
# does carry is stripped in parse_signature and never consulted here,
# since STADE infers shapes from usage, not from declared types. Two
# syntactic signals decide a variable's kind:
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
        # the ELEMENT TYPE of the array being indexed follows from how
        # its indexing RESULT is used, not just from how its own
        # subscripts are used (those are already handled separately by
        # shape_mark_int_from_div_and_index!) -- if `X = A[idx...]` and
        # X is (or becomes) Int64, A itself must be Int64-elemented.
        # This is what correctly classifies a gather/permutation-table
        # array (e.g. a mesh connectivity array used only to index
        # other arrays) as array_int instead of defaulting to
        # array_float, even though nothing ever assigns it an int
        # literal directly.
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
#     needed later. Flagged per :assign whenever `var` is a VALUE-NEEDED
#     variable (see snap_value_needed_vars): its primal value feeds
#     some local partial derivative somewhere in the kernel, reached
#     through anything other than a chain of +/- from that use's
#     statement root. This single test replaces the old two-rule split
#     (self-reference vs. cross-statement read) -- self-reference is
#     just the special case where the "somewhere in the kernel" turns
#     out to be the write's own statement, and needs no separate
#     handling: `x = x*y` marks x needed from its own rhs; `x = x+y`
#     does not, from its own rhs, exactly like Hascoet's op_add/op_sub
#     TBR rule (see ADTBRAnalyzer.collectZonesUsedByDiffRhs) makes
#     neither operand of `+`/`-` ever need its value, self-reference or
#     not. Being conservative for any non-+/- operator, and for `if`
#     conditions unconditionally, only ever costs an unnecessary
#     push/pop pair; no loop-nesting condition narrows any of it, since
#     a write's own backward code runs at that write's mirrored
#     position in the reverse walk and any other value-needing use
#     could be processed earlier or later in that walk regardless of
#     what loop (if any) either one sits in.
#   :branch -- one per `if`, unconditionally: the reverse sweep must
#     replay whichever arm the forward sweep actually took.
#   :tripcount -- a loop's bounds reference a variable reassigned
#     elsewhere in the kernel, so its trip count could be gone by the
#     time the reverse sweep needs to replay it. Deliberately
#     conservative: doesn't try to prove the reassignment happens
#     strictly after or outside this loop, just flags it regardless.
#
# Iteration-independent elision: a non-self-referencing write gets no
# site at all when it's the sole assign site for `var` anywhere in the
# kernel, has no sequential-loop ancestor, and is never read before it
# runs -- see snap_check_assign!'s own comment. This is exactly the
# class of writes Hascoet et al. (2001) identify as needing no tape at
# all inside a genuinely independent loop.

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

# does `var`'s VALUE (not just its syntactic presence) feed some local
# partial derivative reachable from `expr`'s root? Top-down, mirroring
# ADTBRAnalyzer.collectZonesUsedByDiffRhs: `needed` starts false at a
# statement's rhs root and stays false through any nesting of +/- (a
# constant +-1 partial never needs an operand's value, whichever side
# of a `-` it's on, unary or binary) -- so a bare copy or a chain of
# sums/differences never marks anything needed. Any OTHER call --
# `*`, `/`, `^`, or any other function -- is treated as genuinely
# nonlinear: `needed` flips to true (and stays true) for everything
# inside it, since a generic partial can depend on any of that
# operator's arguments. `needed` never resets from true back to false
# on the way back down, matching that once inside a nonlinear
# operator's argument, that whole argument's value is live regardless
# of what's nested further inside it.
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

# every variable whose value is needed SOMEWHERE in the kernel -- the
# union, over every :assign rhs (walked from `needed = false`) and
# every :if condition (walked from `needed = true`, deliberately not
# refined: the reverse sweep never re-evaluates a condition, so this
# is conservative rather than load-bearing, and narrowing it isn't
# this analysis' job), of snap_var_value_needed!'s result
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

# every variable assigned via a scalar (non-array) :assign somewhere
# INSIDE a loop, at any nesting depth -- gated on in_loop rather than
# collecting every :assign in the kernel, since only a write that can
# execute more than once (i.e. sits inside some :for) can ever leave
# a DIFFERENT value behind at different points in the kernel's single
# execution. A var whose only :assign sites are all at top level,
# outside every loop, is a plain one-shot constant for the whole
# kernel run and can never need trip-count-style snapshot/restore --
# including it here would wrongly flag every loop bound that merely
# references it, forcing a push/pop pair whose pop then corrupts the
# loop's own value (see keep_push_pop history for how such a
# mis-scoped snapshot broke a real kernel's adjoint).
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

# one forward-order pass, emitting sites as they're found. A write
# needs a site whenever `var` is read anywhere else in the kernel --
# regardless of whether this particular write sits inside a loop,
# outside one, or which loop: what matters is only whether some OTHER
# statement's read of `var` could see a different value once forward
# and backward walk it in opposite orders, and that risk exists
# equally for a write inside a loop restoring a read inside a
# DIFFERENT (sibling or enclosing) block, a write outside a loop
# restoring a read inside one, or a write and read both at the top
# level. Over-snapshotting a var that turns out not to have needed it
# costs a harmless extra push/pop pair, never a correctness bug --
# every snapshot's push and pop occupy the mirrored position in the
# forward/backward walk, so nesting is always self-consistent
# regardless of which subset of writes get one.
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

# a write needs a site iff `var in value_needed` (see
# snap_value_needed_vars) -- subsumes both the old self-reference and
# cross-statement rules in one test. Also gated by the iteration-
# independent elision below (sole write, no ENCLOSING LOOP AT ALL --
# not just no sequential ancestor: a write inside any loop, sequential
# or not, still re-executes on every iteration and needs its per-
# iteration value restored for a later reverse-sweep read, so `in_loop`
# must cover every :for ancestor -- no self-reference, never read
# before it runs).
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

# is `lhs = rhs` an identity-preserving self-update -- one where
# d(new)/d(old) = 1, so old-lhs and new-lhs are the SAME QUANTITY FOR
# THE SHADOW's PURPOSES (agen_backward_assign's distribute step can
# skip resetting the shadow to 0, since accumulating into it further
# is exactly right). This says nothing about the snapshot decision
# itself -- that's now snap_value_needed_vars's job, and it already
# reaches the right answer for this shape without needing to know
# about it specifically (`+`/`-` never propagate `needed`, self-
# reference or not). Two shapes qualify here: lhs as any direct top-
# level `+` operand, or lhs as specifically the left/minuend operand
# of a binary `-` (the right operand does not qualify -- d(a-b)/db =
# -1, not an identity). Either way, lhs's exact slot must not appear
# anywhere else in rhs (a different index of the same array is a
# different slot and doesn't disqualify it).
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
# `snap_value_needed_vars` decides "does var's value matter anywhere"
# at whole-variable granularity. `snap_fwd_walk!` refines this to a
# per-write decision using the DIRECTION that actually matters for
# TBR: a write w needs its pre-write value snapshotted iff (a) w is
# self-referencing (its own rhs reads var -- the write's own backward
# code needs var-old to distribute the adjoint to w's OTHER operands),
# or (b) some nonlinear read of var occurred STRICTLY BEFORE w in
# forward execution order (matching the existing snap_read_before
# helper's own direction). A nonlinear read that happens AFTER w reads
# w's own NEW value, not the value w is about to destroy, so it plays
# no part in w's own snapshot need -- that read's write (if it has
# one) has since overwritten w's contribution already; w's write is
# only rewound for the benefit of something EARLIER in forward order.
# Shadow analysis only for now -- computed and cross-checked against
# snap_plan's existing decisions (new decisions must always be a
# subset of the old ones), not yet wired into codegen. See
# skill-stade's site-level TBR rollout plan.
#
# `seen` = vars with a nonlinear read (or unconditional if-cond read)
# strictly before the current point. Returns the seen-set as it stood
# just AFTER `body` finished (its OUT set), and records into
# `decisions` (a Dict keyed by agen_site_key(body, idx)) whether each
# :assign site needs a snapshot.
function snap_fwd_walk!(body, seen, active_map, decisions)
    for idx in eachindex(body)
        stmt = body[idx]
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            # local_reads: vars read NONLINEARLY within this rhs alone
            # (needed=false at the root, same rule as everywhere else)
            # -- self-reference only forces a snapshot when var's own
            # occurrence is one of these, not merely present under a
            # linear +/- (pure accumulation needs no old value: house
            # style's der_rule comment, d(new)/d(old)=1).
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

# Loop forward fixed point: a read near the TOP of a loop body is, for
# every iteration but the first, actually preceded by the PREVIOUS
# iteration's reads from later in the body (iteration i+1's statement
# 1 runs after iteration i's statement N) -- so a write near the top
# can need a snapshot due to a read near the bottom, one iteration
# back. `seen` only grows across passes (monotone, bounded by the
# finite variable universe), so this terminates. A final pass re-walks
# with the converged seen-set so `decisions` reflects the fixed point
# (covering iteration >=2), not the first-iteration-only guess.
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

function agen_emit(kernel, lin_plan, snapshot_plan; keep_push_pop::Bool = true, push_pop = nothing, ii_plan = nothing)
    active_map = act_analyze(kernel)
    layout = nothing
    value_needed = exempt = stacks = nothing
    if !keep_push_pop
        # Tier B kernels (a ragged/data-dependent loop bound --
        # agen_tier_b_offender) are no longer refused here: agen_layout
        # (Phase A-D) resolves as much as it can into closed-form,
        # GPU-eligible ragged-block tables, falling back to plain
        # :stack semantics (agen_indexed_layout's own tainted_stacks,
        # reused as agen_layout's sub-engine) only for whatever a
        # block genuinely can't resolve -- see the "Tier B
        # (implemented...)" comment above agen_local_position and
        # agen_layout's own docstring.
        # NOTE: ii_plan/fuse_ii_loops has not been validated in
        # combination with keep_push_pop=false. agen_layout/
        # agen_stack_map are not ii_plan-aware, so a fused var's stack
        # would still be sized and allocated here as if unfused; it
        # would simply never be written to or read from (since
        # agen_emit_ii_loop never calls push/pop for it), which should
        # be harmless but is untested -- treat this combination as
        # unsupported until it's actually exercised.
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
    # fuse_ii_loops can leave a stack with zero remaining push!/pop!
    # calls anywhere in the generated body (every write-site that
    # would have used it got fused away) -- checked from the actual
    # generated code, never predicted ahead of time, since a var's
    # ii_plan coverage does NOT by itself guarantee its stack is fully
    # unused (see agen_drop_unused_stack_args's own comment). Scoped
    # to keep_push_pop=true, matching fuse_ii_loops's existing scope.
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
    # `tier_b_extra_args` (Phase D) is the SAME table/total/value-table
    # name list agen_init_emit returned for this kernel, appended here
    # in that exact order so it lines up with initstacks_*'s own
    # return tuple -- see agen_init_emit's own docstring for why this
    # keeps val_init_stacks' generic splat-the-whole-tuple convention
    # working unmodified. Empty under keep_push_pop=true or a kernel
    # with no ragged block.
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

# one `nm = Vector{T}(...)` init statement -- growable for
# `keep_push_pop` or a genuinely-tainted (Tier B fallback) stack;
# presized from `layout.sizes` for a pure Tier A stack; presized from
# `layout.block_totals` for a stack resolved into one or more Tier B
# ragged-block tables (Phase D) -- see agen_layout's own docs for what
# populates each of these three, mutually exclusive, dicts. Factored
# out of agen_init_emit purely to keep that function's own
# comprehension a plain one-call-per-element form.
function agen_init_alloc_stmt(nm, kind, keep_push_pop::Bool, layout)
    tainted = layout !== nothing && nm in layout.tainted_stacks
    grow = keep_push_pop || tainted
    size_expr = grow ? nothing : (haskey(layout.sizes, nm) ? layout.sizes[nm] : layout.block_totals[nm])
    return Expr(:(=), nm, agen_stack_alloc_expr(kind, grow, size_expr))
end

# Returns `(expr, table_names, tot_names, val_names)`: the extra three
# are empty under keep_push_pop=true or a kernel with no ragged block
# at all (`layout.blocks` empty), and otherwise name every extra
# per-stack table/total/value-table `agen_tier_b_kernel_skeleton`
# (Phase B) builds -- callers (agen_emit/stade_hvp) append these to
# both `initstacks_*`'s own return AND `<name>_b`/`<name>_hv`'s
# argument list, in the SAME order, so val_init_stacks' generic
# splat-the-whole-tuple convention keeps working unmodified: Phase D
# adds no special-cased plumbing to the validation/calling machinery,
# only more return values and more parameters, kept in lockstep here.
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
    # expression actually references -- see skill-stade.md's
    # keep_push_pop entry for why the minimal set (rather than the
    # kernel's full argument list) was chosen
    fargs = keep_push_pop ? Symbol[] : layout.free_vars
    sizing_stmts = Any[]
    total_of = Dict{Symbol,Symbol}()
    if !keep_push_pop && !isempty(layout.tainted_stacks)
        (sizing_stmts, total_of) = agen_tier_b_sizing_stmts(kernel, active_map, value_needed, exempt, stacks, layout.tainted_stacks; push_pop = push_pop)
        # the sizing skeleton may reference kernel arguments the Tier A
        # size formulas alone never would (e.g. a multigrid smoother's
        # own iteration-count arguments, never part of any closed-form
        # offset) -- widen the signature to match, same free-vars-of-
        # what-we-actually-reference
        # principle as the plain Tier A case above
        sizing_free = Set{Symbol}()
        for s in sizing_stmts
            agen_collect_expr_vars!(s, sizing_free)
        end
        fargs = sort(union(fargs, intersect(sizing_free, kernel.sig.args)); by = string)
    end
    # Tier B ragged-block tables (Phase D): built once here, exactly
    # like the fallback sizing pass above but for stacks that resolved
    # into one or more agen_layout blocks instead of falling back
    # entirely. agen_tier_b_kernel_skeleton already interleaves every
    # block's own table-construction loop with whatever scalar prelude
    # surrounds it in the kernel body (Phase B) -- nothing more is
    # needed here beyond widening the signature the same way.
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
    # Tier B: a tainted stack (see agen_use_stack_push) has no size
    # formula at all -- layout.sizes/layout.block_totals deliberately
    # have no entry for it -- so it always allocates growable, exactly
    # like keep_push_pop's own true-case, regardless of the
    # kernel-wide flag; a computed __sz_* total (from the fallback
    # sizing pass above) additionally gets it a sizehint! right after,
    # to avoid push!'s own repeated reallocation as it grows -- see
    # agen_tier_b_sizing_stmts.
    alloc_stmts = Any[]
    for nm in names
        push!(alloc_stmts, agen_init_alloc_stmt(nm, kind_of[nm], keep_push_pop, layout))
        haskey(total_of, nm) && push!(alloc_stmts, Expr(:call, :sizehint!, nm, total_of[nm]))
    end
    body = vcat(sizing_stmts, table_stmts, alloc_stmts)
    push!(body, emit_return_scalars(vcat(names, table_names, tot_names, val_names)))
    return (Expr(:function, Expr(:call, fname, fargs...), Expr(:block, body...)), table_names, tot_names, val_names)
end

# :array/:value stacks hold the popped Float64 scalar itself (every
# push is of one exact lhs reference, never a whole-array copy);
# :branch/:tripcount stacks hold Int64 flags/bounds. Under
# keep_push_pop=false, `size_expr` sizes the allocation up front
# (Vector{T}(undef, size_expr)) instead of growing via push! -- every
# stack still holds exactly the same element type either way.
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

# ---- block-boundary scalar restoration ------------------------------
#
# A scalar var written ONLY inside a nested sub-:for/:if of `body`
# (never as a top-level statement of `body` itself) has no existing
# restore mechanism reaching a LATER, SIBLING statement's own read
# within the SAME `body` when the enclosing loop repeats -- see
# skill-stade.md's writeup on this bug (cavgx/cavgy/cavgz built by one
# `for i_loc` loop, read unchanged by a later sibling `for i_loc` loop,
# both inside a repeating `for i_cell`). The per-write-statement
# push/pop machinery restores a var to whatever it held immediately
# BEFORE that write, benefiting whatever runs immediately before it in
# the reversed sweep -- exactly right for a var read again within the
# SAME loop that writes it (see `vere` elsewhere in this file), but
# wrong for a sibling loop's read, since nothing ever re-establishes
# the var's post-loop value before that read's own backward code runs.
# `agen_nested_write_vars` finds the candidates: variables whose
# writes are all strictly nested within body's own sub-loops/sub-ifs.
function agen_nested_write_vars(body, kinds)
    vars = Set{Symbol}()
    for stmt in body
        if stmt.kind == :for
            union!(vars, agen_collect_reassigned(stmt.body, true))
        elseif stmt.kind == :if
            # this function's own concern (a var written only inside
            # some sub-loop/sub-if of `body`) is unrelated to
            # agen_collect_reassigned's in_loop gating (which exists
            # only to keep tripcount-snapshot candidates restricted to
            # vars that can vary across a loop's own iterations) --
            # force in_loop=true here so this call keeps collecting
            # every assign in the subtree unconditionally, exactly as
            # before that gating was added.
            union!(vars, agen_collect_reassigned(stmt.then, true))
            union!(vars, agen_collect_reassigned(stmt.els, true))
        end
    end
    return Set(v for v in vars if kinds[v] == :scalar_float)
end

# True iff EVERY assignment to `var` within `body` (recursively) sits
# inside an ii_plan-covered (:independent or :reduction) loop. Once
# inside such a loop, everything nested further inside it counts as
# covered too, regardless of depth -- matches how ii_plan's own
# classification already covers a whole stmt.body recursively (e.g.
# cgen_locally_assigned_scalars/ii_escapes_nested), so a var written
# only within nested :for/:if structure inside an already-covered
# loop needs no separate re-proof here.
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

# The subset that actually needs an extra push (at the end of `body`,
# forward) and matching pop (at the start of `body`'s own backward
# processing): value-needed, not already exempt from snapshotting
# entirely, and with an allocated (:value, var) stack to push/pop on.
# Sorted for a deterministic push/pop order between the two sites.
#
# `ii_plan` (nothing by default -- every existing caller stays
# unaffected) excludes a var whenever agen_ii_covered_write_check
# proves EVERY write-site of it is inside an ii_plan-covered loop.
# This is required, not just an optimization: without it, a fully-
# contained fused var still gets an unconditional push/pop here,
# since this function's own criterion (written only inside some
# nested sub-structure of `body`, value-needed) has no knowledge of
# ii_plan's own, strictly more precise proof. A var with multiple
# write-sites where only SOME are ii_plan-covered (e.g. a fresh reset
# sitting outside the classified loop, as a sibling statement) still
# correctly stays a candidate -- agen_ii_covered_write_check requires
# ALL write-sites covered, not just one.
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

# a write to `var` needs a push (forward) / pop-restore (backward)
# whenever `var in value_needed` -- see agen_value_needed_vars. This
# already subsumes self-reference (a self-referencing statement's own
# rhs is itself one of the statements value_needed was built from) --
# no separate check is needed here.
# Polymorphic on `value_needed`'s type: a `Set{Symbol}` (default,
# whole-variable) behaves exactly as before, ignoring `key`. A
# `Dict{Any,Bool}` (site-level TBR, keyed by agen_site_key) looks up
# THIS statement's own decision instead -- see agen_value_needed_sites
# / snap_value_needed_sites and skill-stade's site-level TBR rollout.
function agen_needs_snapshot(lhs, rhs, var, value_needed, key = nothing)
    value_needed isa AbstractDict && return get(value_needed, key, false)
    return var in value_needed
end

# see snap_fwd_walk!'s comment -- identical logic, duplicated here for
# the same purity-rule reason as every other agen_-prefixed pair in
# this file. Shadow analysis only for now; must be checked against
# snap_value_needed_sites for exact agreement (Dict equality, not just
# subset) before either can be wired into real codegen -- forward push
# (driven by the snap_* side today, prospectively this side) and
# backward pop (driven by this agen_* side via agen_needs_snapshot)
# must decide identically at every site or push/pop counts desync.
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

# a :for statement's own lo/hi/step free variables -- factored out so
# agen_tier_b_walk's detection and agen_layout_walk!'s taint-marking
# (below) can never drift apart on what counts as "this loop's bound
# variables": the two MUST agree exactly, since taint-marking is what
# implements Tier B support for the exact loops agen_tier_b_offender
# would otherwise have refused on.
function agen_for_bound_vars(stmt)
    bound_vars = Set{Symbol}()
    agen_collect_expr_vars!(stmt.lo, bound_vars)
    agen_collect_expr_vars!(stmt.hi, bound_vars)
    agen_collect_expr_vars!(stmt.step, bound_vars)
    return bound_vars
end

# true if `expr` contains a ref (`arr[...]`) to any array-kinded var --
# used by the Tier B sizing pass (agen_tier_b_sizing_stmts) to decide
# whether a scalar assign is safe to replicate into a data-free
# skeleton: an array-free RHS is exactly the set of assigns that can
# possibly matter to a loop bound or branch condition downstream,
# since skill-jade's own house style never lets a bound/condition
# reference an array directly.
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

# vars whose sole write anywhere in the kernel qualifies for the
# elision (mirrors snap_check_assign!'s per-statement test exactly,
# just collected as a Set{Symbol} instead of gating site creation
# directly -- var alone is enough to key it, since "sole assign site"
# means there is only ever one statement this could refer to).
# `in_loop` must be true beneath ANY :for ancestor, not just a
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

# ---- keep_push_pop=false: Tier A/B sizing + :indexed emission ------
# See skill-stade.md's `keep_push_pop` entry for the full derivation
# (index formula, offset accounting, Tier A/B split). Summary:
#
# Every snapshot SITE (one push/pop occurrence -- a specific
# syntactic :assign/:for/:if location, not a variable) gets a
# compile-time-CONSTRUCTED, runtime-EVALUATED index into its
# (possibly shared) stack: `base_offset + local_position`.
# `base_offset` is the running sum of the "local multiplicities"
# (product of enclosing loops' trip counts) of every OTHER site
# mapped to the same stack, processed in a fixed forward-declaration
# order -- computed once by `agen_indexed_layout`, walking
# kernel.body with EXACTLY the traversal order/gating
# `agen_forward_body`'s own push-emission uses (mirrored here rather
# than shared, same purity-rule reason every other agen_-prefixed
# duplicate in this file mirrors another stage's logic). `local_position`
# is a 1-based row-major flattening of the CURRENT enclosing loop
# nest, recomputed fresh (never stored) at every push/pop call site
# from `ectx.loop_ctx` -- correct at both the forward and the
# backward call site for one syntactic location because a reversed
# backward loop still iterates the SAME symbolic (lo, hi, step)
# range, only its runtime direction differs (see
# agen_backward_body's `:for` case, which threads the SAME
# stmt.lo/hi/step onto loop_ctx regardless of reverse_it).
#
# Each occurrence is identified by a KEY -- (the kernel.body-side
# Vector containing its statement, that statement's position within
# it, and, for :tripcount only, which bound variable) -- rather than
# by counting occurrences in visitation order. A running-counter
# scheme was tried and rejected: `agen_backward_body`'s branch-scalar
# hoisting (see its own comment) deliberately emits some backward
# code OUT of naive-reversal order, which breaks any scheme that
# assumes backward visitation is a strict reverse enumeration of
# forward visitation. A key computed from the SAME underlying
# kernel.body object at both call sites (threaded through
# `agen_backward_body` as `primal_body`, structurally mirroring
# `plan`/`lin_plan` one-for-one) has no such assumption to break.
#
# The branch-stack's "discard pop" (see agen_backward_body's hoisting
# section) exists ONLY to keep push!/pop!'s single shared stack
# pointer in sync -- with no such pointer in :indexed mode, it is
# simply omitted there: the pushed slot goes unread, a harmless
# over-snapshot (same philosophy as snap_*'s own iteration-independent
# elision).
#
# Tier A (implemented): every enclosing loop's trip count is a
# closed-form expression of kernel arguments/constants -- offset and
# size are each a single formula, computed once by
# `agen_indexed_layout` and embedded directly in the generated code.
#
# Tier B (implemented, growable-buffer step -- see skill-stade.md):
# `agen_tier_b_offender`'s own detection (a loop whose bound var is
# ever reassigned inside an ANCESTOR sequential loop) is reused by
# `agen_layout_walk!` to mark every stack touched by an occurrence
# nested inside such a loop as TAINTED (`layout.tainted_stacks`).
# Tainted stacks fall all the way back to `:stack` (push!/pop!,
# growable `Vector`) semantics, unconditionally, regardless of
# `keep_push_pop` -- exactly the one already-correct mechanism that
# makes reversing a ragged loop's bound possible at all: the LIFO
# stack lets each occurrence's runtime trip count be recovered at its
# matching backward site (`n = pop!(tripcount_stack)`-style) without
# ever needing a closed-form size or offset formula for it. A stack
# with NO tainted occurrence keeps full Tier A `:indexed` treatment
# unchanged. See skill-stade.md's Tier B entry for why this is a
# per-STACK (not per-occurrence) decision, and for the follow-up that
# would replace the growable fallback with true ahead-of-time sizing.

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

# 1-based row-major flat position within one occurrence's own local
# block, from the CURRENT enclosing loop nest (outermost first) --
# degenerates to the literal 1 for a non-loop site. Verified against a
# hand-derived nested-loop offset formula like `(i_seq_ - 1) * n_inner
# + (i_x - 1)` (a single global +1, not one per level -- see
# skill-stade.md).
function agen_local_position(loop_ctx)
    isempty(loop_ctx) && return 1
    terms = Any[agen_mul_exprs(agen_pos0(loop_ctx[i]), agen_stride(loop_ctx, i)) for i in eachindex(loop_ctx)]
    return agen_add_exprs(agen_sum_exprs(terms), 1)
end

# ---- Tier B detection ------------------------------------------------
# a loop's bound-determining symbol is ever an assignment target
# inside an ANCESTOR sequential loop -- see skill-stade.md's Tier B
# section (a multigrid solver's ragged level-size halving sequence is
# the confirmed real instance). Returns the offending bound var, or
# `nothing` if the kernel is fully Tier A.
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

# ---- layout construction ---------------------------------------------
# walks kernel.body ONE time, in EXACTLY agen_forward_body's own
# push-gating order/conditions, recording each occurrence's stack,
# key, and local multiplicity; then folds those into per-stack
# running-sum base offsets and total sizes.
#
# `seq0`/`in_ragged0` seed the walk's `seq_reassigned`/`in_ragged`
# state instead of starting fresh -- defaults preserve every existing
# caller's exact behavior unchanged. Used by `agen_ragged_block` (see
# below) to reuse this ENTIRE function, verbatim, as the "AL-scoped,
# single-owner" sub-engine for one ragged block's own body: seeding
# `seq0` with AL's own newly-introduced reassignments makes any
# doubly-nested raggedness within AL surface in THIS call's own
# `tainted_stacks`, rather than needing a second table-building
# mechanism to handle AL-within-AL.
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
    return (offsets = offsets, sizes = sizes, free_vars = sort(collect(free); by = string), tainted_stacks = tainted_stacks)
end

# `seq_reassigned`/`in_ragged` mirror agen_tier_b_walk's own recursion
# exactly (same bound-var test via agen_for_bound_vars, same
# stmt.sequential-gated growth of seq_reassigned on descent into a
# :for) -- but instead of stopping at the first offender, every
# occurrence recorded while `in_ragged` is true gets its stack added
# to `tainted_stacks`. `in_ragged` starts false and is OR'd in (never
# cleared) on descent, so a loop nested inside a ragged ancestor stays
# tainted regardless of its own bound -- occurrence COUNT, not just
# occurrence content, is what a ragged ancestor puts in doubt.
function agen_layout_walk!(body, kinds, active_map, value_needed, reassigned, exempt, stacks, loop_ctx, occ_mult, key_order, tainted_stacks, seq_reassigned, in_ragged, push_pop = nothing)
    for (idx, stmt) in enumerate(body)
        if stmt.kind == :assign
            var = stmt.lhs isa Symbol ? stmt.lhs : stmt.lhs.args[1]
            # see agen_forward_body's matching comment: gate on the LHS
            # var's own activity (active_map[var]), not this write's
            # rhs activity, so a destructive inactive-rhs write (e.g. a
            # per-iteration array reset) still gets a slot sized here
            # exactly when snap_plan itself would create a site for it.
            # `push_pop` (site-level TBR), when given, takes over from
            # `value_needed` here exactly as it does in
            # agen_forward_body's own push gate -- see
            # agen_push_pop_source's comment.
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
    # block-boundary restoration -- see agen_block_boundary_vars.
    # Mirrors the extra push agen_forward_body emits at the end of
    # this same `body`, at this same point in the walk (after all of
    # body's own statements, using the CURRENT loop_ctx -- which
    # already includes this body's own enclosing loop frame, pushed
    # by the caller before recursing here -- giving exactly the "once
    # per enclosing-loop iteration" multiplicity this occurrence needs).
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

# ---- Tier B ragged-block layout (closed-form, GPU-eligible) ----------
# One "ragged block" = one ancestor sequential loop AL whose body
# contains a ragged descendant it ALONE governs -- see skill-stade.md's
# Tier B section. `stmt` must already be a `:for`; returns `nothing` if
# it isn't a genuine AL (no descendant it alone governs -- an ordinary
# sequential loop with no raggedness inside just gets walked normally
# by `agen_layout` below, no block needed).
#
# AL's OWN body is laid out via the EXISTING, unmodified
# agen_indexed_layout (steps 1/2's boolean in_ragged/tainted_stacks
# machinery) -- reused verbatim as the "AL-scoped, single-owner"
# sub-engine. Critically, the sub-call's `seq0` seed is the INCOMING
# `seq_reassigned` (from OUTSIDE AL) alone, NOT unioned with
# `own_reassigned`: a var reassigned as a TOP-LEVEL statement of
# `stmt.body` itself (e.g. a level-size update right in a sequential
# loop's own body) is fixed for the entire duration of ONE AL
# iteration -- from the sub-call's own perspective (which treats
# `stmt.body` as a fresh top-level body, loop_ctx starting at `[]`),
# that assignment sits OUTSIDE every loop `stmt.body` itself contains,
# so it must NOT count as "reassigned inside an ancestor sequential
# loop" for the sub-call's own detection. Only a reassignment that's
# itself nested inside a FURTHER `:for` within `stmt.body` (a genuine
# AL-within-AL) should register there -- and `own_reassigned` was
# deliberately computed with `in_loop=true` from the start specifically
# so the PRE-CHECK just below sees `stmt.body`'s top-level assignments
# as "ancestor-reassigned" (correct for "is this AL genuine" purposes,
# treating AL itself as the ancestor) -- reusing it as the sub-call's
# own seed would incorrectly make the sub-call see its OWN top-level
# assignments as already-ragged, wrongly tainting every stack touched
# anywhere inside AL rather than just the truly-nested-deeper ones.
#
# `reassigned` (for :tripcount_stack site detection, a different,
# whole-kernel-wide concept -- see agen_tripcount_bound_vars) is passed
# straight through unchanged: it's already computed once, globally, by
# the caller (agen_collect_reassigned(kernel.body)), and always already
# a superset of `own_reassigned` (agen_collect_reassigned's own
# recursion computes the exact same thing for this same `:for`
# subtree when walking from the top), so no extra union is needed.
#
# Returns `(header, local_offsets, local_sizes, ineligible_stacks)` --
# `header` is AL's own (necessarily closed-form, since AL itself isn't
# ragged) loop header; `local_offsets`/`local_sizes` are exactly
# agen_indexed_layout's own `offsets`/`sizes` fields, scoped to
# `stmt.body` alone (i.e. relative to AL's own frame, NOT including
# it); `ineligible_stacks` is that same call's own `tainted_stacks` --
# stacks genuinely governed by a DIFFERENT or DEEPER owner within AL
# (true AL-within-AL), which `agen_layout` routes to the old
# whole-kernel push!/pop! fallback instead of attempting a nested
# block for them (Phase A's single-level scope restriction).
function agen_ragged_block(stmt, kinds, active_map, value_needed, reassigned, exempt, stacks, seq_reassigned; push_pop = nothing)
    stmt.sequential || return nothing
    own_reassigned = agen_collect_reassigned(stmt.body, true)
    agen_tier_b_walk(stmt.body, own_reassigned) === nothing && return nothing
    sub = agen_indexed_layout((body = stmt.body,), kinds, active_map, value_needed, reassigned, exempt, stacks;
                               push_pop = push_pop, seq0 = seq_reassigned, in_ragged0 = false)
    return (header = (var = stmt.var, lo = stmt.lo, hi = stmt.hi, step = stmt.step), body = stmt.body,
            local_offsets = sub.offsets, local_sizes = sub.sizes, ineligible_stacks = sub.tainted_stacks)
end

# ---- Tier B: top-level layout with ragged blocks (Phase A) -----------
# NEW top-level counterpart to agen_indexed_layout -- not yet called by
# agen_emit/stade_hvp (see skill-stade.md's Tier B section for the
# phased rollout this belongs to). Walks kernel.body ONCE, in program
# order, maintaining a per-stack `current` running offset expression
# exactly like agen_indexed_layout's own `running` variable -- except
# `current[stack]` is allowed to become a runtime expression
# (referencing a ragged block's own computed total -- a `__tot_*`
# symbol Phase B's table-building code will define) rather than
# staying closed-form throughout. That's what lets a stack have
# MULTIPLE ragged blocks in sequence (e.g. a multigrid solver's own
# descend and ascend passes both writing to the same stack) chain
# correctly: block 2's
# own `base` is block 1's `base + total_sym`, read back out of
# `current` exactly the way a plain Tier A occurrence's base already
# works today.
#
# A `:for` with `stmt.sequential` is tested via agen_ragged_block: if
# it's a genuine ragged block, its body is NOT walked by this
# recursion at all -- entirely delegated to the sub-engine, and only
# the block's own result (per-stack local_offsets/local_sizes,
# ineligible_stacks) feeds back in. Otherwise it's walked exactly like
# any other loop -- unchanged from agen_layout_walk!'s own behavior
# for a non-ragged loop.
#
# Returns `(offsets, sizes, blocks, block_totals, tainted_stacks,
# free_vars)`:
# - `offsets[key]` is either `(stack, expr)` (static Tier A -- same
#   shape agen_indexed_layout already returns) or `(:ragged, stack,
#   block_id, local_offset_expr)` for an occurrence inside a block.
# - `sizes[stack]` only exists for a stack untouched by ANY block
#   (pure Tier A, same as today) -- a block-touched stack's total
#   needs its `__tot_*` symbols defined first (Phase B), so it can't
#   be a plain closed-form size yet; see `block_totals` instead.
# - `blocks[block_id] = (header, body, base::Dict{Symbol,Any},
#   local_sizes::Dict{Symbol,Any}, total_sym::Dict{Symbol,Symbol},
#   depth::Int)` -- `base[stack]` is the (possibly runtime) offset
#   expression in effect when this block begins; `total_sym[stack]` is
#   the symbol Phase B's codegen must define, holding this block's own
#   computed total contribution to that stack; `depth` is how many
#   loop_ctx frames were already open when AL's OWN :for was reached
#   (i.e. every outer, non-AL loop enclosing it, but not AL's own
#   frame) -- Phase C's agen_site_index uses it to know which prefix
#   of `ectx.loop_ctx` (built by the real forward/backward body walk,
#   which -- unlike this layout's own loop_ctx -- DOES push a frame
#   for AL itself) to skip: `ectx.loop_ctx[depth+2:end]` is exactly
#   the "inside AL, relative to AL's own frame" slice this block's
#   `local_offsets`/`local_sizes` were computed against.
# - `block_totals[stack]` is the FINAL running total expression for a
#   block-touched stack (its eventual allocation size, once Phase B
#   defines every `__tot_*` symbol it references) -- the block-touched
#   counterpart to `sizes`.
# - `tainted_stacks` is exactly the OLD (steps 1/2) fallback set --
#   stacks that must stay on push!/pop! entirely, because
#   agen_ragged_block found a deeper/different owner inside one of
#   their blocks (`ineligible_stacks`). Per-stack, not per-occurrence,
#   exactly like steps 1/2's own tainting: if ANY occurrence sharing a
#   stack is ineligible, the WHOLE stack falls back, even its other
#   occurrences that individually resolved fine.
function agen_layout(kernel, kinds, active_map, value_needed, reassigned, exempt, stacks; push_pop = nothing)
    offsets = Dict{Any,Any}()
    current = Dict{Symbol,Any}()
    blocks = Dict{Any,Any}()
    block_of = Dict{Any,Int}()
    block_touched = Set{Symbol}()
    ineligible = Set{Symbol}()
    block_counter = Ref(0)
    agen_layout_walk_top!(kernel.body, kinds, active_map, value_needed, reassigned, exempt, stacks, Any[], offsets, current, blocks, block_of, block_touched, ineligible, Set{Symbol}(), block_counter, push_pop)
    # a block-local scalar (n, nc, ...) referenced by a local_offset/
    # local_size formula is NOT safe to read as a bare in-scope
    # variable at an arbitrary push/pop site: unlike agen_forward_body
    # (which always preserves original program order, so a scalar
    # like `nc` is guaranteed already (re)computed by the time
    # anything downstream reads it), agen_backward_body reverses
    # PER-STATEMENT order too -- a block-boundary occurrence whose
    # forward statement came LAST in the block's body becomes the
    # FIRST thing the reversed loop executes, potentially before any
    # of the scalar recompute/tripcount-pop machinery that (in
    # forward order) preceded it. See agen_tier_b_value_tables_stmts
    # below for the fix: every block-local scalar a formula depends on
    # gets its OWN per-iteration value table too (built alongside the
    # prefix/total tables, Phase B), and agen_site_index (Phase C)
    # substitutes a bare reference to it with a lookup into that
    # table -- removing the dependency on program order entirely, the
    # same way the prefix/total tables already removed it for offsets.
    #
    # A variable that's a kernel ARGUMENT is normally safe to exclude
    # (always available, unchanged, everywhere) -- EXCEPT when it's
    # also in `reassigned` (a ragged-block level-size argument is
    # exactly this: an argument whose role is just an initial
    # placeholder, immediately overwritten -- e.g. `n = n * 2` as its
    # own kernel body's first
    # statement). Such an argument is really just a pre-declared LOCAL
    # by the time any ragged block reads it, with the identical
    # program-order hazard as any other block-local scalar -- so it
    # must stay a value_var too, not be filtered out as "always safe."
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
        # a block's own header (lo/hi/step) is what Phase B's table
        # allocation (`Vector{Int}(undef, <AL's own trip count>)`) and
        # its own `for <header>` loop need -- e.g. a multigrid solver's
        # own level count never appears in any local_size formula (only
        # in the loop bound itself), so it would otherwise be silently
        # missing from initstacks_*'s eventual signature (Phase D).
        agen_collect_expr_vars!(blk.header.lo, free)
        agen_collect_expr_vars!(blk.header.hi, free)
        agen_collect_expr_vars!(blk.header.step, free)
    end
    # a block's own local_size formula legitimately references
    # internal, ragged-controlling locals (n, nc, ...) that are only
    # ever valid INSIDE code that has already run the sizing/table
    # pass defining them (Phase B) -- never real kernel arguments, so
    # never valid as a signature parameter (unlike a pure Tier A
    # size, which by construction only ever references true kernel
    # arguments already). Filtered here, once, rather than trusting
    # every future caller to filter it themselves.
    free = intersect(free, Set(kernel.sig.args))
    return (offsets = offsets, sizes = sizes, blocks = blocks, block_of = block_of, block_totals = block_totals,
            tainted_stacks = ineligible, free_vars = sort(collect(free); by = string))
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

# Tier B sizing pass -- see the "Tier B (implemented..." comment above
# agen_local_position. A tainted stack's buffer stays push!/pop!-based
# (agen_use_stack_push is unchanged by this), so this does NOT presize
# a true :indexed buffer or unlock GPU-splitting -- it only lets
# `initstacks_*`/hvp's shadow-stack init `sizehint!` the buffer ahead
# of time, avoiding push!'s own repeated reallocation as a growable
# Vector grows. Because sizehint! is a pure performance hint (Vector's
# push!/pop! semantics are correct for ANY hinted size, including 0 or
# an overestimate), this pass has no correctness bar to clear -- it
# only needs to be a REASONABLE estimate, which is why it's safe to
# build via a fresh, independently-scoped walk (agen_tier_b_sizing_walk)
# rather than reusing agen_layout_walk!'s own bookkeeping.
#
# Returns (stmts, total_of) where `stmts` is a self-contained,
# data-free replica of kernel.body: every :for/:if kept verbatim
# (their headers are themselves scalar expressions that must actually
# execute to get real trip counts/branch outcomes), every array
# :assign dropped, every scalar :assign kept unless its RHS reads an
# array (see agen_expr_reads_array) -- and at each occurrence site
# landing on a tainted stack, `__sz_<stack> += 1`. `total_of` maps
# stack name -> the local variable holding its final count.
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

# ---- Tier B ragged-block table construction (Phase B) ----------------
# NEW, not yet wired into initstacks_*/hvp_shadow_stack_inits (see
# skill-stade.md's Tier B section for the phased rollout). Builds the
# actual code that computes, for one ragged block (Phase A's
# `agen_layout`/`agen_ragged_block` output), a per-stack prefix TABLE
# -- one entry per AL iteration, holding the cumulative offset BEFORE
# that iteration -- plus the block's own final total contribution.
#
# Replicates AL's own SCALAR control state (the reassignment
# recurrence that varies per-AL-iteration -- e.g. a multigrid level's
# own size-halving chain) via
# `agen_tier_b_block_skeleton`, below -- but UNLIKE step 2's
# agen_tier_b_sizing_walk (which replicates a WHOLE kernel body,
# occurrence-counting one increment at a time by actually iterating
# every ragged loop for real), this stops at any nested `:for`
# entirely: everything inside a ragged loop's own body is already
# captured by Phase A's closed-form `local_size` formula -- a function
# of the CURRENT scalar state this skeleton produces -- so
# re-iterating it here would be redundant, genuinely-O(n)-per-block-
# iteration work for no additional information. That's the entire
# payoff of Phase A's table design: sizing no longer needs to touch
# the ragged loops themselves at all, only replicate the handful of
# scalar reassignment statements that govern their bounds.
#
# Recurses through :if (a reassignment can be gated -- e.g. a
# window-count decrement inside a conditional retire step), keeps a
# scalar :assign unless its RHS reads an array (agen_expr_reads_array
# -- same rationale as step 2: an array's value never affects a loop
# bound or a local_size
# formula, by skill-jade's own house style), drops every array
# :assign, and drops a nested :for's STRUCTURE entirely (not even an
# empty loop -- there is deliberately no recursive case for :for here).
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

# One block's own table-construction statements -- `blk` is one entry
# of `agen_layout(...).blocks` (carries `header`, `body`, `local_sizes`,
# `total_sym`; `base` is used elsewhere, by whatever eventually reads
# `agen_layout`'s `offsets`/`block_totals`, not needed here). Emits,
# for every stack this block touches, in one shared loop over AL's own
# header (so the skeleton replica -- and any branch it takes -- is
# built exactly once per iteration, not once per stack):
#
#   prefix_<stack>_<block_id> = Vector{Int}(undef, <AL's own trip count>)
#   __tot_<stack>_<block_id> = 0
#   for <AL's own header>
#       prefix_<stack>_<block_id>[pos0+1] = __tot_<stack>_<block_id>   # BEFORE this iteration's own contribution
#       ... <AL's own scalar skeleton, shared across every stack> ...
#       __tot_<stack>_<block_id> = __tot_<stack>_<block_id> + <local_size[stack]>
#   end
#
# `agen_pos0(header)+1` as the table WRITE index is deliberately the
# exact same formula Phase C's table READ (at a push/pop site inside
# this block) will use -- both come from agen_local_position's own
# 0-based agen_pos0, so a write at real iteration k and a later read
# at that same iteration k are guaranteed to agree, the same
# guarantee Tier A's own pos0/stride formulas already rely on (see the
# "Tier A (implemented)" note: recomputed fresh from loop structure at
# each site, never cached, so both directions can't independently
# drift).
# Builds one block's table-construction statements: the offset/total
# tables described above (unchanged), PLUS -- for every block-local
# scalar a local_offset/local_size formula depends on
# (`blk.value_vars`, computed in agen_layout) -- a per-iteration VALUE
# table (`val_<var>_<block_id>`), written every iteration right after
# the scalar skeleton updates it. This is what lets agen_site_index
# (Phase C) resolve such a variable via a table lookup instead of a
# bare in-scope reference, which is NOT safe at an arbitrary push/pop
# site under agen_backward_body's per-statement reversal -- see the
# comment in agen_layout above where `value_vars` is computed for why.
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
    loop_body = Any[Expr(:(=), Expr(:ref, table_name(s), idx_expr), blk.total_sym[s]) for s in stack_names]
    append!(loop_body, agen_tier_b_block_skeleton(blk.body, kinds))
    for v in value_vars
        push!(loop_body, Expr(:(=), Expr(:ref, value_table_name(v), idx_expr), v))
    end
    for s in stack_names
        push!(loop_body, Expr(:(=), blk.total_sym[s], agen_add_exprs(blk.total_sym[s], blk.local_sizes[s])))
    end
    push!(decls, emit_forloop(header.var, header.lo, header.hi, header.step, loop_body))
    return decls
end

# Full-kernel scalar skeleton with ragged blocks spliced in. This is
# THE actual Phase B deliverable (not agen_tier_b_block_stmts alone --
# that only covers one block's own loop; a kernel's static prelude
# before/between/after blocks -- e.g. a multigrid solver's own level-
# size initialization before its first block -- still has to run to
# get those locals initialized correctly before any block's own
# skeleton first reads them). Mirrors agen_layout_walk_top!'s OWN
# top-level traversal decisions EXACTLY (same agen_ragged_block-
# classified `:for`
# statements, at the same points, via `layout.block_of`) so this
# skeleton's structure can never drift from what agen_layout used to
# build `blocks`/`offsets` in the first place -- a mismatch here would
# silently corrupt which block's table-construction code runs where.
#
# Where agen_layout_walk_top! delegated an entire `:for` to
# agen_ragged_block and stopped recursing into it, this emits that
# block's own table-construction statements (agen_tier_b_block_stmts)
# in its place -- looked up via `layout.block_of[key]`, so this walker
# never needs to re-run agen_ragged_block's (non-trivial) detection
# itself. Everywhere else -- ordinary scalar prelude between/around
# blocks, any non-AL loop's own structure -- follows the same
# keep-scalar/drop-array rule as agen_tier_b_block_skeleton, except a
# non-AL `:for`'s own structure is KEPT (recursed into, since scalar
# state can still evolve inside it) -- unlike a `:for` nested INSIDE a
# block, which agen_tier_b_block_skeleton already drops outright (that
# case never reaches this function at all: it's inside `blk.body`,
# walked separately, once, by agen_tier_b_block_stmts).
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

# the correct, complete way to get initstacks_*'s eventual Tier B
# argument list (Phase D): every kernel argument referenced ANYWHERE
# in the actual skeleton code that will run, INCLUDING prelude
# statements outside any block (e.g. a level-size initialization) that
# `agen_layout`'s own `free_vars` field can't see, since it's built
# before the skeleton exists. Computed from the skeleton itself rather
# than enumerated case-by-case, so it can't silently miss a kernel arg
# that only ever appears in an unblocked prelude statement.
function agen_tier_b_skeleton_free_vars(skeleton_stmts, kernel_args)
    free = Set{Symbol}()
    for s in skeleton_stmts
        agen_collect_expr_vars!(s, free)
    end
    return sort(collect(intersect(free, Set(kernel_args))); by = string)
end

# ---- push/pop emission strategy --------------------------------------
# `ectx` is the small thread-through-the-recursion value described
# above -- exactly analogous to cgen_body's own owner/kernels
# threading. `loop_ctx` is temporarily extended (push!/pop!) around a
# :for's own recursion, in both agen_forward_body and
# agen_backward_body. Under keep_push_pop=true, `layout` is never
# consulted (ectx.keep_push_pop short-circuits first).
agen_ectx_stack() = (keep_push_pop = true, loop_ctx = Any[], layout = nothing, push_pop = nothing, ii_plan = nothing)

# resolves which value_needed-like info the per-statement push/pop
# gate should consult: the site-level Dict (ectx.push_pop) when one
# was computed, else falls back to the ordinary whole-variable Set
# (`value_needed`) -- exactly today's behavior when the flag is off.
# Never affects agen_block_boundary_vars/agen_exempt_vars/
# agen_if_branch_scalar_vars, which stay on the plain `value_needed`
# Set always -- those model a different (block-scope / branch-hoist)
# question site-level TBR doesn't cover yet.
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

function agen_site_index(ectx, key)
    entry = ectx.layout.offsets[key]
    if entry[1] === :ragged
        # Tier B (Phase C): a ragged-block occurrence -- see the
        # "Tier B ragged-block layout" note above agen_ragged_block,
        # and blocks[block_id]'s own depth field for why the
        # loop_ctx slice below is `depth+2:end`, not `depth+1:end`.
        # All three terms are PURE expressions (table lookups keyed
        # by the current loop variable, plus an ordinary position
        # formula) -- deliberately never a mutating index (see
        # skill-stade.md's Tier B "Known blocking issue" note: an
        # index with an embedded mutation gets evaluated twice by
        # hvp_double_stmt, since it reuses the same index sub-Expr for
        # both the shadow and primal writes as two separate
        # statements).
        (_, stack, block_id, local_offset) = entry
        blk = ectx.layout.blocks[block_id]
        table_name = Symbol("prefix_" * string(stack) * "_" * string(block_id))
        table_idx = agen_add_exprs(agen_pos0(blk.header), 1)
        # `local_offset` may reference a block-local scalar (n, nc,
        # ...) -- unsafe to read as a bare in-scope variable here: see
        # the value_vars comment in agen_layout for why
        # agen_backward_body's per-statement reversal can reach a
        # push/pop before that scalar's own recompute/tripcount-pop,
        # for a ragged occurrence specifically (this dependency never
        # existed under plain push!/pop!). Substituting a table lookup
        # for each one removes the dependency on program order
        # entirely -- the same `table_idx` as the offset table itself,
        # since agen_tier_b_block_stmts writes both at the identical
        # point in the same per-iteration loop (Phase B).
        subst = Dict{Symbol,Any}(v => Expr(:ref, Symbol("val_" * string(v) * "_" * string(block_id)), table_idx) for v in blk.value_vars)
        local_offset = agen_substitute_vars(local_offset, subst)
        inner_ctx = ectx.loop_ctx[(blk.depth + 2):end]
        # agen_local_position's own formula can ALSO reference a
        # block-local scalar directly -- an inner frame's `hi` is
        # literally the loop bound symbol (e.g. n, for `for
        # i_seq_j = 1:n`) -- so it needs the identical substitution,
        # not just `local_offset` above.
        local_position = agen_substitute_vars(agen_local_position(inner_ctx), subst)
        # `blk.base[stack]` is the offset this whole block starts at
        # (0, or a prior block's/static prefix's own total, for a
        # stack touched by more than one block -- e.g. a stack touched
        # by both a descend and an ascend pass in the same solver).
        # The prefix table itself is only ever relative to "within this
        # block alone" (Phase B always starts each block's own
        # __tot_* at 0) -- omitting `base` here would make every
        # multi-block stack's later blocks silently overlap its
        # earlier ones' index range instead of continuing after them.
        # `base` is always a plain symbol/kernel-arg expression (never
        # a block-local scalar), since it's built from `current`,
        # which only ever holds closed-form/kernel-arg/`__tot_*`
        # terms -- see agen_layout_static_record!/agen_layout_walk_top!
        # -- so it needs no substitution of its own.
        return agen_add_exprs(agen_add_exprs(agen_add_exprs(get(blk.base, stack, 0), Expr(:ref, table_name, table_idx)), local_offset), local_position)
    end
    (_, offset) = entry
    return agen_add_exprs(offset, agen_local_position(ectx.loop_ctx))
end

# true whenever `stack_name` should use plain push!/pop! rather than
# an :indexed direct write/read (formula-based Tier A, or Phase C's
# table-based Tier B ragged block) -- either the whole kernel is in
# :stack mode, or this specific stack is in `ectx.layout.tainted_stacks`
# because some occurrence on it couldn't be resolved into either
# (agen_indexed_layout's plain Tier A math, or agen_layout's Tier B
# ragged-block tables -- see the "Tier B ragged-block layout" note
# above agen_ragged_block for what lands here: genuine AL-within-AL,
# Phase A's single-level scope restriction). `ectx.layout` is only
# ever `nothing` when `ectx.keep_push_pop` is already true (see
# agen_ectx_stack/agen_emit/stade_hvp), so the `!== nothing` guard is
# just defensive. Works unchanged whichever layout function built
# `ectx.layout` -- agen_indexed_layout (Tier A only) or agen_layout
# (Tier A + Tier B ragged blocks) both expose `tainted_stacks` in the
# same shape.
agen_use_stack_push(ectx, stack_name) = ectx.keep_push_pop ||
    (ectx.layout !== nothing && stack_name in ectx.layout.tainted_stacks)

# `key` is unused (and may be `nothing`) whenever
# agen_use_stack_push(ectx, stack_name) is true, matching every call
# site below that only ever computes a real key inside the :indexed
# branch's own guard
function agen_emit_push(stack_name::Symbol, value, ectx, key)
    agen_use_stack_push(ectx, stack_name) && return Expr(:call, :push!, stack_name, value)
    return Expr(:(=), Expr(:ref, stack_name, agen_site_index(ectx, key)), value)
end

# returns the RHS expr only -- caller wraps `lhs = <this>`, matching
# how a plain `pop!(stack)` was always just an rhs expr too
function agen_emit_pop(stack_name::Symbol, ectx, key)
    agen_use_stack_push(ectx, stack_name) && return Expr(:call, :pop!, stack_name)
    return Expr(:ref, stack_name, agen_site_index(ectx, key))
end

# ---- Phase 3 cleanup: drop stack args left unused by fusion --------
# A var covered by an ii_plan site does NOT always mean its stack
# becomes fully unused -- a separate mechanism (agen_block_boundary_
# vars, threading a var's value correctly across a repeating
# ancestor's own iterations) can still need it, unrelated to whether
# the specific accumulation write-site inside the classified loop
# still pushes. A blanket "drop every ii_plan-covered var's stack"
# would have been a real bug (an undefined-variable error at best).
# The safe approach: generate the adjoint body FIRST (unchanged), then
# check what it ACTUALLY still references, and only drop what
# provably has zero remaining push!/pop! calls anywhere in the
# output -- correct by construction, since it's reading the
# already-correct generated code rather than trying to predict it.
#
# Scoped to keep_push_pop=true only (matching fuse_ii_loops's own
# existing scope limitation) -- under keep_push_pop=false, a stack
# push/pop is `Expr(:ref, stack_name, idx)`, not `push!`/`pop!` calls,
# and interacts with Tier A/B layout sizing this doesn't attempt to
# handle.
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

# ---- ii_* Phase 3 codegen: :independent fusion only ----------------
# Builds ONE un-reversed loop (same header as the primal, never
# reversed -- a fused loop only ever traverses its own range once, at
# the same point in program order the primal always did) whose body
# is: the primal recompute of stmt.body, immediately followed by that
# same body's own backward differentiation in reverse statement order,
# with the loop itself never reversed.
#
# The vars this fuses (vn_local) are removed from value_needed for
# both nested calls, which suppresses their push/pop. This is done via
# a LOCALLY-SCOPED value_needed Set, not a kernel-wide exempt Set,
# because the same variable name can be reused, unrelated, by a
# different loop elsewhere in the kernel -- a kernel-wide, name-based
# exemption would incorrectly suppress protection for that other
# usage. Scoping the modification to this call, which only ever visits
# statements inside stmt.body, avoids that.
#
# `:reduction` and `:mixed` sites are deliberately not fused here. A
# standalone reduction's adjoint total isn't fully accumulated until
# after its downstream consumer's own (logically later, hence
# backward-sweep-earlier) code has run; fusing the reduction's own
# forward+backward per-iteration would read that total too early,
# silently producing an understated result rather than erroring.
# `agen_forward_body`/`agen_backward_body` only dispatch here for
# `:independent`; `:reduction`/`:mixed` use their own, separate
# handling (see the comments at their own call sites).
#
# Known, deliberate scope limitation: `stacks`/`agen_stack_map`/
# `agen_init_emit` don't know which stacks fusion makes unused on
# their own -- that's handled separately, via a post-hoc scan of the
# generated code (see the "Phase 3 cleanup" section below) plus
# `agen_block_boundary_vars`'s own ii_plan-awareness.
#
# `agen_ii_override_ectx` is REQUIRED for correctness, not just a
# nicety, because site-level TBR is always active: `agen_push_pop_
# source` consults `ectx.push_pop` (a per-site Dict) instead of
# `value_needed` whenever it's set, so excluding a var from
# `value_needed` alone has NO effect on the actual push/pop decision.
# Without this override, a fusion-covered write can still get pushed
# (because the TBR analysis, unaware of fusion, decided it needs
# protecting) with nothing to pop it -- for `:reduction` specifically,
# that leaves a genuinely unmatched push every iteration, permanently
# growing the stack and corrupting later, unrelated pops. This builds
# a new ectx whose push_pop Dict (a copy of whatever TBR already
# decided) additionally forces `false` for every site writing a
# vn_local var -- applied everywhere value_needed is locally modified
# for ii_plan purposes, not just here.
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
#
# agen_emit_ii_loop plays two different roles. At the FORWARD position
# (`:independent`) its `fwd` half is the kernel's actual primal
# execution and must be emitted whole. At the BACKWARD position
# (`:reduction`/`:mixed`) the primal already ran in the ordinary
# forward sweep, so `fwd` there is purely a RECOMPUTE: its only job is
# to re-establish the fused scalars that were never snapshotted.
#
# Re-executing the whole body in that role is wrong, not merely
# wasteful: an accumulating array write (`y[i] = y[i] + t * t`) applies
# a second time and leaves the array holding twice its forward value.
# Verified directly -- gradients stay bit-identical while the array
# diverges, which is why no finite-difference test ever caught it. It
# is harmless today only because the escaping-array-write gate keeps
# anything outside the loop from observing the corruption; it is not
# harmless in general, and it is what would block any future extension
# past that gate.
#
# The filter keeps every SCALAR assignment (so index scalars like
# `i_node = i_cell_to_node[...]` come along, and the recompute matches
# the previous whole-body behaviour exactly for scalars) and drops
# every array write. Dropping array writes is only sound when no
# recomputed scalar reads an array this body itself writes -- see
# ii_body_scalar_reads_own_array_write, which refuses classification
# in that case rather than silently recomputing from a post-forward
# array value.
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
            # gate on the LHS var's own activity (active_map[var]), not
            # this statement's rhs activity -- a write whose own rhs is
            # a plain inactive literal (e.g. `mup[i] = 0.0`) can still
            # DESTROY a value that some other, earlier-in-forward-order
            # statement needs for its own nonlinear derivative (e.g. a
            # divisor read later re-differentiated wrt a different
            # var). Gating on this statement's own rhs activity misses
            # exactly that case; matching snap_check_assign!'s own gate
            # (active_map[var], not this write's rhs) is what keeps
            # this in sync with snap_plan's site list. An int-kinded
            # lhs never owns a :value/:array site regardless of
            # activity -- only :tripcount covers int reassignment.
            # `exempt` skips the push entirely for a write snap_plan
            # itself would also elide -- see agen_exempt_vars.
            if kinds[var] in (:scalar_float, :array_float) && get(active_map, var, false) && agen_needs_snapshot(stmt.lhs, stmt.rhs, var, agen_push_pop_source(value_needed, ectx), agen_site_key(body, idx)) && !(var in exempt)
                push!(exprs, agen_emit_push(stacks[(agen_snapshot_kind(stmt.lhs), var)], stmt.lhs, ectx, agen_site_key(body, idx)))
            end
            push!(exprs, Expr(:(=), stmt.lhs, stmt.rhs))
        elseif stmt.kind == :for
            # ii_plan-covered site (only when fuse_ii_loops=true; ectx.
            # ii_plan is nothing otherwise, making this whole branch a
            # no-op): `:independent` is built as one fused, un-reversed
            # loop (agen_emit_ii_loop) instead of the ordinary push-
            # then-later-reverse treatment. `:reduction` keeps its
            # ordinary forward-sweep position and structure (the primal
            # value is often still needed elsewhere), just with its
            # reduction var(s) excluded from `value_needed` so nothing
            # pushes their old value -- see agen_emit_ii_loop's own
            # comment for why. `:mixed` gets the same treatment as
            # `:reduction` -- both halves excluded from push, nothing
            # fused at this position -- see the :mixed branch below for
            # why splitting them across positions is unsafe.
            key = agen_site_key(body, idx)
            ii_kind = ectx.ii_plan === nothing ? nothing : get(ectx.ii_plan, key, nothing)
            if ii_kind === :independent && lin_body !== nothing && unsafe !== nothing
                push!(exprs, agen_emit_ii_loop(stmt, lin_body[idx], kinds, active_map, value_needed, reassigned, stacks, exempt, unsafe; ectx = ectx))
            elseif ii_kind === :mixed && lin_body !== nothing && unsafe !== nothing
                # NOT split across positions: a var safe to fuse here
                # (vn_ind) can be read by a var deferred to the
                # backward position (vn_red)'s own accumulation term,
                # and vn_ind's own "collect every contribution, then
                # distribute" step needs both contributions available
                # before it runs -- but the vn_red-side one isn't
                # available yet. So this treats `:mixed` the same way
                # `:reduction` treats vn_red: exclude the whole
                # vn_local set from push here (still correct, still
                # the real benefit), but defer ALL differentiation to
                # the backward position instead of fusing any of it.
                local_names = cgen_locally_assigned_scalars(stmt.body)
                redvars = cgen_scalar_reduction_vars(stmt.body)
                vn_local = Set(v for v in intersect(value_needed, union(local_names, redvars)) if get(active_map, v, false))
                loop_value_needed = setdiff(value_needed, vn_local)
                loop_ectx = agen_ii_override_ectx(ectx, stmt.body, vn_local)
                for bv in agen_tripcount_bound_vars(stmt, reassigned)
                    push!(exprs, agen_emit_push(stacks[(:tripcount, bv)], bv, ectx, agen_site_key(body, idx, bv)))
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
                    push!(exprs, agen_emit_push(stacks[(:tripcount, bv)], bv, ectx, agen_site_key(body, idx, bv)))
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
            then_exprs = vcat(Any[agen_emit_push(nm, 1, ectx, key)],
                               agen_forward_body(stmt.then, kinds, active_map, value_needed, reassigned, stacks, exempt;
                                                  ectx = ectx,
                                                  lin_body = lin_body === nothing ? nothing : lin_body[idx].then,
                                                  unsafe = unsafe))
            els_exprs = vcat(Any[agen_emit_push(nm, 0, ectx, key)],
                              agen_forward_body(stmt.els, kinds, active_map, value_needed, reassigned, stacks, exempt;
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
        push!(exprs, agen_emit_push(stacks[(:value, var)], var, ectx, agen_site_key(body, 0, var)))
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

# the constant a branch-snapshotted scalar falls back to on whichever
# side of the :if doesn't assign it -- the nearest preceding sibling
# assign to the same var in this exact block. Only a literal-Number
# rhs is trusted here (not merely "inactive"): recompute must not
# depend on any other variable's current value, which the reverse
# sweep could already have disturbed by this point.
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
    # block-boundary restoration (see agen_block_boundary_vars above):
    # restore each such var here, at the very start of this body's own
    # backward processing, from the matching push agen_forward_body
    # emitted at the end of this same body -- must run before anything
    # else below, since those pushes/pops are precisely what makes
    # this body's OWN nested-loop-written value visible again instead
    # of whatever a later (i.e. previously-processed, in reverse
    # order) sibling iteration of the ENCLOSING loop left behind.
    for var in agen_block_boundary_vars(primal_body, kinds, value_needed, exempt, stacks; ii_plan = ectx.ii_plan)
        push!(exprs, Expr(:(=), var, agen_emit_pop(stacks[(:value, var)], ectx, agen_site_key(primal_body, 0, var))))
    end
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
    # Branch-snapshotted scalars (x = 0.0; if cond: x = expr end)
    # whose own primal VALUE -- not just their shadow -- is read by a
    # DIFFERENT statement later in this same block, for a nonlinear
    # term's own partial (e.g. a divisor whose adjoint needs the
    # literal numerator it divides). A plain reverse walk restores
    # such a scalar only once it reaches the :if itself, which comes
    # AFTER that read in the reverse walk (the :if precedes the read
    # in forward order) -- one iteration too late, so the read picks
    # up whatever the deeper, already-processed iteration left
    # behind. Recomputed here instead, forward-style, from the
    # freshly popped branch flag: safe because a snapshot site's rhs
    # (an array read, or the literal-constant sibling it falls back
    # to) never itself depends on anything the reverse sweep has
    # touched yet at this point. Walked in reverse-plan order so
    # multiple qualifying :if's in this block still pop branch_stack
    # in the same relative LIFO order a plain reversal would.
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
        push!(exprs, Expr(:(=), flag, agen_emit_pop(stacks[(:branch, :cond)], ectx, agen_site_key(primal_body, idx))))
        branch_flags[idx] = flag
        hoisted_vars[idx] = Set(v for (v, _, _, _, _) in resolved)
        for (var, then_expr, els_expr, then_pushed, els_pushed) in resolved
            snm = stacks[(:value, var)]
            # the discard-pop below exists ONLY to keep push!/pop!'s
            # single shared stack pointer in sync -- :indexed mode has
            # no such pointer, so it's simply omitted there; the
            # pushed slot goes unread, a harmless over-snapshot (see
            # skill-stade.md's keep_push_pop entry)
            if ectx.keep_push_pop
                if then_pushed && els_pushed
                    push!(exprs, Expr(:(=), :__snap_discard, agen_emit_pop(snm, ectx, nothing)))
                elseif then_pushed
                    push!(exprs, emit_if(Expr(:call, :(==), flag, 1), Any[Expr(:(=), :__snap_discard, agen_emit_pop(snm, ectx, nothing))], Any[]))
                elseif els_pushed
                    push!(exprs, emit_if(Expr(:call, :(==), flag, 0), Any[Expr(:(=), :__snap_discard, agen_emit_pop(snm, ectx, nothing))], Any[]))
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
                # this loop's own reduction var(s) never needed a push
                # (agen_forward_body's :for dispatch excluded them from
                # value_needed) -- their actual adjoint is emitted HERE
                # instead, reusing agen_emit_ii_loop exactly as
                # :independent does, but appended at this loop's
                # ordinary, unfused backward-sweep position rather than
                # replacing the forward sweep. That's what makes it
                # safe unlike fusing at the forward position: by the
                # time the reverse sweep reaches this loop's own
                # position, everything logically after it in the
                # primal has already run its own backward code, so the
                # accumulated shadow this loop distributes is already
                # complete. No tripcount pop either, same reasoning as
                # :independent.
                push!(exprs, agen_emit_ii_loop(primal_body[idx], stmt, kinds, active_map, value_needed, reassigned, stacks, exempt, unsafe; ectx = ectx, recompute = true))
            elseif ii_kind === :recompute
                # Same dispatch as :reduction -- the loop keeps its
                # ordinary forward position and its ordinary backward
                # position; only its snapshots are replaced by the
                # filtered recompute. Nothing moves, which is what lets
                # this kind exist for a loop with an escaping array
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
                    push!(exprs, Expr(:(=), bv, agen_emit_pop(stacks[(:tripcount, bv)], ectx, agen_site_key(primal_body, idx, bv))))
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
                push!(exprs, Expr(:(=), :__branch, agen_emit_pop(nm, ectx, agen_site_key(primal_body, idx))))
                push!(exprs, emit_if(Expr(:call, :(==), :__branch, 1), then_exprs, els_exprs))
            end
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
# Also true whenever `body` itself has a block-boundary var (see
# agen_block_boundary_vars): that push happens once per iteration of
# whatever loop `body` is the direct body of, same as any other.
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
        # restore this write's overwritten old value whenever
        # active_map[var] does -- matching agen_forward_body's push
        # gate and snap_check_assign!'s own gate -- NOT stmt.active
        # (this write's own rhs activity): a write can destroy a value
        # some other, earlier-in-forward-order statement's nonlinear
        # derivative still needs even when this particular write's own
        # rhs is a plain inactive literal (e.g. a per-iteration array
        # reset). `exempt`/`skip_restore` mirror the forward sweep's
        # own skips -- no push ever happened for those, so there is
        # nothing to pop.
        if get(active_map, var, false) && agen_needs_snapshot(stmt.lhs, stmt.tree.expr, var, agen_push_pop_source(value_needed, ectx), key) && !(var in exempt) && !(var in skip_restore)
            nm = stacks[(agen_snapshot_kind(stmt.lhs), var)]
            push!(exprs, Expr(:(=), stmt.lhs, agen_emit_pop(nm, ectx, key)))
        end
        if stmt.active
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
        elseif !is_accum
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

function hvp_shadow_stack_inits(sites, shadow_of, keep_push_pop::Bool = true, layout = nothing, total_of = Dict{Symbol,Symbol}())
    exprs = Any[]
    seen = Set{Symbol}()
    for s in sites
        s.kind in (:array, :value) || continue
        nm = agen_site_stack_name(s)
        nm in seen && continue
        push!(seen, nm)
        # a shadow stack is exactly as large as its primal counterpart.
        # Lands folded directly into this `_hv` function's own body
        # (never a separate initstacks_*_hv). Uses `length(nm)` on the
        # PRIMAL stack -- which `initstacks_*` already allocated with
        # the correct size and passed in as `_hv`'s own argument --
        # rather than re-evaluating `layout.sizes`/`layout.block_totals`
        # itself: a Tier B size formula can legitimately reference a
        # block-local scalar (e.g. a coarse-grid count reused both
        # before its ancestor loop, where it's a plain Tier A
        # constant, and inside it, where it's ragged) that's only ever
        # safe to read AT THE POINT the real forward sweep or Phase B's
        # own skeleton would compute it -- never at this function's
        # very top, before anything has run. `length(nm)` sidesteps
        # that entirely: it needs nothing but the already-built primal
        # stack, so it's correct regardless of which kind of size
        # formula (or none, for a tainted/growable stack) produced it.
        grow = keep_push_pop || (layout !== nothing && nm in layout.tainted_stacks)
        alloc = agen_stack_alloc_expr(:value, grow, grow ? nothing : Expr(:call, :length, nm))
        push!(exprs, Expr(:(=), shadow_of[nm], alloc))
        haskey(total_of, nm) && push!(exprs, Expr(:call, :sizehint!, shadow_of[nm], total_of[nm]))
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

# matches agen_stack_alloc_expr's own output: the keep_push_pop=true
# empty form Vector{Float64}()/Vector{Int64}() (1 arg: just the curly),
# and the keep_push_pop=false pre-sized form Vector{Float64}(undef,
# size_expr)/Vector{Int64}(undef, size_expr) (3 args: curly, :undef,
# an arbitrary size expression) -- size_expr itself is never validated
# here since it's just carried verbatim into the emitted rhs (cgen_body
# re-emits every :assign statement's rhs unexamined).
cgen_is_stack_alloc(rhs) = rhs isa Expr && rhs.head == :call &&
    rhs.args[1] isa Expr && rhs.args[1].head == :curly && rhs.args[1].args[1] == :Vector &&
    (length(rhs.args) == 1 || (length(rhs.args) == 3 && rhs.args[2] == :undef))

# the pre-sized subset of cgen_is_stack_alloc -- the only form that
# ever gets read/written from inside a *split* device kernel (see the
# device-residency note above cgen_stack_device_expr/jgen_stack_device_expr
# below). The empty, growing keep_push_pop=true form never needs this:
# every loop touching it contains a push!/pop!, and skill-stade.md's
# stack safety rule means such a loop is never split onto the device in
# the first place, so that Vector legitimately stays a plain host Vector
# (which is also the only Julia Vector variant push!/pop! works on at
# all -- a device array supports neither).
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
# (e.g. an array-length argument), not just what the body touches --
# then subtracts every scalar the body assigns locally (see
# cgen_locally_assigned_scalars), since those are per-iteration
# temporaries, not caller-supplied arguments, even though they're
# "used" in the body-wide sense cgen_collect_vars! collects
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

# a scalar assigned anywhere inside a loop's own body (at any nesting
# depth, including a nested loop's own loop variable) is always a
# local temporary, never a caller-supplied argument. This holds
# specifically *because* the enclosing loop is iteration-independent:
# a genuine cross-iteration read of a not-yet-assigned scalar would be
# exactly the loop-carried dependency that classification already
# rules out, so anything assigned as a bare Symbol lhs is guaranteed
# to be initialized fresh within the same iteration before any read.
# Array names never qualify (skill-jade forbids in-kernel allocation,
# so an array symbol is always caller-supplied) -- only a bare-Symbol
# assignment target counts, never an array-ref lhs.
#
# EXCEPTION: a *self-referencing* bare-Symbol assignment (`cb = cb +
# ...`, reading its own lhs among the rhs's flattened +/- terms) is
# not a fresh-per-iteration local at all -- it's a cross-thread scalar
# reduction (an adjoint accumulator like `cb`/`dxb`/`dtb` is the
# common case), the exact same "accumulate in place" pattern
# cgen_device_assign already special-cases for an *array-indexed* lhs
# (`x[k] = x[k] + v`). Excluding it here keeps it a free var (see
# cgen_free_vars), so it's still passed as a kernel argument instead
# of silently vanishing from the generated signature; the atomic
# rewrite that makes accumulating it across threads actually safe
# lives in cgen_device_assign/jgen_device_assign, driven by
# cgen_scalar_reduction_vars below.
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

# The subset of a (to-be device-split) loop body's free vars that are
# scalar cross-thread reductions (see the self-referencing exception
# above) -- these are exactly the vars cgen_emit/jgen_emit must box
# into a 1-element device array before any kernel launch can read or
# atomically write them, and the ones cgen_device_assign/
# jgen_device_assign must rewrite into an atomic add against that
# boxed array rather than an ordinary per-thread-local assignment.
#
# NOT every self-referencing bare-Symbol assignment qualifies, though
# -- a per-thread private running sum (`s = 0.0` then `s = s + ...`
# inside a nested SEQUENTIAL loop, e.g. an inlined convolution's own
# reduction over kernel taps) is self-referencing too, but it's fully
# reinitialized ("fresh-init'd") within the same device-split
# iteration before it's ever read, so it's thread-private and safe as
# an ordinary per-thread local -- exactly cgen_locally_assigned_scalars'
# own criterion. Only a var with a self-referencing assignment
# SOMEWHERE and no fresh (non-self-referencing) init ANYWHERE in the
# same body has no local initializer at all -- its value can only come
# from outside the loop (a genuine cross-thread accumulator, e.g. an
# adjoint scalar like `cb`). Hence: self-referenced vars, minus
# whatever cgen_locally_assigned_scalars already deems local.
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
# A loop marked i_seq_ (skill-jade's prefix convention) is normally
# left entirely on the host (cgen_body's/jgen_body's `!stmt.sequential`
# gate) because "sequential" covers BOTH genuine recurrences
# (order-dependent, e.g. a shift recurrence `u[i]=c*u[i-1]`) AND loops
# with no real cross-iteration coupling at all beyond a commutative/
# associative accumulation -- either INTO a shared scalar (a simple
# dot-product's `loss[1]=loss[1]+u[i]*v[i]`) or OUT to per-iteration-
# unique array slots (a reverse-mode adjoint sweep's `ub[i]=ub[i]+
# v[i]*lossb[1]`, the exact shape STADE's own AD transform emits for
# such a kernel's adjoint). skill-jade's naming rule only says
# "carries state across iterations", it doesn't distinguish any of
# these. Leaving a loop like this unsplit means it runs as an ordinary
# host-side Julia loop, indexing whatever arrays the caller passed one
# element at a time -- fine for host Arrays, but a `CUDA.allowscalar
# (false)` crash the moment those arguments are device arrays
# (confirmed against a live GPU run of exactly this kernel shape,
# forward and reverse sweep both).
#
# The one thing that's NEVER safe to parallelize is a genuine VALUE
# recurrence: the same array read at one loop-var-dependent index and
# written at a DIFFERENT loop-var-dependent index (a shift recurrence
# `u[i]=c*u[i-1]` -- iteration i needs iteration i-1's write). Notably,
# an array that's read and written at the SAME index expression every
# time (loss[1]=loss[1]+...; ub[i]=ub[i]+...) is NOT a recurrence in
# this sense -- nothing about it depends on what order iterations run
# in, so it's safe to hand to cgen_device_assign exactly as-is: a
# thread-invariant index (loss[1]) becomes an atomic add (already
# implemented, unchanged); a loop-var-dependent index (ub[i]) becomes
# an ordinary per-thread-unique write (also already implemented,
# unchanged) -- the same trust cgen_device_assign already extends to
# every INDEPENDENT loop's adjoint accumulators (e.g. `cb[k]=cb[k]+v`).
# A same-array read/write at differing indices where at least one
# depends on the loop var is refused. So is a bare-Symbol
# self-reference that isn't additive (a multiplicative or other
# non-+/- recurrence, e.g. `p = p * r`) -- cgen_self_referencing_assign
# already only recognizes +/- forms, so anything else self-referencing
# is a real recurrence too.
#
# This intentionally does NOT try to prove indices are collision-free
# across threads for cases like a graph kernel's per-edge scatter-add
# into a shared-by-receiver-node accumulator (`agg[k]=agg[k]+
# messages[...]`, k derived from a per-edge destination lookup) -- that
# read/write pair is at the SAME expression each time (not a
# recurrence by the definition above), so it's allowed through, and
# cgen_device_assign's EXISTING thread-dependent-index fast path
# decides atomic-vs-plain for it exactly as it already does for any
# independent loop with a data-dependent scatter target. Whether that
# specific write ends up atomic or plain is unchanged by this feature
# either way -- it's the same accepted trade-off cgen_device_assign's
# own docs already flag for independent loops (kernel author's
# responsibility to avoid or explicitly synchronize genuine
# collisions), just now also reachable from a loop that used to be
# sequential. Anything this can't prove free of a genuine recurrence
# keeps today's behavior (stays sequential, host-side) rather than
# guessing -- e.g. a simple shift recurrence (`u[i]=c*u[i-1]`) or a
# Jacobi-style sweep (`mup[i_node]` overwritten with no self-reference
# at all while `up` is read at a DIFFERENT node index derived from the
# same outer sequential loop across sweeps).
# All scalar symbols assigned anywhere in body (self-referencing or
# not, any nesting depth) -- used only to invalidate cgen_body's/
# jgen_body's known_consts tracking after an :if statement, since a
# branch could reassign a tracked var to something non-constant and
# cgen_locally_assigned_scalars alone (non-self-referencing assigns
# only) wouldn't catch that.
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

function cgen_reduction_only_loop(body::Vector{NamedTuple}, loopvar::Symbol, known_consts::Dict{Symbol,Any} = Dict{Symbol,Any}())
    local_names = cgen_locally_assigned_scalars(body)
    # A locally-assigned scalar that fails the plain "already defined"
    # walk below MAY still be safe if it provably converges to the
    # SAME literal constant on every control-flow path through the
    # loop body (e.g. a var where both the `__branch==1` and
    # `else` arms end with `wb = 0.0`, so `wb` is 0.0 at the end of
    # every iteration regardless of which branch ran) -- PROVIDED that
    # constant also matches the value `wb` is known to hold coming
    # into the loop in the first place (known_consts, populated by
    # cgen_body from a plain top-level `wb = 0.0` literal assignment
    # immediately preceding this loop in program order). Given both,
    # by induction every iteration -- including the first -- starts
    # with wb==0.0, so injecting `wb = 0.0` as the very first statement
    # of the per-thread kernel body reproduces the sequential
    # semantics exactly. Without a matching known_consts entry, we have
    # no way to know the loop's TRUE entering value (cgen_body only
    # hands this function the loop's own body, not its surrounding
    # scope), so this stays unproven and the loop is refused, same as
    # before this extension existed.
    synth = Dict{Symbol,Any}()
    for v in local_names
        haskey(known_consts, v) || continue
        c = cgen_loop_convergent_constant(body, v)
        c !== nothing && c == known_consts[v] && (synth[v] = c)
    end
    cgen_reduction_only_scalar_walk(body, Set{Symbol}(), local_names, synth)[1] || return nothing
    writes = Dict{Any,Vector{Any}}()
    reads = Dict{Any,Vector{Any}}()
    cgen_collect_array_accesses!(body, writes, reads)
    # Deliberately NOT scoped to "differing index expressions that
    # mention the loop's own variable" -- a sweep-style recurrence
    # (a Jacobi-style relaxation, or a timestep loop reading its own
    # previous step) carries its cross-iteration coupling through an
    # INNER loop's index (e.g. `up[i_node]` written during sweep k,
    # read during sweep k+1's traversal of the very same i_node range)
    # with no mention
    # of the outer i_seq_ variable anywhere in either index at all.
    # The loop var doesn't have to appear in an index for reading and
    # writing the same array at two different index expressions to be
    # a genuine recurrence -- so ANY structural mismatch between a
    # write index and a read index of the same array disqualifies the
    # loop, unconditionally.
    for (arr, widxs) in writes
        ridxs = get(reads, arr, Any[])
        for w in widxs, r in ridxs
            w == r && continue
            return nothing
        end
    end
    return synth
end

# Is `var` assigned anywhere within body, at any nesting depth
# (self-referencing or not)? Used only to bail out of the convergence
# analysis below when a NESTED loop touches the same variable --
# proving convergence through a sub-loop's own repeated execution
# needs its own fixed-point argument, which isn't needed for any
# corpus kernel today, so it's simplest and safest to just decline.
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

# The value `var` provably holds at the END of one traversal of body,
# if that's the SAME literal on every control-flow path -- else
# `nothing`. Walks in program order threading a running "state" that's
# one of :unchanged (not touched by any statement seen so far along
# this path, so it still holds whatever value flowed in from outside
# body), :unknown (touched, but not provably a single literal -- e.g.
# assigned a computed expression, or an if/else where the two arms
# disagree), or a Number (touched, and every touch since the last
# branch point agrees on this exact literal).
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
            cgen_var_assigned_anywhere(stmt.body, var) && (state = :unknown)
        end
    end
    return state
end

# Program-order walk, scoped to exactly the vars cgen_locally_assigned_
# scalars already calls "thread-private" (i.e. NOT in
# cgen_scalar_reduction_vars, since those are boxed+atomic and their
# correct initial value comes from the host-side box, unaffected by
# where in the loop body their self-reference sits). For a
# thread-private var, requires every self-referencing use to be
# preceded, textually within THIS loop's own body, by SOME assignment
# to it (fresh or self-referencing, doesn't matter which -- once it's
# been assigned at all within the body, cgen_device_body's ordinary
# per-thread-local codegen is correct) -- OR to be listed in `synth`
# (cgen_reduction_only_loop's convergent-constant proof), in which case
# it's treated as already defined from the very start of the walk,
# since cgen_kernel_def will inject `var = synth[var]` as the kernel
# body's first statement. This is a strictly narrower (safer)
# requirement than cgen_locally_assigned_scalars itself applies, and
# deliberately so: that function is documented as correct only
# "because the enclosing loop is iteration-independent" (see its own
# comment) -- a precondition this function exists specifically to
# stand in for once a SEQUENTIAL loop is under consideration, where it
# doesn't automatically hold.
#
# Concretely this catches a reverse sweep of exactly this shape: `wb` is
# additively self-referencing (`wb = wb + lossb[1]`) AND has a fresh
# reset (`wb = 0.0`) later in the SAME body, on every control-flow
# path -- so cgen_locally_assigned_scalars (correctly, for ITS
# purpose) calls it thread-private, not a cross-thread reduction
# target. That's true value-wise (wb really does start every iteration
# at 0, since both branches reset it before the loop moves on) -- but
# the loop's ORIGINAL, unsplit source only has ONE textual `wb = 0.0`,
# sitting BEFORE the loop entirely, not as the loop body's own first
# statement. A per-thread kernel only receives the loop BODY, so its
# very first line (`wb = wb + lossb[1]`) would read a `wb` neither
# this kernel nor its caller ever assigned -- confirmed against a live
# GPU run of the JACC target, where it manifested as a device-side
# KernelException (undefined variable, caught by JACC's stricter
# runtime check) rather than silently computing garbage, but CUDA.jl's
# laxer runtime checking on the exact same generated shape means it
# can NOT be relied on to fail loudly there either. cgen_reduction_
# only_loop's convergence proof (see its own comment) now recovers
# this case instead of just refusing it: `wb`'s missing fresh-init is
# synthesized as a literal `wb = 0.0` at the top of the per-thread
# kernel body, since it's provably the value every iteration starts
# with anyway.
function cgen_reduction_only_scalar_walk(body::Vector{NamedTuple}, defined_in::Set{Symbol}, local_names::Set{Symbol}, synth::Dict{Symbol,Any} = Dict{Symbol,Any}())
    defined = union(defined_in, Set{Symbol}(keys(synth)))
    for stmt in body
        if stmt.kind == :assign && stmt.lhs isa Symbol
            if stmt.lhs in local_names && cgen_expr_contains(stmt.rhs, stmt.lhs)
                stmt.lhs in defined || return (false, defined)
            end
            push!(defined, stmt.lhs)
        elseif stmt.kind == :if
            ok_then, defined_then = cgen_reduction_only_scalar_walk(stmt.then, defined, local_names, synth)
            ok_then || return (false, defined)
            ok_els, defined_els = cgen_reduction_only_scalar_walk(stmt.els, defined, local_names, synth)
            ok_els || return (false, defined)
            defined = intersect(defined_then, defined_els)
        elseif stmt.kind == :for
            ok, _ = cgen_reduction_only_scalar_walk(stmt.body, defined, local_names, synth)
            ok || return (false, defined)
        end
    end
    return (true, defined)
end

# collects every array read/write index expression (as a Vector of the
# :ref node's index args, for structural `==` comparison) reachable
# anywhere within body, at any nesting depth -- deliberately flat/
# order-insensitive since cgen_reduction_only_loop only needs to know
# WHICH index expressions a given array is touched at, not in what
# sequence, to detect a genuine same-array differing-index recurrence.
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
# gpu_backend :: (suffix, kernel_tag, launch_macro::Symbol,
#                 threads_kw::Symbol, blocks_kw::Symbol, tid_rhs::Expr,
#                 atomic_macro::Expr, allowscalar_macro::Expr, preamble::String,
#                 default_precision::Type{<:AbstractFloat},
#                 precision_locked::Bool, precision_lock_reason::String)
#
# Everything above this point (parsing, free-var collection, stack-op
# detection, loop-splitting decision, +/- flattening for atomic
# detection) is genuinely backend-agnostic -- it never mentions CUDA.
# What differs between CUDA.jl, AMDGPU.jl, and Metal.jl is all
# syntactic, not semantic: the launch macro name, the two launch
# keyword names (`threads`/`blocks`, `groupsize`/`gridsize`,
# `threads`/`groups` -- all three resolve to the exact same
# `cld(n_iter, blocksize)` formula for the second one, so nothing about
# the arithmetic below differs), the thread-index intrinsic (Metal's
# `thread_position_in_grid().x` is already the global index -- simpler
# than CUDA/AMDGPU's two-intrinsic affine combination, but still just
# an Expr for the same tid_rhs slot), the atomic macro's owning module,
# the allowscalar macro's owning module (cgen_body wraps any host-side
# statement run that touches a device array element-wise -- i.e. never
# made it into a split-off kernel -- in one of these; see the comment
# above cgen_body), and the `using` preamble. Adding a further backend
# (oneAPI.jl, ...)
# should only ever mean adding one more of these constructors -- if it
# turns out to need a change anywhere else in this section, that's a
# sign the new backend isn't actually the same programming model and
# deserves its own prefix instead of being forced in here.
#
# default_precision/precision_locked/precision_lock_reason exist
# because Metal is not just "Float64 works but is slower" the way
# switching precision is for CUDA/AMDGPU: Apple GPUs have no FP64
# hardware at all, and Metal.jl enforces this in software too --
# MtlArray flatly refuses to be constructed with Float64 elements, and
# any kernel whose arithmetic touches a Float64 fails to *compile*.
# precision_locked=true makes stade_gpu refuse an explicit request for
# anything but default_precision outright, at code-generation time,
# rather than silently emitting Julia source that's guaranteed to fail
# once the caller actually tries to build their input arrays or run
# the kernel on real Metal hardware.

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

# thread_position_in_grid().x is already a global 1-based index (no
# manual block/thread affine combination needed, unlike CUDA/AMDGPU);
# `groups=` (not the older, pre-v1.9-preview `grid=`) is the current
# launch keyword for the group count, verified against Metal.jl's
# current stable docs/README/source examples, all of which
# consistently use threads=/groups= and the unsuffixed
# thread_position_in_grid().x form (the _1d/_2d/_3d-suffixed
# intrinsics are deprecated as of Metal.jl v1.9). Metal.@atomic exists
# and is documented as working the same way CUDA.jl's @atomic does.
#
# `^` is not accounted for here even though Metal has had a real,
# separate compiler bug where `Float32 ^ Integer` is computed in
# double precision internally regardless of what Julia's own types say
# (JuliaGPU/Metal.jl#552) -- that's a backend compiler bug, not an
# Expr-level promotion issue cgen_convert_precision's operand-casting
# rewrite could fix, so it's recorded here as a caveat rather than
# "handled": avoid `^` in the innermost loops of a kernel bound for
# Metal until you've confirmed the fix status against your Metal.jl
# version.
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

# device residency for a keep_push_pop=false stack: `agen_init_emit`
# always writes the stack allocation as a plain host `Vector{T}(undef,
# size_expr)`, since agen_ (skill-jade rule 3's "no top-level const"
# aside) has no notion of a GPU backend at all -- it's cgen_'s own job,
# same as everything else backend-specific, to turn that into a real
# on-device allocation once it's known which device will actually read
# and write it. This only ever fires for the pre-sized :indexed form
# (cgen_is_sized_stack_alloc): that's the only stack shape a split
# device kernel ever touches directly (see cgen_is_sized_stack_alloc's
# own comment for why the push!/pop! form never needs this). `undef`
# is preserved rather than switched to a zero-fill: every element gets
# written by a stack push before its first read (the entire point of
# the forward sweep), so a device-side `undef` allocation costs nothing
# and matches the plain-Vector behavior this replaces exactly.
function cgen_stack_device_expr(rhs::Expr, backend)
    T = rhs.args[1].args[2]
    size_expr = rhs.args[3]
    return Expr(:call, Expr(:curly, backend.arrtype, T), :undef, size_expr)
end

# ---- idiomatic scalar-reduction detection (keep_all_atomic=false) --
# Recognizes the narrow, syntactically-exact shape
#     target = target ± f(arr_1[loopvar], arr_2[loopvar], ..., free scalars)
# as the WHOLE and ONLY statement in a loop body already proven safe to
# split by cgen_reduction_only_loop -- i.e. a bare dot-product-style
# `loss[1]=loss[1]+u[i]*v[i]`, NOT a case with an :if picking the
# per-iteration term (a SECOND statement, so length(body)==1 fails
# and this correctly declines -- same "prove a narrow case, refuse
# otherwise" discipline as cgen_reduction_only_loop itself, and the
# caller falls back to today's atomic-kernel codegen unchanged, exactly
# as if keep_all_atomic had been true). Returns
# (target::Expr, op::Symbol, arrs::Vector{Symbol}, term) or `nothing`;
# the caller decides how to turn (arrs, term, loopvar) into an actual
# replacement expression, since that differs by target (see
# cgen_idiomatic_reduction_value / jgen_idiomatic_reduction_value).
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

# True iff `loopvar` occurs anywhere in `e` OUTSIDE of a :ref's index
# position -- i.e. the per-iteration term uses the loop variable itself
# for something other than indexing one of the arrays it reads
# (cgen_idiomatic_scalar_reduction already verified every :ref's index
# IS exactly loopvar; this only needs to rule out a bare use elsewhere,
# e.g. a hypothetical `u[i] + i`).
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

# CUDA/AMDGPU/Metal: `dot`/`sum(abs2, ·)` and `mapreduce` all dispatch
# on the array TYPE of their argument (GPUArrays.jl's generic
# mapreduce/reduce machinery, or -- for CUDA.jl/AMDGPU.jl specifically
# -- a vendor cuBLAS/rocBLAS dot method when one exists), so the same
# call works correctly on every one of these backends: `dot`/`sum` pick
# whichever underlying implementation is fastest for that array type at
# RUNTIME, with no backend-conditional codegen needed here. `dot` and
# `sum` both come from `using LinearAlgebra`/Base, present in every GPU
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

# JACC: no confirmed BLAS-level dot/sum acceleration to special-case
# for, so every matched shape (dot-shaped or not) goes through the same
# `JACC.@parallel_reduce range=N f(args...)` primitive -- JACC's own
# portable, atomic-free reduction abstraction, replacing the
# `@parallel_for` + `Atomix.@atomic` kernel this loop would otherwise
# get. Unlike the CUDA/AMDGPU/Metal path, `term` needs no substitution:
# JACC's own convention is a closure whose first parameter IS the loop
# index (see the range=N example in JACC's docs), so reusing loopvar's
# own name as that parameter and leaving every `arr[loopvar]` in `term`
# exactly as written is already the correct closure body.
function jgen_idiomatic_reduction_value(arrs::Vector{Symbol}, term, loopvar::Symbol, n_iter)
    closure = Expr(:->, Expr(:tuple, loopvar, arrs...), term)
    return Expr(:macrocall, Expr(:., :JACC, QuoteNode(Symbol("@parallel_reduce"))), nothing,
                Expr(:(=), :range, n_iter), Expr(:call, closure, arrs...))
end

# ---- JACC idiomatic-reduction write-back (keep_all_atomic=false) ---
# `target` (an `arr[idx...]` ref -- see cgen_idiomatic_scalar_reduction)
# must never be read or written from the host once `arr` may be a JACC
# device array (see jgen_body's doc comment for the live-GPU failure
# this fixes). This performs the `target += value` accumulate inside a
# trivial one-thread device kernel instead, reusing the exact
# `Atomix.@atomic target += ...` shape jgen_device_assign already emits
# for reduce_vars. `value` is always pre-signed by the caller (see
# jgen_body) so this only ever needs `+=`, matching jgen_device_assign's
# own convention. `wfargs` (from cgen_collect_expr_vars! on `target`)
# already includes `target`'s own array symbol plus every free var used
# in its index expression(s) -- nothing else is touched by the kernel.
# `__jgen_redval` is a synthetic parameter name for the reduction
# result. It's passed straight through as a `CuArray{Float64,1}` --
# `JACC.@parallel_reduce` already returns one directly (confirmed via a
# live GPU diagnostic; it is NOT a host Float64 despite reading like one
# in the original code's `loss[1] = loss[1] + JACC.@parallel_reduce(...)`)
# -- and read via `[1]` here, on-device, exactly like every other kernel
# in this file that reads a boxed scalar (e.g. `loss[1]`, `lossd[1]`).
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


# True iff `e` contains an `Expr(:ref, ...)` anywhere -- i.e. an
# element-wise array index -- at any depth. Used by cgen_body to spot
# host-side statements that touch what may be a device array; scalar
# kernel arguments and local scalar temporaries never appear as the
# base of a :ref (skill-jade's grammar only ever indexes an array or a
# stack), so this is exactly the "is this legal under allowscalar(false)"
# test, no kind/type lookup needed.
function cgen_expr_has_ref(e)
    e isa Expr || return false
    e.head == :ref && return true
    return any(cgen_expr_has_ref, e.args)
end

# ---- host-side body walk: splits off one device kernel per eligible
#      iteration-independent loop. Anything left over -- a statement
#      with no enclosing loop at all, or one whose enclosing loop
#      stayed host-sequential because splitting it would be unsound
#      (a genuine recurrence) -- runs
#      as ordinary host-side Julia. If that statement touches a device
#      array element-wise (cgen_expr_has_ref on its lhs/rhs), it's
#      exactly the pattern CUDA.allowscalar(false) in this file's own
#      preamble is designed to reject: confirmed as a live-GPU crash
#      for the unsplit-loop case (see the note a few hundred lines up),
#      and true by the same argument for the zero-loop case, which
#      never even reaches a loop to fail to split.
#
# Fix: buffer up each maximal run of consecutive :assign statements at
# a given nesting level: if any statement in the run touches an array,
# emit the whole run inside one backend.allowscalar_macro block; if
# none do (e.g. a purely scalar prelude computing a derived count)
# emit the run exactly as before, unwrapped. A run is flushed (in
# order) whenever a :stackpush/:if/:for statement is reached, and
# once more at the end of the body, so statement order is preserved
# exactly -- the wrapping never reorders anything, it only brackets
# spans that were already going to run sequentially on the host.
# Grouping into runs (rather than wrapping each statement in its own
# block) doesn't change how many host<->device transfers happen --
# @allowscalar only lifts a task-local check around whatever runs
# inside it, each individual scalar getindex!/setindex! still pays its
# own transfer -- it just avoids emitting one macro block per line.
#
# Not handled: an :if condition itself indexing a device array with no
# enclosing loop (e.g. a hypothetical top-level `if u[1] > 0.0`). Real
# kernels that branch on a scalar kernel argument, never an array
# read, are unaffected, and no test kernel exercises the array-
# condition shape, so it's flagged here rather than guessed at.
#
# keep_all_atomic (default true): when false, a splittable loop is
# first offered to cgen_idiomatic_scalar_reduction -- if it matches the
# narrow single-statement pure-scalar-reduction shape, the whole
# loop+kernel+launch is replaced by one `dot`/`sum(abs2,·)`/`mapreduce`
# call (see cgen_idiomatic_reduction_value) instead of a synthesized
# atomic-accumulate kernel. Anything that doesn't match that shape
# (a branching accumulator, any gather-indexed scatter-
# add) is completely unaffected either way -- this flag only ever
# changes which of two ALREADY-equivalent primal computations gets
# emitted for a provably pure scalar reduction, never what a kernel
# computes.
function cgen_body(body::Vector{NamedTuple}, kernels::Vector{Expr}, owner::Symbol, backend, reduce_vars::Set{Symbol}; keep_all_atomic::Bool = true)
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
            # tracked purely so a LATER sequential loop in this same
            # body can prove a locally-assigned scalar's true entering
            # value (cgen_reduction_only_loop's convergent-constant
            # check) -- see how a `wb = 0.0` right before its reverse
            # sweep gets used this way. Only a bare literal counts;
            # anything else invalidates (rather than tracks) the
            # symbol, since
            # we have no way to prove IT stayed constant either.
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
            push!(exprs, emit_if(cond, cgen_body(stmt.then, kernels, owner, backend, reduce_vars; keep_all_atomic), cgen_body(stmt.els, kernels, owner, backend, reduce_vars; keep_all_atomic)))
            for v in cgen_all_assigned_scalars(vcat(stmt.then, stmt.els))
                delete!(known_consts, v)
            end
        elseif stmt.kind == :for
            flush_pending!()
            synth = !stmt.sequential ? Dict{Symbol,Any}() : cgen_reduction_only_loop(stmt.body, stmt.var, known_consts)
            if synth !== nothing && !cgen_contains_stackop(stmt.body)
                red = keep_all_atomic ? nothing : cgen_idiomatic_scalar_reduction(stmt.body, stmt.var)
                if red !== nothing
                    target, op, arrs, term = red
                    value = cgen_idiomatic_reduction_value(arrs, term, stmt.var)
                    push!(pending, Expr(:(=), target, Expr(:call, op, target, value)))
                    pending_has_array = true
                else
                    idx = length(kernels) + 1
                    fargs = cgen_free_vars(stmt, stmt.var)
                    loop_reduce_vars = cgen_scalar_reduction_vars(stmt.body)
                    union!(reduce_vars, loop_reduce_vars)
                    push!(kernels, cgen_kernel_def(stmt, owner, idx, fargs, backend, loop_reduce_vars, synth))
                    push!(exprs, cgen_launch_expr(stmt, owner, idx, fargs, backend))
                end
            else
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, cgen_body(stmt.body, kernels, owner, backend, reduce_vars; keep_all_atomic)))
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

# true iff `x` provably depends on the thread var ONLY through pure
# arithmetic -- i.e. it's safe to trust as per-thread-unique. Two
# things break that proof, either directly or transitively through a
# chained scalar let-binding: (1) any array READ anywhere in the
# expression (a `:ref` node) -- a gather through caller-supplied data
# (e.g. an edge-to-node lookup array) can return the same value for
# two different thread-var values, so nothing downstream of it can be
# trusted injective no matter how much arithmetic follows; (2) any symbol
# that's itself thread-dependent (`thread_dep`) but NOT already proven
# injective (`injective_dep`) -- e.g. `agg_off = (d_node - 1) *
# n_msg_feat` is pure arithmetic on its face, but `d_node` came from
# that same array read, so the taint propagates through it. A symbol
# that's thread-INdependent (not in thread_dep at all -- an ordinary
# loop-invariant caller argument) is always fine, injective or not,
# since it's constant across threads.
function cgen_expr_injective_ok(x, injective_dep::Set{Symbol}, thread_dep::Set{Symbol})
    x isa Symbol && return !(x in thread_dep) || (x in injective_dep)
    x isa Expr || return true
    x.head == :ref && return false
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
            # a write's index can be computed through a same-body
            # scalar let-binding one or more hops from the thread
            # variable (`yi = f(idx); arr[yi] = ...`), not just written
            # literally in the index expression itself -- extend
            # thread_dep transitively so cgen_device_assign's occurs-
            # check sees through it. Tracks CURRENT dependency per
            # variable (removed on a reassignment that breaks the
            # chain), not a once-true-always-true flag. injective_dep
            # is tracked the same way, one level stricter (see
            # cgen_expr_injective_ok): a chain stays injective only as
            # long as every hop is pure arithmetic on already-injective
            # symbols, with no array read anywhere in the chain.
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

# a write races across threads unless the enclosing device loop's own
# thread-mapped variable occurs (at any depth) in the write's index, OR
# the index is computed from it through a chain of same-body scalar
# let-bindings (`thread_dep` -- see cgen_device_body) -- a real
# occurs-check, not shallow top-level membership. ANY
# thread-invariant-indexed write from a parallel loop is a race
# regardless of whether it's an additive accumulation pattern
# (x[k]=x[k]+v) or a plain replacement -- an accumulation is safe to
# fix with an atomic +=; a replacement has no such fix (concurrent
# threads writing DIFFERENT values to the same fixed slot is a race
# no atomic wrapper resolves), so it's refused outright rather than
# silently emitted as an ordinary, unprotected write. This has never
# been hit in practice (every case tested so far with a
# thread-invariant index was additive) -- refusing loudly here is
# cheap insurance against a future sizing/offset bug (e.g. from
# keep_push_pop=false's Tier A/B arithmetic) silently parallelizing
# something wrong instead of getting caught at generation time.
#
# A thread-DEPENDENT index is not automatically race-free either,
# though: "depends on the thread var" and "is injective in the thread
# var" are different claims, and only the second one actually implies
# per-thread-uniqueness. A graph kernel's edge-to-node scatter-add
# (`agg[k] = agg[k] + messages[...]`, `k` derived from a per-edge
# destination lookup) is thread-dependent but NOT injective -- two
# different edges (threads) can gather the same receiver node,
# landing on the same `k`. `injective_dep` (see
# cgen_expr_injective_ok/cgen_device_body) distinguishes this from the
# `ub[i_seq_x]`-style case where the index IS the thread var (or a
# pure-arithmetic function of it, e.g. a div/mod loop-unravel), which
# stays a plain per-thread-unique write exactly as before. An
# additive write whose index depends on the thread var only through a
# non-injective (gathered) chain gets the SAME atomic treatment as the
# thread-invariant case above, rather than assuming uniqueness that
# was never proven. A non-additive (overwrite) write through such a
# gathered index is intentionally left untouched by this distinction --
# an atomic can't fix a plain-replacement race either way, so that
# case stays the same accepted kernel-author-responsibility trade-off
# already documented on cgen_reduction_only_loop for any data-dependent
# scatter target.
function cgen_device_assign(stmt, thread_var::Symbol, thread_dep::Set{Symbol}, injective_dep::Set{Symbol}, backend, reduce_vars::Set{Symbol})
    if stmt.lhs isa Symbol && stmt.lhs in reduce_vars
        # scalar cross-thread reduction (e.g. an adjoint accumulator
        # like `cb`/`dxb`/`dtb`) -- cgen_emit already boxed this free
        # var into a 1-element device array before any kernel launch,
        # so the accumulation becomes an atomic add against index 1,
        # the exact same treatment as the array-indexed self-reference
        # case below, just pre-boxed to a known-size-1 array instead of
        # relying on an existing caller-supplied index.
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
            thread_invariant && error("cgen_device_assign: write to `$(stmt.lhs)` inside a GPU-split loop has an index that doesn't depend on the loop's own thread variable (`$thread_var`), even transitively through same-body scalar let-bindings, and isn't an additive accumulation -- this is a data race across threads, not something an atomic wrapper can fix. See skill-stade.md's cgen_device_assign hardening note.")
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
    host_body = cgen_body(gk.body, kernels, gk.name, backend, reduce_vars; keep_all_atomic)
    isempty(kernels) || pushfirst!(host_body, :(nthread_per_block = 256))
    # Every scalar cross-thread reduction free var (see
    # cgen_scalar_reduction_vars) must already be a 1-element device
    # array before the first kernel launch that atomically writes it,
    # and must be unboxed back to a plain host scalar before it can be
    # returned -- both transfers are whole-array host<->device copies
    # (CuArray([v]) / Array(v)[1]), never an element-wise scalar index
    # into a still-device-resident array, so this is legal under
    # allowscalar(false). Sorted for deterministic codegen output,
    # matching cgen_free_vars' own convention.
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

# opt-in, applied only when the caller passes precision=T (T != Float64)
# to stade_gpu/stade_cuda/stade_amdgpu/stade_gpu_file -- converts every
# Float64 literal it finds to T, leaving Int-typed loop/index arithmetic
# (trip counts, thread-index offsets, nthread_per_block) untouched since
# a bare literal walk only ever matches AbstractFloat leaves. Applied
# only to the freshly-generated host/kernel Exprs cgen_emit just built --
# it never touches the input file on disk, so re-running at Float64
# always reproduces the exact double-precision behavior of the source
# kernel with nothing to undo.
#
# Deliberately just a literal walk, no operand-forcing rewrite: every
# array/scalar the kernel operates on is caller-supplied (skill-jade
# rule 8 -- no in-kernel allocation), so its precision is the caller's
# responsibility to get right, not something STADE should silently
# paper over by injecting T(...) casts around every division or
# transcendental call. One known consequence worth knowing about: a
# handful of Base operations return Float64 unconditionally when *both*
# their operands happen to be Integer, with no Float64 literal anywhere
# in the source for a tree walk to catch -- true division of two
# Integers (`2/2 -> Float64`) and any transcendental intrinsic applied
# to an Integer argument (`sqrt(2) -> Float64`), both via Base's generic
# `f(x::Real) = f(float(x))` fallback, where `float(::Integer)` always
# means Float64, never T. That's a real Integer-argument case, not a
# caller-precision one (an Int stays semantically exact regardless of
# T), so if a kernel's index/loop-bound arithmetic ever flows into a
# bare `/` or transcendental call with no float operand alongside it,
# the result is Float64 even under precision=Float32 -- on a
# precision_locked backend (Metal) that then fails to compile rather
# than silently running in double precision, so it surfaces immediately
# rather than corrupting results.
function cgen_convert_precision(expr, ::Type{T}) where {T<:AbstractFloat}
    tname = Symbol(string(T))
    if expr isa AbstractFloat
        return T(expr)
    # STADE never emits the bare Symbol :Float64 anywhere except as a
    # stack's element-type marker -- the curly type parameter in a
    # cgen_/jgen_ device stack allocation (Vector{Float64}, CuArray{Float64},
    # ROCArray{Float64}, MtlArray{Float64} -- see cgen_stack_device_expr)
    # or the first positional argument of JACC.zeros(Float64, size_expr)
    # (see jgen_stack_device_expr). This isn't the caller-precision
    # question the comment above opts out of retyping for -- these
    # stacks are STADE's own allocations, never caller-supplied, so
    # retyping them to match every other downcast float in the same
    # output is still STADE's job. Both sit inside an ordinary Expr that
    # the generic recursion branch below already walks arg-by-arg, so a
    # single, context-free rewrite here is enough -- without it, a
    # Metal.jl device kernel that reads from an un-retyped Float64 stack
    # fails to compile (Apple GPUs have no FP64 hardware). `:Int64` is
    # deliberately left untouched: a :branch/:tripcount stack's element
    # type is never precision-converted for any backend, mirroring how
    # every other Int-typed loop/index expression in this function is
    # left alone by the AbstractFloat-only literal walk above.
    elseif expr === :Float64
        return tname
    elseif expr isa Expr
        return Expr(expr.head, [cgen_convert_precision(a, T) for a in expr.args]...)
    end
    return expr
end


# ==================== jgen_* =======================================
# JACC.jl codegen. JACC replaces CUDA.jl/AMDGPU.jl/Metal.jl's "write a
# kernel + a vendor launch macro + vendor thread-index intrinsics"
# model with a single plain function taking the loop index as its
# first argument, dispatched via `JACC.@parallel_for range=N f(args...)`
# (JACC.jl v1.x; the pre-1.0/v0.0.x line used a plain function call
# `JACC.parallel_for(N, f, args...)` instead -- see skill-stade.md's
# jgen_ v1.x migration note if targeting an older JACC pin) -- which
# vendor backend actually executes it is chosen once per Julia
# *project*, outside any code jgen_ generates, via Preferences.jl. That
# is a different enough model from cgen_'s gpu_backend (no launch
# macro, no thread-index intrinsic to synthesize, and no way to know
# the eventual backend at generation time at all) that it gets its own
# prefix rather than being forced into gpu_backend, per the rule
# skill-stade.md already states for cgen_ itself: if a new target
# needs a change outside the shared machinery, it isn't the same
# programming model and doesn't belong under the same prefix.
#
# jgen_ reuses cgen_'s already backend-agnostic front end directly
# rather than duplicating it: cgen_ingest, cgen_free_vars,
# cgen_contains_stackop, cgen_trip_count, cgen_loopvar_from_tid,
# cgen_flatten_sum, cgen_expr_contains, cgen_sum_excluding, and
# cgen_convert_precision all operate purely on the parsed cgen_kernel/
# statement shape with a documented, frozen contract -- none of them
# know what a gpu_backend even is, so calling them isn't reaching into
# cgen_'s private internals the way skill-stade.md's purity rule warns
# against; it's using the shared utility layer cgen_ and jgen_ both
# sit on. Only the emit step below differs.
#
# Atomics: JACC adopted Atomix.@atomic as its own cross-backend atomic
# primitive (JACC's changelog: "Add support for Atomix.@atomic"), so
# the atomic-vs-plain-write decision is identical to cgen_'s -- only
# the macro target is fixed to Atomix instead of varying by backend.
#
# Precision is deliberately NOT locked or defaulted the way Metal's
# gpu_backend is: STADE cannot know, at generation time, which of
# JACC's five backends a given output file will eventually run under
# -- that choice is deferred to runtime, on a machine STADE never
# sees, which is the entire point of JACC. Pretending to guarantee
# precision safety the way cgen_backend_metal does would be actively
# misleading here. stade_jacc defaults to Float64 (a no-op, same as
# cgen_'s unlocked backends) and leaves it to the caller to pass
# precision=Float32 if a Metal-configured JACC deployment is a real
# possibility for the output.
#
# Bounds checking inside a split-off loop is deliberately omitted:
# JACC.@parallel_for range=N f(args...) is documented and exemplified
# (JuliaGPU/JACC.jl's own current stable docs) without an internal `i <=
# length(...)` guard inside the kernel function, unlike a raw @cuda/
# @roc/@metal launch which can overshoot to block granularity. This is
# taken on documentation/example evidence, not verified against a
# running JACC install -- no GPU hardware, of any vendor, has been
# available to actually run anything cgen_/jgen_ produce, on any
# backend, at any point. If a real run shows a guard is needed for
# some backend/version, add it the same way cgen_kernel_def does.

jgen_kernel_fname(owner::Symbol, idx::Int) = Symbol("jacc_kernel_" * string(owner) * "_" * string(idx) * "!")

# same device-residency need as cgen_stack_device_expr, but JACC has
# no vendor-specific array constructor to reach for -- and, per the
# section comment above, deliberately can't know at generation time
# which vendor a given output will even run on. `JACC.zeros(T, N)` is
# JACC.jl's own documented, portable, backend-dispatched allocator
# (juliagpu.github.io/JACC.jl/stable's own example: `JACC.zeros(Float32,
# N)`), so it's the only allocation call this can safely emit -- there
# is no documented `undef`-style JACC allocator to fall back to, unlike
# cgen_'s vendor `Array{T}(undef, N)` constructors, so this zero-fills
# instead. That's a harmless no-op in practice (every element is
# overwritten by a stack push before its first read), just a very
# slightly more expensive allocation than the `undef` cgen_ backends use.
function jgen_stack_device_expr(rhs::Expr)
    T = rhs.args[1].args[2]
    size_expr = rhs.args[3]
    return Expr(:call, Expr(:., :JACC, QuoteNode(:zeros)), T, size_expr)
end

# keep_all_atomic: same meaning and same detector as cgen_body's (see
# its doc comment) -- a matched loop is replaced by one
# `JACC.@parallel_reduce` call (jgen_idiomatic_reduction_value) instead
# of a synthesized `@parallel_for` + `Atomix.@atomic` per-element kernel.
#
# CORRECTED (was wrong): an earlier version of this comment claimed no
# allowscalar-style handling was needed here, unlike cgen_body's
# CUDA/AMDGPU/Metal path. A live GPU run (validate_corpus_gpu on a
# kernel with a pure scalar-reduction shape, JACC backend,
# keep_all_atomic=false) proved that false:
# `JACC.@parallel_reduce` itself returns a plain host scalar (safe to
# read), but WRITING that scalar back into `target` (an `arr[idx]`
# ref whose `arr` is, at runtime, a JACC-device-backed array for any
# scalar_float argument -- see validate_corpus_gpu.jl's
# GPU_BACKEND_SPECS) via a bare host-side `target = target op value`
# is a host getindex/setindex! against a device array. Unlike
# CUDA.jl/AMDGPU.jl/Metal.jl (which cgen_body wraps in
# backend.allowscalar_macro), JACC has no such escape hatch, so this
# raised "Scalar indexing is disallowed" for such a kernel's own
# adjoint/jacc (and, as a cascade, hvp/jacc). Fix: never write `target`
# from the host at all -- perform the accumulate itself on-device via a
# synthesized range=1 kernel (jgen_reduction_writeback_kernel /
# jgen_reduction_writeback_launch below), reusing the exact
# `Atomix.@atomic target += value` shape jgen_device_assign already
# uses for reduce_vars accumulation -- the same shape keep_all_atomic=
# true independently reaches (via its per-element atomic kernel) and
# which the live GPU run confirmed works.
function jgen_body(body::Vector{NamedTuple}, kernels::Vector{Expr}, owner::Symbol, reduce_vars::Set{Symbol}; keep_all_atomic::Bool = true)
    exprs = Any[]
    known_consts = Dict{Symbol,Any}()
    for stmt in body
        if stmt.kind == :stackpush
            push!(exprs, Expr(:call, :push!, stmt.stack, stmt.value))
        elseif stmt.kind == :assign
            rhs = cgen_is_sized_stack_alloc(stmt.rhs) ? jgen_stack_device_expr(stmt.rhs) : stmt.rhs
            push!(exprs, Expr(:(=), stmt.lhs, rhs))
            if stmt.lhs isa Symbol
                if stmt.rhs isa Number
                    known_consts[stmt.lhs] = stmt.rhs
                else
                    delete!(known_consts, stmt.lhs)
                end
            end
        elseif stmt.kind == :if
            push!(exprs, emit_if(stmt.cond, jgen_body(stmt.then, kernels, owner, reduce_vars; keep_all_atomic), jgen_body(stmt.els, kernels, owner, reduce_vars; keep_all_atomic)))
            for v in cgen_all_assigned_scalars(vcat(stmt.then, stmt.els))
                delete!(known_consts, v)
            end
        elseif stmt.kind == :for
            synth = !stmt.sequential ? Dict{Symbol,Any}() : cgen_reduction_only_loop(stmt.body, stmt.var, known_consts)
            if synth !== nothing && !cgen_contains_stackop(stmt.body)
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
                    # `JACC.@parallel_reduce(...)` must be evaluated as its
                    # own top-level host statement -- never nested inside
                    # another JACC launch macro's call-argument list.
                    # `JACC.@parallel_for` parses `f(args...)` syntactically
                    # and forwards `args...` into its own device-launch/
                    # compile machinery; it does not guarantee host-side
                    # pre-evaluation of a macro-call argument the way a
                    # plain function call would. Nesting the reduce directly
                    # in the launch call (an earlier version of this fix)
                    # produced `GPUCompiler.InvalidIRError` compiling
                    # `_parallel_for_cuda(...)` on a live GPU run -- a
                    # nested kernel-launch-and-synchronize construct
                    # reachable from device-compiled code. The shape used
                    # everywhere else in this file (including the original
                    # working code and the passing tangent/jacc path) is:
                    # `@parallel_reduce` only ever as the RHS of a plain
                    # host statement. `widx` is unique per write-back site
                    # within this owner's kernel list, so the temp name
                    # can't collide with another reduction in the same body.
                    redvar = Symbol("__jgen_redval_", widx)
                    push!(exprs, Expr(:(=), redvar, signed_value))
                    # NOT boxed via JACC.array here, despite that being
                    # jgen_emit's own reduce_vars convention for a genuine
                    # host scalar: a live GPU diagnostic
                    # (`typeof(JACC.@parallel_reduce(...))`) confirmed
                    # `JACC.@parallel_reduce` already returns a
                    # device-resident `CuArray{Float64,1}` directly, NOT a
                    # host Float64 -- despite reading naturally as one in
                    # the original (pre-fix) code's
                    # `loss[1] = loss[1] + JACC.@parallel_reduce(...)`,
                    # which only worked because Julia dispatches `+` on
                    # mixed scalar/array types via broadcasting-adjacent
                    # promotion in that specific host-arithmetic context.
                    # Wrapping it again via `JACC.array([redvar])` (an
                    # earlier version of this fix) tried to build a
                    # `CuArray` whose element type is itself a `CuArray`
                    # -- illegal ("CuArray only supports element types
                    # that are allocated inline"), confirmed by a live
                    # run. `redvar` is passed straight through as the
                    # kernel argument; the kernel indexes `[1]` on it
                    # on-device (see jgen_reduction_writeback_kernel),
                    # exactly like reading any other boxed scalar
                    # (`loss[1]`, `lossd[1]`) elsewhere in this file.
                    push!(kernels, jgen_reduction_writeback_kernel(owner, widx, target, wfargs))
                    push!(exprs, jgen_reduction_writeback_launch(owner, widx, wfargs, redvar))
                else
                    idx = length(kernels) + 1
                    fargs = cgen_free_vars(stmt, stmt.var)
                    loop_reduce_vars = cgen_scalar_reduction_vars(stmt.body)
                    union!(reduce_vars, loop_reduce_vars)
                    push!(kernels, jgen_kernel_def(stmt, owner, idx, fargs, loop_reduce_vars, synth))
                    push!(exprs, jgen_launch_expr(stmt, owner, idx, fargs))
                end
            else
                push!(exprs, emit_forloop(stmt.var, stmt.lo, stmt.hi, stmt.step, jgen_body(stmt.body, kernels, owner, reduce_vars; keep_all_atomic)))
            end
            delete!(known_consts, stmt.var)
        end
    end
    return exprs
end

# JACC hands the loop index in directly as the split-off function's
# first parameter -- no thread-index intrinsic to bind, unlike
# cgen_kernel_def, since JACC.@parallel_for range=N already
# guarantees the index range. cgen_loopvar_from_tid still does the
# affine lo/step remapping (JACC's own 1:N index space vs. the
# original loop's actual lo/step/hi), same as it does for cgen_.
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

# JACC.jl v1.x: JACC.@parallel_for range=N f(args...) -- a macro with a
# `range=` keyword-style argument, not a plain function call (that was
# the pre-1.0/v0.0.x API; see skill-stade.md's jgen_ v1.x migration
# note). The underlying kernel function's own signature is unchanged
# (index still its own first parameter, supplied internally by the
# macro) -- only the launch call's shape moved.
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
            # mirrors cgen_device_body's thread_dep/injective_dep
            # tracking exactly -- see that function's comment. This
            # brings the JACC target up to the same transitive
            # occurs-check parity cgen_device_assign already has,
            # rather than jgen_device_assign's previous bare
            # (non-transitive) thread_var check.
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

# identical decision to cgen_device_assign, reusing cgen_'s own
# occurs-check, injectivity-check, and sum-flattening helpers -- only
# the atomic macro's target module is fixed rather than coming from a
# backend descriptor. The scalar-reduction branch mirrors
# cgen_device_assign's exactly: reduce_vars was already boxed into a
# 1-element JACC.array by jgen_emit before any kernel launch, so `cb =
# cb + other` becomes an atomic add against index 1 of that boxed
# array. The array-ref branch now also mirrors cgen_device_assign's
# thread-invariant refusal AND its injective_dep distinction (see that
# function's comment) -- previously this used a bare, non-transitive
# `thread_var` occurs-check with no refusal path at all for a genuine
# non-additive race, which was a real (if narrower) gap of its own.
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
            thread_invariant && error("jgen_device_assign: write to `$(stmt.lhs)` inside a GPU-split loop has an index that doesn't depend on the loop's own thread variable (`$thread_var`), even transitively through same-body scalar let-bindings, and isn't an additive accumulation -- this is a data race across threads, not something an atomic wrapper can fix. See skill-stade.md's cgen_device_assign hardening note.")
        end
    end
    return Expr(:(=), stmt.lhs, stmt.rhs)
end

jgen_host_fname(name::Symbol) = Symbol(string(name) * "_jacc")

function jgen_emit(gk; keep_all_atomic::Bool = true)
    kernels = Expr[]
    reduce_vars = Set{Symbol}()
    host_body = jgen_body(gk.body, kernels, gk.name, reduce_vars; keep_all_atomic)
    # Mirrors cgen_emit's box/unbox handling, JACC v1.x API: JACC.array
    # for a host->device whole-array transfer, JACC.to_host for the
    # reverse -- see the "JACC v1.x API" section of
    # skill-runpod-julia-cuda-jacc's SKILL.md.
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
    # pinned to the v1.x line: jgen_launch_expr emits the v1.x
    # `@parallel_for range=N` macro form (a breaking change from
    # v0.0.x's plain-function-call API) -- an unpinned Pkg.add here
    # would silently drift onto whatever major version is newest,
    # exactly the class of mismatch that motivated this pin.
    return "import Pkg\nhaskey(Pkg.project().dependencies, \"JACC\") || Pkg.add(name = \"JACC\", version = \"1\")\nhaskey(Pkg.project().dependencies, \"Atomix\") || Pkg.add(\"Atomix\")\nimport JACC\nimport Atomix\nJACC.@init_backend\n"
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

# replicates stade_tangent's own three-call pipeline (act_analyze ->
# lin_build -> tgen_emit) directly, rather than calling stade_tangent
# itself -- val_generate_baseline needs a tangent to self-check a
# candidate baseline, and calling into stade_ (the layer that itself
# composes val_'s baseline/validation machinery) would make the two
# layers mutually dependent. Calling the same underlying pipeline
# stages stade_tangent wraps keeps val_ resting only on the codegen
# pipeline, never on the top-level orchestration layer built on it.
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
# parsed straight from source (not a skill-jade sig) -- used to learn
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

function val_random_int_args(sig; lo::Int = 2, hi::Int = 5)
    return Dict{Symbol,Int}(a => rand(lo:hi) for a in sig.args if sig.kinds[a] == :scalar_int)
end

# ---- avoiding near-zero/negative divisors in random baselines ------
#
# A float argument used anywhere as the divisor of a `/` (e.g. a
# "volume"/"weight"/"mass"-like quantity a kernel divides by) has no
# guarantee of staying away from zero -- or even staying positive --
# under plain `randn()`. That's not usually just imprecision: an
# iterative relaxation (Jacobi-style correction loops, etc.) dividing
# by a near-zero or sign-flipped value on every step is a classic
# divergence trigger, unrelated to whether the generated derivative
# code is correct. This is a general, kernel-agnostic property (many
# numerical kernels divide by a physically-positive quantity) rather
# than anything up.jl-specific, so it's detected the same way
# val_arg_ndims detects usage shape: a static scan of `kernel.body`.
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
    # val_grow_shapes sizes every array_float/array_int arg's every
    # dimension to the SAME N (a uniform grid -- see its own comment).
    # array_int content is drawn from 1:idx_cap rather than 1:N: a
    # direct-indexing array_int arg (e.g. a mesh connectivity table
    # indexing straight into another array) is safely in-bounds with
    # idx_cap==N, but a *compressed* id (e.g. a graph node id later
    # scaled by a stride, `(id-1)*n_feat+k`, before it indexes an
    # array) needs idx_cap far below N instead -- val_grow_shapes
    # searches both independently and passes idx_cap through; falling
    # back to N here keeps any caller that never passed idx_cap
    # (direct-indexing behaviour) identical to before. Falls back to
    # 1 if there are no array args at all (so no array_int arg could
    # exist either).
    N = isempty(shapes) ? 1 : minimum(minimum(s) for s in Base.values(shapes))
    cap = idx_cap === nothing ? N : idx_cap
    divisors = val_divisor_args(kernel)
    # a positive-but-wide-spread divisor (e.g. abs(randn())+0.5) still
    # lets the RATIO between two independent divisor-like args (e.g. a
    # "cell volume" divided into a "node volume", as in an iterative
    # relaxation's per-step gain) land far from 1 -- and an iterative
    # loop amplifies whatever that ratio is on every one of its steps.
    # Narrowing the spread specifically for divisor args (still always
    # positive, still random, just closer to a common scale) keeps
    # that per-step gain closer to 1 without hard-coding anything
    # about what the ratio "means" for any particular kernel.
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

# grows one trial size N (same N along every dimension, for every
# array_float/array_int arg) until the primal runs without a
# BoundsError. Oversized dimensions are harmless -- only undersized
# ones are wrong -- so this always converges on a *safe* (if not
# minimal) size without any static index-range analysis, and stays
# fully general across arbitrarily shaped index expressions.
#
# idx_cap (the array_int content range) is searched independently of
# N (the array-size grid), outer loop over idx_cap around the inner
# N loop -- tying them together (idx_cap==N) makes a compressed-id
# argument scaled by a stride before indexing (see val_random_values)
# unsatisfiable at any N, since the array it indirectly indexes needs
# to grow faster than the id range itself. Starting idx_cap small and
# growing it outward still finds direct-indexing kernels' previous
# idx_cap==N solution immediately, since idx_cap<=N is all that case
# ever required.
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

# orchestrates a full random baseline: random ints, a compiled primal
# probed to find safe array sizes, then final random Float64/Int
# content at those sizes. A few retries with fresh int draws guard
# against a rare unlucky combination tripping an error unrelated to
# array sizing -- and, when self_check is on, against a combination
# that runs cleanly but is nonetheless *semantically* degenerate for
# this particular kernel (e.g. integer control args whose implicit
# relationship the kernel never validates, like a multigrid depth
# that doesn't fit the requested fine-grid size). There is no general,
# kernel-agnostic way to detect that up front; instead, a quick
# tangent-vs-finite-difference check is run against the candidate
# baseline itself, and the whole (int_args, shapes, values) triple is
# discarded and redrawn if it fails -- using the tangent oracle as a
# cheap, generic "is this input point even sane" filter, rather than
# encoding any kernel-specific domain knowledge.
function val_generate_baseline(kernel, primal_expr::Expr;
                                scale::Float64 = 1.0, int_lo::Int = 2, int_hi::Int = 5,
                                grow_start::Int = 4, grow_max::Int = 512, attempts::Int = 5,
                                self_check::Bool = true, self_check_trials::Int = 2,
                                self_check_epsilon::Float64 = 1e-6, self_check_rtol::Float64 = 1e-3,
                                max_output_magnitude::Float64 = 1e6)
    primal_fn = val_compile(primal_expr)
    obs_fn = self_check ? val_compile(val_primal_observing_expr(kernel, primal_expr)) : nothing
    tangent_fn = self_check ? val_compile(val_build_tangent(kernel)) : nothing
    last_err = nothing
    # a scalar_int arg is very often an iteration count (a relaxation
    # loop bound, a refinement-pass count, etc.) -- more iterations
    # means more chances for any per-step amplification, however
    # small, to compound into a blown-up result under otherwise
    # perfectly reasonable random data (see the diverging-relaxation
    # comment below). Narrowing the *upper* end of the scalar_int
    # range specifically after a divergence -- rather than just
    # redrawing at the same range -- directly targets that, without
    # assuming anything about what any particular scalar_int arg
    # means. Allowed to fall below the caller's own `int_lo` as a
    # last resort (down to 1, never 0 -- a zero-iteration loop would
    # trivially "pass" by not exercising the kernel at all): a working
    # baseline with a smaller iteration count than requested is far
    # more useful than no baseline at all. Reset for a non-divergence
    # failure, since those aren't evidence the range itself is the
    # problem.
    cur_hi = int_hi
    for attempt in 1:attempts
        int_args = val_random_int_args(kernel.sig; lo = min(int_lo, cur_hi), hi = cur_hi)
        try
            grown = val_grow_shapes(kernel, primal_fn, int_args; start = grow_start, max_size = grow_max)
            values = val_random_values(kernel, grown.shapes, int_args; scale = scale, idx_cap = grown.idx_cap)
            if self_check
                x0 = val_flatten(kernel, values)
                f_eval_vec = x -> val_call_primal_observed(obs_fn, kernel, int_args,
                                                            val_unflatten(kernel, int_args, values, x))
                # a candidate whose own output has blown up (e.g. a
                # Jacobi-style relaxation loop diverging for a random,
                # physically meaningless input) makes h=epsilon finite
                # differences numerically meaningless -- their step's
                # own contribution to the function value falls below
                # floating-point precision at that magnitude, so the
                # FD "reference" itself is garbage, not the exactly-
                # computed adjoint/tangent being compared against it.
                # Reject and redraw before even reaching the tangent
                # self-check below, same as any other bad candidate.
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

# unlike io_read_kernel_bundle (an unkeyed, order-preserving list for
# reading a mixed bag of differently-named generated artifacts), this
# is for reading a *corpus* of original kernel definitions that may
# call each other -- keyed by each kernel's own parsed name, erroring
# on a duplicate, ready to hand straight to inl_inline_calls. Works
# unchanged for a single-kernel file (one entry) too.
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

# shared by io_read_corpus_entry and every stade_*_file writer below:
# resolves which kernel in a corpus is the *entry* -- the one whose
# overall behavior the file is actually about. For a single-kernel
# file that's just its one kernel. For a multi-kernel corpus file, the
# convention every single-kernel file already follows (a file named
# `foo.jl` defines a kernel named `foo`) extends to say the entry
# kernel is named after the file itself.
function io_corpus_entry_name(path::String, kernels::Dict{Symbol,Expr})
    length(kernels) == 1 && return first(keys(kernels))
    entry_name = Symbol(splitext(basename(path))[1])
    haskey(kernels, entry_name) ||
        error("io_corpus_entry_name: $path defines $(length(kernels)) kernels but none is named :$(entry_name) (the file's own basename) -- a multi-kernel file needs an entry kernel named after the file")
    return entry_name
end

# every val_*/stade_validate_* function below only ever takes a
# single primal Expr -- this is the one place that bridges a
# possibly-multi-kernel FILE down to that single Expr, so nothing
# downstream needs to know or care whether the file it came from had
# one kernel or several. For a single-kernel file, returns that
# kernel's Expr exactly as io_read_kernel always did (no inlining --
# there's nothing to inline). For a multi-kernel corpus file, inlines
# the whole call graph (inl_inline_calls) and returns the entry
# kernel (io_corpus_entry_name).
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

# corpus counterpart of io_write_kernel_file: for every name in
# primal_exprs (sorted for a deterministic file layout -- today that's
# a single entry from every stade_*_file caller below, but the
# function stays general), bundles that name's own generated parts,
# if any (already assembled by the caller, e.g. `[tangent_expr]` or
# `[initstacks_expr, adjoint_expr]` -- same shape io_write_kernel_file
# already expects) followed by its primal_exprs entry, exactly as
# handed in -- this function has no opinion on whether that primal is
# an original as-authored body or an inlined one; stade_tangent_file
# et al. pass the entry kernel's fully-inlined primal (the flattened
# body the derivative was actually generated from), so the written
# file is just that one kernel's derivative plus its own primal, with
# no other corpus member appearing at all. Reduces to
# io_write_kernel_file's own output for a single-entry call.
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

# io_read_baseline_yaml is kernel-agnostic (plain text in, numbers
# out) so it has no way to know which `values:` entries are really
# array_int args (e.g. a mesh connectivity table) rather than
# array_float ones -- everything comes back parsed as Float64. Any
# caller that has `kernel` in scope must coerce those specific entries
# back to Int before using them as array indices (round rather than a
# raw Int(...) truncation, purely for exact-integer-valued Float64 ->
# Int robustness against any future non-integral-looking formatting;
# the values were always written as whole numbers in the first place).
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
    # both kwargs accepted, documented, and otherwise ignored --
    # tgen_* never emits push!/pop! at all (every active statement
    # gets a shadow line directly, no stacks), so there is nothing for
    # either keep_push_pop's storage strategy or fuse_ii_loops'
    # fusion to apply to. Pure interface-consistency no-ops, letting a
    # caller iterate uniformly over stade_tangent/stade_adjoint/
    # stade_hvp without special-casing tangent mode. See skill-
    # stade.md's keep_push_pop entry.
    kernel = parse_override_indep_dep(parse_kernel(expr), independents, dependents)
    active_map = act_analyze(kernel)
    lin_plan = lin_build(kernel, active_map)
    return tgen_emit(kernel, lin_plan)
end

# computes the site-level TBR decision from BOTH independently
# duplicated implementations (snap_* and agen_*, per skill-stade's
# purity rule) and asserts they agree exactly before returning either
# -- the permanent guard, so a future silent divergence between the
# two hand-maintained copies fails loudly right here instead of
# corrupting a gradient. Returns (snap_sites, agen_sites): snap_plan
# consumes its own, agen_emit/hvp_emit consume theirs, keeping each
# stage's own duplicate as its sole input per the stage-purity
# convention -- this function is the one place allowed to compare them.
# Always run -- site-level TBR is no longer an opt-in flag, it's how
# every snapshot decision in this file is made, unconditionally.
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
    # Mirrors agen_emit's own post-hoc cleanup exactly (see its
    # comment for the full reasoning), applied here independently
    # because hvp_expr, not adjoint_expr, is what this function
    # returns paired with initstacks_expr. Required for consistency,
    # not just tidiness: stade_adjoint's own initstacks_* (built via
    # agen_emit) already drops a stack once fusion leaves it fully
    # unused, and validation code that shares one kernel's initstacks_*
    # across both its adjoint and hvp calls needs hvp_expr's own
    # signature to match what that shared initstacks_* actually
    # provides. hvp_expr's fwd/bwd portions are built from the exact
    # same agen_forward_body/agen_backward_body calls, same ectx, same
    # ii_plan, as the adjoint path -- hvp_double_body only ever adds
    # shadow push!/pop! calls on separate, unlisted shadow-stack names
    # (never removes or renames an original stack reference), so
    # scanning hvp_expr independently for used stack names lands on
    # the same unused set the adjoint side would find, not a
    # potentially different one.
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

# multi-kernel entry points: inline the whole corpus's call graph away
# (inl_inline_calls), then defer to the existing single-kernel
# function above, unchanged, per kernel. Independents/dependents
# overrides still don't belong here -- a caller who needs them can run
# inl_inline_calls directly and call stade_tangent/stade_adjoint/
# stade_hvp per kernel.
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

# reads any number of kernels from one file (a lone kernel, or a
# corpus of kernels that call each other -- see inl_*), differentiates
# only the corpus's entry kernel (io_corpus_entry_name: the file's own
# basename for a multi-kernel corpus, its one kernel otherwise) against
# its whole call graph inlined away, and writes back out just that one
# kernel: its generated derivative parts, followed by its own INLINED
# primal (the call graph already flattened, exactly the body the
# derivative was generated from) -- not a per-kernel bundle of every
# original definition the corpus happened to contain. One code path
# handles both: a single-kernel file is just a one-entry corpus, where
# inlining is a no-op and this reduces to differentiating that lone
# kernel exactly as before.
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

# Expr in, cuda_plan out -- accepts either a plain skill-jade kernel
# or one of STADE's own generated functions (see cgen_ingest), for
# whichever GPU backend descriptor is passed in. precision=nothing
# (the default) means "use this backend's own default_precision" --
# Float64 (a no-op) for CUDA/AMDGPU, Float32 for Metal. Passing an
# explicit precision overrides that, except for a precision_locked
# backend, where anything but its own default_precision is a hard
# error at generation time rather than a silent guarantee that'll only
# surface as a failure once the caller tries to compile/run the result.
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

# path in, path out. Reads every function def in in_path (a plain
# kernel file, or a stade_tangent_file/stade_adjoint_file output) and
# writes one file: every device kernel first, then every host
# function in original file order. `precision` applies uniformly to
# every function converted in this call (initstacks_/adjoint/primal
# alike, for a stade_adjoint_file output) -- for per-function control,
# call stade_gpu directly on each def instead. The input file on disk
# is only ever read, never rewritten, so precision=nothing is always
# available again on the next call with nothing to reset.
#
# keep_all_atomic (default true, unchanged behavior): pass false to
# let a provably pure scalar reduction (see cgen_idiomatic_scalar_
# reduction/cgen_body's doc comment) generate as a `dot`/`sum(abs2,·)`/
# `mapreduce` call instead of a hand-rolled atomic-accumulate kernel.
# Left at its default, every reduction generates exactly as before --
# useful when stepping through a generated kernel's atomics is itself
# what you're debugging.
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

# JACC has no gpu_backend value at all -- there's only one JACC target
# from cgen_/jgen_'s point of view, since which vendor it actually
# runs on is chosen later, outside this call entirely. precision has
# no locked default here for the same reason (see jgen_* section
# comment): Float64 unless the caller explicitly asks otherwise.
# keep_all_atomic: same meaning as stade_gpu_file's (see its doc
# comment); on the JACC target a matched reduction becomes one
# `JACC.@parallel_reduce` call instead of `@parallel_for` +
# `Atomix.@atomic` -- see jgen_body's doc comment.
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
# Numerically validates a generated tangent/adjoint/hvp file against
# central finite differences of the primal, using a baseline that is
# auto-generated once and cached to a YAML file next to the kernel (or
# a user-supplied one, read via the same public entry point). See the
# val_* banner above for what each mode actually checks.

# the function that reads a baseline YAML and performs the check --
# usable directly by a caller pointing at their own hand-written file.
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
        # under keep_push_pop=false, `initstacks_*`'s own signature is
        # whatever agen_emit actually built it with (a minimal free-var
        # set for Tier A, extended with Phase D's table/total/value
        # args for a Tier B ragged block) -- read it back from the
        # GENERATED Expr itself rather than recomputing it here, so
        # this can never drift from whatever agen_init_emit decided.
        stack_arg_names = keep_push_pop ? Symbol[] : Symbol.(adjoint_out.initstacks.args[1].args[2:end])
        return val_validate_adjoint(kernel, primal_expr, adjoint_out, baseline;
                                     trials = trials, epsilon = epsilon, rtol = rtol, stack_arg_names = stack_arg_names)
    else
        adjoint_out = stade_adjoint(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
        hvp_out = stade_hvp(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
        stack_arg_names = keep_push_pop ? Symbol[] : Symbol.(adjoint_out.initstacks.args[1].args[2:end])
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
                                       self_check::Bool = true)
    primal_expr = io_read_corpus_entry(in_path)
    kernel = parse_kernel(primal_expr)
    baseline = val_generate_baseline(kernel, primal_expr; scale = scale, int_lo = int_lo, int_hi = int_hi,
                                      self_check = self_check)
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

# Sibling to stade_validate_adjoint_file for a third-party (not
# STADE-generated) adjoint, e.g. one produced by Tapenade: same
# baseline machinery and the same val_validate_adjoint oracle,
# but the adjoint/initstacks come from `adjoint_path` instead of
# from calling stade_adjoint on the primal. stade_validate_adjoint_file
# itself can't be reused as-is for this -- it always regenerates
# STADE's own adjoint internally and has no way to take an adjoint
# file as input -- so this reuses everything beneath it instead:
# io_read_baseline_yaml/stade_generate_baseline_file for the baseline,
# and val_validate_adjoint (with its stack_arg_names hook, added for
# exactly this case) for the numerical check itself.
# Expects `adjoint_path` to bundle a `<name>_b` function and an
# `initstacks_<name>_b` function (any argument list), matching
# Tapenade's own naming convention.
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

let
    helper = :(function stub_scale(a, b)
        tmp = a * b
        a = tmp
        return nothing
    end)
    caller = :(function stub_caller(x, y, n)
        stub_scale(x, y)
        return nothing
    end)
    kernels = Dict{Symbol,Expr}(:stub_scale => helper, :stub_caller => caller)

    inlined = inl_inline_calls(kernels)
    caller_src = string(inlined[:stub_caller])
    @assert occursin("tmp_stub_scale_c1", caller_src)   # renamed local spliced in
    @assert !occursin("stub_scale(x, y)", caller_src)   # call statement gone

    tangent_out = stade_tangent_corpus(kernels)
    adjoint_out = stade_adjoint_corpus(kernels)
    hvp_out     = stade_hvp_corpus(kernels)
    @assert tangent_out[:stub_caller] isa Expr
    @assert adjoint_out[:stub_caller].adjoint isa Expr && adjoint_out[:stub_caller].initstacks isa Expr
    @assert hvp_out[:stub_caller].hvp isa Expr && hvp_out[:stub_caller].initstacks isa Expr
    println("STADE.jl inl_* stage round-tripped a two-kernel stub call graph through all three codegen modes OK")
end

let
    trivial = :(function stub(x, n, y)
        return nothing
    end)
    tangent_out = stade_tangent(trivial)
    adjoint_out = stade_adjoint(trivial)

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

    f32_plan = stade_cuda(:(function stub2(u, v, n)
        for i_x = 1:n
            v[i_x] = 2.5 * u[i_x]
        end
        return nothing
    end); precision = Float32)
    @assert any(l -> l isa Float32, f32_plan.kernels[1].args[2].args) ||
            occursin("2.5f0", string(f32_plan.kernels[1]))
    f64_plan = stade_cuda(:(function stub2(u, v, n)
        for i_x = 1:n
            v[i_x] = 2.5 * u[i_x]
        end
        return nothing
    end))
    @assert occursin("2.5", string(f64_plan.kernels[1])) && !occursin("2.5f0", string(f64_plan.kernels[1]))
    println("precision=Float32 downcasts generated code OK; default stays Float64 OK")

    metal_plan = stade_metal(:(function stub3(u, v, n)
        for i_x = 1:n
            v[i_x] = u[i_x] / 2.0 + sqrt(2.0) * 1.5
        end
        return nothing
    end))
    @assert String(metal_plan.host.args[1].args[1]) == "stub3_metal"
    ksrc = string(metal_plan.kernels[1])
    # no more operand-forcing T(...) wraps (see cgen_convert_precision's
    # comment on why) -- just the literal walk: every bare Float64
    # literal in the source (2.0, 2.0, 1.5) downcasts to f0, and nothing
    # else is rewritten
    @assert occursin("thread_position_in_grid", ksrc) && occursin("1.5f0", ksrc) &&
            occursin("2.0f0", ksrc) && !occursin("Float32(", ksrc)
    @assert occursin("@metal", string(:(@metal threads = 1 groups = 1 f()))) # sanity on macro name only
    threw = false
    try
        stade_metal(:(function stub3(u, v, n) return nothing end); precision = Float64)
    catch
        threw = true
    end
    @assert threw
    println("cgen_* round-tripped a kernel through the Metal backend OK; precision_locked correctly rejected Float64")

    jacc_plan = stade_jacc(:(function stub4(u, v, n)
        for i_x = 1:n
            v[i_x] = v[i_x] + 2.0 * u[i_x]
            v[1] = v[1] + u[i_x]
        end
        return nothing
    end))
    @assert String(jacc_plan.host.args[1].args[1]) == "stub4_jacc"
    hsrc = string(jacc_plan.host)
    @assert occursin("JACC.@parallel_for", hsrc) && occursin("range = ", hsrc)
    ksrc = string(jacc_plan.kernels[1])
    @assert occursin("Atomix.@atomic", ksrc) && !occursin("threadIdx", ksrc) && !occursin("blockIdx", ksrc)
    println("jgen_* round-tripped a kernel through the JACC target OK (split loop -> parallel_for + kernel, atomic write via Atomix)")

    # cgen_reduction_only_loop: a pure scalar reduction (`i_seq_`,
    # commutative/associative accumulation only) is now split onto the
    # device with an atomic add against the boxed accumulator, exactly
    # like the real-GPU failure mode this feature fixes for a bare
    # dot-product-style kernel -- no more "for i_seq_t" left on the
    # host for this shape.
    reduce_plan = stade_cuda(:(function stub5(u, v, n)
        acc = 0.0
        for i_seq_t = 1:n
            acc = acc + u[i_seq_t]
        end
        v[1] = acc
        return nothing
    end))
    rhsrc = string(reduce_plan.host)
    @assert occursin("@cuda", rhsrc) && !occursin("for i_seq_t", rhsrc)
    rksrc = string(reduce_plan.kernels[1])
    @assert occursin("CUDA.@atomic", rksrc)
    println("cgen_reduction_only_loop split a pure scalar-reduction i_seq_ loop onto the device OK (atomic accumulate, no host loop left)")

    # A per-iteration-unique array accumulation (a reverse-sweep
    # shape: ub[i]=ub[i]+... -- read/write at the SAME index
    # every time, so NOT a recurrence) also splits, with a plain
    # (non-atomic) per-thread write, since the index is injective in
    # the loop var -- matches cgen_device_assign's existing trusted
    # fast path for independent loops' adjoint accumulators.
    unique_plan = stade_cuda(:(function stub7(ub, v, lossb, n)
        for i_seq_x = n:-1:1
            ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
        end
        return nothing
    end))
    @assert occursin("@cuda", string(unique_plan.host)) && !occursin("for i_seq_x", string(unique_plan.host))
    uksrc = string(unique_plan.kernels[1])
    @assert !occursin("CUDA.@atomic", uksrc)
    println("cgen_reduction_only_loop split a per-thread-unique array accumulation onto the device OK (plain write, no atomic needed, no host loop left)")

    # A scatter-add accumulation through a GATHERED (data-dependent)
    # index -- a graph kernel's edge-to-node aggregation shape: the
    # target index comes from an array read, so two different edges
    # (threads) CAN land on the same `agg_off` (unlike the `ub[i_seq_x]`
    # case just above, where the index IS the loop var). This must NOT
    # take the plain-write fast path -- it needs an atomic add, since
    # per-thread-uniqueness was never proven, only "depends on the
    # thread var" was. This is the exact case cgen_reduction_only_loop's
    # own comment names as accepted-but-unproven; injective_dep's job is
    # making sure it's proven safe (via atomic) rather than left as an
    # assumption.
    scatter_plan = stade_cuda(:(function stub8(agg, dst, msg, n)
        for i_seq_e = 1:n
            d = dst[i_seq_e]
            agg[d] = agg[d] + msg[i_seq_e]
        end
        return nothing
    end))
    @assert occursin("@cuda", string(scatter_plan.host)) && !occursin("for i_seq_e", string(scatter_plan.host))
    sksrc = string(scatter_plan.kernels[1])
    @assert occursin("CUDA.@atomic", sksrc)
    println("cgen_reduction_only_loop correctly wraps a gather-indexed scatter-add in an atomic OK (index depends on thread var but isn't provably injective -- graph-scatter regression guard)")

    # Same scatter-add shape, JACC target -- jgen_device_assign now
    # carries the same thread_dep/injective_dep tracing as cgen_'s, so
    # this must also come out atomic rather than plain.
    scatter_plan_jacc = stade_jacc(:(function stub8j(agg, dst, msg, n)
        for i_seq_e = 1:n
            d = dst[i_seq_e]
            agg[d] = agg[d] + msg[i_seq_e]
        end
        return nothing
    end))
    jksrc = string(scatter_plan_jacc.kernels[1])
    @assert occursin("Atomix.@atomic", jksrc)
    println("cgen_reduction_only_loop correctly wraps a gather-indexed scatter-add in an atomic OK (JACC target parity)")

    # A genuine recurrence (array read of a value a PREVIOUS iteration
    # of the same i_seq_ loop wrote, e.g. a shift recurrence `u[i]=c*u[i-1]`)
    # must stay sequential/host-side -- cgen_reduction_only_loop
    # refuses to split it because the read and write of `u` happen at
    # two DIFFERENT loop-var-dependent index expressions (i_seq_k vs
    # i_seq_k-1), so no per-thread trick could fix it.
    recur_plan = stade_cuda(:(function stub6(u, c, n)
        for i_seq_k = 2:n
            u[i_seq_k] = c * u[i_seq_k - 1]
        end
        return nothing
    end))
    @assert occursin("for i_seq_k", string(recur_plan.host)) && isempty(recur_plan.kernels)
    println("cgen_reduction_only_loop correctly refused to split a genuine recurrence OK (stays host-side, unchanged from prior behavior)")

    # A sweep-style recurrence (a Jacobi-style relaxation, or a
    # timestep loop reading its own previous step): the outer i_seq_
    # variable itself never appears in either index -- the coupling is
    # entirely through an INNER loop's index, reading a NEIGHBOR's
    # value written during the
    # previous outer sweep (mup[i_k] written at i_k, but up read at
    # nbr[i_k] -- a genuinely different index expression). Must still
    # be refused.
    sweep_plan = stade_cuda(:(function stub8(up, mup, nbr, n)
        for i_seq_sweep = 1:10
            for i_k = 1:n
                i_nbr = nbr[i_k]
                mup[i_k] = 0.5 * up[i_nbr]
            end
            for i_k = 1:n
                up[i_k] = mup[i_k]
            end
        end
        return nothing
    end))
    @assert occursin("for i_seq_sweep", string(sweep_plan.host))
    println("cgen_reduction_only_loop correctly refused a sweep-style recurrence carried through an inner loop's index OK (stays host-side)")

    # A real failure mode this resolves, via the convergent-constant
    # extension: `wb` is additively self-referencing (a pure-reduction
    # shape on its own) AND has a fresh reset later in the SAME body
    # on every control-flow path -- cgen_locally_assigned_scalars
    # correctly calls that thread-private (a common legitimate
    # per-thread-scratch idiom), and its only textual initializer sits
    # OUTSIDE the loop.
    # But every path through the loop body resets it to the SAME
    # literal (0.0) that its pre-loop initializer ALSO assigns -- so
    # cgen_reduction_only_loop can prove wb is 0.0 at the top of every
    # iteration and synthesize that as the kernel body's first
    # statement. Confirmed against a live GPU run of the naive
    # (pre-extension) version of this shape crashing both backends
    # (JACC threw KernelException; CUDA generated the exact same
    # use-before-def kernel silently) -- this test locks in the fix.
    wb_plan = stade_cuda(:(function stub9(loss, ub, u, n)
        wb = 0.0
        for i_seq_x = n:-1:1
            wb = wb + loss[1]
            if u[i_seq_x] > 0.0
                ub[i_seq_x] = ub[i_seq_x] + wb
                wb = 0.0
            else
                wb = 0.0
            end
        end
        return nothing
    end))
    @assert occursin("@cuda", string(wb_plan.host)) && !occursin("for i_seq_x", string(wb_plan.host))
    wbksrc = string(wb_plan.kernels[1])
    @assert occursin("wb = 0.0", wbksrc)
    println("cgen_reduction_only_loop split a convergent-constant wb shape OK (convergent-constant proof: synthesized wb=0.0 at kernel start)")

    # Negative case: the loop's OWN convergent constant (0.0) doesn't
    # match what the pre-loop initializer actually assigns (1.0 here)
    # -- unprovable, must stay host-side. (A contrived example: real
    # STADE-generated code wouldn't disagree like this, but the
    # eligibility check can't assume that -- it must verify.)
    wb_mismatch_plan = stade_cuda(:(function stub11(loss, ub, u, n)
        wb = 1.0
        for i_seq_x = n:-1:1
            wb = wb + loss[1]
            if u[i_seq_x] > 0.0
                ub[i_seq_x] = ub[i_seq_x] + wb
                wb = 0.0
            else
                wb = 0.0
            end
        end
        return nothing
    end))
    @assert occursin("for i_seq_x", string(wb_mismatch_plan.host))
    println("cgen_reduction_only_loop correctly refused when the pre-loop value doesn't match the loop's convergent constant OK (stays host-side)")

    # Negative case: pre-loop initializer isn't a literal at all (some
    # arg-derived value) -- no known_consts entry to check against, so
    # no proof is possible regardless of what the loop converges to.
    wb_nonliteral_plan = stade_cuda(:(function stub12(loss, ub, u, w0, n)
        wb = w0
        for i_seq_x = n:-1:1
            wb = wb + loss[1]
            if u[i_seq_x] > 0.0
                ub[i_seq_x] = ub[i_seq_x] + wb
                wb = 0.0
            else
                wb = 0.0
            end
        end
        return nothing
    end))
    @assert occursin("for i_seq_x", string(wb_nonliteral_plan.host))
    println("cgen_reduction_only_loop correctly refused when the pre-loop value isn't a literal OK (stays host-side)")

    # Regression guard: a PURE reduction var with no reset anywhere in
    # the loop body (dotprod_b's/stub5's `acc`) must still split --
    # it's boxed+atomic, so its correct initial value comes from the
    # host-side box regardless of where in the body its self-reference
    # sits, unlike wb above.
    pure_reduce_plan = stade_cuda(:(function stub10(u, v, n)
        acc = 0.0
        for i_seq_t = 1:n
            acc = acc + u[i_seq_t]
        end
        v[1] = acc
        return nothing
    end))
    @assert occursin("@cuda", string(pure_reduce_plan.host)) && !occursin("for i_seq_t", string(pure_reduce_plan.host))
    println("cgen_reduction_only_loop still splits a pure reduction var with no in-loop reset OK (regression guard)")
end
# ============================================================
# ii_* -- eligibility analysis and codegen for fusing iteration-
# independent-loop adjoint generation ("II-loop fusion", named after
# Tapenade's II_LOOP), enabled via fuse_ii_loops=true.
#
# Classification granularity: outermost eligible loop first. A loop
# nest is classified as a single fusion unit whenever the outer loop's
# own full body (including everything nested inside it) passes; only
# if the outer loop fails does the walk recurse into its direct :for
# children to look for a smaller eligible unit. This lets a value
# built by one sibling inner loop and consumed by another, both nested
# in the same outer iteration, classify as :independent without any
# special-casing: it never escapes the outer loop's own body, only an
# inner one, and escape is checked at the granularity being tested.
# ============================================================

# ---- shared helpers (not duplicated -- pure structural walks, not a
# TBR/codegen decision, so Hard Rule 7's duplication rationale
# doesn't apply to them any more than agen_site_key's already-shared
# status does) ----
#
# Escape detection tracks program order rather than doing a flat
# "does this name get read anywhere else in the kernel" check, which
# is unsound: the same variable name can be reused, entirely
# unrelated, in a different loop nest elsewhere in the kernel, and a
# flat check would wrongly flag a later loop's own fresh, non-self-
# referencing overwrite as an escaping read of the earlier loop's
# value. A fresh overwrite kills the dependency -- nothing downstream
# of it can be reading the earlier loop's contribution any more.

# Walks `body` in forward program order, threading `alive` (vars
# still known to carry `target`'s own contribution) exactly the way
# snap_fwd_walk!/agen_fwd_walk_loop! thread `seen` -- returns the
# updated alive-set (their OUT set) and accumulates escapes into
# `escaped` in place. A var is removed from `alive` the moment a
# FRESH (non-self-referencing) assignment to it is seen (it no longer
# carries target's value from that point on); a read of a still-alive
# var is recorded as an escape. `:if` is handled conservatively: a var
# survives past the branch only if it survives BOTH arms, and a read
# inside either arm is still an escape regardless of which arm
# actually runs at runtime.
#
# True occurrence detector -- unlike agen_var_value_needed!/snap_var_
# value_needed!, this counts a read regardless of whether it's linear
# (a +/- operand) or nonlinear. This matters because a purely LINEAR
# downstream read of a fused var (a straight copy, no +/-/* involved)
# still contributes to that var's shadow when its own backward code
# runs (the adjoint of an identity copy is itself an identity pass-
# through, not zero), and that contribution must land before a fused
# loop's own backward code reads the shadow -- which fusion cannot
# guarantee for anything outside the loop, regardless of whether the
# read was linear. The "value-needed" (nonlinear-only) concept this
# file uses elsewhere answers a different, narrower question (does
# the OLD value need protecting for someone else's own partial
# derivative), correct for its own purpose but the wrong tool here.
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

# Walks `body` in forward program order, threading `alive` (vars
# still known to carry `target`'s own contribution) exactly the way
# snap_fwd_walk!/agen_fwd_walk_loop! thread `seen` -- returns the
# updated alive-set (their OUT set) and accumulates escapes into
# `escaped` in place. A var is removed from `alive` the moment a
# FRESH (non-self-referencing) assignment to it is seen (it no longer
# carries target's value from that point on); ANY read (linear or
# nonlinear -- see ii_expr_reads) of a still-alive var is recorded as
# an escape. `:if` is handled conservatively: a var survives past the
# branch only if it survives BOTH arms (matches snap_fwd_walk!'s own
# conservative merge elsewhere in this file), and a read inside either
# arm is still an escape regardless of which arm actually runs at
# runtime. A write's own LHS index expressions (for an array target)
# are checked too -- an alive scalar used as an index elsewhere is
# still a genuine read of it.
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
#
# A sound scope check has to handle two different cases depending on
# whether an ancestor level is the literal kernel top level (executes
# once, no wraparound) or itself a repeating `:for` (executes possibly
# many times, so a read positioned textually BEFORE target within that
# same ancestor body can still observe target's contribution -- on the
# next ancestor iteration).
#
# `:if` ancestors are handled too (a target loop living inside an `:if`
# branch). An `:if` branch does not repeat on its own -- one evaluation
# runs at most one of its two branches exactly once -- so within the
# branch itself there is no wraparound: `repeating = false` for a
# branch-body level, unconditionally. The `:if` statement's own
# position within its own container, one level further out, follows
# the ordinary :for-vs-kernel-top rule instead -- if the whole `:if`
# sits inside a repeating ancestor, it gets re-evaluated each
# iteration, the same wraparound concern a bare :for ancestor already
# has, one level further out. The sibling branch is never examined at
# all: if `:then` ran (meaning target ran), `:els` provably did not
# run in that same evaluation, so nothing inside it could ever observe
# target's contribution. This falls out automatically from only ever
# recursing into the one branch that actually contains target.
#
# `ii_find_ancestor_path(body, target, body_repeats)` returns the
# chain of (containing body, index, does-this-body-itself-repeat)
# triples from innermost to outermost, ending at the kernel body
# itself (whose own `body_repeats` is always false). Every :for and
# :if ancestor is covered, so `nothing` only means target isn't
# reachable from `body` at all.
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

# vars from `vars` that escape `target`, handling `target` at any
# `:for`/`:if`-mixed nesting depth (falls back to the top-level-only
# case when target is already top-level, since then the path has
# length 1 and that single level's `body_repeats` is false).
#
# At each level whose containing body itself repeats: a full
# wraparound pass (same-position "after" siblings, then wrapping to
# "before" siblings on the next repetition) detects every reachable
# read -- one pass suffices, since `alive` is a fixed set of names
# established once before this runs, not something that itself needs
# to converge. What propagates outward to the next ancestor level uses
# only the same-position "after" segment's kill effect: a consumer
# outside this entire level only ever observes whatever the last
# repetition's own tail end left behind -- a "before" kill would only
# matter for a next repetition that, for the last one, never happens,
# so it must not be allowed to protect the var from being checked
# further out. A non-repeating level (an :if branch taken in
# isolation, or the kernel body itself) skips the wraparound pass
# entirely -- there is no next repetition to wrap around to.
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

# True iff `body` (recursively, through :for/:if) writes to any
# active array that is read -- at all, linear or nonlinear, see
# ii_expr_reads's own comment for why nonlinear-only is the wrong
# test -- anywhere in `kernel_body` outside `body` itself.
#
# This matters because agen_emit_ii_loop fuses a loop's entire
# backward differentiation, not just the scalar vn_ind/vn_red subset
# proven safe. An array write inside the loop body whose array is read
# anywhere else would get its backward code fused too, moving it to
# run before its true downstream consumer's own (logically later,
# hence backward-sweep-earlier) code has run -- silently wrong, not an
# error. This refuses the whole loop rather than fusing only the
# proven-safe statements and leaving the array write at its normal
# position; the latter would need agen_backward_body to skip
# individual statements within a body, not whole bodies -- real,
# not-yet-attempted future work (see the plan notes for how much
# stack-elimination coverage this specifically costs today).
#
# Order-aware, mirroring ii_escapes_nested's own design for scalars,
# rather than a blanket "read anywhere else in the kernel" check: a
# read positioned before the loop, at a non-repeating level, could
# never actually observe this write's contribution, since it already
# ran before the write happened. Reuses ii_find_ancestor_path exactly
# as the scalar side does, with the same repeating-vs-not distinction
# (a repeating ancestor needs the wraparound "after ++ before" check;
# the literal kernel top level only needs "after").
#
# Deliberately does NOT model any "kill" for arrays, unlike the scalar
# side's fresh-overwrite-kills-alive logic: proving a later write to
# the same array safely overwrites (kills) whatever this write
# contributed would require proving index equality between the two
# writes, which this analysis doesn't attempt -- so an array, once
# checked, stays "alive" all the way out to the kernel top level, with
# no early exit. This is strictly more conservative than the scalar
# side, and deliberately so.
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
#
# A snapshot exists to carry a primal value from forward time to
# backward time. The alternative is to recompute it there instead --
# which moves no differentiation at all, so unlike fusion it stays
# sound in the presence of an escaping array write. `ii_recomputable`
# is the proof obligation: re-executing `loop_body`'s own statements
# at the backward position must reproduce `var`'s forward value.
#
# The cone of inputs must bottom out only in things whose forward
# value is still intact when the backward sweep arrives: values never
# assigned anywhere in the kernel, this nest's own loop indices, or
# other members of the recomputed chain. Two deliberate
# conservatisms, both of which have bitten this codebase before in
# other guises:
#   - An ARRAY is refused whenever it's assigned anywhere in the
#     kernel, even if the writes are inside this same loop: proving a
#     read observes this loop's own write rather than a foreign one
#     needs index reasoning this analysis doesn't attempt.
#   - A SCALAR is refused if it is LIVE-IN to loop_body -- read before
#     any write of it, so re-execution would restart from whatever the
#     variable happens to hold at backward time rather than from its
#     forward-time initial value. This is what separates a reduction
#     accumulator whose reset sits outside the loop (live-in, refused)
#     from the same accumulator classified one level further out, with
#     its reset inside (not live-in, accepted). Note that a scalar
#     assigned again in some later, unrelated part of the kernel is
#     NOT a problem: the recompute overwrites it before reading it.
#     Only the ordering within loop_body matters.
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
            v in written && return false
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

# True iff any var in `vars` is assigned inside a nested :for of
# `body` (at any depth below body's own top level).
#
# agen_emit_ii_loop builds its fused loop as vcat(fwd, bwd) at the
# CLASSIFIED loop's own level only: the entire forward nest runs, then
# the entire backward nest. A fused scalar assigned inside a nested
# :for therefore holds only its LAST inner-iteration value by the time
# the backward nest reads it -- every backward iteration then uses the
# wrong primal. (A var assigned at the classified body's own top level
# is fine: it holds one value for the whole iteration, whatever the
# nest below reads it. An :if is fine too -- both halves re-evaluate
# the same branch.) Refusing here rather than teaching
# agen_emit_ii_loop to interleave per level is the conservative
# choice; interleaving is real, not-yet-attempted future work.
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

# True iff some scalar assignment in `body` reads an array that `body`
# itself writes. agen_ii_recompute_stmts drops array writes from the
# backward-position recompute, so such a scalar would be rebuilt from
# the array's POST-forward contents rather than the value it held when
# the statement first ran -- identical for a write-once-per-index
# array, different for an accumulating one, and not worth
# distinguishing here.
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

# True iff `body` still contains a push site after the vn_local
# exclusion -- a snapshot, branch flag, or tripcount that
# agen_forward_body would emit anyway.
#
# This only matters for the classifications that execute the body
# TWICE: `:reduction`/`:mixed` keep the ordinary forward loop AND emit
# agen_emit_ii_loop (primal ++ backward) at the backward position, so
# any surviving push runs in both, leaving the forward one orphaned
# and restoring the wrong value in the second. `:independent` replaces
# the forward loop entirely, so its push and pop stay matched within
# the one fused iteration and it needs no such gate.
#
# Keyed by the SITE-level TBR decisions, exactly as agen_forward_body's
# own push gate is (agen_push_pop_source reads ectx.push_pop, which is
# this same Dict). A whole-variable approximation is unusable here: a
# pure accumulation like `res[i] = res[i] + auxres` is value-needed as
# a variable yet emits no push at all, and treating it as one refuses
# every loop containing a scatter-accumulate -- which is most of the
# loops this analysis exists to classify. `exempt` is deliberately not
# consulted: agen_collect_exempt_vars! never exempts a write inside a
# loop, which every site here is.
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
    synth = cgen_reduction_only_loop(stmt.body, stmt.var, known_consts)
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
    vn_red = intersect(vn_local, redvars)
    vn_ind = setdiff(vn_local, redvars)
    # Each half is checked independently against ITS OWN safety
    # condition rather than requiring the whole loop to be purely one
    # shape or the other -- a loop can genuinely contain both a pure
    # reduction accumulator and a fully-contained independent chain at
    # once (they don't interact), and there's no reason to refuse the
    # whole loop just because it isn't homogeneous.
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
    synth = cgen_reduction_only_loop(stmt.body, stmt.var, known_consts)
    synth === nothing && return :none
    local_names = cgen_locally_assigned_scalars(stmt.body)
    redvars = cgen_scalar_reduction_vars(stmt.body)
    vn_local = Set(v for v in intersect(value_needed, union(local_names, redvars)) if get(active_map, v, false))
    isempty(vn_local) && return :none
    # see ii_fused_var_in_nested_for -- vcat(fwd, bwd) is only valid
    # when no fused var is live across a nested loop boundary.
    ii_fused_var_in_nested_for(stmt.body, vn_local, plan) && return :none
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

# `known_consts` is built LOCALLY, fresh at the top of every call (one
# per body-list), mirroring cgen_body's own exact convention rather
# than being threaded down from a caller -- each body-list gets its
# own independent Dict, populated only by literal scalar assigns
# preceding a candidate loop within that same body. This is required,
# not just tidier: with known_consts always empty, an unrelated self-
# referencing-with-reset accumulator anywhere in a loop's body causes
# cgen_reduction_only_loop to refuse the ENTIRE loop outright -- even
# when the var this analysis actually cares about (a genuinely
# independent, value-needed scalar elsewhere in the same body) has
# nothing to do with that unrelated accumulator at all.
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

# ---- ii_* stress tests -------------------------------------------
# Regression guards for classification edge cases found while
# hardening the eligibility analysis.

let
    # A loop nested under :if, fully contained (no escape).
    k = parse_kernel(:(function ii_stress_ifnest(x, y, z, cond_arr, n, out)
        if cond_arr[1] > 0.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
        else
            for i = 1:n
                z[i] = z[i] + x[i]
            end
        end
        out[1] = x[1]
        return nothing
    end))
    plan = stade_ii_plan_check(k)
    @assert length(plan) == 1 && only(values(plan)) == :independent "ii_stress_ifnest: expected one :independent site, got $plan"
    println("ii_plan correctly classifies a fully-contained loop nested under :if OK")

    # A loop mixing a genuine reduction var (self-referencing, no
    # fresh reset) with a genuine, locally-contained independent var
    # as its value-needed writes -- the two don't interact, so both
    # halves are checked against their own condition independently
    # rather than requiring the whole loop to be homogeneous.
    k2 = parse_kernel(:(function ii_stress_mixed(x, y, n, acc_out)
        acc = 0.0
        for i = 1:n
            t = x[i] * x[i]
            y[i] = y[i] + t * t
            acc = acc + x[i]
        end
        acc_out[1] = acc * acc
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert length(plan2) == 1 && only(values(plan2)) == :mixed "ii_stress_mixed: expected one :mixed site, got $plan2"
    println("ii_plan correctly classifies a loop mixing reduction and independent value-needed vars as :mixed OK")

    # Positive control, run alongside the two cases above so a future
    # change that breaks classification entirely (e.g. returns :none
    # for everything) doesn't silently pass by having nothing left to
    # fail: a clean top-level independent loop must still classify.
    k3 = parse_kernel(:(function ii_stress_positive_control(x, y, n)
        for i = 1:n
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
        end
        return nothing
    end))
    plan3 = stade_ii_plan_check(k3)
    @assert length(plan3) == 1 && only(values(plan3)) == :independent "ii_stress_positive_control: expected exactly one :independent site, got $plan3"
    println("ii_plan still correctly classifies a clean top-level independent loop OK (positive control)")
end

# ---- ii_* adversarial regression tests ----------------------------
# These deliberately target specific soundness/completeness
# dimensions of agen_ii_classify/ii_escapes that don't necessarily
# show up in ordinary kernels -- corpus testing alone doesn't
# establish general correctness on its own.

let
    # A1: a genuine cross-loop dependency (var read by later code with
    # no intervening fresh overwrite) must still be caught as an escape.
    k = parse_kernel(:(function ii_adv_genuine_escape(x, y, n, out)
        for i = 1:n
            t = x[i] * x[i]
            y[i] = y[i] + t
        end
        out[1] = t * t
        return nothing
    end))
    plan = stade_ii_plan_check(k)
    @assert isempty(plan) "ii_adv_genuine_escape: expected refusal (t genuinely escapes with no intervening kill), got $plan"

    # A2: a genuine array recurrence buried inside a loop whose scalar
    # content looks perfectly independent -- the array-index-mismatch
    # check must still refuse the whole loop regardless of how clean
    # the scalar part is.
    k2 = parse_kernel(:(function ii_adv_buried_recurrence(u, x, n)
        for i = 2:n
            t = x[i] * x[i]
            s = t / 2.0
            u[i] = u[i - 1] + s * s
        end
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert isempty(plan2) "ii_adv_buried_recurrence: expected refusal (genuine array recurrence), got $plan2"

    # A3: trip count bound reassigned elsewhere in the kernel -- must
    # still classify, documenting a deliberate invariant: a fused loop
    # only ever executes its own header once, at the same point in
    # program order the primal always did, so it never needs the
    # bound protected against a later reassignment the way a
    # separately-reversed second traversal would. If codegen ever
    # starts re-reading a bound a second time, this assertion should
    # catch it.
    k3 = parse_kernel(:(function ii_adv_unstable_tripcount(x, y, n, m_arg)
        m = m_arg
        for i = 1:m
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
        end
        m = 0
        return nothing
    end))
    plan3 = stade_ii_plan_check(k3)
    @assert length(plan3) == 1 && only(values(plan3)) == :independent "ii_adv_unstable_tripcount: expected one :independent site, got $plan3"

    # A4: an independent scalar with two fresh assignment sites in the
    # same iteration, on different :if arms -- must still classify.
    k4 = parse_kernel(:(function ii_adv_two_assign_sites(x, y, n)
        for i = 1:n
            t = 0.0
            if x[i] > 0.0
                t = x[i] * x[i]
            else
                t = -x[i] * x[i]
            end
            y[i] = y[i] + t * t
        end
        return nothing
    end))
    plan4 = stade_ii_plan_check(k4)
    @assert length(plan4) == 1 && only(values(plan4)) == :independent "ii_adv_two_assign_sites: expected one :independent site, got $plan4"

    # A5: a value-needed array write (not scalar) inside an otherwise-
    # eligible loop, consumed later elsewhere -- the array write must
    # stay out of scope by construction (array-ref lhs is never in
    # local_names/redvars), and the loop's own fully-contained scalar
    # must not be affected by the array's own escape.
    k5 = parse_kernel(:(function ii_adv_array_escape(x, arr, n, out)
        for i = 1:n
            t = x[i] * x[i]
            arr[i] = arr[i] + t
        end
        out[1] = arr[1] * arr[1]
        return nothing
    end))
    plan5 = stade_ii_plan_check(k5)
    @assert isempty(plan5) "ii_adv_array_escape: expected empty, got $plan5"

    println("ii_plan adversarial regression suite (5 corpus-independent cases) OK")
end

# ---- ii_* nested-:for extension regression tests -------------------
# Target a loop nested inside a genuinely ineligible ancestor still
# being judged on its own.

let
    # NF1: inner loop's locals fully contained -- outer i_seq_k fails
    # (genuine array recurrence, u[k]=u[k-1]+1.0), forcing the walker
    # to judge the inner `for i` loop on its own. Must now succeed.
    k1 = parse_kernel(:(function ii_nf1_contained(u, x, y, n, m)
        for i_seq_k = 2:m
            u[i_seq_k] = u[i_seq_k - 1] + 1.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
        end
        return nothing
    end))
    plan1 = stade_ii_plan_check(k1)
    @assert length(plan1) == 1 && only(values(plan1)) == :independent "ii_nf1_contained: expected one :independent site, got $plan1"

    # NF2: wraparound escape -- a nonlinear read of `t` positioned
    # BEFORE the inner loop, within the SAME repeating i_seq_k body,
    # reachable on the NEXT ancestor iteration. Must stay refused.
    k2 = parse_kernel(:(function ii_nf2_wraparound(u, x, y, n, m, out)
        for i_seq_k = 2:m
            out[i_seq_k] = t * t
            u[i_seq_k] = u[i_seq_k - 1] + 1.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
        end
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert isempty(plan2) "ii_nf2_wraparound: expected refusal (wraparound escape), got $plan2"

    # NF3: forward escape -- nonlinear read of `t` AFTER the inner
    # loop, same repeating ancestor body, no wraparound needed. Must
    # stay refused.
    k3 = parse_kernel(:(function ii_nf3_forward(u, x, y, n, m, out)
        for i_seq_k = 2:m
            u[i_seq_k] = u[i_seq_k - 1] + 1.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
            out[i_seq_k] = t * t
        end
        return nothing
    end))
    plan3 = stade_ii_plan_check(k3)
    @assert isempty(plan3) "ii_nf3_forward: expected refusal (forward escape), got $plan3"

    # NF4: two levels of :for nesting below the ineligible ancestor --
    # confirms multi-level path propagation, not just single-level.
    k4 = parse_kernel(:(function ii_nf4_two_levels(u, x, y, n, m, p)
        for i_seq_k = 2:m
            u[i_seq_k] = u[i_seq_k - 1] + 1.0
            for k2 = 1:p
                for i = 1:n
                    t = x[i] * x[i]
                    s = t / 2.0
                    y[i] = y[i] + s * s
                end
            end
        end
        return nothing
    end))
    plan4 = stade_ii_plan_check(k4)
    # both the inner loop and the k2 loop wrapping it now classify: the
    # walker is post-order, so k2 sees its nested loop already covered
    # and is no longer refused by ii_fused_var_in_nested_for. The larger
    # unit is strictly better; what matters is that both are eligible.
    @assert length(plan4) == 2 && all(v -> v == :independent, values(plan4)) "ii_nf4_two_levels: expected the inner loop AND its k2 wrapper to classify, got $plan4"

    println("ii_plan nested-:for extension regression suite (4 cases) OK")
end

# ---- ii_* :if-nesting extension regression tests -------------------

let
    # IF1: fully-contained locals in an :if.then branch -- must
    # classify. `else` uses a SEPARATE array so a same-name self-
    # reference in the sibling branch doesn't trip the array-escape
    # check's own branch-unaware scope.
    k1 = parse_kernel(:(function ii_if1_contained(x, y, z, n, cflag)
        if cflag[1] > 0.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
        else
            z[1] = z[1] + 1.0
        end
        return nothing
    end))
    plan1 = stade_ii_plan_check(k1)
    @assert length(plan1) == 1 && only(values(plan1)) == :independent "ii_if1_contained: expected one :independent site, got $plan1"

    # IF2: nonlinear read of `t` AFTER the loop but still within the
    # SAME branch, before the :if closes. Must stay refused.
    k2 = parse_kernel(:(function ii_if2_same_branch_after(x, y, z, n, cflag, out)
        if cflag[1] > 0.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
            out[1] = t * t
        else
            z[1] = z[1] + 1.0
        end
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert isempty(plan2) "ii_if2_same_branch_after: expected refusal (same-branch escape), got $plan2"

    # IF3: nonlinear read AFTER the whole :if statement closes. Must
    # stay refused -- if the condition is true, this read really would
    # observe the loop's contribution.
    k3 = parse_kernel(:(function ii_if3_post_if(x, y, z, n, cflag, out)
        if cflag[1] > 0.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
        else
            z[1] = z[1] + 1.0
        end
        out[1] = t * t
        return nothing
    end))
    plan3 = stade_ii_plan_check(k3)
    @assert isempty(plan3) "ii_if3_post_if: expected refusal (post-if escape), got $plan3"

    # IF4: the sibling branch reuses the same variable name `t`,
    # entirely unrelated to the `:then` branch's own `t`. Must NOT be
    # treated as an escape -- the two branches are mutually exclusive,
    # so nothing in `:els` can ever observe `:then`'s contribution.
    k4 = parse_kernel(:(function ii_if4_other_branch_safe(x, y, z, n, cflag, out)
        if cflag[1] > 0.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
        else
            t = z[1] * z[1]
            out[1] = t * t
        end
        return nothing
    end))
    plan4 = stade_ii_plan_check(k4)
    @assert length(plan4) == 1 && only(values(plan4)) == :independent "ii_if4_other_branch_safe: expected one :independent site (sibling branch must not count), got $plan4"

    # IF5: the whole :if sits inside a repeating ancestor (i_seq_k,
    # itself ineligible via a genuine array recurrence, forcing the
    # walker down into the :if), with a nonlinear read of `t`
    # positioned BEFORE the :if within that SAME repeating body --
    # reachable on the NEXT ancestor iteration. Must stay refused.
    k5 = parse_kernel(:(function ii_if5_repeating_ancestor(u, x, y, z, n, m, cflag, out)
        for i_seq_k = 2:m
            u[i_seq_k] = u[i_seq_k - 1] + 1.0
            out[i_seq_k] = t * t
            if cflag[i_seq_k] > 0.0
                for i = 1:n
                    t = x[i] * x[i]
                    s = t / 2.0
                    y[i] = y[i] + s * s
                end
            else
                z[1] = z[1] + 1.0
            end
        end
        return nothing
    end))
    plan5 = stade_ii_plan_check(k5)
    @assert isempty(plan5) "ii_if5_repeating_ancestor: expected refusal (wraparound escape through a repeating :if ancestor), got $plan5"

    # IF6: :if nested inside a repeating (ineligible) ancestor, fully
    # contained -- confirms the combination (repeating ancestor +
    # :if + :for) works, not just each piece in isolation.
    k6 = parse_kernel(:(function ii_if6_nested_contained(u, x, y, z, n, m, cflag)
        for i_seq_k = 2:m
            u[i_seq_k] = u[i_seq_k - 1] + 1.0
            if cflag[i_seq_k] > 0.0
                for i = 1:n
                    t = x[i] * x[i]
                    s = t / 2.0
                    y[i] = y[i] + s * s
                end
            else
                z[1] = z[1] + 1.0
            end
        end
        return nothing
    end))
    plan6 = stade_ii_plan_check(k6)
    @assert length(plan6) == 1 && only(values(plan6)) == :independent "ii_if6_nested_contained: expected one :independent site, got $plan6"

    println("ii_plan :if-nesting extension regression suite (6 cases) OK")
end

# ---- ii_* known_consts threading + mixed-classification regression -

let
    # KC1: an UNRELATED wb-style self-referencing-with-reset
    # accumulator, sitting in the SAME loop body as a genuinely
    # independent, value-needed chain (`t`/`s`) that has nothing to do
    # with `wb` at all. Before known_consts threading, this loop was
    # refused OUTRIGHT (cgen_reduction_only_loop's own Phase 1 proof
    # failed on `wb`, regardless of what else was in the body) --
    # confirmed directly via cgen_reduction_only_loop(body, var,
    # Dict()) returning `nothing` vs Dict(:wb=>0.0) accepting it.
    # Must now succeed for `t`/`s`.
    k1 = parse_kernel(:(function ii_kc1_unrelated_wb(x, y, loss, n)
        wb = 0.0
        for i = 1:n
            wb = wb + loss[1]
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
            if x[i] > 0.0
                wb = 0.0
            else
                wb = 0.0
            end
        end
        return nothing
    end))
    plan1 = stade_ii_plan_check(k1)
    @assert length(plan1) == 1 && only(values(plan1)) == :independent "ii_kc1_unrelated_wb: expected one :independent site, got $plan1"

    # KC2: same shape, but the pre-loop value (1.0) does NOT match
    # what `wb` converges to inside the loop (0.0) -- unprovable, must
    # stay refused.
    k2 = parse_kernel(:(function ii_kc2_mismatch(x, y, loss, n)
        wb = 1.0
        for i = 1:n
            wb = wb + loss[1]
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
            if x[i] > 0.0
                wb = 0.0
            else
                wb = 0.0
            end
        end
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert isempty(plan2) "ii_kc2_mismatch: expected refusal (pre-loop value doesn't match convergent constant), got $plan2"

    # KC3: pre-loop value is a variable, not a literal -- no
    # known_consts entry possible regardless, must stay refused.
    k3 = parse_kernel(:(function ii_kc3_nonliteral(x, y, loss, n, wb0)
        wb = wb0
        for i = 1:n
            wb = wb + loss[1]
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
            if x[i] > 0.0
                wb = 0.0
            else
                wb = 0.0
            end
        end
        return nothing
    end))
    plan3 = stade_ii_plan_check(k3)
    @assert isempty(plan3) "ii_kc3_nonliteral: expected refusal (pre-loop value not a literal), got $plan3"

    println("ii_plan known_consts threading regression suite (3 cases) OK")
end

let
    # M1: mixed case where the INDEPENDENT half genuinely escapes --
    # whole loop must still refuse, not silently accept the reduction
    # half alone.
    k1 = parse_kernel(:(function ii_mixed_neg_ind_escapes(x, y, n, acc_out, out)
        acc = 0.0
        for i = 1:n
            t = x[i] * x[i]
            y[i] = y[i] + t * t
            acc = acc + x[i]
        end
        acc_out[1] = acc * acc
        out[1] = t * t
        return nothing
    end))
    plan1 = stade_ii_plan_check(k1)
    @assert isempty(plan1) "ii_mixed_neg_ind_escapes: expected refusal (independent half escapes), got $plan1"

    # M2: mixed case where the REDUCTION half is read nonlinearly
    # INSIDE the loop (violates pure-reduction shape) -- whole loop
    # must still refuse.
    k2 = parse_kernel(:(function ii_mixed_neg_red_inside_nonlinear(x, y, n, acc_out)
        acc = 0.0
        for i = 1:n
            t = x[i] * x[i]
            y[i] = y[i] + t * t
            acc = acc + x[i]
            y[i] = y[i] + acc * acc
        end
        acc_out[1] = acc * acc
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert isempty(plan2) "ii_mixed_neg_red_inside_nonlinear: expected refusal (reduction half read nonlinearly inside loop), got $plan2"

    println("ii_plan mixed-classification negative-control regression suite (2 cases) OK")
end

# ---- Phase 3 codegen correctness regression tests -------------------
# Both found via central-difference validation on real kernels, not
# anticipated by any adversarial kernel built for the eligibility
# oracle alone -- eligibility never generates code, so it never
# exercised whether fusing a proven-safe scalar chain silently sweeps
# in something else that isn't safe. Locked in here as permanent
# regressions checked via actual central-difference validation, not
# just classification results.

let
    # Array-write-escape regression: a loop containing a genuinely
    # independent scalar chain (`t`/`s`) ALONGSIDE an array write
    # (`r[i] = t`) whose array is read elsewhere (`out[1] = r[1] /
    # 2.0`) must be refused entirely -- agen_emit_ii_loop fuses a
    # loop's WHOLE backward differentiation, not just the proven-safe
    # scalar subset, so an escaping array write would have its
    # backward code moved too early if this weren't caught.
    k1 = parse_kernel(:(function ii_reg_array_escape(x, r, y, n, out)
        for i = 1:n
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
            r[i] = t
        end
        out[1] = r[1] / 2.0
        return nothing
    end))
    plan1 = stade_ii_plan_check(k1)
    @assert all(v -> v == :recompute, values(plan1)) "ii_reg_array_escape: expected refusal of every FUSING kind (r escapes to out[1]); :recompute is permitted here because it moves no adjoint, got $plan1"

    # Regression for the array write that does NOT escape (r never
    # read outside the loop) -- must still classify. Distinguishes
    # "any array write refuses" (too conservative, would be a
    # regression in completeness) from "any ESCAPING array write
    # refuses" (the actual, correct behavior).
    k2 = parse_kernel(:(function ii_reg_array_no_escape(x, r, y, n)
        for i = 1:n
            t = x[i] * x[i]
            r[i] = t
            s = r[i] / 2.0
            y[i] = y[i] + s * s
        end
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert length(plan2) == 1 && only(values(plan2)) == :independent "ii_reg_array_no_escape: expected one :independent site, got $plan2"

    # Linear-read-escape regression: a scalar chain fully contained
    # EXCEPT for a purely LINEAR downstream read (`z = z + t`, a
    # straight accumulation, no +/-/*/etc. combining it nonlinearly)
    # outside the loop. agen_var_value_needed!-based escape detection
    # (used by every OTHER stage in this file for its own, different
    # purpose) would miss this, since a linear read never marks
    # `needed=true` -- but the read still contributes to `t`'s shadow,
    # and that contribution must land before a fused loop's own
    # backward code reads it. Must be refused.
    k3 = parse_kernel(:(function ii_reg_linear_escape(x, y, n, out)
        for i = 1:n
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
        end
        out[1] = out[1] + t
        return nothing
    end))
    plan3 = stade_ii_plan_check(k3)
    @assert all(v -> v == :recompute, values(plan3)) "ii_reg_linear_escape: expected refusal of every FUSING kind (t read linearly outside the loop); :recompute is permitted here because it moves no adjoint, got $plan3"

    println("ii_plan Phase 3 codegen correctness regression suite (3 cases) OK")
end

let
    # End-to-end numerical regression: fuse_ii_loops=true must produce
    # numerically correct gradients (matches fuse_ii_loops=false to
    # within the same central-difference tolerance) for a kernel that
    # actually classifies :independent and gets fused. This is the
    # one test in this file that exercises agen_emit_ii_loop itself,
    # not just ii_plan classification -- everything else here checks
    # eligibility only. Uses a temp file since stade_validate_adjoint_
    # file's baseline-caching path is file-based.
    mktempdir() do dir
        path = joinpath(dir, "ii_e2e_stub.jl")
        write(path, """
        function ii_e2e_stub(x, y, n)
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
            return nothing
        end
        """)
        plan = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path)))
        @assert length(plan) == 1 && only(values(plan)) == :independent "ii_e2e_stub: expected one :independent site, got $plan"
        r_unfused = stade_validate_adjoint_file(path; trials = 10, fuse_ii_loops = false)
        r_fused   = stade_validate_adjoint_file(path; trials = 10, fuse_ii_loops = true)
        @assert r_unfused.ok "ii_e2e_stub: unfused baseline itself failed central-difference validation, max_rel_err=$(r_unfused.max_rel_err)"
        @assert r_fused.ok "ii_e2e_stub: fused adjoint failed central-difference validation, max_rel_err=$(r_fused.max_rel_err)"
    end
    println("ii_plan Phase 3 end-to-end numerical regression (agen_emit_ii_loop) OK")
end

# ---- ii_* array-side order-aware escape analysis regression tests --
# Mirrors the scalar side's own design (ii_escapes_nested), but
# deliberately WITHOUT modeling any "kill" for arrays (see the
# function's own comment for why) -- strictly more conservative than
# the scalar side.

let
    # B1: the SAME array name read by a genuinely UNRELATED, EARLIER
    # statement (before the loop, non-repeating level) -- a read that
    # already ran before the write happened could never observe it.
    # Must now be accepted (was wrongly refused by the blanket check).
    k1 = parse_kernel(:(function ii_arr_b1_before_unrelated(x, r, y, n, out)
        out[1] = r[1]
        for i = 1:n
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
            r[i] = t
        end
        return nothing
    end))
    plan1 = stade_ii_plan_check(k1)
    @assert length(plan1) == 1 && only(values(plan1)) == :independent "ii_arr_b1_before_unrelated: expected one :independent site, got $plan1"

    # B2: genuine read AFTER the loop, non-repeating level -- must
    # stay refused.
    k2 = parse_kernel(:(function ii_arr_b2_after(x, r, y, n, out)
        for i = 1:n
            t = x[i] * x[i]
            s = t / 2.0
            y[i] = y[i] + s * s
            r[i] = t
        end
        out[1] = r[1]
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert all(v -> v == :recompute, values(plan2)) "ii_arr_b2_after: expected refusal of every FUSING kind (genuine after-loop escape); :recompute is permitted here because it moves no adjoint, got $plan2"

    # B3: nested inside a repeating (ineligible) ancestor, read
    # positioned BEFORE the write within the SAME repeating body --
    # wraparound escape, must stay refused.
    k3 = parse_kernel(:(function ii_arr_b3_wraparound(u, x, r, y, n, m, out)
        for i_seq_k = 2:m
            out[i_seq_k] = r[1]
            u[i_seq_k] = u[i_seq_k - 1] + 1.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
                r[i] = t
            end
        end
        return nothing
    end))
    plan3 = stade_ii_plan_check(k3)
    @assert all(v -> v == :recompute, values(plan3)) "ii_arr_b3_wraparound: expected refusal of every FUSING kind (wraparound escape); :recompute is permitted here because it moves no adjoint, got $plan3"

    # B4: nested inside a repeating ancestor, read AFTER the write
    # within the SAME repeating body -- must stay refused.
    k4 = parse_kernel(:(function ii_arr_b4_same_level_after(u, x, r, y, n, m, out)
        for i_seq_k = 2:m
            u[i_seq_k] = u[i_seq_k - 1] + 1.0
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
                r[i] = t
            end
            out[i_seq_k] = r[1]
        end
        return nothing
    end))
    plan4 = stade_ii_plan_check(k4)
    @assert all(v -> v == :recompute, values(plan4)) "ii_arr_b4_same_level_after: expected refusal of every FUSING kind (same-repeating-level escape); :recompute is permitted here because it moves no adjoint, got $plan4"

    # B5: fully contained, array never read anywhere else at all.
    k5 = parse_kernel(:(function ii_arr_b5_contained(x, r, y, n)
        for i = 1:n
            t = x[i] * x[i]
            r[i] = t
            s = r[i] / 2.0
            y[i] = y[i] + s * s
        end
        return nothing
    end))
    plan5 = stade_ii_plan_check(k5)
    @assert length(plan5) == 1 && only(values(plan5)) == :independent "ii_arr_b5_contained: expected one :independent site, got $plan5"

    println("ii_plan array-side order-aware escape regression suite (5 cases) OK")
end

# ---- ii_* :reduction codegen end-to-end numerical regression tests -
# Run against real generated code via central-difference validation,
# not just ii_plan classification.

let
    mktempdir() do dir
        # Simple case: each iteration's own contribution lands in a
        # distinct downstream slot, so summation order is irrelevant
        # and fused/unfused should match to the bit.
        path1 = joinpath(dir, "ii_red_e2e_simple.jl")
        write(path1, """
        function ii_red_e2e_simple(x, y, n, out)
            s = 0.0
            for i = 1:n
                s = s + x[i] * x[i]
            end
            out[1] = s * s
            return nothing
        end
        """)
        plan1 = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path1)))
        @assert length(plan1) == 1 && only(values(plan1)) == :reduction "ii_red_e2e_simple: expected one :reduction site, got $plan1"
        r1_unfused = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = false)
        r1_fused   = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = true)
        @assert r1_unfused.ok "ii_red_e2e_simple: unfused baseline failed, max_rel_err=$(r1_unfused.max_rel_err)"
        @assert r1_fused.ok "ii_red_e2e_simple: fused adjoint failed, max_rel_err=$(r1_fused.max_rel_err)"

        # Harder case: every iteration's contribution accumulates into
        # the same shared downstream slot -- floating-point summation
        # order genuinely differs between the un-reversed (fused) and
        # reversed (unfused) traversals here, so this is the real test
        # that the distribution loop's un-reversed order still gives a
        # numerically correct (not just bit-identical-by-luck) answer.
        path2 = joinpath(dir, "ii_red_e2e_shared.jl")
        write(path2, """
        function ii_red_e2e_shared(x, z, n, out)
            s = 0.0
            for i = 1:n
                s = s + x[i] * x[i]
            end
            for i = 1:n
                z[1] = z[1] + s * x[i]
            end
            out[1] = z[1]
            return nothing
        end
        """)
        plan2 = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path2)))
        @assert length(plan2) == 1 && only(values(plan2)) == :reduction "ii_red_e2e_shared: expected one :reduction site, got $plan2"
        r2_unfused = stade_validate_adjoint_file(path2; trials = 10, fuse_ii_loops = false)
        r2_fused   = stade_validate_adjoint_file(path2; trials = 10, fuse_ii_loops = true)
        @assert r2_unfused.ok "ii_red_e2e_shared: unfused baseline failed, max_rel_err=$(r2_unfused.max_rel_err)"
        @assert r2_fused.ok "ii_red_e2e_shared: fused adjoint failed, max_rel_err=$(r2_fused.max_rel_err)"
    end
    println("ii_plan Phase 3 :reduction end-to-end numerical regression (2 cases) OK")
end

# ---- ii_* Phase 3 cleanup: unused stack removal regression tests ---
# A var covered by an ii_plan site does not always mean its stack
# becomes fully unused -- agen_block_boundary_vars's own, separate
# mechanism (unrelated to fusion) can still keep it alive, and
# correctly so when not every write-site of that var is ii_plan-
# covered.

let
    mktempdir() do dir
        # Fully covered: single write-site, entirely inside the
        # classified loop, never escaping -- stack must be fully
        # eliminated from both signatures.
        path1 = joinpath(dir, "ii_cleanup_full.jl")
        write(path1, """
        function ii_cleanup_full(x, y, n)
            for i = 1:n
                t = x[i] * x[i]
                s = t / 2.0
                y[i] = y[i] + s * s
            end
            return nothing
        end
        """)
        r1 = stade_adjoint(io_read_corpus_entry(path1); fuse_ii_loops = true)
        sig1 = string(r1.adjoint.args[1])
        @assert !occursin("t_stack", sig1) && !occursin("s_stack", sig1) "ii_cleanup_full: expected t_stack/s_stack fully removed, got $sig1"
        @assert isempty(r1.initstacks.args[1].args[2:end]) "ii_cleanup_full: expected initstacks_* to take no stack args, got $(r1.initstacks.args[1])"
        r1v_false = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = false)
        r1v_true  = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = true)
        @assert r1v_false.ok && r1v_true.ok "ii_cleanup_full: validation failed after stack removal"

        # Partially covered: a reset OUTSIDE the classified loop (a
        # sibling statement) means NOT every write-site is covered --
        # the stack must stay.
        path2 = joinpath(dir, "ii_cleanup_partial.jl")
        write(path2, """
        function ii_cleanup_partial(x, y, out, n, m)
            for j = 1:m
                s = 0.0
                for i = 1:n
                    s = s + x[i, j] * x[i, j]
                end
                for i = 1:n
                    y[i, j] = y[i, j] + s * x[i, j]
                end
            end
            out[1] = y[1, 1]
            return nothing
        end
        """)
        r2 = stade_adjoint(io_read_corpus_entry(path2); fuse_ii_loops = true)
        sig2 = string(r2.adjoint.args[1])
        # `s = 0.0` used to be an uncovered write-site because the
        # enclosing `j` loop was refused (y escapes to out[1]). It now
        # classifies as :recompute, so every write-site of `s` IS
        # covered and the stack goes -- the coverage improvement this
        # kind exists to produce. The validation below is what keeps
        # this honest.
        @assert !occursin("s_stack", sig2) "ii_cleanup_partial: expected s_stack to be removed (every write-site is now ii_plan-covered via the enclosing :recompute loop), got $sig2"
        r2v_false = stade_validate_adjoint_file(path2; trials = 10, fuse_ii_loops = false)
        r2v_true  = stade_validate_adjoint_file(path2; trials = 10, fuse_ii_loops = true)
        @assert r2v_false.ok && r2v_true.ok "ii_cleanup_partial: validation failed"
    end
    println("ii_plan Phase 3 unused-stack-removal regression suite (2 cases) OK")
end

# ---- ii_* :mixed codegen end-to-end numerical regression tests -----
# The cross-dependency case (M2 below) is the one that actually caught
# a real bug: a first version of :mixed split vn_ind's differentiation
# to the forward position and vn_red's to the backward position
# (mirroring how :independent and :reduction each work alone). That
# passed a minimal test (M1, which happens not to have any vn_ind var
# read by a vn_red statement) but failed central-difference validation
# once tested against exactly this cross-dependency: a vn_ind var's
# own "collect every contribution, then distribute" step ran too early
# (at the forward position), before its vn_red-side contribution (only
# available later, at the backward position) had even been computed,
# silently dropping that contribution and leaking the shadow instead
# of distributing it. Fixed by NOT splitting -- `:mixed` now gets the
# same treatment as `:reduction` (everything deferred to the backward
# position) rather than attempting the split at all. M1 is kept as a
# positive control that the simpler case still works; M2 is the actual
# regression that would catch the split-based bug returning.

let
    mktempdir() do dir
        # M1: vn_ind (`t`) never read by the vn_red (`acc`) statement
        # -- no cross-dependency, the simplest possible :mixed case.
        path1 = joinpath(dir, "ii_mixed_e2e_simple.jl")
        write(path1, """
        function ii_mixed_e2e_simple(x, y, n, acc_out)
            acc = 0.0
            for i = 1:n
                t = x[i] * x[i]
                y[i] = y[i] + t * t
                acc = acc + x[i]
            end
            acc_out[1] = acc * acc
            return nothing
        end
        """)
        plan1 = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path1)))
        @assert length(plan1) == 1 && only(values(plan1)) == :mixed "ii_mixed_e2e_simple: expected one :mixed site, got $plan1"
        r1_unfused = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = false)
        r1_fused   = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = true)
        @assert r1_unfused.ok "ii_mixed_e2e_simple: unfused baseline failed, max_rel_err=$(r1_unfused.max_rel_err)"
        @assert r1_fused.ok "ii_mixed_e2e_simple: fused adjoint failed, max_rel_err=$(r1_fused.max_rel_err)"

        # M2: vn_ind (`diff`) IS read by the vn_red (`s2`) statement's
        # own accumulation term (a variance-style computation), nested
        # inside a genuinely sequential outer loop, which is what
        # forces the inner loop to be classified on its own rather
        # than subsumed into a larger :independent unit.
        path2 = joinpath(dir, "ii_mixed_e2e_crossdep.jl")
        write(path2, """
        function ii_mixed_e2e_crossdep(u, x, y, out, d, n_rows)
            for i_seq_row = 2:n_rows
                u[i_seq_row] = u[i_seq_row - 1] + 1.0
                s2 = 0.0
                for j = 1:d
                    diff = x[(i_seq_row - 1) * d + j]
                    y[(i_seq_row - 1) * d + j] = y[(i_seq_row - 1) * d + j] + diff * diff
                    s2 = s2 + diff * diff
                end
                row_var = s2 / d
                out[i_seq_row] = row_var
            end
            return nothing
        end
        """)
        plan2 = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path2)))
        @assert length(plan2) == 1 && only(values(plan2)) == :mixed "ii_mixed_e2e_crossdep: expected one :mixed site, got $plan2"
        r2_unfused = stade_validate_adjoint_file(path2; trials = 10, fuse_ii_loops = false)
        r2_fused   = stade_validate_adjoint_file(path2; trials = 10, fuse_ii_loops = true)
        @assert r2_unfused.ok "ii_mixed_e2e_crossdep: unfused baseline failed, max_rel_err=$(r2_unfused.max_rel_err)"
        @assert r2_fused.ok "ii_mixed_e2e_crossdep: fused adjoint failed (max_rel_err=$(r2_fused.max_rel_err)) -- this is exactly the cross-dependency shape that broke the earlier split-based design"
    end
    println("ii_plan Phase 3 :mixed end-to-end numerical regression (2 cases) OK")
end

# ---- ii_* site-level TBR + fuse_ii_loops interaction regression ----
# Found by direct user report: fuse_ii_loops=true combined with
# site-level TBR produced genuinely wrong gradients, not just a missed
# optimization. Root cause: agen_push_pop_source ignores `value_needed`
# entirely once ectx.push_pop is set (site-level TBR's own, ii_plan-
# unaware per-site Dict takes over the push/pop decision completely)
# -- so excluding a var from value_needed, this file's only mechanism
# for suppressing a push/pop pair everywhere else, had zero effect
# once site-level TBR was active. For `:reduction` specifically this
# left a genuinely unmatched push at the forward position (the stale
# decision still said to push, but fusion's whole design assumes
# nothing there ever needs popping), permanently growing the stack and
# corrupting every later, unrelated pop. Fixed by agen_ii_override_ectx.

let
    mktempdir() do dir
        path = joinpath(dir, "ii_tbr_interaction.jl")
        write(path, """
        function ii_tbr_interaction(x, y, n, acc_out)
            acc = 0.0
            for i = 1:n
                acc = acc + x[i] * x[i]
            end
            acc_out[1] = acc * acc
            return nothing
        end
        """)
        r = stade_validate_adjoint_file(path; trials = 10, fuse_ii_loops = true)
        @assert r.ok "ii_tbr_interaction: fuse_ii_loops=true failed (max_rel_err=$(r.max_rel_err)) -- this is exactly the stack-imbalance shape that produced wrong gradients before agen_ii_override_ectx existed"
    end
    println("ii_plan Phase 3 site-level TBR interaction regression OK")
end

# ---- ii_* fuse_ii_loops on stade_hvp/stade_tangent regression -----
# fuse_ii_loops was originally wired only into stade_adjoint;
# extended here to stade_tangent (a documented no-op, tgen_* has no
# stacks to fuse in the first place, matching keep_push_pop's own
# existing treatment there) and stade_hvp (genuinely functional --
# hvp_emit reuses the exact same agen_forward_body/agen_backward_body
# calls the adjoint path does, just with ii_plan previously hardcoded
# to nothing instead of threaded through).
#
# A real consistency bug was found and fixed while wiring this in, not
# assumed away: hvp_emit's own fwd/bwd portion never went through the
# post-hoc unused-stack cleanup that stade_adjoint's own agen_emit
# already applies, so once fusion left a stack unused, the generated
# `_hv` function's signature and a shared `initstacks_*`'s own return
# tuple went out of sync -- any caller (validation code included) that
# passes one kernel's initstacks_* output to both its adjoint and hvp
# functions would hit a MethodError from the extra trailing stack
# arguments the hvp function no longer needed but still declared.
# Fixed by applying the same cleanup independently to hvp_expr.

let
    mktempdir() do dir
        # Reduction case: exercises hvp_double_body's own handling of
        # the backward-position distribution loop, not just a fused
        # forward-position loop.
        path = joinpath(dir, "ii_hvp_e2e_reduction.jl")
        write(path, """
        function ii_hvp_e2e_reduction(x, y, n, out)
            s = 0.0
            for i = 1:n
                s = s + x[i] * x[i]
            end
            out[1] = s * s
            return nothing
        end
        """)
        plan = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path)))
        @assert length(plan) == 1 && only(values(plan)) == :reduction "ii_hvp_e2e_reduction: expected one :reduction site, got $plan"

        # Signature consistency: stade_adjoint's own cleaned-up
        # initstacks_* must exactly match what stade_hvp's own cleaned-
        # up hvp function expects -- this is the exact shape of the
        # bug found above (a MethodError from mismatched stack args).
        a = stade_adjoint(io_read_corpus_entry(path); fuse_ii_loops = true)
        h = stade_hvp(io_read_corpus_entry(path); fuse_ii_loops = true)
        @assert string(a.initstacks.args[1]) == string(h.initstacks.args[1]) "ii_hvp_e2e_reduction: adjoint and hvp initstacks_* signatures diverged: $(a.initstacks.args[1]) vs $(h.initstacks.args[1])"
        @assert isempty(a.initstacks.args[1].args[2:end]) "ii_hvp_e2e_reduction: expected s_stack fully removed from initstacks_*, got $(a.initstacks.args[1])"

        r_unfused = stade_validate_hvp_file(path; trials = 10, fuse_ii_loops = false)
        r_fused   = stade_validate_hvp_file(path; trials = 10, fuse_ii_loops = true)
        @assert r_unfused.ok "ii_hvp_e2e_reduction: unfused baseline failed, max_rel_err=$(r_unfused.max_rel_err)"
        @assert r_fused.ok "ii_hvp_e2e_reduction: fused hvp failed, max_rel_err=$(r_fused.max_rel_err)"

        # stade_tangent's own no-op: both settings must still validate.
        rt_false = stade_validate_tangent_file(path; trials = 10, fuse_ii_loops = false)
        rt_true  = stade_validate_tangent_file(path; trials = 10, fuse_ii_loops = true)
        @assert rt_false.ok && rt_true.ok "ii_hvp_e2e_reduction: stade_tangent's fuse_ii_loops no-op broke validation"
    end
    println("ii_plan fuse_ii_loops on stade_hvp/stade_tangent regression OK")
end
# ---- ii_* surviving-snapshot regression (:reduction/:mixed) --------
# `:reduction`/`:mixed` keep the ordinary forward loop AND emit
# agen_emit_ii_loop (primal ++ backward) at the backward position, so
# the body runs TWICE. Any push site inside it that the vn_local
# exclusion doesn't cover therefore fires in both: the forward one is
# never popped, and the second execution restores the wrong value.
# S1 is the shape that produced a genuinely wrong gradient
# (max_rel_err ~2.0, not a near-miss); S2 is the branch-flag variant
# of the same double-push. S3 is the positive
# control that `:independent` -- which REPLACES the forward loop
# rather than adding to it, keeping push and pop matched inside one
# fused iteration -- is deliberately not gated by this.

let
    mktempdir() do dir
        # S1: `z[i] = z[i] * x[i]` is self-referencing under `*`, so it
        # needs an :array snapshot, and z is not in vn_local (it's an
        # array, not a locally-assigned scalar), so nothing excludes it.
        path1 = joinpath(dir, "ii_survsnap_array.jl")
        write(path1, """
        function ii_survsnap_array(x, z, n, acc_out)
            acc = 0.0
            for i = 1:n
                z[i] = z[i] * x[i]
                acc = acc + z[i]
            end
            acc_out[1] = acc * acc
            return nothing
        end
        """)
        plan1 = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path1)))
        @assert isempty(plan1) "ii_survsnap_array: expected no classified site (surviving array snapshot inside a :reduction body), got $plan1"
        r1 = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = true)
        @assert r1.ok "ii_survsnap_array: fused adjoint failed (max_rel_err=$(r1.max_rel_err)) -- a :reduction body executes twice, so a surviving push orphans its forward copy"

        # S2: an :if inside the body -- agen_forward_body pushes a
        # branch flag unconditionally, so the same double-push applies.
        path2 = joinpath(dir, "ii_survsnap_branch.jl")
        write(path2, """
        function ii_survsnap_branch(x, n, acc_out)
            acc = 0.0
            for i = 1:n
                if x[i] > 0.0
                    acc = acc + x[i]
                else
                    acc = acc - x[i]
                end
            end
            acc_out[1] = acc * acc
            return nothing
        end
        """)
        plan2 = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path2)))
        @assert isempty(plan2) "ii_survsnap_branch: expected no classified site (branch flag pushed inside a :reduction body), got $plan2"
        r2 = stade_validate_adjoint_file(path2; trials = 10, fuse_ii_loops = true)
        @assert r2.ok "ii_survsnap_branch: fused adjoint failed (max_rel_err=$(r2.max_rel_err))"

        # S3: positive control -- :independent replaces the forward
        # loop entirely, so a surviving snapshot inside it stays
        # matched and must NOT be refused.
        path3 = joinpath(dir, "ii_survsnap_indep_ok.jl")
        write(path3, """
        function ii_survsnap_indep_ok(x, y, z, n)
            for i = 1:n
                z[i] = z[i] * x[i]
                t = z[i] * z[i]
                y[i] = y[i] + t * t
            end
            return nothing
        end
        """)
        plan3 = stade_ii_plan_check(parse_kernel(io_read_corpus_entry(path3)))
        @assert length(plan3) == 1 && only(values(plan3)) == :independent "ii_survsnap_indep_ok: :independent must stay classified despite a surviving snapshot, got $plan3"
        r3 = stade_validate_adjoint_file(path3; trials = 10, fuse_ii_loops = true)
        @assert r3.ok "ii_survsnap_indep_ok: fused adjoint failed (max_rel_err=$(r3.max_rel_err))"
    end
    println("ii_plan surviving-snapshot regression suite (3 cases) OK")
end

# ---- ii_recomputable adversarial suite ----------------------------
# The proof obligation for recompute-at-backward-time. Each case
# targets one way re-execution can silently diverge from the forward
# values, rather than a happy path.

let
    kernel_of(ex) = parse_kernel(ex)
    first_loop(body) = body[findfirst(s -> s.kind == :for, body)]

    # R1: chain bottoming out entirely in never-written arguments.
    k1 = kernel_of(:(function ii_recomp_clean(x, y, n)
        for i = 1:n
            t = x[i] * x[i]
            y[i] = y[i] + t * t
        end
        return nothing
    end))
    @assert ii_recomputable(k1, first_loop(k1.body).body, :t) "R1: a chain over never-written args must be recomputable"

    # R2: chain reads an array the kernel writes elsewhere -- its
    # forward contents are gone by backward time.
    k2 = kernel_of(:(function ii_recomp_written_arr(x, z, y, n)
        for i = 1:n
            z[i] = x[i] * 2.0
        end
        for i = 1:n
            t = z[i] * z[i]
            y[i] = y[i] + t * t
        end
        return nothing
    end))
    @assert !ii_recomputable(k2, k2.body[2].body, :t) "R2: a chain reading a kernel-written array must be refused"

    # R3: accumulator whose reset sits OUTSIDE the candidate loop --
    # live-in, so re-execution restarts from the wrong value.
    k3 = kernel_of(:(function ii_recomp_livein(x, n, out)
        acc = 0.0
        for i = 1:n
            acc = acc + x[i] * x[i]
        end
        out[1] = acc * acc
        return nothing
    end))
    @assert !ii_recomputable(k3, first_loop(k3.body).body, :acc) "R3: a live-in accumulator must be refused"

    # R4: the same accumulator one level out, with its reset INSIDE
    # the candidate loop -- no longer live-in, so it is recomputable.
    # This is the whole reason classification wants to move outward.
    k4 = kernel_of(:(function ii_recomp_reset_inside(x, n, m, out)
        for i_seq_o = 1:m
            acc = 0.0
            for i = 1:n
                acc = acc + x[i] * x[i]
            end
            out[i_seq_o] = acc * acc
        end
        return nothing
    end))
    @assert ii_recomputable(k4, first_loop(k4.body).body, :acc) "R4: an accumulator whose reset is inside the loop must be recomputable"

    # R5: a reassigned kernel ARGUMENT read by the chain -- trusting
    # "it's an argument, so it's safe" was a real bug in the Tier B work.
    k5 = kernel_of(:(function ii_recomp_reassigned_arg(x, y, a, n)
        a = a * 2.0
        for i = 1:n
            t = x[i] * a
            y[i] = y[i] + t * t
        end
        return nothing
    end))
    @assert !ii_recomputable(k5, k5.body[2].body, :t) "R5: a chain reading a reassigned kernel argument must be refused"

    # R6: chain through nested loop indices -- re-established by the
    # re-executed headers, so recomputable.
    k6 = kernel_of(:(function ii_recomp_nested_idx(x, y, n, m)
        for i = 1:n
            for j = 1:m
                t = x[(i - 1) * m + j] * x[(i - 1) * m + j]
                y[(i - 1) * m + j] = y[(i - 1) * m + j] + t * t
            end
        end
        return nothing
    end))
    @assert ii_recomputable(k6, first_loop(k6.body).body, :t) "R6: a chain over nested loop indices must be recomputable"

    # R7: read inside an :if condition still counts as a read, so a
    # var first observed there is live-in.
    k7 = kernel_of(:(function ii_recomp_if_livein(x, y, n)
        s = 1.0
        for i = 1:n
            if s > 0.0
                s = x[i] * x[i]
            end
            y[i] = y[i] + s * s
        end
        return nothing
    end))
    @assert !ii_recomputable(k7, first_loop(k7.body).body, :s) "R7: a var read in an :if condition before any write is live-in"
    println("ii_recomputable adversarial suite (7 cases) OK")
end

# ---- ii_* nested-loop-boundary regression -------------------------
# agen_emit_ii_loop fuses as vcat(fwd, bwd) at the CLASSIFIED loop's
# own level: the whole forward nest, then the whole backward nest. A
# fused scalar assigned inside a NESTED :for therefore survives into
# the backward nest holding only its last inner-iteration value, and
# every backward iteration silently differentiates against it. N1 is
# that shape (it produced max_rel_err ~1.7, not a near-miss); the
# correct behaviour is to refuse the outer unit and let the walker
# fall back to the inner loop, which is flat and genuinely safe. N2
# is the positive control that a fused var assigned at the classified
# body's OWN top level is unaffected -- it holds one value for the
# whole iteration, whatever the nest below it reads.
#
# The pre-existing nested-:for suite above covers the other
# direction (finding a smaller eligible unit inside a failed outer
# loop) and asserts on classification only, so it never exercised
# this.

let
    mktempdir() do dir
        path1 = joinpath(dir, "ii_nestbound_inner.jl")
        write(path1, """
        function ii_nestbound_inner(x, y, n, m)
            for i = 1:n
                for j = 1:m
                    t = x[(i - 1) * m + j] * x[(i - 1) * m + j]
                    y[(i - 1) * m + j] = y[(i - 1) * m + j] + t * t
                end
            end
            return nothing
        end
        """)
        k1 = parse_kernel(io_read_corpus_entry(path1))
        plan1 = stade_ii_plan_check(k1)
        inner_key = agen_site_key(k1.body[1].body, 1)
        # The inner loop must be classified; the outer one is allowed to
        # be as well, but ONLY because the inner is -- that is exactly
        # what ii_fused_var_in_nested_for's plan-awareness permits, and
        # the fused outer loop then delegates to the inner fused form
        # rather than reading a stale last-iteration value. If the inner
        # were ever refused, the outer must be too.
        @assert haskey(plan1, inner_key) "ii_nestbound_inner: the inner loop must be classified, got $plan1"
        outer_key1 = agen_site_key(k1.body, 1)
        @assert !haskey(plan1, outer_key1) || haskey(plan1, inner_key) "ii_nestbound_inner: the outer loop may only be classified when its nested loop is, got $plan1"
        r1u = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = false)
        r1f = stade_validate_adjoint_file(path1; trials = 10, fuse_ii_loops = true)
        @assert r1u.ok "ii_nestbound_inner: unfused baseline failed, max_rel_err=$(r1u.max_rel_err)"
        @assert r1f.ok "ii_nestbound_inner: fused adjoint failed (max_rel_err=$(r1f.max_rel_err)) -- a fused var live across a nested loop boundary keeps only its last inner value"

        # N2: `t` assigned at the classified loop's own top level and
        # merely READ inside the nested loop -- must stay classified.
        path2 = joinpath(dir, "ii_nestbound_toplevel.jl")
        write(path2, """
        function ii_nestbound_toplevel(x, y, n, m)
            for i = 1:n
                t = x[i] * x[i]
                for j = 1:m
                    y[(i - 1) * m + j] = y[(i - 1) * m + j] + t * t
                end
            end
            return nothing
        end
        """)
        k2 = parse_kernel(io_read_corpus_entry(path2))
        plan2 = stade_ii_plan_check(k2)
        outer_key = agen_site_key(k2.body, 1)
        @assert length(plan2) == 1 && haskey(plan2, outer_key) "ii_nestbound_toplevel: a fused var written at the classified body's own top level must stay classified, got $plan2"
        r2f = stade_validate_adjoint_file(path2; trials = 10, fuse_ii_loops = true)
        @assert r2f.ok "ii_nestbound_toplevel: fused adjoint failed, max_rel_err=$(r2f.max_rel_err)"
    end
    println("ii_plan nested-loop-boundary regression suite (2 cases) OK")
end

# ---- ii_* backward-recompute array-corruption regression ----------
# agen_emit_ii_loop's `fwd` half is the real primal at the FORWARD
# position (:independent) but only a RECOMPUTE at the BACKWARD one
# (:reduction/:mixed), where the primal already ran in the ordinary
# forward sweep. Re-executing the whole body there applies every
# accumulating array write a second time, leaving the array at twice
# its forward value. Gradients stay bit-identical while this happens,
# so no finite-difference test can catch it -- C1 compares the array
# STATE after _b against the unfused baseline instead. C2 is the
# guard for the case that makes dropping array writes unsafe (a
# recomputed scalar reading an array the same body writes), which is
# refused rather than recomputed from post-forward contents.

let
    src = :(function ii_recomp_corrupt(x, y, n, acc_out)
        acc = 0.0
        for i = 1:n
            t = x[i] * x[i]
            y[i] = y[i] + t * t
            acc = acc + x[i]
        end
        acc_out[1] = acc * acc
        return nothing
    end)
    plan = stade_ii_plan_check(parse_kernel(src))
    @assert length(plan) == 1 && only(values(plan)) == :mixed "ii_recomp_corrupt: expected one :mixed site, got $plan"
    states = Dict{Bool,Any}()
    for fuse in (false, true)
        out = stade_adjoint(src; fuse_ii_loops = fuse)
        tag = fuse ? "f" : "u"
        Base.eval(Main, Meta.parse("begin\n" *
            replace(io_expr_to_source(out.initstacks), "initstacks_ii_recomp_corrupt_b" => "ii_rc_init_" * tag) * "\n" *
            replace(io_expr_to_source(out.adjoint), "ii_recomp_corrupt_b" => "ii_rc_b_" * tag) * "\nend"))
        n = 5
        x = [0.3, -0.7, 1.1, 0.5, -0.2]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        yb = [0.5, -0.5, 0.25, 1.5, -1.0]
        xb = zeros(n); ao = [0.0]; aob = [1.0]
        st = Base.invokelatest(Base.eval(Main, Symbol("ii_rc_init_" * tag)))
        args = Any[x, xb, y, yb, n, ao, aob]
        st isa Tuple ? append!(args, st) : (st === nothing || push!(args, st))
        Base.invokelatest(Base.eval(Main, Symbol("ii_rc_b_" * tag)), args...)
        states[fuse] = (xb = copy(xb), y = copy(y))
    end
    @assert maximum(abs.(states[true].xb .- states[false].xb)) < 1e-12 "ii_recomp_corrupt: fused gradient diverged from the unfused baseline"
    @assert maximum(abs.(states[true].y .- states[false].y)) < 1e-12 "ii_recomp_corrupt: fused _b left the primal array in a different state than the unfused baseline -- the backward-position recompute re-applied an accumulating array write"

    # C2: a recomputed scalar reads an array this same body writes --
    # dropping the array write would rebuild it from post-forward
    # contents, so the loop must not be classified at all.
    k2 = parse_kernel(:(function ii_recomp_reads_own_write(x, z, n, acc_out)
        acc = 0.0
        for i = 1:n
            z[i] = z[i] + x[i]
            t = z[i] * z[i]
            acc = acc + t
        end
        acc_out[1] = acc * acc
        return nothing
    end))
    plan2 = stade_ii_plan_check(k2)
    @assert isempty(plan2) "ii_recomp_reads_own_write: expected refusal (recomputed scalar reads an array the body writes), got $plan2"
    println("ii_plan backward-recompute array-corruption regression suite (2 cases) OK")
end