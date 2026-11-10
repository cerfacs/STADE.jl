# Battle-test library: oracle for loop offload status (plan STADE_offload_battle_plan_2026-11-09.md).
# Nothing here runs generated code on a GPU. It reads generated source only.
# Statuses: K = loop mapped to threads, T = loop serial inside a thread, H = loop on the host,
# X = loop absent from the generated code (library call or removed).
using SHA

# ---------------------------------------------------------------- subject files and tables
# Table lines in a subject file:
#   # LOOP <var> <I|S> <K|T|H>      a for loop: class and expected status
#   # WHILE <var> <K|T|H>          a while loop, named by the first symbol of its condition
function bt_read_table(path::String)
    loops = Dict{Symbol,Tuple{Symbol,Symbol}}()
    whiles = Dict{Symbol,Symbol}()
    for ln in eachline(path)
        m = match(r"^#\s*LOOP\s+(\w+)\s+([IS])\s+([KTH])\s*$", ln)
        m === nothing || (loops[Symbol(m.captures[1])] = (Symbol(m.captures[2]), Symbol(m.captures[3])))
        m = match(r"^#\s*WHILE\s+(\w+)\s+([KTH])\s*$", ln)
        m === nothing || (whiles[Symbol(m.captures[1])] = Symbol(m.captures[2]))
    end
    return (loops = loops, whiles = whiles)
end

bt_stmts(b) = b isa Expr && b.head === :block ? [s for s in b.args if !(s isa LineNumberNode)] : Any[b]
bt_fn_body(f::Expr) = bt_stmts(f.args[2])

# ---------------------------------------------------------------- polynomial normal form of an index
# poly: Dict{Vector{Symbol},Int}, key = sorted multiset of atoms, [] = constant.
bt_padd(a, b, s = 1) = (r = copy(a); for (k, v) in b; r[k] = get(r, k, 0) + s * v; r[k] == 0 && delete!(r, k); end; r)
function bt_pmul(a, b)
    r = Dict{Vector{Symbol},Int}()
    for (k1, v1) in a, (k2, v2) in b
        k = sort(vcat(k1, k2)); r[k] = get(r, k, 0) + v1 * v2
        r[k] == 0 && delete!(r, k)
    end
    return r
end
function bt_poly(e)
    if e isa Integer; return e == 0 ? Dict{Vector{Symbol},Int}() : Dict(Symbol[] => Int(e)); end
    if e isa Symbol; return Dict([e] => 1); end
    if e isa Expr && e.head === :call
        op = e.args[1]; a = e.args[2:end]
        if op === :+ ; return reduce((x, y) -> bt_padd(x, y), [bt_poly(t) for t in a]); end
        if op === :- && length(a) == 1; return bt_padd(Dict{Vector{Symbol},Int}(), bt_poly(a[1]), -1); end
        if op === :- ; return bt_padd(bt_poly(a[1]), reduce((x, y) -> bt_padd(x, y), [bt_poly(t) for t in a[2:end]]), -1); end
        if op === :* ; return reduce(bt_pmul, [bt_poly(t) for t in a]); end
    end
    if e isa Expr && e.head === :block; return bt_poly(bt_stmts(e)[end]); end
    return Dict([Symbol("@@", string(e))] => 1)          # opaque atom
end
bt_atoms(p) = Set(s for k in keys(p) for s in k)

# affine decomposition in v: (k, rest) with poly = k*v + rest, or nothing when v is not linear
function bt_aff(p, v::Symbol)
    k = 0; rest = Dict{Vector{Symbol},Int}()
    for (m, c) in p
        n = count(==(v), m)
        n == 0 ? (rest[m] = c) : (n == 1 && length(m) == 1 ? (k += c) : return nothing)
    end
    # an opaque atom that mentions v makes the subscript unusable
    for s in bt_atoms(rest); occursin(String(v), String(s)) && startswith(String(s), "@@") && return nothing; end
    return (k, rest)
end

# ---------------------------------------------------------------- access collection
bt_is_arr(e) = e isa Expr && e.head === :ref
function bt_reads!(acc, e)
    if e isa Expr
        if e.head === :ref; push!(acc, e); foreach(a -> bt_reads!(acc, a), e.args[2:end])
        else foreach(a -> bt_reads!(acc, a), e.args); end
    end
    return acc
end
function bt_syms!(acc, e)
    e isa Symbol && push!(acc, e)
    e isa Expr && foreach(a -> bt_syms!(acc, a), e.head === :call ? e.args[2:end] : e.args)
    return acc
end
bt_same(a, b) = string(a) == string(b)
# accumulation: A[idx] = A[idx] + e  or  = e + A[idx]  (also for scalars: s = s + e)
function bt_accum_target(lhs, rhs)
    rhs isa Expr && rhs.head === :call && rhs.args[1] === :+ && length(rhs.args) == 3 || return false
    return bt_same(rhs.args[2], lhs) || bt_same(rhs.args[3], lhs)
end
function bt_accum_other(lhs, rhs)
    return bt_same(rhs.args[2], lhs) ? rhs.args[3] : rhs.args[2]
end

# collect sites in a statement list: sites = (kind=:w/:r/:acc, arr, subs, vars_in_scope)
function bt_collect!(sites, scal_w, scal_r, stmts, inner::Set{Symbol}; defined = Set{Symbol}())
    for s in stmts
        s isa LineNumberNode && continue
        if s isa Expr && s.head === :(=)
            lhs, rhs = s.args
            if bt_is_arr(lhs)
                if bt_accum_target(lhs, rhs)
                    push!(sites, (kind = :acc, arr = lhs.args[1], subs = lhs.args[2:end]))
                    other = bt_accum_other(lhs, rhs)
                    for r in bt_reads!(Any[], other); push!(sites, (kind = :r, arr = r.args[1], subs = r.args[2:end])); end
                else
                    push!(sites, (kind = :w, arr = lhs.args[1], subs = lhs.args[2:end]))
                    for r in bt_reads!(Any[], rhs); push!(sites, (kind = :r, arr = r.args[1], subs = r.args[2:end])); end
                end
                for r in Iterators.flatten(bt_reads!(Any[], x) for x in lhs.args[2:end]); push!(sites, (kind = :r, arr = r.args[1], subs = r.args[2:end])); end
            else
                lhs isa Symbol || continue
                # scalar: reads before the write, accumulation
                rd = bt_syms!(Set{Symbol}(), rhs)
                for r in bt_reads!(Any[], rhs); push!(sites, (kind = :r, arr = r.args[1], subs = r.args[2:end])); end
                if bt_accum_target(lhs, rhs)
                    push!(scal_w, (name = lhs, acc = true, cond = false))
                    push!(scal_r, (name = lhs, def = copy(defined), imp = true))
                    rd2 = bt_syms!(Set{Symbol}(), bt_accum_other(lhs, rhs)); for q in rd2; push!(scal_r, (name = q, def = copy(defined), imp = false)); end
                else
                    for q in rd; push!(scal_r, (name = q, def = copy(defined), imp = false)); end
                    push!(scal_w, (name = lhs, acc = false, cond = false))
                end
                push!(defined, lhs)
            end
        elseif s isa Expr && s.head === :for
            var = s.args[1].args[1]; rng = s.args[1].args[2]
            for q in bt_syms!(Set{Symbol}(), rng); push!(scal_r, (name = q, def = copy(defined), imp = false)); end
            push!(inner, var); push!(defined, var)
            d2 = copy(defined)
            bt_collect!(sites, scal_w, scal_r, bt_stmts(s.args[2]), inner; defined = d2)
            # scalars written in the inner loop are not definitely defined afterwards (zero trips)
        elseif s isa Expr && s.head === :while
            for q in bt_syms!(Set{Symbol}(), s.args[1]); push!(scal_r, (name = q, def = copy(defined), imp = false)); end
            bt_collect!(sites, scal_w, scal_r, bt_stmts(s.args[2]), inner; defined = copy(defined))
        elseif s isa Expr && (s.head === :if || s.head === :elseif)
            for q in bt_syms!(Set{Symbol}(), s.args[1]); push!(scal_r, (name = q, def = copy(defined), imp = false)); end
            for r in bt_reads!(Any[], s.args[1]); push!(sites, (kind = :r, arr = r.args[1], subs = r.args[2:end])); end
            arms = Any[bt_stmts(s.args[2])]
            length(s.args) >= 3 && push!(arms, s.args[3] isa Expr && s.args[3].head === :elseif ? Any[s.args[3]] : bt_stmts(s.args[3]))
            defs = Vector{Set{Symbol}}()
            for a in arms; d = copy(defined); bt_collect!(sites, scal_w, scal_r, a, inner; defined = d); push!(defs, d); end
            if length(arms) == 2; union!(defined, intersect(defs[1], defs[2])); end
        elseif s isa Expr && s.head === :return
        else
            for q in bt_syms!(Set{Symbol}(), s); push!(scal_r, (name = q, def = copy(defined), imp = false)); end
        end
    end
end

# dimension test of one pair for loop variable v.  returns :nodep, (:dep, delta), :any, :maybe
function bt_dim(s1, s2, v::Symbol, inner::Set{Symbol})
    p1 = bt_poly(s1); p2 = bt_poly(s2)
    a1 = bt_aff(p1, v); a2 = bt_aff(p2, v)
    (a1 === nothing || a2 === nothing) && return :maybe
    (k1, r1), (k2, r2) = a1, a2
    diff = bt_padd(r1, r2, -1)
    mentions_inner = !isempty(intersect(bt_atoms(diff), inner))
    if k1 == k2 && k1 != 0
        isempty(diff) && return :nodep
        mentions_inner && return :maybe
        if length(diff) == 1 && haskey(diff, Symbol[])
            c = diff[Symbol[]]
            return (-c) % k1 == 0 ? (:dep, -c ÷ k1) : :nodep
        end
        return :maybe
    elseif k1 == 0 && k2 == 0
        isempty(diff) && return :any
        mentions_inner && return :any                     # assumes the inner ranges overlap (cross-check only)
        length(diff) == 1 && haskey(diff, Symbol[]) && return :nodep
        return :maybe
    end
    return :maybe
end
function bt_pair(subs1, subs2, v, inner)
    length(subs1) == length(subs2) || return :maybe
    res = [bt_dim(a, b, v, inner) for (a, b) in zip(subs1, subs2)]
    any(==(:nodep), res) && return :nodep
    any(==(:maybe), res) && return :maybe
    deps = [r[2] for r in res if r isa Tuple]
    if isempty(deps); return :dep; end                  # no dimension involves v: same element every iteration
    allequal(deps) ? (first(deps) == 0 ? :nodep : :dep) : :nodep
end

# class of a for loop: :I, :S or :U (unknown, needs the table).  stmts = loop body
function bt_derived(var::Symbol, stmts)
    der = Set{Symbol}([var])
    changed = true
    while changed
        changed = false
        function walk(ss)
            for s in ss
                s isa Expr || continue
                if s.head === :(=) && s.args[1] isa Symbol && !(s.args[1] in der) && !isempty(intersect(bt_syms!(Set{Symbol}(), s.args[2]), der))
                    push!(der, s.args[1]); changed = true
                elseif s.head === :for || s.head === :while; walk(bt_stmts(s.args[2]))
                elseif s.head === :if; for a in s.args[2:end]; a isa Expr && walk(bt_stmts(a)); end; end
            end
        end
        walk(stmts)
    end
    delete!(der, var)
    return der
end
# scalars assigned from an array read: a gathered index
function bt_gathered(stmts)
    g = Set{Symbol}()
    function walk(ss)
        for s in ss
            s isa Expr || continue
            if s.head === :(=) && s.args[1] isa Symbol && !isempty(bt_reads!(Any[], s.args[2])); push!(g, s.args[1])
            elseif s.head === :for || s.head === :while; walk(bt_stmts(s.args[2]))
            elseif s.head === :if; for a in s.args[2:end]; a isa Expr && walk(bt_stmts(a)); end; end
        end
    end
    walk(stmts)
    return g
end
function bt_classify_loop(var::Symbol, body, lo_hi_syms)
    der = bt_derived(var, bt_stmts(body))
    gath = bt_gathered(bt_stmts(body))
    sites = Any[]; sw = Any[]; sr = Any[]; inner = Set{Symbol}()
    pairf(a, b, v, inn) = any(sb -> !isempty(intersect(bt_syms!(Set{Symbol}(), sb), der)), vcat(collect(a), collect(b))) ? :maybe : bt_pair(a, b, v, inn)
    bt_collect!(sites, sw, sr, bt_stmts(body), inner; defined = Set{Symbol}())
    unknown = false
    # scalars: written in the body and read before a definite write -> carried, unless pure accumulation
    wnames = Set(w.name for w in sw)
    for name in wnames
        name in inner && continue
        ws = [w for w in sw if w.name == name]
        rs = [r for r in sr if r.name == name]
        allacc = all(w -> w.acc, ws)
        if allacc
            # accumulation scalar must not be read elsewhere in the body
            any(r -> !r.imp, rs) && return :S
        else
            for r in rs; (name in r.def) || return :S; end
        end
    end
    # arrays
    arrs = unique(s.arr for s in sites)
    for A in arrs
        S = [s for s in sites if s.arr == A]
        writes = [s for s in S if s.kind != :r]
        isempty(writes) && continue
        for w in writes
            # a plain write through a subscript that is not affine in v (a gather) cannot be proven injective
            if w.kind == :w
                for sb in w.subs
                    sb isa Symbol && sb != var && !(sb in inner) && (sb in gath) && return :S
                    ps = bt_poly(sb)
                    if any(t -> startswith(String(t), "@@"), bt_atoms(ps)) || (bt_aff(ps, var) === nothing && var in bt_atoms(ps))
                        unknown = true
                    end
                end
            end
        end
        # a plain write whose subscripts do not mention the loop variable rewrites the same elements each iteration
        for w in writes
            w.kind == :w || continue
            any(sb -> !isempty(intersect(bt_syms!(Set{Symbol}(), sb), Set([var]) ∪ der)), w.subs) || return :S
            any(sb -> !isempty(intersect(bt_syms!(Set{Symbol}(), sb), der)), w.subs) && (unknown = true)
        end
        for w in writes, o in S
            (w === o) && w.kind == :acc && continue
            if w.kind == :acc && o.kind == :acc && bt_same(Expr(:ref, w.arr, w.subs...), Expr(:ref, o.arr, o.subs...)); continue; end
            (w.kind == :acc && o.kind == :r && false) && continue
            if w.kind == :acc && o.kind == :acc; r = pairf(w.subs, o.subs, var, inner); r == :maybe && (unknown = true); continue; end
            r = pairf(w.subs, o.subs, var, inner)
            r == :dep && return :S
            r == :maybe && (unknown = true)
        end
        # write-write self pair of a plain write site
        for w in writes
            w.kind == :w || continue
            r = pairf(w.subs, w.subs, var, inner)
            r == :dep && return :S
        end
    end
    return unknown ? :U : :I
end

# classes of every for loop and while loop of a subject, by variable name
function bt_classify_subject(f::Expr)
    cls = Dict{Symbol,Symbol}()
    function walk(stmts)
        for s in stmts
            s isa Expr || continue
            if s.head === :for
                v = s.args[1].args[1]
                cls[v] = bt_classify_loop(v, s.args[2], nothing)
                walk(bt_stmts(s.args[2]))
            elseif s.head === :while
                walk(bt_stmts(s.args[2]))
            elseif s.head === :if || s.head === :elseif
                for a in s.args[2:end]; a isa Expr && (a.head === :elseif ? walk(Any[a]) : walk(bt_stmts(a))); end
            end
        end
    end
    walk(bt_fn_body(f))
    return cls
end

# ---------------------------------------------------------------- oracle: expected statuses (plan section 4.3)
# table: loops var => (class, E).  Returns var => status computed from the classes alone.
function bt_expected(f::Expr, classes::Dict{Symbol,Symbol})
    exp = Dict{Symbol,Symbol}()
    function walk(stmts, ctx)
        for s in stmts
            s isa Expr || continue
            if s.head === :for
                v = s.args[1].args[1]
                c = get(classes, v, :S)
                if ctx === :host
                    if c === :I; exp[v] = :K; walk(bt_stmts(s.args[2]), :kernel)
                    else exp[v] = :H; walk(bt_stmts(s.args[2]), :host); end
                else
                    exp[v] = :T; walk(bt_stmts(s.args[2]), :kernel)
                end
            elseif s.head === :while
                v = bt_syms!(Any[], s.args[1])
                w = first(bt_syms!(Symbol[], s.args[1]))
                exp[w] = ctx === :host ? :H : :T
                walk(bt_stmts(s.args[2]), ctx)
            elseif s.head === :if || s.head === :elseif
                for a in s.args[2:end]; a isa Expr && (a.head === :elseif ? walk(Any[a], ctx) : walk(bt_stmts(a), ctx)); end
            end
        end
    end
    walk(bt_fn_body(f), :host)
    return exp
end

# ---------------------------------------------------------------- status of loops in generated code
const BT_THREAD = Set([:__tid, :__tidr, :__rb_t, :__jx, :__ly, :__jacc_i, :__jacc_j, :__jacc_k, :threadIdx, :blockIdx])
bt_mentions_lane(e) = e isa Symbol ? (e in (:__ly, :__jx)) : (e isa Expr ? any(bt_mentions_lane, e.args) : false)
bt_mentions_tid(e) = e isa Symbol ? (e in (:__tid, :__tidr, :__rb_t, :__jacc_i, :__jacc_j, :__jacc_k)) : (e isa Expr ? any(bt_mentions_tid, e.args) : false)
bt_mentions_thread(e) = e isa Symbol ? (e in BT_THREAD) : (e isa Expr ? any(bt_mentions_thread, e.args) : false)
bt_count(d, v, s) = (d[v][s] = get(d[v], s, 0) + 1)

bt_has_mapreduce(e) = e isa Expr && ((e.head === :call && e.args[1] === :mapreduce) || (e.head === :macrocall && e.args[1] isa Expr && e.args[1].head === :. && String(e.args[1].args[2].value) == "@parallel_reduce") || any(bt_has_mapreduce, e.args))
function bt_scan_host!(obs, e, vars, whilevars, allow; in_allow = false, rmode = false)
    e isa Expr || return
    if e.head === :if && length(e.args) == 3 && bt_has_mapreduce(e.args[3]) && !bt_has_mapreduce(e.args[2])
        # reduction twin: host loop below reduction_threshold, device mapreduce above it
        bt_scan_host!(obs, e.args[2], vars, whilevars, allow; rmode = true)
        bt_scan_host!(obs, e.args[3], vars, whilevars, allow)
        return
    end
    if e.head === :for && rmode && e.args[1] isa Expr && e.args[1].args[1] in vars
        bt_count(obs, e.args[1].args[1], :R)
    elseif e.head === :for && e.args[1] isa Expr && e.args[1].head === :(=) && e.args[1].args[1] in vars
        v = e.args[1].args[1]; bt_count(obs, v, :H); in_allow && (allow[1] += 0)
    elseif e.head === :while
        for w in bt_syms!(Symbol[], e.args[1]); if w in whilevars; bt_count(obs, w, :H); break; end; end
    elseif e.head === :-> && e.args[1] isa Symbol && e.args[1] in vars
        bt_count(obs, e.args[1], :K)                     # device mapreduce over the loop variable
    elseif e.head === :macrocall && e.args[1] isa Expr && e.args[1].head === :. && String(e.args[1].args[2].value) == "@allowscalar"
        allow[1] += 1
    end
    foreach(a -> bt_scan_host!(obs, a, vars, whilevars, allow; rmode = rmode), e.args)
end
function bt_scan_kernel!(obs, e, vars, whilevars)
    e isa Expr || return
    if e.head === :(=) && e.args[1] isa Symbol && e.args[1] in vars && bt_mentions_thread(e.args[2])
        bt_count(obs, e.args[1], :K)
    elseif e.head === :for && e.args[1] isa Expr && e.args[1].head === :(=) && e.args[1].args[1] in vars
        v = e.args[1].args[1]
        rng = e.args[1].args[2]
        bt_count(obs, v, bt_mentions_lane(rng) && !bt_mentions_tid(rng) ? :L : bt_mentions_thread(rng) ? :K : :T)
        bt_scan_kernel!(obs, e.args[2], vars, whilevars)      # the header is not a thread assignment
        return
    elseif e.head === :while
        for w in bt_syms!(Symbol[], e.args[1]); if w in whilevars; bt_count(obs, w, :T); break; end; end
    end
    foreach(a -> bt_scan_kernel!(obs, a, vars, whilevars), e.args)
end
# plan has .host (Expr) and .kernels (Vector{Expr}); returns var => Dict(status => count), and allowscalar count
function bt_observe(plan, vars, whilevars)
    obs = Dict{Symbol,Dict{Symbol,Int}}(v => Dict{Symbol,Int}() for v in union(vars, whilevars))
    allow = [0]
    bt_scan_host!(obs, plan.host, vars, whilevars, allow)
    for k in plan.kernels; bt_scan_kernel!(obs, k, vars, whilevars); end
    for v in keys(obs); isempty(obs[v]) && (obs[v][:X] = 1); end
    return obs, allow[1]
end

# ---------------------------------------------------------------- generation in each mode and backend
const BT_PIN_CUDA = (warp_atomics = false, shared_private = false, coarsen = 0, row_block = 0, col_block = 0,
                     reduce_block = false, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0)
const BT_PIN_JACC = (inbounds = false, dead_scalars = false, split_reduce = 0, warp_atomics = false, jacc_contraction = 0)

function bt_mode_expr(primal::Expr, mode::Symbol; fuse::Bool = true)
    mode === :primal && return primal
    mode === :tangent && return STADE.stade_tangent(primal; keep_push_pop = false, fuse_ii_loops = fuse)
    mode === :adjoint && return STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = fuse).adjoint
    mode === :hvp && return STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = fuse).hvp
    error("mode")
end
function bt_plan(expr::Expr, backend::Symbol, config::Symbol)
    if backend === :cuda
        return config === :pinned ? STADE.stade_gpu(expr, STADE.cgen_backend_cuda(); BT_PIN_CUDA...) :
                                    STADE.stade_gpu(expr, STADE.cgen_backend_cuda())
    else
        return config === :pinned ? STADE.stade_jacc(expr; BT_PIN_JACC...) : STADE.stade_jacc(expr)
    end
end
# plan text for hashing and size
bt_text(plan) = string(plan.host) * join(string.(plan.kernels))
bt_hash(plan) = bytes2hex(sha256(bt_text(plan)))[1:16]

# ---------------------------------------------------------------- nest generator (families F3 and F11)
# pattern: String over I and S, one letter per loop, outermost first.
# I loop: the body touches element l_k only.  S loop: the body reads element l_k - 1 of the written array.
# tri: loop k >= 2 runs 1:l_(k-1) (or 2:l_(k-1)).  fuse: the leading run of I loops becomes one loop over the flat range.
# wrapif: the body sits in a data-dependent branch.  tmp: the body goes through a scalar temporary.
# sibling: an independent loop follows the nest.  pre: a sequential loop precedes the nest.
function bt_nest_source(name::String, pattern::String; tri::Bool = false, fuse::Bool = false,
                        wrapif::Bool = false, tmp::Bool = false, sibling::Bool = false, pre::Bool = false)
    d = length(pattern)
    vars = [Symbol("l", k) for k in 1:d]
    ns = [Symbol("n", k) for k in 1:d]
    lead = 0
    while lead < d && pattern[lead + 1] == 'I'; lead += 1; end
    dofuse = fuse && lead >= 2 && !tri
    idx(shift) = join([k in shift ? "$(vars[k]) - 1" : String(vars[k]) for k in 1:d], ", ")
    sdims = [k for k in 1:d if pattern[k] == 'S']
    reads = isempty(sdims) ? "a * x[$(idx(Int[]))]" :
            join(["0.5 * y[$(idx([k]))]" for k in sdims], " + ") * " + x[$(idx(Int[]))]"
    wr = "y[$(idx(Int[]))]"
    classes = Dict{Symbol,Symbol}()
    extra = sibling || pre
    lines = String["function $name(x, y, " * (extra ? "z, u, " : "") * "a, " * join(ns, ", ") * (extra ? ", np)" : ")")]
    ind = "    "
    if pre
        push!(lines, "$(ind)for p1 = 2:np"); push!(lines, "$(ind)    u[p1] = 0.5 * u[p1 - 1] + 1.0"); push!(lines, "$(ind)end")
        classes[:p1] = :S
    end
    k0 = 1
    if dofuse
        push!(lines, "$(ind)for fz = 1:" * join(ns[1:lead], " * "))
        ind *= "    "
        for k in 1:lead
            stride = k < lead ? join(ns[k+1:lead], " * ") : ""
            push!(lines, "$ind$(vars[k]) = " * (k < lead ? "mod(div(fz - 1, $stride), $(ns[k])) + 1" : "mod(fz - 1, $(ns[k])) + 1"))
        end
        classes[:fz] = :I
        k0 = lead + 1
    end
    for k in k0:d
        lo = pattern[k] == 'S' ? 2 : 1
        hi = tri && k >= 2 ? String(vars[k-1]) : String(ns[k])
        push!(lines, "$(ind)for $(vars[k]) = $lo:$hi")
        classes[vars[k]] = pattern[k] == 'S' ? :S : :I
        ind *= "    "
    end
    if wrapif
        push!(lines, "$(ind)if x[$(idx(Int[]))] > 0.0")
        push!(lines, "$(ind)    $wr = $reads")
        push!(lines, "$(ind)else")
        push!(lines, "$(ind)    $wr = x[$(idx(Int[]))]")
        push!(lines, "$(ind)end")
    elseif tmp
        push!(lines, "$(ind)t0 = $reads")
        push!(lines, "$(ind)$wr = t0 * 1.5")
    else
        push!(lines, "$ind$wr = $reads")
    end
    for k in d:-1:k0
        ind = ind[1:end-4]; push!(lines, ind * "end")
    end
    if dofuse; ind = ind[1:end-4]; push!(lines, ind * "end"); end
    if sibling
        push!(lines, "$(ind)for q1 = 1:np"); push!(lines, "$(ind)    z[q1] = u[q1] * 2.0"); push!(lines, "$(ind)end")
        classes[:q1] = :I
    end
    push!(lines, "    return nothing"); push!(lines, "end")
    return join(lines, "\n") * "\n", classes
end

# parse a generated subject source into its function Expr
bt_parse_fn(src::String) = first(x for x in Meta.parseall(src).args if x isa Expr && x.head === :function)
