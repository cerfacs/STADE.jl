# ==================== validate_primal_outputs.jl =====================
# Checks `primal_outputs = false` of stade_adjoint: the generator drops the forward sweep, and the stores of primal arrays that nothing reads, when the backward sweep
# does not need them. The primal arrays of the kernel are then left at their input values.
#   - units: every kernel runs the default adjoint and the `primal_outputs = false` adjoint on the same random arrays. The adjoint (shadow) arrays must agree exactly.
#     When the pass reports a change, every array that the kernel writes must still hold its input values. When it reports none, the text must equal the default text.
#     Kernels: ping-pong stencil steps (the forward sweep goes), a one-step stencil with recomputed indices, a nonlinear map into a fresh array, a reduction into a
#     length-1 array, in-place updates that the backward sweep needs (kept), a running state, a conditional, a chain of reassignments.
#   - SABOTAGE: edits of the new code, each of which one check must catch.
#   julia validate_primal_outputs.jl        units, sabotage
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# (label, expected: true when the pass must change the code, the arrays that must hold their input values then, source)
const PO_UNIT = [
    ("ping-pong stencil steps", true, [:u, :v], """
function f(u, v, n)
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:3
        for i = 2:n - 1
            v[i] = u[i] + 0.25 * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        end
        for i = 2:n - 1
            u[i] = v[i] + 0.25 * (v[i - 1] - 2.0 * v[i] + v[i + 1])
        end
    end
    return nothing
end"""),
    ("a one-step periodic stencil with recomputed indices", true, [:un], """
function f(u, un, n)
    nn = n * 1
    for idx = 1:nn
        i = mod(idx - 1, n) + 1
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        c = u[idx]
        un[idx] = c + 0.1 * (u[im] + u[ip] - 2.0 * c)
    end
    return nothing
end"""),
    ("a nonlinear map into a fresh array", true, [:un], """
function f(u, w, un, n)
    for i = 1:n
        un[i] = u[i] * u[i] * w[i] + sin(u[i])
    end
    return nothing
end"""),
    ("a sum of squares into a length-1 array", true, [:out], """
function f(u, out, n)
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end"""),
    ("a tanh layer then a linear output layer (the store of the output and its inner-loop sum are dropped)", true, [:y], """
function f(x, w1, w2, h, y, n)
    for j = 1:n
        s = 0.0
        for i = 1:n
            s = s + w1[i] * x[j] * x[i]
        end
        h[j] = tanh(s)
    end
    for j = 1:n
        s = 0.0
        for i = 1:n
            s = s + w2[i] * h[i] * h[j]
        end
        y[j] = s
    end
    return nothing
end"""),
    ("an in-place nonlinear update that the backward sweep needs", false, Symbol[], """
function f(u, n)
    for t = 1:2
        for i = 1:n
            u[i] = u[i] * u[i] + 0.1
        end
    end
    return nothing
end"""),
    ("a running state that the next iteration reads", true, [:un], """
function f(u, un, n)
    s = 0.0
    for i = 1:n
        s = 0.9 * s + u[i]
        un[i] = s * s
    end
    return nothing
end"""),
    ("a conditional store", true, [:un], """
function f(u, un, n)
    for i = 1:n
        t = u[i]
        if t > 0.0
            un[i] = t * t
        else
            un[i] = 2.0 * t
        end
    end
    return nothing
end"""),
    ("a chain of reassignments into a fresh array", true, [:un], """
function f(u, w, un, n)
    for i = 1:n
        s = u[i] * w[i]
        s = s + 0.1 * s * s
        s = s * s + w[i]
        un[i] = s
    end
    return nothing
end"""),
    ("a second product of an array that the first loop wrote", true, [:un], """
function f(u, v, un, n)
    for i = 1:n
        v[i] = u[i] * u[i]
    end
    for i = 1:n
        un[i] = v[i] * v[i] + u[i]
    end
    return nothing
end"""),
]

# the arrays of an adjoint function: every non-stack array random, returns (after, before) by name
function po_stack_names(init::Expr)
    out = Symbol[]
    for st in init.args[2].args
        st isa Expr && st.head === :(=) && st.args[1] isa Symbol && st.args[2] isa Expr && st.args[2].head === :call && :undef in st.args[2].args && push!(out, st.args[1])
    end
    return out
end

function po_run(adj, kpp::Bool, n::Int, seed)
    fexpr, init = adj.adjoint, adj.initstacks
    m1, m2 = Module(), Module()
    f = Core.eval(m1, fexpr)
    fi = Core.eval(m2, init)
    narg = length(init.args[1].args) - 1
    stk = Base.invokelatest(fi, (narg == 0 ? () : (n,))...)
    stk = stk === nothing ? Any[] : stk isa Tuple ? collect(stk) : Any[stk]
    names = kpp ? Symbol[] : po_stack_names(init)
    args = fexpr.args[1].args[2:end]
    rng = MersenneTwister(seed)
    vals = Any[]
    after = Pair{Symbol,Any}[]
    before = Pair{Symbol,Any}[]
    si = 0
    for a in args
        if a === :n
            push!(vals, n)
        elseif startswith(string(a), "__") || endswith(string(a), "_stack") || endswith(string(a), "_stk") || (a in names)
            si += 1
            push!(vals, stk[si])
        else
            arr = randn(rng, max(n, 1))
            push!(vals, arr)
            push!(before, a => copy(arr))
            push!(after, a => arr)
        end
    end
    Base.invokelatest(f, vals...)
    GC.@preserve m1 m2 nothing  # the modules must outlive the call: a function of a collected module crashes the runtime
    return after, before
end

function po_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, want, unchanged, src) in PO_UNIT
        for kpp in (true, false)
            tag = "$(label) (keep_push_pop = $(kpp))"
            try
                ex = Meta.parse(src)
                a = STADE.stade_adjoint(ex; keep_push_pop = kpp, fuse_ii_loops = true)
                b = STADE.stade_adjoint(ex; keep_push_pop = kpp, fuse_ii_loops = true, primal_outputs = false)
                same = string(a.adjoint) == string(b.adjoint)
                ok_flag = b.primal_pruned == want && (b.primal_pruned || same) && !a.primal_pruned
                worst = 0.0
                kept_ok = true
                for n in (2, 3, 7)
                    ra, _ = po_run(a, kpp, n, n)
                    rb, before = po_run(b, kpp, n, n)
                    for ((na, xa), (nb, xb)) in zip(ra, rb)
                        if endswith(string(na), "b")
                            worst = max(worst, maximum(abs.(xa .- xb)))
                        end
                    end
                    for ((nb, xb), (n0, x0)) in zip(rb, before)
                        nb in unchanged && b.primal_pruned && (kept_ok &= (xb == x0))
                    end
                end
                check(ok_flag && worst == 0.0 && kept_ok, "$(tag): changed = $(b.primal_pruned) (want $(want)), adjoint difference $(worst), primal arrays kept = $(kept_ok)")
            catch e
                check(false, "$(tag): error $(first(split(sprint(showerror, e), "\n")))")
            end
        end
    end
    return bad
end


# the whole corpus: the adjoint with `primal_outputs = false` must give exactly the gradient of the default adjoint on the same baseline and seed
function po_corpus_checks(names = nothing; modes = (true, false), shard = (1, 1))
    bad = 0
    changed = 0
    total = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    dir = joinpath(@__DIR__, "val-corpus")
    files = sort(filter(f -> endswith(f, ".jl"), readdir(dir)))
    names === nothing || (files = [f for f in files if replace(f, ".jl" => "") in names])
    files = [f for (i, f) in enumerate(files) if mod(i - 1, shard[2]) == shard[1] - 1]
    mktempdir() do tmp
        for f in files
            path = joinpath(dir, f)
            for kpp in modes
                tag = "$(f) (keep_push_pop = $(kpp))"
                try
                    primal = STADE.io_read_corpus_entry(path)
                    kernel = STADE.parse_kernel(primal)
                    yp = joinpath(tmp, replace(f, ".jl" => "_$(kpp).yaml"))
                    STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = 3, int_hi = 5)
                    baseline = STADE.io_read_baseline_yaml(yp)
                    STADE.val_coerce_int_arrays!(kernel, baseline.values)
                    seed = STADE.val_random_values_like(kernel, baseline.values)
                    grads = Any[]
                    pruned = false
                    for po in (true, false)
                        out = STADE.stade_adjoint(primal; keep_push_pop = kpp, fuse_ii_loops = true, primal_outputs = po)
                        po || (pruned = out.primal_pruned)
                        fn = STADE.val_compile(out.adjoint)
                        init = STADE.val_compile(out.initstacks)
                        sn = kpp ? Symbol[] : Symbol[a for a in out.initstacks.args[1].args[2:end]]
                        x0 = STADE.val_flatten(kernel, baseline.values)
                        vals = STADE.val_unflatten(kernel, baseline.int_args, baseline.values, x0)
                        push!(grads, STADE.val_call_adjoint(fn, init, kernel, baseline.int_args, vals, seed; stack_arg_names = sn))
                    end
                    total += 1
                    pruned && (changed += 1)
                    check(grads[1] == grads[2], "$(tag): gradients equal, changed = $(pruned)")
                catch e
                    check(false, "$(tag): error $(first(split(sprint(showerror, e), "\n")))")
                end
            end
        end
    end
    println("corpus: $(changed) of $(total) adjoints changed")
    return bad
end


# handwritten descriptors for the two guards that no generated adjoint reaches: a table name in the backward sweep (only a ragged loop bound makes one) and a stack
# operation that feeds a scalar nothing reads
function po_handwritten_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    sig = (name = :f, args = [:u, :n], kinds = Dict(:u => :array_float, :n => :scalar_int, :x => :scalar_float, :y => :scalar_float), independents = [:u], dependents = [:u])
    kern = (sig = sig, body = Any[(kind = :assign, lhs = :x, rhs = 1.0), (kind = :assign, lhs = :y, rhs = 2.0)])
    # a table name that the backward sweep mentions: the forward sweep stays
    bwd = Any[:(ub[1] = ub[1] + tab[1])]
    check(!STADE.agen_forward_skippable(kern, Any[], bwd, Symbol[], [:tab]), "a table name in the backward sweep keeps the forward sweep")
    check(STADE.agen_forward_skippable(kern, Any[], bwd, Symbol[], [:other]), "an unrelated table name does not")
    # a scalar assigned from a stack pop is kept, a plain dead scalar is dropped
    body = Any[:(x = pop!(s)), :(y = 2.0 * 3.0), :(ub[1] = ub[1] + 1.0)]
    (out, changed) = STADE.agen_prune_dead_primal(kern, body, [:s])
    check(changed && length(out) == 2 && out[1] == :(x = pop!(s)), "a pop feeding an unread scalar is kept, the plain dead scalar goes ($(length(out)) statement(s) left)")
    return bad
end

function validate_primal_outputs(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += po_unit_checks()
        bad += po_handwritten_checks()
    end
    if :corpus in phases
        println("-- corpus")
        bad += po_corpus_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("a name that the backward sweep reads before it writes is accepted", "x in used && agen_needs_entry(seq, x) && return false", "nothing"),
            ("a stack that the backward sweep touches is accepted", "any(in(used), stack_names) && return false", "nothing"),
            ("a table of the backward sweep is accepted", "any(in(used), extra_names) && return false", "nothing"),
            ("every use of a name counts as a write before the read", "e isa Symbol && return e == x\n    e isa Expr || return false\n    if e.head == :block", "e isa Symbol && return false\n    e isa Expr || return false\n    if e.head == :block"),
            ("a forward sweep that holds adjoint statements is dropped", "k in (:scalar_float, :array_float) && agen_shadow(v) in fused && return false", "nothing"),
            ("a store into an array that is read is dropped", "dead_a = Set{Symbol}(a for a in warrays if !(a in reads))", "dead_a = Set{Symbol}(a for a in warrays)"),
            ("an assignment to a scalar that is read is dropped", "dead_s = Set{Symbol}(a for a in wscalars if !(a in reads))", "dead_s = Set{Symbol}(a for a in wscalars)"),
            ("a stack operation is dropped with its scalar", "(:push! in rs || :pop! in rs) || return nothing", "return nothing"),
            ("the keyword is ignored by stade_adjoint", "ii_plan = ii_plan, remat = remat, primal_outputs = primal_outputs)\nend\n\nfunction stade_hvp", "ii_plan = ii_plan, remat = remat)\nend\n\nfunction stade_hvp"),
            ("the forward sweep is never dropped", "skipped = !primal_outputs && agen_forward_skippable(kernel, fwd, bwd, agen_stack_names(sites), tier_b_extra_args)", "skipped = false"),
            ("dead stores are never pruned", "    if !primal_outputs\n        (body, dropped) = agen_prune_dead_primal(kernel, body, agen_stack_names(sites))", "    if false\n        (body, dropped) = agen_prune_dead_primal(kernel, body, agen_stack_names(sites))"),
            ("the forward sweep is dropped with the default setting", "skipped = !primal_outputs && agen_forward_skippable", "skipped = agen_forward_skippable"),
            ("an array written by the kernel is not a candidate", "warrays = Set{Symbol}(a for a in written if get(kernel.sig.kinds, a, :none) in (:array_float, :array_int))", "warrays = Set{Symbol}()"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases[intersect(eachindex(cases), slice)]
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && (check(true, "anchor \"$label\": found once"); continue)
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_primal_outputs.jl"))))
                    n = po_unit_checks() + po_handwritten_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all primal output checks passed" : "*** $bad primal output check(s) failed ***")
    return bad
end
abspath(PROGRAM_FILE) == abspath(joinpath(@__DIR__, "validate_primal_outputs.jl")) && exit(validate_primal_outputs(; phases = (:units, :sabotage)) == 0 ? 0 : 1)
