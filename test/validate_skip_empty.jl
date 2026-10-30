#=
    validate_skip_empty(dir="val-corpus"; only=nothing, sabotage=true)

Checks of the removal of launches of empty kernels (skip_empty, CUDA).

  1. UNIT: `cgen_empty_body` accepts a kernel of scalar assignments, loops and branches over pure expressions and refuses a store, an atomic, a call with an
     effect and a `while`. `cgen_skip_empty` deletes the plain launches of an empty kernel, also inside a loop, and the definition once nothing names it.
  2. CORPUS: with the keyword on no launch of an empty kernel is left in the initstacks plan or the adjoint plan, a kernel with a store is never removed,
     and `skip_empty = false` writes the v0.4.48 text, which still has the launch.
  3. SABOTAGE: a gate that calls every kernel empty must fail the checks.
=#
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

function validate_skip_empty(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    cuda = STADE.cgen_backend_cuda()
    kern(src) = Meta.parse(src)
    e1 = kern("function k1!(n, nf)\n    __tid = 1\n    if __tid > div(n - 1, 1) + 1\n        return nothing\n    end\n    for f = 1:nf\n    end\n    return nothing\nend")
    e2 = kern("function k2!(a, n)\n    __tid = 1\n    a[__tid] = 1.0\n    return nothing\nend")
    e3 = kern("function k3!(a, n)\n    __tid = 1\n    CUDA.@atomic a[__tid] += 1.0\n    return nothing\nend")
    e4 = kern("function k4!(a, n)\n    __tid = 1\n    foo(a)\n    return nothing\nend")
    e5 = kern("function k5!(a, n)\n    t = 1\n    while t < n\n        t = t + 1\n    end\n    return nothing\nend")
    e6 = kern("function k6!(a, n)\n    s = a[1] + n\n    if s > 0\n        s = s * 2\n    end\n    return nothing\nend")
    check(STADE.cgen_empty_body(e1.args[2]) && STADE.cgen_empty_body(e6.args[2]), "scalar assignments, loops and branches over pure expressions are empty")
    check(!STADE.cgen_empty_body(e2.args[2]) && !STADE.cgen_empty_body(e3.args[2]) && !STADE.cgen_empty_body(e4.args[2]) && !STADE.cgen_empty_body(e5.args[2]), "a store, an atomic, a call with an effect and a while are not")
    host = kern("function h(a, n, nf)\n    @cuda threads = 256 blocks = 1 k1!(n, nf)\n    @cuda threads = 256 blocks = 1 k2!(a, n)\n    for t = 1:3\n        @cuda threads = 256 blocks = 1 k1!(n, nf)\n    end\n    return nothing\nend")
    nh, nk, gone = STADE.cgen_skip_empty(host, Expr[e1, e2], cuda)
    count_launch(h, nm) = (c = 0; walk(x) = x isa Expr && (x.head === :macrocall && x.args[end] isa Expr && x.args[end].head === :call && x.args[end].args[1] === nm && (c += 1); foreach(walk, x.args)); walk(h); c)
    check(gone == [:k1!] && count_launch(nh, :k1!) == 0 && count_launch(nh, :k2!) == 1, "the plain launches of an empty kernel go, also inside a loop, and the one with a store stays")
    check(length(nk) == 1 && nk[1].args[1].args[1] === :k2!, "the definition of the removed kernel goes and the other stays")
    nh2, nk2, gone2 = STADE.cgen_skip_empty(kern("function h2(a, n, nf)\n    @cuda threads = 256 blocks = 1 k2!(a, n)\n    return nothing\nend"), Expr[e2], cuda)
    check(isempty(gone2) && length(nk2) == 1, "a plan with no empty kernel comes back as it was")
    # corpus
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    total = 0
    for name in names, mode in (:adjoint, :hvp)
        primal = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        g = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false) : STADE.stade_adjoint(primal; keep_push_pop = false)
        for (lab, ex) in (("initstacks", g.initstacks), (string(mode), mode == :hvp ? g.hvp : g.adjoint))
            on = STADE.stade_gpu(ex, cuda; inbounds = false); off = STADE.stade_gpu(ex, cuda; inbounds = false, skip_empty = false)
            empties_off = [k.args[1].args[1] for k in off.kernels if k.head === :function && STADE.cgen_empty_body(k.args[2])]
            empties_on = [k.args[1].args[1] for k in on.kernels if k.head === :function && STADE.cgen_empty_body(k.args[2])]
            total += length(empties_off)
            if !isempty(empties_off)
                check(isempty(empties_on) && all(n -> count_launch(on.host, n) == 0 && count_launch(off.host, n) >= 1, empties_off) && length(on.kernels) == length(off.kernels) - length(empties_off), "$name [$lab]: $(length(empties_off)) empty launch(es) removed")
            else
                check(all(string(a) == string(b) for (a, b) in zip(on.kernels, off.kernels)) && string(on.host) == string(off.host), "$name [$lab]: no empty kernel, same text")
            end
        end
    end
    println("\n$total empty kernel(s) in the corpus plans")
    check(total > 0, "at least one corpus plan has an empty kernel")
    if sabotage
        for (label, sab, cases) in (("every kernel is empty", "cgen_empty_body(e) = true", ["dotprod"]),
                                    ("a store counts as pure", "cgen_empty_body(e) = e isa Expr ? !(e.head === :while) : true", ["dotprod"]))
            script = """
                include($(repr(joinpath(@__DIR__, "validate_skip_empty.jl"))))
                Core.eval(STADE, Meta.parse($(repr(sab))))
                println("RESULT ", try validate_skip_empty($(repr(dir)); only = $(repr(cases)), sabotage = false) catch; 1 end)
            """
            o = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
            m = match(r"RESULT (\d+)", o)
            check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"$label\": a check fails")
        end
    end
    println("\n", bad == 0 ? "all skip-empty checks passed" : "*** $bad skip-empty check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_skip_empty(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
