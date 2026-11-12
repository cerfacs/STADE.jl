# Regression for v0.5.14: a scalar boxed into a 1-element device array stays a box in host code.
#   Defect: phc_gs1d (counted Gauss-Seidel). The adjoint of the scalar input h2 is boxed for the kernel of the reverse sweep, while the forward-sweep reverse
#   loop stays on the host and ran `h2b = h2b + f[i1] * c`: CuArray + Float64, a MethodError on the V100 in both modes.
#   Fix: cgen_host_box_refs rewrites each host reference to `h2b[1]` under one allowscalar block and leaves launch arguments alone.
# Reads generated code. No device is used. The device check is the V100 run of phc_gs1d.
include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

hb_bare(e, vars, inlaunch = false) = e isa Symbol ? (!inlaunch && e in vars ? 1 : 0) :
    e isa Expr ? (e.head == :macrocall && any(a -> a isa Symbol && string(a) in ("@cuda", "@parallel_for") || a isa Expr && a.head == :. && string(a.args[end].value) in ("@cuda", "@parallel_for"), e.args[1:1]) ? 0 :
                  e.head == :ref && length(e.args) == 2 && e.args[1] in vars ? 0 : sum(hb_bare(a, vars) for a in e.args; init = 0)) : 0

function validate_host_box_refs(; verbose = true)
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    asc = Expr(:., :CUDA, QuoteNode(Symbol("@allowscalar")))
    vars = Set([:hb])
    # unit: a launch keeps the box, a bare host statement is wrapped, a statement already inside allowscalar is not wrapped twice, a condition is wrapped
    raw = Meta.parse("""begin
        @cuda threads = 4 kern!(a, hb)
        for i = 1:n
            hb = hb + a[i]
        end
        if hb > 0.0
            x = 1.0
        end
        CUDA.@allowscalar begin
            hb = hb + 1.0
        end
    end""")
    out = STADE.cgen_host_box_refs(raw, vars, asc)
    txt = string(out)
    say(occursin("kern!(a, hb)", txt), "launch argument rewritten: $txt")
    say(occursin("hb[1] = hb[1] + a[i]", txt) && occursin("allowscalar", txt), "bare statement not rewritten: $txt")
    say(occursin("hb[1] > 0.0", txt), "condition not rewritten: $txt")
    say(count("@allowscalar", txt) == 3, "expected 3 allowscalar blocks, got $(count("@allowscalar", txt)): $txt")
    say(hb_bare(out, vars) == 0, "bare uses left after the rewrite")
    # sabotage: the detector must see the defect in the unrewritten form
    say(hb_bare(raw, vars) >= 3, "sabotage: detector missed the bare uses of the unrewritten form ($(hb_bare(raw, vars)))")
    say(STADE.cgen_host_box_refs(raw, Set{Symbol}(), asc) === raw, "empty box set must return the expression unchanged")
    say(STADE.cgen_host_box_refs(raw, vars, nothing) === raw, "no allowscalar macro must return the expression unchanged")
    # generated code of the counted corpus: no bare use of a boxed scalar in any host function, on CUDA and on JACC
    dir = joinpath(@__DIR__, "counted-corpus")
    bk = STADE.cgen_backend_cuda()
    seen_gs1d = false
    for f in sort(readdir(dir))
        endswith(f, ".jl") || continue
        e = STADE.io_read_corpus_entry(joinpath(dir, f))
        for mode in (:adjoint, :hvp)
            g = mode == :hvp ? STADE.stade_hvp(e; keep_push_pop = false) : STADE.stade_adjoint(e; keep_push_pop = false)
            gk = mode == :hvp ? g.hvp : g.adjoint
            pi_ = STADE.stade_gpu(g.initstacks, bk)
            pg = STADE.stade_gpu(gk, bk; init_hosts = Dict{Symbol,Expr}(pi_.host.args[1].args[1] => pi_.host))
            t = STADE.io_expr_to_source(pg.host)
            boxed = Set(Symbol(m.captures[1]) for m in eachmatch(r"(\w+) = CuArray\(\[\1\]\)", t))
            for v in boxed
                say(!occursin(Regex("\\b" * string(v) * " = " * string(v) * " [-+*/]"), t), "$f $mode cuda: bare accumulation of the boxed scalar $v")
            end
            if f == "phc_gs1d.jl" && mode == :adjoint
                seen_gs1d = true
                say(occursin("h2b[1] = h2b[1] +", t), "phc_gs1d cuda: the host sweep does not accumulate into h2b[1]")
                say(!occursin(r"\bh2b = h2b \+", t), "phc_gs1d cuda: bare h2b accumulation remains")
                say(occursin(r"cuda_kernel_\w+!?\(.*\bh2b\b", t), "phc_gs1d cuda: the launch no longer passes the box")
            end
            pj = STADE.stade_jacc(gk; init_hosts = Dict{Symbol,Expr}(g.initstacks.args[1].args[1] => g.initstacks))
            tj = STADE.io_expr_to_source(pj.host)
            for m in eachmatch(r"(\w+) = JACC\.array\(\[\1\]\)", tj)
                v = m.captures[1]
                say(!occursin(Regex("\\b" * v * " = " * v * " [-+*/]"), tj), "$f $mode jacc: bare accumulation of the boxed scalar $v")
            end
        end
    end
    say(seen_gs1d, "phc_gs1d is missing from the counted corpus")
    println(bad == 0 ? "all $checks host box reference checks passed" : "$bad of $checks host box reference checks NOT OK")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_host_box_refs() || exit(1)
end
