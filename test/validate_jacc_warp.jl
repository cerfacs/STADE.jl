include(joinpath(@__DIR__, "sabotage_run.jl"))
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_jacc_warp(dir="val-corpus"; only=nothing, sabotage=true)

Structural gate for the warp-aggregated atomics of the JACC path (v0.4.71). No GPU needed; `validate_jacc_gpu` runs of the V100 check the numbers.

For every corpus kernel, adjoint and HVP (`keep_push_pop = false`, `fuse_ii_loops = true`), with `split_reduce = 0` and with the split twin on:

  1. EQUIVALENCE: replacing each `__stade_wa_add!(A, i, v)` call of the default output by `Atomix.@atomic A[i] += v` gives exactly the output of
     `warp_atomics = false`. The pass changes nothing but the atomic sites, in the host, in the plain kernels and in the twins.
  2. HELPERS: the definitions are present exactly when a call is; they have the CUDA device array method and the generic Atomix method, and no other CUDA call
     reaches a kernel through them; `warp_atomics = false` has none.
  3. SABOTAGE: with the rewrite putting the index and the value in the wrong order, check 1 must fail on a kernel with an atomic.
"""
function validate_jacc_warp(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    wa = STADE.cgen_warp_add_name()
    atomix = Expr(:., :Atomix, QuoteNode(Symbol("@atomic")))
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    function unwa(e)
        e isa Expr || return e
        args = Any[unwa(a) for a in e.args]
        if e.head == :call && length(args) == 4 && args[1] === wa
            return Expr(:macrocall, atomix, nothing, Expr(:(+=), Expr(:ref, args[2], args[3]), args[4]))
        end
        return Expr(e.head, args...)
    end
    ncalls(e) = e isa Expr ? (e.head == :call && !isempty(e.args) && e.args[1] === wa ? 1 : 0) + sum(ncalls(a) for a in e.args; init = 0) : 0
    src(e) = STADE.io_expr_to_source(e)
    total = 0; kernels_with = 0
    for name in names
        pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp))
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = true) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            for split in (0, 16)
                kw = (inbounds = false, dead_scalars = false, split_reduce = split, split_min_inner = 0, split_max_out = 10^9)
                on = try STADE.stade_jacc(e; kw...) catch; continue end
                off = STADE.stade_jacc(e; warp_atomics = false, kw...)
                n = sum(ncalls(k) for k in on.kernels; init = 0)
                total += n; n > 0 && (kernels_with += 1)
                same = length(on.kernels) == length(off.kernels) && all(src(unwa(a)) == src(b) for (a, b) in zip(on.kernels, off.kernels)) && src(on.host) == src(off.host)
                check(same, "$name [$mode, split $split]: $n warp call(s), output equals the plain atomics after substitution")
                check((n > 0) == !isempty(on.helpers), "$name [$mode, split $split]: helpers present iff a call is ($(length(on.helpers)) definition(s))")
                check(isempty(off.helpers) && !any(k -> STADE.cgen_expr_calls(k, wa), off.kernels), "$name [$mode, split $split]: warp_atomics = false has no helper and no call")
                # a warp call never has an address that differs between adjacent lanes (class :D), a plain atomic always has one (the gate: the helper only pays where the lanes can agree)
                gate_ok = true
                for k in on.kernels
                    k.head === :function || continue
                    cls = STADE.jgen_warp_classes(k)
                    function walk(x)
                        x isa Expr || return
                        if x.head == :call && length(x.args) == 4 && x.args[1] === wa
                            STADE.jgen_warp_class(x.args[3], cls) === :D && (gate_ok = false)
                        elseif x.head == :macrocall && length(x.args) == 3 && x.args[1] == atomix
                            st = STADE.cgen_atomic_site(x)
                            st !== nothing && STADE.jgen_warp_class(st.idx, cls) !== :D && (gate_ok = false)
                        end
                        foreach(walk, x.args)
                    end
                    walk(k)
                end
                check(gate_ok, "$name [$mode, split $split]: no warp call has a lane-distinct address, every plain atomic left has one")
            end
        end
    end
    println("\n$total warp call(s) in $kernels_with kernel plan(s)")
    check(total > 0, "at least one corpus kernel emits a warp call")
    # the gate on a hand-written kernel: which addresses reach the helper
    function gate_case(label, body, want_warp, want_plain)
        k = Meta.parse("function jacc_kernel_g(__jacc_i, A, B, idx, n, v)\n$body\nreturn nothing\nend")
        t = STADE.jgen_warp_atomics(k)
        nw = ncalls(t)
        np = length(findall("Atomix.@atomic", src(t)))
        check(nw == want_warp && np == want_plain, "gate: $label ($nw warp, $np plain; want $want_warp, $want_plain)")
    end
    gate_case("a fixed address", "Atomix.@atomic A[3] += v", 1, 0)
    gate_case("the thread index", "Atomix.@atomic A[__jacc_i] += v", 0, 1)
    gate_case("a scalar assigned from the thread index", "j = __jacc_i + 1\nAtomix.@atomic A[j] += v", 0, 1)
    gate_case("a chain of two scalars", "j = __jacc_i\nm = 2 * j\nAtomix.@atomic A[m] += v", 0, 1)
    gate_case("a loop variable with fixed bounds", "for k = 1:n\nAtomix.@atomic A[k] += v\nend", 1, 0)
    gate_case("a loop variable bounded by the thread index", "for k = 1:__jacc_i\nAtomix.@atomic A[k] += v\nend", 1, 0)
    gate_case("an index read at the thread index (a gather index may repeat)", "q = idx[__jacc_i]\nAtomix.@atomic A[q] += v", 1, 0)
    gate_case("an index read at a fixed position", "q = idx[2]\nAtomix.@atomic A[q] += v", 1, 0)
    gate_case("one of each", "Atomix.@atomic A[1] += v\nAtomix.@atomic B[__jacc_i] += v", 1, 1)
    gate_case("div of the thread index by a uniform value (lanes agree in runs)", "r = div(__jacc_i - 1, n) + 1\nAtomix.@atomic A[r] += v", 1, 0)
    gate_case("mod of the thread index (distinct lanes)", "c = mod(__jacc_i - 1, n) + 1\nAtomix.@atomic A[c] += v", 0, 1)
    gate_case("a flattened 2-D position (mod part + div part times a width)", "c = mod(__jacc_i - 1, n) + 1\nr = div(__jacc_i - 1, n) + 1\nAtomix.@atomic A[c + (r - 1) * n] += v", 0, 1)
    gate_case("the same position shifted by a neighbour offset", "c = mod(__jacc_i - 1, n) + 1\nr = div(__jacc_i - 1, n) + 1\nAtomix.@atomic A[c + (r - 1) * n + 1] += v", 0, 1)
    gate_case("a row index times a width plus an inner loop variable", "r = div(__jacc_i - 1, n) + 1\nfor k = 1:n\nAtomix.@atomic A[(r - 1) * n + k] += v\nend", 1, 0)
    gate_case("the thread index scaled by a literal", "Atomix.@atomic A[3 * __jacc_i] += v", 0, 1)
    gate_case("the sum of two lane-distinct values", "Atomix.@atomic A[__jacc_i + mod(__jacc_i, 7)] += v", 1, 0)
    gate_case("a scalar assigned twice, once from the thread index", "j = 1\nj = __jacc_i\nAtomix.@atomic A[j] += v", 1, 0)
    hs = join(src.(STADE.cgen_warp_helper_exprs(; jacc = true)), "\n")
    check(occursin("arr::CUDA.CuDeviceArray, idx::Integer, val::Float64", hs) && occursin("arr::CUDA.CuDeviceArray, idx::Integer, val::Float32", hs) && occursin("Atomix.@atomic arr[idx] += val", hs),
          "the JACC helpers: typed methods for CUDA device arrays, Atomix for any other array")
    hc = join(src.(STADE.cgen_warp_helper_exprs()), "\n")
    check(!occursin("CuDeviceArray", hc) && !occursin("Atomix", hc), "the CUDA helpers are unchanged: no type annotation, no Atomix")
    if sabotage
        sab = "function jgen_warp_atomics(e)\n    e isa Expr || return e\n    atomix = Expr(:., :Atomix, QuoteNode(Symbol(\"@atomic\")))\n" *
              "    if e.head === :macrocall && length(e.args) == 3 && e.args[1] == atomix\n        site = cgen_atomic_site(e)\n" *
              "        site === nothing || return Expr(:call, cgen_warp_add_name(), site.arr, jgen_warp_atomics(site.val), jgen_warp_atomics(site.idx))\n    end\n" *
              "    return Expr(e.head, Any[jgen_warp_atomics(a) for a in e.args]...)\nend"
        script = """
            include($(repr(joinpath(@__DIR__, "validate_jacc_warp.jl"))))
            Core.eval(STADE, Meta.parse($(repr(sab))))
            println("RESULT ", validate_jacc_warp($(repr(dir)); only = ["idm_tile"], sabotage = false))
        """
        out = sab_read(script)
        m = match(r"RESULT (\d+)", out)
        check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"index and value swapped\": idm_tile fails the equivalence check")
        script2 = """
            include($(repr(joinpath(@__DIR__, "validate_jacc_warp.jl"))))
            STADE.jgen_warp_classes(k::Expr) = Dict{Symbol,Symbol}()
            println("RESULT ", validate_jacc_warp($(repr(dir)); only = ["idm_tile"], sabotage = false))
        """
        out2 = sab_read(script2)
        m2 = match(r"RESULT (\d+)", out2)
        check(m2 !== nothing && parse(Int, m2.captures[1]) > 0, "sabotage \"no address is lane-distinct\": the gate checks fail")
        script3 = """
            include($(repr(joinpath(@__DIR__, "validate_jacc_warp.jl"))))
            STADE.jgen_warp_classes(k::Expr) = Dict{Symbol,Symbol}(:__jacc_i => :D)
            println("RESULT ", validate_jacc_warp($(repr(dir)); only = ["idm_tile"], sabotage = false))
        """
        out3 = sab_read(script3)
        m3 = match(r"RESULT (\d+)", out3)
        check(m3 !== nothing && parse(Int, m3.captures[1]) > 0, "sabotage \"the class does not follow assignments\": the gate checks fail")
    end
    println("\n", bad == 0 ? "all JACC warp atomics checks passed" : "*** $bad JACC warp atomics check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_jacc_warp(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
