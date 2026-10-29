isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_atomic_hoist(dir="val-corpus"; only=nothing, sabotage=true)

Structural gate for loop-invariant atomic hoisting in the device backends. No GPU needed; `validate_corpus_gpu.jl` checks the numbers on a device.

Three groups of checks:

  1. UNITS: `cgen_emit_device_loop` on hand-written loops. A site hoists when its index is invariant in the loop and the loop holds no other use of its
     array; it must stay put when the index reads an array, uses the loop variable or a symbol assigned in the loop, when the loop reads the array, when
     the value mentions it, or when the array also has a non-invariant site. The flush is guarded by a flag, names do not collide with the body, the number
     of hoisted sites is capped, and `hoist = false` is the plain loop.
  2. ROUND TRIP: for every corpus kernel, adjoint and HVP, CUDA and JACC, inverting the hoist on the default output gives exactly the output with
     `hoist_atomics = false`. The pass changes nothing but the atomic sites.
  3. SABOTAGE: each guard removed in a copy of STADE.jl must make a unit check fail.
"""
function hoist_unit_checks(S)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    wa = S.cgen_warp_add_name()
    src(e) = S.io_expr_to_source(e)
    ncalls(e) = e isa Expr ? (e.head == :call && !isempty(e.args) && e.args[1] === wa ? 1 : 0) + (e.head == :macrocall && length(e.args) == 3 && e.args[3] isa Expr && e.args[3].head == :(+=) ? 1 : 0) + sum(ncalls(a) for a in e.args; init = 0) : 0
    loop_of(out) = only(e for e in out if e isa Expr && e.head == :for)
    flushes(out) = [e for e in out if e isa Expr && e.head == :if]
    emit(body; hoist = true, step = 1) = S.cgen_emit_device_loop(:e, 1, :n, step, Any[body...], hoist)
    wacall(a, i, v) = Expr(:call, wa, a, i, v)
    plain(body) = Any[S.emit_forloop(:e, 1, :n, 1, Any[body...])]

    # 1. invariant site, call form
    out = emit([wacall(:A, :k, :(x[e]))])
    check(length(flushes(out)) == 1 && ncalls(loop_of(out)) == 0 && ncalls(out[end]) == 1, "invariant call-form site: one flush after the loop, none inside")
    # 2. the loop reads the array
    body = [wacall(:A, :k, :(x[e])), :(t = A[3])]
    check(src(Expr(:block, emit(body)...)) == src(Expr(:block, plain(body)...)), "a read of the array in the loop blocks the hoist")
    # 3. index uses the loop variable
    body = [wacall(:A, :e, :(x[e]))]
    check(length(emit(body)) == 1 && ncalls(loop_of(emit(body))) == 1, "an index that uses the loop variable stays")
    # 4. index uses a symbol assigned in the loop
    body = [:(kk = k + e), wacall(:A, :kk, :(x[e]))]
    check(length(emit(body)) == 1, "an index that uses a symbol assigned in the loop stays")
    # 5. index reads an array
    body = [wacall(:A, :(ia[k]), :(x[e]))]
    check(length(emit(body)) == 1, "an index that reads an array stays")
    # 6. the value mentions the array
    body = [wacall(:A, :k, :(A[2] + x[e]))]
    check(length(emit(body)) == 1, "a value that mentions the array stays")
    # 7. one array with an invariant and a varying site; another array hoists in the same loop
    body = [wacall(:A, :k, :(x[e])), wacall(:A, :e, :(x[e])), wacall(:B, :k, :(x[e]))]
    out = emit(body)
    check(length(flushes(out)) == 1 && ncalls(loop_of(out)) == 2 && ncalls(out[end]) == 1 && occursin("B", src(out[end])), "only the array whose every site is invariant hoists")
    # 8. a site under an if
    body = [Expr(:if, :(x[e] > 0.0), Expr(:block, wacall(:A, :k, :(x[e]))))]
    out = emit(body)
    check(length(flushes(out)) == 1 && ncalls(loop_of(out)) == 0 && occursin("__ha1_A = 1", src(loop_of(out))), "a site under an if hoists, and the flag is set under the same if")
    # 9. the same key twice
    body = [wacall(:A, :k, :(x[e])), wacall(:A, :k, :(2.0 * x[e]))]
    out = emit(body)
    check(length(flushes(out)) == 1 && ncalls(loop_of(out)) == 0, "two sites with one key share one accumulator")
    # 10. macro forms (CUDA without warp helper, JACC)
    for (label, mac) in (("CUDA.@atomic", Expr(:., :CUDA, QuoteNode(Symbol("@atomic")))), ("Atomix.@atomic", Expr(:., :Atomix, QuoteNode(Symbol("@atomic")))))
        body = [Expr(:macrocall, mac, nothing, Expr(:(+=), :(A[k]), :(x[e])))]
        out = emit(body)
        check(length(flushes(out)) == 1 && ncalls(loop_of(out)) == 0 && occursin(split(label, ".")[end], src(out[end])), "$label form hoists and flushes with the same macro")
    end
    # 11. cap
    body = Any[wacall(Symbol("A", i), :k, :(x[e])) for i in 1:20]
    out = emit(body)
    check(length(flushes(out)) == S.cgen_hoist_max_sites() && ncalls(loop_of(out)) == 20 - S.cgen_hoist_max_sites(), "at most $(S.cgen_hoist_max_sites()) sites hoist per loop")
    # 12. rollback
    body = [wacall(:A, :k, :(x[e]))]
    check(src(Expr(:block, emit(body; hoist = false)...)) == src(Expr(:block, plain(body)...)), "hoist = false is the plain loop")
    # 13. a name already used in the body
    body = [:(__ac1_A = 1.0), wacall(:A, :k, :(x[e] + __ac1_A))]
    out = emit(body)
    s = src(Expr(:block, out...))
    check(length(flushes(out)) == 1 && occursin("__ac2_A", s), "an accumulator name already in the body is not reused")
    # 14. the flush is guarded by the flag, so a loop that never ran issues nothing
    out = emit([wacall(:A, :k, :(x[e]))])
    check(occursin("!= 0", src(out[end])) && occursin("__ha1_A = 0", src(out[1] isa Expr ? Expr(:block, out[1:2]...) : out[1])), "the flush runs only when the loop ran: flag cleared before the loop, tested after")
    # 15. descending loop
    out = emit([wacall(:A, :k, :(x[e]))]; step = -1)
    check(length(flushes(out)) == 1 && occursin("-1", src(loop_of(out))), "a descending loop hoists the same way")
    return bad
end

# Inverse of the hoist on generated code: rewrites `acc = acc + v; flag = 1` back to the atomic, drops the setup and the flush.
function hoist_invert(S, e)
    wa = S.cgen_warp_add_name()
    is_flush(x) = x isa Expr && x.head == :if && length(x.args) == 2 && x.args[1] isa Expr && x.args[1].head == :call && x.args[1].args[1] == :!= &&
                  x.args[1].args[2] isa Symbol && startswith(string(x.args[1].args[2]), "__ha") && x.args[2] isa Expr && x.args[2].head == :block &&
                  length(x.args[2].args) == 1 && S.cgen_atomic_site(x.args[2].args[1]) !== nothing
    function find_flush(x)
        x isa Expr || return nothing
        for (i, a) in enumerate(x.args)
            is_flush(a) && return (x, i)
        end
        for a in x.args
            r = find_flush(a)
            r === nothing || return r
        end
        return nothing
    end
    function rewrite(x, acc, flag, tmpl)
        x isa Expr || return x
        out = Any[]
        i = 1
        args = x.args
        while i <= length(args)
            a = args[i]
            if a isa Expr && a.head == :(=) && a.args[1] === acc && a.args[2] isa Expr && a.args[2].head == :call && a.args[2].args[1] == :+ && a.args[2].args[2] === acc &&
               i < length(args) && args[i + 1] == Expr(:(=), flag, 1)
                push!(out, S.cgen_atomic_rebuild(tmpl, a.args[2].args[3]))
                i += 2
            elseif a isa Expr && a.head == :(=) && a.args[1] in (acc, flag) && (a.args[2] == 0.0 || a.args[2] == 0)
                i += 1
            else
                push!(out, rewrite(a, acc, flag, tmpl))
                i += 1
            end
        end
        return Expr(x.head, out...)
    end
    while true
        r = find_flush(e)
        r === nothing && return e
        parent, i = r
        fl = parent.args[i]
        tmpl = fl.args[2].args[1]
        acc = S.cgen_atomic_site(tmpl).val
        flag = fl.args[1].args[2]
        deleteat!(parent.args, i)
        e = rewrite(e, acc, flag, tmpl)
    end
end

function validate_atomic_hoist(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    println("-- unit checks")
    bad += hoist_unit_checks(STADE)
    println("-- round trip over the corpus")
    cuda = STADE.cgen_backend_cuda()
    src(e) = STADE.io_expr_to_source(e)
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    hoisted_plans = 0; plans = 0; sites = 0
    for name in names
        pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp)), fuse in (false, true)
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = fuse) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            for (label, mk) in (("cuda", h -> STADE.stade_gpu(e, cuda; hoist_atomics = h, shared_private = false)), ("jacc", h -> STADE.stade_jacc(e; hoist_atomics = h)))
                on = mk(true); off = mk(false)
                plans += 1
                n = sum(count_hoists(k) for k in on.kernels; init = 0)
                sites += n; n > 0 && (hoisted_plans += 1)
                same = length(on.kernels) == length(off.kernels) && all(src(hoist_invert(STADE, deepcopy(a))) == src(b) for (a, b) in zip(on.kernels, off.kernels)) && src(on.host) == src(off.host)
                check(same, "$name [$mode f=$(Int(fuse)) $label]: $n hoisted site(s), inverting the hoist gives the output of hoist_atomics = false")
            end
        end
    end
    println("\n$sites hoisted site(s) in $hoisted_plans of $plans plan(s)")
    only === nothing && check(sites > 0, "at least one corpus kernel hoists")
    if sabotage
        repo = abspath(joinpath(@__DIR__, ".."))
        text = read(joinpath(repo, "src", "STADE.jl"), String)
        cases = [
            ("every use of the array must be a site", "keys = Any[k for k in keys if cgen_count_symbol(whole, k.arr) == inv_count[k.arr]]", "keys = keys"),
            ("index must not use a loop-assigned symbol or the loop variable", "isempty(intersect(cgen_expr_symbols(site.idx), assigned))", "true"),
            ("index must not read an array", "!cgen_expr_has_ref(site.idx) && isempty", "isempty"),
            ("loop variable counts as assigned", "Set{Symbol}([var]))", "Set{Symbol}())"),
            ("flush guarded by the flag", "push!(post, emit_if(Expr(:call, :!=, flags[n], 0), Any[cgen_atomic_rebuild(e, accs[n])], Any[]))", "append!(post, Any[cgen_atomic_rebuild(e, accs[n])])"),
            ("accumulator names avoid the body", "while cgen_count_symbol(whole, Symbol(\"__ac\", i, \"_\", k.arr)) > 0 || Symbol(\"__ac\", i, \"_\", k.arr) in accs", "while false"),
            ("cap on hoisted sites", "return length(keys) > cgen_hoist_max_sites() ? keys[1:cgen_hoist_max_sites()] : keys", "return keys"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases
                count_old = length(findall(old, text))
                if count_old != 1
                    check(false, "sabotage \"$label\": anchor found $count_old times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    include($(repr(joinpath(@__DIR__, "validate_atomic_hoist.jl"))))
                    println("RESULT ", hoist_unit_checks(STADE))
                """
                out = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"$label\": a unit check fails")
            end
        end
    end
    println("\n", bad == 0 ? "all atomic hoist checks passed" : "*** $bad atomic hoist check(s) failed ***")
    return bad
end

# hoisted sites in one kernel: flags set to 1 beside an accumulation
count_hoists(e) = e isa Expr ? (e.head == :(=) && e.args[1] isa Symbol && startswith(string(e.args[1]), "__ha") && e.args[2] == 1 ? 1 : 0) + sum(count_hoists(a) for a in e.args; init = 0) : 0

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_atomic_hoist(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
