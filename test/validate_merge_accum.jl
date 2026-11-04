# ==================== validate_merge_accum.jl =====================
# Checks `cgen_merge_accum`, the GPU-stage pass that merges two accumulations into one array element (`A[i] = A[i] + e1` ... `A[i] = A[i] + e2` into one statement).
#   - units: handwritten functions, run before and after the pass on the same random arrays (they must agree to rounding), and the number of accumulation statements
#     must be the expected one. Cases that must merge (a scalar statement between, three in a row, inside a loop body) and cases that must not (the array read between,
#     the index or the first value reassigned between, a loop between, a second value that reads the array).
#   - corpus: the adjoint of every corpus kernel, run before and after the pass, gradients equal to rounding. The number of adjoints that change is printed.
#   - SABOTAGE: edits of the pass, each of which one unit must catch.
#   julia validate_merge_accum.jl        units, sabotage
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# (label, expected accumulation statements on `a` after the pass, source)
const MA_UNIT = [
    ("a scalar statement between", 1, """
function f(a, b, c, n)
    for i = 1:n
        j = i
        t = b[i] * 2.0
        a[j] = a[j] + t
        s = c[i] + 1.0
        a[j] = a[j] + s
    end
    return nothing
end"""),
    ("three in a row", 1, """
function f(a, b, c, n)
    for i = 1:n
        a[i] = a[i] + b[i]
        a[i] = a[i] + c[i]
        a[i] = a[i] + b[i] * c[i]
    end
    return nothing
end"""),
    ("a plain statement between that does not touch the first value", 1, """
function f(a, b, c, n)
    q = 0.0
    for i = 1:n
        a[i] = a[i] + b[i]
        q = q + c[i]
        a[i] = a[i] + q
    end
    return nothing
end"""),
    ("the array is read between", 2, """
function f(a, b, c, n)
    for i = 1:n
        a[i] = a[i] + b[i]
        t = a[i] * 2.0
        a[i] = a[i] + t
    end
    return nothing
end"""),
    ("the first value is reassigned between", 2, """
function f(a, b, c, n)
    for i = 1:n
        t = b[i]
        a[i] = a[i] + t
        t = c[i]
        a[i] = a[i] + 1.0
    end
    return nothing
end"""),
    ("the index is reassigned between", 2, """
function f(a, b, c, n)
    for i = 1:n
        k = i
        a[k] = a[k] + b[i]
        k = n + 1 - i
        a[k] = a[k] + c[i]
    end
    return nothing
end"""),
    ("a loop between", 2, """
function f(a, b, c, n)
    for i = 1:n
        a[i] = a[i] + b[i]
        for m = 1:2
            c[i] = c[i] * 0.5
        end
        a[i] = a[i] + c[i]
    end
    return nothing
end"""),
    ("the first value reads the array (it is evaluated before either update)", 1, """
function f(a, b, c, n)
    for i = 1:n
        a[i] = a[i] + a[i] * 0.5
        a[i] = a[i] + b[i]
    end
    return nothing
end"""),
    ("the second value reads the array", 2, """
function f(a, b, c, n)
    for i = 1:n
        a[i] = a[i] + b[i]
        a[i] = a[i] + a[i] * 0.5
    end
    return nothing
end"""),
    ("another array written between", 2, """
function f(a, b, c, n)
    for i = 1:n
        a[i] = a[i] + b[i]
        c[i] = c[i] + 1.0
        a[i] = a[i] + c[i]
    end
    return nothing
end"""),
    ("different elements", 2, """
function f(a, b, c, n)
    for i = 1:n
        a[i] = a[i] + b[i]
        a[n + 1 - i] = a[n + 1 - i] + c[i]
    end
    return nothing
end"""),
]

function ma_count_accum(e, arr::Symbol)
    n = 0
    function walk(x)
        x isa Expr || return
        if x.head == :(=) && STADE.cgen_accum_parts(x) !== nothing && STADE.cgen_accum_parts(x)[1].args[1] == arr
            n += 1
        end
        foreach(walk, x.args)
    end
    walk(e)
    return n
end

function ma_run(fexpr::Expr, n::Int, seed)
    m = Module()
    f = Core.eval(m, fexpr)
    rng = MersenneTwister(seed)
    args = Any[]
    out = Any[]
    for a in fexpr.args[1].args[2:end]
        if a === :n
            push!(args, n)
        else
            arr = randn(rng, n)
            push!(args, arr)
            push!(out, arr)
        end
    end
    Base.invokelatest(f, args...)
    GC.@preserve m nothing  # the module must outlive the call
    return out
end

function ma_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, want, src) in MA_UNIT
        try
            ex = Meta.parse(src)
            mx = STADE.cgen_merge_accum(ex)
            got = ma_count_accum(mx, :a)
            worst = 0.0
            for n in (3, 8)
                ra = ma_run(ex, n, n)
                rb = ma_run(mx, n, n)
                for (x, y) in zip(ra, rb)
                    worst = max(worst, maximum(abs.(x .- y)))
                end
            end
            check(got == want && worst <= 1e-12, "$(label): $(got) accumulation statement(s) (want $(want)), difference $(worst)")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    return bad
end

# the corpus: gradients of the adjoint before and after the pass agree to rounding
function ma_corpus_checks(names = nothing; modes = (true, false), shard = (1, 1))
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
                    out = STADE.stade_adjoint(primal; keep_push_pop = kpp, fuse_ii_loops = true)
                    merged = STADE.cgen_merge_accum(out.adjoint)
                    grads = Any[]
                    for ad in (out.adjoint, merged)
                        fn = STADE.val_compile(ad)
                        init = STADE.val_compile(out.initstacks)
                        sn = kpp ? Symbol[] : Symbol[a for a in out.initstacks.args[1].args[2:end]]
                        x0 = STADE.val_flatten(kernel, baseline.values)
                        vals = STADE.val_unflatten(kernel, baseline.int_args, baseline.values, x0)
                        push!(grads, STADE.val_call_adjoint(fn, init, kernel, baseline.int_args, vals, seed; stack_arg_names = sn))
                    end
                    total += 1
                    diff = string(out.adjoint) != string(merged)
                    diff && (changed += 1)
                    g1, g2 = Float64[], Float64[]
                    foreach(x -> append!(g1, x isa Number ? [Float64(x)] : vec(Float64.(x))), grads[1])
                    foreach(x -> append!(g2, x isa Number ? [Float64(x)] : vec(Float64.(x))), grads[2])
                    scale = max(1.0, maximum(abs.(g1)))
                    check(length(g1) == length(g2) && maximum(abs.(g1 .- g2)) <= 1e-10 * scale, "$(tag): gradients agree, changed = $(diff)")
                catch e
                    check(false, "$(tag): error $(first(split(sprint(showerror, e), "\n")))")
                end
            end
        end
    end
    println("corpus: $(changed) of $(total) adjoints changed")
    return bad
end

function validate_merge_accum(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    :units in phases && (println("-- units"); bad += ma_unit_checks())
    :corpus in phases && (println("-- corpus"); bad += ma_corpus_checks())
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("the array read between is ignored", "(st.args[1] in syms1 || arr in rs || :push! in rs || :pop! in rs) && break", "(st.args[1] in syms1 || :push! in rs || :pop! in rs) && break"),
            ("a name of the index or of the first value reassigned between is ignored", "(st.args[1] in syms1 || arr in rs || :push! in rs || :pop! in rs) && break", "(arr in rs || :push! in rs || :pop! in rs) && break"),
            ("a statement that is not a scalar assignment is stepped over", "(st isa Expr && st.head == :(=) && st.args[1] isa Symbol) || break", "true || break"),
            ("a second value that reads the array is merged", "if !(arr in syms2) && !(:push! in syms2 || :pop! in syms2)", "if !(:push! in syms2 || :pop! in syms2)"),
            ("different elements are merged", "if p2 !== nothing && p2[1] == lhs", "if p2 !== nothing && p2[1].args[1] == lhs.args[1]"),
            ("the pass is never applied", "    merge_accum && (expr = cgen_merge_accum(expr))\n    plan = cgen_emit(", "    plan = cgen_emit("),
            ("the JACC path ignores the pass", "    merge_accum && (expr = cgen_merge_accum(expr))\n    plan = jgen_emit(", "    plan = jgen_emit("),
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
                    include($(repr(joinpath(@__DIR__, "validate_merge_accum.jl"))))
                    n = ma_unit_checks() + ma_gpu_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all merge accumulation checks passed" : "*** $bad merge accumulation check(s) failed ***")
    return bad
end

# the pass is applied by the CUDA and the JACC generators, and `merge_accum = false` leaves the text alone
function ma_gpu_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    src = """
function f_b(a, b, c, n)
    for i = 1:n
        j = i
        t = b[i] * 2.0
        a[j] = a[j] + t
        s = c[i] + 1.0
        a[j] = a[j] + s
    end
    return nothing
end"""
    ex = Meta.parse(src)
    for (name, gen) in (("cuda", (e; kw...) -> STADE.stade_cuda(e; kw...)), ("jacc", (e; kw...) -> STADE.stade_jacc(e; kw...)))
        try
            on = gen(ex)
            off = gen(ex; merge_accum = false)
            txt(p) = join((string(k) for k in vcat(p.kernels, [p.host])), "\n")
            a = count("a[j]", txt(on)) + count("(a, j", txt(on))
            b = count("a[j]", txt(off)) + count("(a, j", txt(off))
            check(a < b, "$(name): fewer accumulations into a[j] with the pass ($(a)) than without ($(b))")
        catch e
            check(false, "$(name): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    return bad
end

function validate_merge_accum_all(; phases = (:units, :sabotage))
    bad = validate_merge_accum(; phases)
    :units in phases && (bad += ma_gpu_checks())
    return bad
end
abspath(PROGRAM_FILE) == abspath(joinpath(@__DIR__, "validate_merge_accum.jl")) && exit(validate_merge_accum_all(; phases = (:units, :sabotage)) == 0 ? 0 : 1)
