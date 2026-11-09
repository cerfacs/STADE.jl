include(joinpath(@__DIR__, "sabotage_run.jl"))
# ==================== validate_colblock.jl =====================
# Checks the column-block twin (`col_block`, CUDA, v0.4.65): a flat loop `for idx = 1:R*N` with `i = div(idx - 1, N) + 1`, `j = mod(idx - 1, N) + 1` and an accumulation
# `z[j] = z[j] + e` (the push form of a layer) gets a twin that runs a block of 32 columns x 8 row lanes per tile of 128 rows. The CPU shim has no static shared memory, so
# these checks are on the recognizer and on the generated text; the device result is checked by the V100 parity job (twin against plain kernel on ragged sizes).
#   - units: handwritten kernels, compiled with `stade_cuda`; `cgen_colblock_kernel` is asked for a twin of every kernel. A case lists the number of twins it must give.
#   - mutants: generated kernels edited by hand for a shape no source reaches.
#   - text: the tile sum has all 8 lanes in fixed order, one barrier before it, the column guard on the loop and on the flush, one lane flushes.
#   - launch: the host holds the run-time condition (whole rows, inner length, rows, grid limit) and keeps the plain launch as the fallback; the default leaves the text alone.
#   - SABOTAGE: edits of the recognizer and of the emitter, each of which one check must catch.
#   julia validate_colblock.jl        units, sabotage
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

const CB_UNIT = [
    ("a push-form layer", 1, """
function f(w, v, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        z[j] = z[j] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("two accumulations into two arrays", 1, """
function f(w, v, z, q, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        z[j] = z[j] + w[idx] * v[i]
        q[j] = q[j] + w[idx] * w[idx]
    end
    return nothing
end"""),
    ("a scalar of the row used by the accumulation", 1, """
function f(w, v, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        s = v[i] * 2.0
        z[j] = z[j] + s * w[idx]
    end
    return nothing
end"""),
    ("a store of the element beside the accumulation", 1, """
function f(w, v, g, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        g[idx] = w[idx] * 2.0
        z[j] = z[j] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("the accumulation target moves with the row", 0, """
function f(w, v, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        z[i] = z[i] + w[idx] * v[j]
    end
    return nothing
end"""),
    ("the target is the column plus one", 0, """
function f(w, v, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        z[j + 1] = z[j + 1] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("the target is a fixed element", 0, """
function f(w, v, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        z[1] = z[1] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("the accumulated array is read in the body", 0, """
function f(w, v, g, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        g[idx] = z[j]
        z[j] = z[j] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("a literal inner length", 0, """
function f(w, v, z, n0)
    for idx = 1:n0 * 64
        i = div(idx - 1, 64) + 1
        j = mod(idx - 1, 64) + 1
        z[j] = z[j] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("a row and a column with different lengths", 0, """
function f(w, v, z, n0, n1, n2)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n2) + 1
        z[j] = z[j] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("a descending loop", 0, """
function f(w, v, z, n0, n1)
    for idx = n0 * n1:-1:1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        z[j] = z[j] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("a loop that starts past 1", 0, """
function f(w, v, z, n0, n1)
    for idx = 2:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        z[j] = z[j] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("a branch around the accumulation", 0, """
function f(w, v, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        if w[idx] > 0.0
            z[j] = z[j] + w[idx] * v[i]
        end
    end
    return nothing
end"""),
    ("the row is assigned twice", 0, """
function f(w, v, z, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        i = i + 0
        z[j] = z[j] + w[idx] * v[i]
    end
    return nothing
end"""),
    ("no accumulation", 0, """
function f(w, v, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        g[idx] = w[idx] * v[i] * v[j]
    end
    return nothing
end"""),
    ("five accumulations", 0, """
function f(w, v, a, b, c, d, e, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        a[j] = a[j] + w[idx] * v[i]
        b[j] = b[j] + w[idx] * 2.0
        c[j] = c[j] + w[idx] * 3.0
        d[j] = d[j] + w[idx] * 4.0
        e[j] = e[j] + w[idx] * 5.0
    end
    return nothing
end"""),
]

function cb_twins(src::String; minN::Int = 1)
    plan = STADE.stade_cuda(Meta.parse(src); inbounds = false)
    be = STADE.cgen_backend_cuda()
    n = 0
    for k in plan.kernels
        STADE.cgen_colblock_kernel(k, be, minN) === nothing || (n += 1)
    end
    return n, plan
end

# kernels edited by hand for a shape that no source reaches
function cb_mutants()
    be = STADE.cgen_backend_cuda()
    plan = STADE.stade_cuda(Meta.parse(CB_UNIT[1][3]); inbounds = false)
    base = first(k for k in plan.kernels if STADE.cgen_colblock_kernel(k, be, 1) !== nothing)
    out = Tuple{String,Any}[]
    for (label, stmt) in (("a branch beside the accumulation", :(if w[idx] > 0.0
                              g[idx] = 1.0
                          end)),)
        k = deepcopy(base)
        blk = k.args[2].args
        pos = findfirst(s -> s isa Expr && STADE.cgen_atomic_site(s) !== nothing, blk)
        insert!(blk, pos, stmt)
        push!(out, (label, k))
    end
    k = deepcopy(base)
    blk = k.args[2].args
    pos = findfirst(s -> s isa Expr && STADE.cgen_atomic_site(s) !== nothing, blk)
    blk[pos] = STADE.cgen_atomic_rebuild(Expr(:call, blk[pos].args[1], blk[pos].args[2], 1, blk[pos].args[4]), blk[pos].args[4])
    push!(out, ("the accumulated element is a fixed one", k))
    k = deepcopy(base)
    blk = k.args[2].args
    pos = findfirst(s -> s isa Expr && STADE.cgen_atomic_site(s) !== nothing, blk)
    insert!(blk, pos, Expr(:(=), :n0, 1))
    push!(out, ("a parameter is reassigned", k))
    return out
end

function cb_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    be = STADE.cgen_backend_cuda()
    for (label, k) in cb_mutants()
        got = STADE.cgen_colblock_kernel(k, be, 1) === nothing ? 0 : 1
        check(got == 0, "$(label): $(got) twin(s) (want 0)")
    end
    for (label, want, src) in CB_UNIT
        try
            got, _ = cb_twins(src)
            check(got == want, "$(label): $(got) twin(s) (want $(want))")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    return bad
end

# the twin text: the tile sum, the barrier, the guards, the single flusher
function cb_text_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    try
        be = STADE.cgen_backend_cuda()
        plan = STADE.stade_cuda(Meta.parse(CB_UNIT[2][3]); inbounds = false)
        t = first(STADE.cgen_colblock_kernel(k, be, 1) for k in plan.kernels if STADE.cgen_colblock_kernel(k, be, 1) !== nothing)
        txt = STADE.io_expr_to_source(t.kdef)
        for s in 1:2, l in 1:8
            check(occursin("__cb_s$(s)[Int((threadIdx()).x), $(l)]", txt), "text: site $s adds lane $l of the tile")
        end
        check(count("CUDA.sync_threads()", txt) == 1 && findfirst("CUDA.sync_threads()", txt)[1] < findfirst("__cb_t1 =", txt)[1], "text: one barrier, before the tile sums")
        check(count("if j <= n1", txt) == 1 && occursin("__cb_t == 1 && j <= n1", txt), "text: the loop is guarded by the column and one lane per column flushes")
        check(count("CuStaticSharedArray(Float64, (32, 8))", txt) == 2, "text: one 32 x 8 shared tile per accumulation")
        check(occursin("min(__cb_r0 + 128, div(", txt), "text: the tile ends at the last row")
        check(occursin("__cb_r0 + __cb_t:8:__cb_r1", txt), "text: the row lanes step by 8 from their own row")
        check(!occursin("if __tid >", txt), "text: no thread-index guard of the flat kernel")
    catch e
        check(false, "text: error $(first(split(sprint(showerror, e), "\n")))")
    end
    return bad
end

# the host: the keyword adds the twin and a guarded launch, the default leaves the text alone
function cb_launch_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    try
        ex = Meta.parse(CB_UNIT[1][3])
        on = STADE.stade_cuda(ex; col_block = 256); off = STADE.stade_cuda(ex; col_block = 0)
        txt(p) = join((string(x) for x in vcat(p.kernels, [p.host])), "\n")
        a, b = txt(on), txt(off)
        check(occursin("_k!", a) && !occursin("_k!", b), "launch: the twin is there with col_block = 256 and not with col_block = 0")
        check(occursin(", n1) == 0", a), "launch: the launch tests that the range is whole rows")
        check(occursin(">= 256", a), "launch: the launch tests the inner length against the keyword")
        check(occursin(">= 64", a), "launch: the launch tests the row count")
        check(occursin("<= 8388480", a), "launch: the launch tests the grid limit")
        check(occursin("threads = (32, 8)", a), "launch: the twin runs blocks of 32 x 8")
        check(count("blocks = max(1", a) >= 1, "launch: the plain launch stays as the fallback")
        check(!occursin("CuStaticSharedArray", b), "launch: no static shared tile with col_block = 0")
        on2 = STADE.stade_cuda(ex)
        check(txt(on2) == a, "launch: col_block = 256 is the plan without the keyword (the default)")
        both = STADE.stade_cuda(ex; col_block = 256, row_block = 256)
        check(occursin("_k!", txt(both)) && !occursin("_r!", txt(both)), "launch: with row_block also on, a column kernel gets the column twin only")
        ex2 = Meta.parse(CB_UNIT[5][3])
        none = STADE.stade_cuda(ex2; col_block = 256)
        check(txt(none) == txt(STADE.stade_cuda(ex2; col_block = 0)), "launch: a kernel the recognizer refuses has the same plan with the keyword")
    catch e
        check(false, "launch: error $(first(split(sprint(showerror, e), "\n")))")
    end
    return bad
end

function validate_colblock(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    :units in phases && (println("-- units"); bad += cb_unit_checks() + cb_text_checks() + cb_launch_checks())
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("a literal inner length is accepted", "    N isa Symbol && N in params || return nothing\n", ""),
            ("a row and a column with different lengths are accepted", "    (mods[1].N == N && mods[1].r == divs[1].r) || return nothing\n    N isa Symbol && N in params", "    N isa Symbol && N in params"),
            ("a descending loop is accepted", "    ix.step == 1 || return nothing\n", ""),
            ("a target that is not the column is accepted", "            site.idx === jv || return nothing\n            cgen_count_symbol(whole, site.arr) == 1 || return nothing\n", "            cgen_count_symbol(whole, site.arr) == 1 || return nothing\n"),
            ("the accumulated array may be used elsewhere", "            site.idx === jv || return nothing\n            cgen_count_symbol(whole, site.arr) == 1 || return nothing\n", "            site.idx === jv || return nothing\n"),
            ("a statement of any kind is allowed in the body", "        else\n            return nothing\n        end\n    end\n    isempty(sites) && return nothing\n    length(sites) <= cgen_colblock_max_sites()", "        else\n            push!(keep, s)\n        end\n    end\n    isempty(sites) && return nothing\n    length(sites) <= cgen_colblock_max_sites()"),
            ("a kernel without an accumulation gets a twin", "    isempty(sites) && return nothing\n    length(sites) <= cgen_colblock_max_sites()", "    length(sites) <= cgen_colblock_max_sites()"),
            ("any number of accumulations is accepted", "    length(sites) <= cgen_colblock_max_sites() || return nothing\n", ""),
            ("the row, the column or the flat index may be assigned twice", "        get(asg, v, 0) == 1 || return nothing\n    end\n    # the statements: scalars, stores and atomic sites only; the target of an accumulation is the column itself", "    end\n    # the statements: scalars, stores and atomic sites only; the target of an accumulation is the column itself"),
            ("a parameter may be reassigned", "        (s isa Expr && s.head === :(=) && s.args[1] isa Symbol && s.args[1] in params) && return nothing\n    end\n    base = parts.sig.args[1]\n    sn = string(base)\n    name = Symbol(endswith(sn, \"!\") ? sn[1:end-1] * \"_k!\"", "    end\n    base = parts.sig.args[1]\n    sn = string(base)\n    name = Symbol(endswith(sn, \"!\") ? sn[1:end-1] * \"_k!\""),
            ("the inner length is not tested against the keyword", "_k!\" : sn * \"_k\")\n    nrow = Expr(:call, :div, n_iter, N)\n    conds = Any[Expr(:call, :>=, N, minN), ", "_k!\" : sn * \"_k\")\n    nrow = Expr(:call, :div, n_iter, N)\n    conds = Any["),
            ("the whole-rows condition is dropped", "Expr(:call, :(==), Expr(:call, :mod, n_iter, N), 0), Expr(:call, :>=, nrow, cgen_colblock_min_rows())", "Expr(:call, :>=, nrow, cgen_colblock_min_rows())"),
            ("the minimum row count is dropped", "Expr(:call, :>=, nrow, cgen_colblock_min_rows()), ", ""),
            ("the grid limit is dropped", ", Expr(:call, :<=, nrow, 65535 * cgen_colblock_rows())", ""),
            ("the tile sum drops a lane", "terms = Any[Expr(:ref, sh[k], tx, l) for l in 1:TY]", "terms = Any[Expr(:ref, sh[k], tx, l) for l in 1:TY-1]"),
            ("no barrier before the tile sum", "    push!(post, Expr(:call, Expr(:., :CUDA, QuoteNode(:sync_threads))))\n    fin = Any[]\n    flush = Any[]", "    fin = Any[]\n    flush = Any[]"),
            ("the loop is not guarded by the column", "    guarded = Expr(:if, Expr(:call, :<=, jv, N), Expr(:block, loop))", "    guarded = loop"),
            ("every row lane flushes", "Expr(:if, Expr(:&&, Expr(:call, :(==), :__cb_t, 1), Expr(:call, :<=, jv, N)), Expr(:block, fin..., flush...))", "Expr(:if, Expr(:call, :<=, jv, N), Expr(:block, fin..., flush...))"),
            ("the tile runs past the last row", "Expr(:(=), :__cb_r1, Expr(:call, :min, Expr(:call, :+, :__cb_r0, R), nrow))", "Expr(:(=), :__cb_r1, Expr(:call, :+, :__cb_r0, R))"),
            ("the row lanes start at the same row", "Expr(:call, :(:), Expr(:call, :+, :__cb_r0, :__cb_t), TY, :__cb_r1)", "Expr(:call, :(:), Expr(:call, :+, :__cb_r0, 1), TY, :__cb_r1)"),
            ("the pass is never applied", "    backend.kernel_tag == \"cuda\" && col_block >= 1 && ((host, kernels) = cgen_colblock_apply(host, kernels, backend; minN = Int(col_block)))\n", ""),
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
                    include($(repr(joinpath(@__DIR__, "validate_colblock.jl"))))
                    n = sab_phase_sum($(repr("validate_colblock.jl")), $(repr(label)), [cb_unit_checks, cb_text_checks, cb_launch_checks])
                    println("RESULT ", n)
                """
                out = sab_read(script)
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all column-block checks passed" : "*** $bad column-block check(s) failed ***")
    return bad
end
abspath(PROGRAM_FILE) == abspath(joinpath(@__DIR__, "validate_colblock.jl")) && exit(validate_colblock(; phases = (:units, :sabotage)) == 0 ? 0 : 1)
