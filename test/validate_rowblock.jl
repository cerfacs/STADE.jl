# ==================== validate_rowblock.jl =====================
# Checks the row-block twin (`row_block`, CUDA): a flat loop `for idx = 1:R*N` with `i = div(idx - 1, N) + 1`, `j = mod(idx - 1, N) + 1` and an accumulation into `A[i]` gets a
# twin that runs one block per row. The CPU cannot run the twin (it uses warp shuffles), so these checks are on the generated text and on the recognizer; the device
# result is checked by the V100 parity job (the twin against the plain kernel on random data).
#   - units: handwritten kernels. Each is compiled with `stade_cuda`, and `cgen_rowblock_kernel` is asked for a twin of every kernel. A case lists the number of twins it must give.
#   - launch: the host text holds the run-time condition (`mod(n_iter, N) == 0`, and `n_iter == lo` for a descending loop) and keeps the plain launch as the fallback.
#   - SABOTAGE: edits of the recognizer, each of which one unit must catch.
#   julia validate_rowblock.jl        units, sabotage
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# (label, twins wanted, source)
const RB_UNIT = [
    ("an ascending flat loop with one accumulation", 1, """
function f(w, wb, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        wb[idx] = x[i] * g[j]
        xb[i] = xb[i] + w[idx] * g[j]
    end
    return nothing
end"""),
    ("a descending flat loop", 1, """
function f(w, wb, x, xb, g, n0, n1)
    for idx = n0 * n1:-1:1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        xb[i] = xb[i] + w[idx] * g[j]
    end
    return nothing
end"""),
    ("two accumulations into two arrays", 1, """
function f(w, x, xb, yb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        xb[i] = xb[i] + w[idx] * g[j]
        yb[i] = yb[i] + w[idx] * w[idx]
    end
    return nothing
end"""),
    ("a scalar of the row used by the accumulation", 1, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        s = x[i] * 2.0
        xb[i] = xb[i] + s * w[idx] * g[j]
    end
    return nothing
end"""),
    ("the accumulation target moves with the column", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        xb[j] = xb[j] + w[idx] * x[i]
    end
    return nothing
end"""),
    ("the target is a scalar derived from the column", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        k = j + i
        xb[k] = xb[k] + w[idx]
    end
    return nothing
end"""),
    ("the accumulated array is read in the body", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        g[idx] = xb[i] * w[idx]
        xb[i] = xb[i] + w[idx]
    end
    return nothing
end"""),
    ("two accumulations into one array (merge_accum makes them one)", 1, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        xb[i] = xb[i] + w[idx]
        xb[i] = xb[i] + g[j]
    end
    return nothing
end"""),
    ("the target is the sum of the row and the column", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        xb[i + j] = xb[i + j] + w[idx]
    end
    return nothing
end"""),
    ("the target is one fixed element", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        xb[1] = xb[1] + w[idx] * g[j]
    end
    return nothing
end"""),
    ("two accumulations into one array with a read between them", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        xb[i] = xb[i] + w[idx]
        g[idx] = xb[i]
        xb[i] = xb[i] + g[j]
    end
    return nothing
end"""),
    ("no accumulation, plain stores only", 0, """
function f(w, x, wb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        wb[idx] = x[i] * g[j]
    end
    return nothing
end"""),
    ("the inner length is a literal", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * 64
        i = div(idx - 1, 64) + 1
        j = mod(idx - 1, 64) + 1
        xb[i] = xb[i] + w[idx] * g[j]
    end
    return nothing
end"""),
    ("the row and the column use different lengths", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1 + 1) + 1
        xb[i] = xb[i] + w[idx] * g[j]
    end
    return nothing
end"""),
    ("the loop does not start at 1", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 2:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        xb[i] = xb[i] + w[idx] * g[j]
    end
    return nothing
end"""),
    ("a branch in the body", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        if w[idx] > 0.0
            xb[i] = xb[i] + w[idx] * g[j]
        end
    end
    return nothing
end"""),
    ("the flat index is reassigned", 0, """
function f(w, x, xb, g, n0, n1)
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        i = i + 0
        xb[i] = xb[i] + w[idx] * g[j]
    end
    return nothing
end"""),
]

function rb_twins(src::String; minN::Int = 1)
    ex = Meta.parse(src)
    plan = STADE.stade_cuda(ex; inbounds = false)
    be = STADE.cgen_backend_cuda()
    n = 0
    for k in plan.kernels
        STADE.cgen_rowblock_kernel(k, be, minN) === nothing || (n += 1)
    end
    return n, plan
end

# kernels that the source cannot produce (the array that is accumulated is read in the body, a branch beside the accumulation): the generated kernel of the first
# case with one statement inserted before its atomic site
function rb_mutants()
    be = STADE.cgen_backend_cuda()
    plan = STADE.stade_cuda(Meta.parse(RB_UNIT[1][3]); inbounds = false)
    base = first(k for k in plan.kernels if STADE.cgen_rowblock_kernel(k, be, 1) !== nothing)
    out = Tuple{String,Any}[]
    for (label, stmt) in (("the accumulated array is read in the body", :(g[idx] = xb[i])),
                          ("a branch beside the accumulation", :(if w[idx] > 0.0
                              g[idx] = 1.0
                          end)))
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
    push!(out, ("the accumulated element does not depend on the row", k))
    return out
end

function rb_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    be = STADE.cgen_backend_cuda()
    for (label, k) in rb_mutants()
        got = STADE.cgen_rowblock_kernel(k, be, 1) === nothing ? 0 : 1
        check(got == 0, "$(label): $(got) twin(s) (want 0)")
    end
    for (label, want, src) in RB_UNIT
        try
            got, _ = rb_twins(src)
            check(got == want, "$(label): $(got) twin(s) (want $(want))")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    return bad
end

# the host: the keyword adds the twin and a guarded launch, the default leaves the text alone
function rb_launch_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, k, mustdesc) in (("ascending", 1, false), ("descending", 2, true))
        try
            ex = Meta.parse(RB_UNIT[k][3])
            on = STADE.stade_cuda(ex; row_block = 256); off = STADE.stade_cuda(ex)
            txt(p) = join((string(x) for x in vcat(p.kernels, [p.host])), "\n")
            a, b = txt(on), txt(off)
            check(occursin("_r!", a) && !occursin("_r!", b), "$(label): the twin is there with the keyword and not without it")
            check(occursin(", n1) == 0", a), "$(label): the launch tests that the range is whole rows")
            check(mustdesc ? occursin(r"== n0 \* n1", a) : !occursin(r"== n0 \* n1", a), "$(label): the descending condition (n_iter == lo) is $(mustdesc ? "present" : "absent")")
            check(occursin(">= 256", a), "$(label): the launch tests the inner length against the keyword")
            check(count("blocks = max(1", a) >= 1, "$(label): the plain launch stays as the fallback")
            check(!occursin("shfl_down_sync", b), "$(label): no shuffle without the keyword")
            on2 = STADE.stade_cuda(ex; row_block = 0)
            check(txt(on2) == b, "$(label): row_block = 0 is the plan without the keyword")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    return bad
end

function validate_rowblock(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    :units in phases && (println("-- units"); bad += rb_unit_checks() + rb_launch_checks())
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("a literal inner length is accepted", "    isempty(cgen_split_vars(N)) && return nothing\n", ""),
            ("a row and a column with different lengths are accepted", "    (mods[1].N == N && mods[1].r == divs[1].r) || return nothing\n", ""),
            ("an ascending loop may start past 1", "        lo == 1 && return (lo = 1, step = 1)\n        return nothing\n", "        return (lo = lo, step = 1)\n"),
            ("a target that depends on the column is accepted", "            isempty(intersect(cgen_split_vars(site.idx), tainted)) || return nothing\n", ""),
            ("a target that does not use the row is accepted", "            iv in cgen_split_vars(site.idx) || return nothing\n", ""),
            ("the accumulated array may be used elsewhere", "            cgen_count_symbol(whole, site.arr) == 1 || return nothing\n", ""),
            ("a statement of any kind is allowed in the body", "        else\n            return nothing\n        end\n    end\n    isempty(sites) && return nothing", "        else\n            push!(keep, s)\n        end\n    end\n    isempty(sites) && return nothing"),
            ("a kernel without an accumulation gets a twin", "    isempty(sites) && return nothing\n", ""),
            ("the row, the column or the flat index may be assigned twice", "        get(asg, v, 0) == 1 || return nothing\n", ""),
            ("the descending condition is dropped", "    ix.step == -1 && push!(conds, Expr(:call, :(==), n_iter, ix.lo))\n", ""),
            ("the whole-rows condition is dropped", "Expr(:call, :>, n_iter, 0), Expr(:call, :(==), Expr(:call, :mod, n_iter, N), 0), ", "Expr(:call, :>, n_iter, 0), "),
            ("the inner length is not tested against the keyword", "Expr(:call, :>=, N, minN), ", ""),
            ("the pass is never applied", "    backend.kernel_tag == \"cuda\" && row_block >= 1 && ((host, kernels) = cgen_rowblock_apply(host, kernels, backend; minN = Int(row_block)))\n", ""),
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
                    include($(repr(joinpath(@__DIR__, "validate_rowblock.jl"))))
                    n = rb_unit_checks() + rb_launch_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all row-block checks passed" : "*** $bad row-block check(s) failed ***")
    return bad
end
abspath(PROGRAM_FILE) == abspath(joinpath(@__DIR__, "validate_rowblock.jl")) && exit(validate_rowblock(; phases = (:units, :sabotage)) == 0 ? 0 : 1)
