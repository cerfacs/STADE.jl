include(get(ENV, "STADE_SRC", joinpath(@__DIR__, "..", "src", "STADE.jl")))
using Random

# Differential fuzzer for integer state in the reverse sweep.
#
# Usage: julia fuzz_int_state.jl FIRST_SEED LAST_SEED [OUTDIR]
# STADE_SRC=path/to/STADE.jl runs it against another version of the source.
#
# Each seed makes one random kernel `fz(x, w, idx, cnt, n, steps, nf, out)` and one random input. The kernel is built
# from the shapes that hurt in v0.4.17 to v0.4.19: a counter bumped under a float branch, an integer array read through
# a scalar copy and rewritten in place (plain, self-referring, conditional, indexed by a computed scalar), a loop bound
# read from an integer array that the loop or a later pass rewrites, a `while` over a float bound, an outer pass loop,
# and the same scalar name assigned at several sites. Every index is wrapped by `mod(.., 6) + 1`, so the kernel is
# in range for every draw. Float branches test `x` against thresholds that the draw stays 2e-3 away from, and the
# `while` bound is a half integer, so a finite-difference step never changes control flow.
#
# Oracles, with fuse_ii_loops off and on:
#   fd      the adjoint against central differences of the primal at every entry of x and w (the primal is the truth)
#   tan     the tangent against finite differences        (generic validator)
#   dot     the exact dot-product identity                (generic validator)
#   hvp     the HVP against finite differences of the adjoint
#   nopp    keep_push_pop = false: it must either refuse with the documented reason, or generate code that passes the
#           generic adjoint validator. Anything else is a defect.
# A failing seed writes its kernel and input to OUTDIR/fail_<seed>.jl, which test/int-state-corpus can adopt.

say(args...) = (println(args...); flush(stdout))

const NXF = 6
const THR = [0.35, 0.55, 0.75, 0.95]
fz_pick(rng, v) = v[rand(rng, 1:length(v))]

# an integer expression that is in 1..6 for any value of the scalars in `vars`
function fz_idx_expr(rng, vars)
    v = fz_pick(rng, vars)
    a = fz_pick(rng, [1, 2, 3]); b = fz_pick(rng, [0, 1, 2, 3])
    return "mod($v * $a + $b, 6) + 1"
end

# one float statement over the valid positions `pos`
function fz_float_stmt(rng, pos)
    a = fz_pick(rng, pos); b = fz_pick(rng, pos)
    st = fz_pick(rng, ["s += x[$a] * x[$b]", "s += w[$a] * x[$b]", "w[$a] += x[$b] * x[$a]", "w[$a] = x[$b] * 2.0",
                       "s += x[$a] * x[$a]"])
    return rand(rng) < 0.3 ? ["if x[$b] > $(fz_pick(rng, THR))", "    " * st, "end"] : [st]
end

# integer statements. `pos` grows with every scalar the statement defines.
function fz_int_stmt(rng, pos, vars, names, k_used, gpu = false, ext = false)
    kind = gpu ? fz_pick(rng, [:copy_idx, :copy_idx, :write_idx, :write_idx, :write_idx_via]) :
                 fz_pick(rng, vcat([:copy_idx, :copy_idx, :copy_cnt, :write_idx, :write_idx, :write_idx_self, :write_cnt,
                                    :bump, :write_idx_via], ext ? [:int_acc, :if_int_write, :if_int_float] : Symbol[]))
    a = fz_pick(rng, pos)
    nm = fz_pick(rng, names)
    if kind == :copy_idx
        push!(pos, nm)
        return ["$nm = idx[$a]"]
    elseif kind == :copy_cnt
        push!(pos, nm)
        return ["$nm = cnt[$(rand(rng, 1:3))]"]
    elseif kind == :write_idx
        st = "idx[$a] = $(fz_idx_expr(rng, vars))"
        return rand(rng) < 0.35 ? ["if x[$(fz_pick(rng, pos))] > $(fz_pick(rng, THR))", "    " * st, "end"] : [st]
    elseif kind == :write_idx_self
        return ["idx[$a] = mod(idx[$a] + $(rand(rng, 1:3)), 6) + 1"]
    elseif kind == :write_idx_via
        b = fz_pick(rng, pos)
        push!(pos, nm)
        return ["$nm = idx[$b]", "idx[$a] = $nm"]
    elseif kind == :int_acc
        # an integer that accumulates array elements, and a position taken from it
        k_used[2] = true
        st = ["m = m + idx[$a]"]
        rand(rng) < 0.6 && (push!(st, "$nm = mod(m, 6) + 1"); nm in pos || push!(pos, nm))
        return st
    elseif kind == :if_int_write
        # an integer write under an integer condition
        return ["if idx[$a] > $(rand(rng, 1:5))", "    idx[$(fz_pick(rng, pos))] = $(fz_idx_expr(rng, vars))", "end"]
    elseif kind == :if_int_float
        # a float statement under an integer condition
        return vcat(["if idx[$a] > $(rand(rng, 1:5))"], ["    " * l for l in fz_float_stmt(rng, pos)], ["end"])
    elseif kind == :write_cnt
        c = rand(rng, 1:3)
        return ["cnt[$c] = mod(cnt[$c] + $(rand(rng, 1:3)), 4) + 1"]
    else # :bump
        k_used[1] = true
        st = "k = k + 1"
        stmt = rand(rng) < 0.6 ? ["if x[$(fz_pick(rng, pos))] > $(fz_pick(rng, THR))", "    " * st, "end"] : [st]
        return vcat(stmt, rand(rng) < 0.6 ? ["$nm = mod(k, 6) + 1"] : String[])
    end
end

# the body of a loop: 2 to 5 statements, at least one of them a float statement; a bump reads through `p`
function fz_body(rng, pos0, vars, names, k_used, gpu = false, ext = false)
    pos = copy(pos0)
    out = String[]
    nst = rand(rng, 2:5)
    for q in 1:nst
        if rand(rng) < 0.45
            append!(out, fz_float_stmt(rng, pos))
        else
            st = fz_int_stmt(rng, pos, vars, names, k_used, gpu, ext)
            append!(out, st)
            # `nm = mod(k, 6) + 1` defined by a bump is a valid position
            for l in st
                m = match(r"^(\w+) = mod\(k, 6\) \+ 1$", l)
                m === nothing || (m.captures[1] in pos || push!(pos, m.captures[1]))
            end
        end
    end
    any(l -> occursin("s +=", l) || occursin("w[", l), out) || append!(out, fz_float_stmt(rng, pos))
    return out
end

function fz_loop(rng, vars, names, k_used, uses, gpu = false, ext = false)
    kind = gpu ? :for_n : fz_pick(rng, vcat([:for_n, :for_n, :for_cnt, :for_copy, :while_nf], ext ? [:for_step2, :for_rev] : Symbol[]))
    ind(ls) = ["    " * l for l in ls]
    if kind == :for_n
        uses[:n] = true
        return vcat(["for i = 1:n"], ind(fz_body(rng, ["i"], vcat(vars, ["i"]), names, k_used, gpu, ext)), ["end"])
    elseif kind == :for_step2
        uses[:n] = true
        return vcat(["for i = 1:2:n"], ind(fz_body(rng, ["i"], vcat(vars, ["i"]), names, k_used, gpu, ext)), ["end"])
    elseif kind == :for_rev
        uses[:n] = true
        return vcat(["for i = n:-1:1"], ind(fz_body(rng, ["i"], vcat(vars, ["i"]), names, k_used, gpu, ext)), ["end"])
    elseif kind == :for_cnt
        return vcat(["for i = 1:cnt[$(rand(rng, 1:3))]"], ind(fz_body(rng, ["i"], vcat(vars, ["i"]), names, k_used, gpu, ext)), ["end"])
    elseif kind == :for_copy
        c = fz_pick(rng, ["c1", "c2"]); r = rand(rng, 1:3)
        return vcat(["$c = cnt[$r]", "for i = 1:$c"], ind(fz_body(rng, ["i"], vcat(vars, ["i"]), names, k_used, gpu, ext)), ["end"])
    else
        uses[:nf] = true
        body = fz_body(rng, ["p"], vcat(vars, ["kk"]), names, k_used, gpu, ext)
        return vcat(["kk = 1", "while kk <= nf", "    p = mod(kk, 6) + 1"], ind(body), ["    kk = kk + 1", "end"])
    end
end

# does shape inference type `arr` as an integer array in this body? (a trial kernel with every argument present)
function fz_int_typed(body_lines, arr)
    txt = join(body_lines, "\n")
    args = ["x", "w", "idx", "cnt", "n", "steps", "nf", "out"]
    ex = Meta.parse("function fz(" * join(args, ", ") * ")\n s = 0.0\n" * txt * "\n out[1] = s\n return nothing\nend")
    k = STADE.parse_kernel(ex)
    return get(k.sig.kinds, Symbol(arr), :none) == :array_int
end

# returns the kernel source (with its header comment) and the names of its arguments
# A loop of the restricted kind. A write loop writes `idx` and never reads it; a read loop reads it and never writes it.
# Every write loop comes before every read loop, and no pass loop spans both, so no integer needs a restore.
function fz_gpu_loop(rng, phase, vars, names, uses)
    ind(ls) = ["    " * l for l in ls]
    uses[:n] = true
    pos = ["i"]
    body = String[]
    for _ in 1:rand(rng, 2:4)
        if phase == :write
            if rand(rng) < 0.5
                append!(body, fz_float_stmt(rng, pos))
            else
                st = "idx[i] = $(fz_idx_expr(rng, vcat(vars, ["i"])))"
                append!(body, rand(rng) < 0.35 ? ["if x[i] > $(fz_pick(rng, THR))", "    " * st, "end"] : [st])
            end
        else
            if !("j" in pos) || rand(rng) < 0.25
                nm = fz_pick(rng, names)
                push!(body, "$nm = idx[$(fz_pick(rng, pos))]")
                nm in pos || push!(pos, nm)
            else
                append!(body, fz_float_stmt(rng, pos))
            end
        end
    end
    any(l -> occursin("s +=", l) || occursin("w[", l), body) || append!(body, fz_float_stmt(rng, pos))
    return vcat(["for i = 1:n"], ind(body), ["end"])
end

# Seeds 1000 to 1999 make the restricted kind: only counted loops over `n`, plain and gathered integer writes, no counter
# and no integer array that is read for a bound. These are the kernels `keep_push_pop = false` should accept, so that the
# GPU path is exercised and not refused.
function fz_kernel(seed::Int)
    gpu = 1000 <= seed < 2000
    ext = seed >= 2000
    rng = MersenneTwister(seed)
    names = fz_pick(rng, [["j", "jj"], ["j", "j"], ["j", "q", "jj"]])
    k_used = [false, false]
    uses = Dict{Symbol,Bool}()
    lines = String[]
    nseg = rand(rng, 2:4)
    if gpu
        nw = rand(rng, 1:2); nr = rand(rng, 1:2)
        wrap(ls) = rand(rng) < 0.3 ? vcat(["for t = 1:steps"], ["    " * l for l in ls], ["end"]) : ls
        for _ in 1:nw
            ls = fz_gpu_loop(rng, :write, ["t"], names, uses)
            if rand(rng) < 0.3
                uses[:steps] = true
                ls = vcat(["for t = 1:steps"], ["    " * l for l in ls], ["end"])
            else
                ls = fz_gpu_loop(rng, :write, String[], names, uses)
            end
            append!(lines, ls)
        end
        for _ in 1:nr
            append!(lines, fz_gpu_loop(rng, :read, String[], names, uses))
        end
        nseg = 0
    end
    for sg in 1:nseg
        if rand(rng) < 0.3
            uses[:steps] = true
            inner = String[]
            for _ in 1:rand(rng, 1:2)
                append!(inner, fz_loop(rng, ["t"], names, k_used, uses, gpu, ext))
            end
            append!(lines, vcat(["for t = 1:steps"], ["    " * l for l in inner], ["end"]))
        else
            append!(lines, fz_loop(rng, String[], names, k_used, uses, gpu, ext))
        end
    end
    pre = String[]
    k_used[1] && push!(pre, "k = 1")
    k_used[2] && push!(pre, "m = 0")
    txt = join(vcat(pre, lines), "\n")
    # An array that the kernel only writes, or reads through `mod`, is inferred as a float array, and the generic
    # validators then perturb it as one. A read at the end that is used as a subscript pins the type, and is a
    # read after every write, so it adds no restore of its own.
    typed = String[]
    for (arr, tmp) in (("idx", "jt1"), ("cnt", "jt2"))
        occursin(arr * "[", txt) || continue
        if !fz_int_typed(vcat(pre, lines), arr)
            push!(typed, "$tmp = $arr[1]"); push!(typed, "s += x[$tmp] * x[$tmp]")
        end
    end
    lines = vcat(lines, typed)
    txt = join(vcat(pre, lines), "\n")
    uses_w = occursin("w[", txt)
    uses_cnt = occursin("cnt[", txt)
    uses_n = occursin(r"(1:n\b|1:2:n\b|\bn:-1:1)", txt)
    uses_steps = occursin("steps", txt)
    uses_nf = occursin("nf", txt)
    uses_idx = occursin("idx[", txt)
    args = String["x"]
    uses_w && push!(args, "w")
    uses_idx && push!(args, "idx")
    uses_cnt && push!(args, "cnt")
    uses_n && push!(args, "n")
    uses_steps && push!(args, "steps")
    uses_nf && push!(args, "nf")
    push!(args, "out")
    src = IOBuffer()
    println(src, "# fz(", join(args, ", "), ")")
    println(src, "#\n# Random kernel, seed $seed, made by test/fuzz_int_state.jl.\n#")
    println(src, "# x: float array of length 6\n# out: length-1 output array; out[1] receives the sum")
    println(src, "function fz(", join(args, ", "), ")")
    println(src, "    s = 0.0")
    for l in vcat(pre, lines)
        println(src, "    ", l)
    end
    println(src, "    out[1] = s\n    return nothing\nend")
    return String(take!(src)), args
end

function fz_args(seed::Int, args)
    rng = MersenneTwister(seed + 7919)
    x = zeros(NXF)
    for i in 1:NXF
        while true
            v = 0.2 + rand(rng)
            all(abs(v - t) > 2e-3 for t in THR) && (x[i] = v; break)
        end
    end
    vals = Dict{Symbol,Any}(:x => x, :w => 0.5 .+ rand(rng, NXF), :idx => rand(rng, 1:NXF, NXF), :cnt => rand(rng, 1:3, 3),
                            :n => rand(rng, 3:NXF), :steps => rand(rng, 1:3), :nf => rand(rng, 2:NXF) + 0.5, :out => zeros(1))
    return [Symbol(a) => vals[Symbol(a)] for a in args]
end

dup_arg(v::AbstractArray) = copy(v)
dup_arg(v) = v

function fz_primal(m, args)
    vals = Any[dup_arg(v) for (_, v) in args]
    Base.invokelatest(getfield(m, :fz), vals...)
    return vals[findfirst(p -> p.first == :out, args)][1]
end

function fz_adjoint(m, args)
    fb = getfield(m, :fz_b)
    init = getfield(m, :initstacks_fz_b)
    names = [n for n in Base.method_argnames(first(methods(fb))) if !startswith(string(n), "#")]
    primal = Dict(args)
    stacks = Base.invokelatest(init)
    stacks = stacks === nothing ? () : (stacks isa Tuple ? stacks : (stacks,))
    si = 0
    vals = Any[]; adj = Dict{Symbol,Any}(); scalar_adj = Symbol[]
    for nm in names
        s = string(nm)
        if haskey(primal, nm)
            push!(vals, dup_arg(primal[nm]))
        elseif endswith(s, "_stack")
            si += 1; push!(vals, stacks[si])
        elseif endswith(s, "b") && haskey(primal, Symbol(s[1:end-1]))
            base = Symbol(s[1:end-1]); pv = primal[base]
            if pv isa AbstractArray
                buf = base == :out ? ones(size(pv)) : zeros(size(pv))
            else
                buf = 0.0; push!(scalar_adj, base)
            end
            adj[base] = buf; push!(vals, buf)
        else
            error("cannot map generated argument `$nm`")
        end
    end
    r = Base.invokelatest(fb, vals...)
    rs = r === nothing ? () : (r isa Tuple ? r : (r,))
    for (b, v) in zip(scalar_adj, rs); adj[b] = v; end
    return adj
end

function fz_fd(src, args, fuse)
    dir = mktempdir(); dst = joinpath(dir, "fz_b.jl")
    STADE.stade_adjoint_file(src, dst; keep_push_pop = true, fuse_ii_loops = fuse)
    m = Module(:FuzzCase)
    Base.include(m, src); Base.include(m, dst)
    adj = fz_adjoint(m, args)
    bad = 0; worst = 0.0
    for nm in (:x, :w)
        haskey(Dict(args), nm) || continue
        base = Dict(args)[nm]
        for e in 1:length(base)
            function with(d)
                alt = [p.first == nm ? (nm => (c = copy(base); c[e] += d; c)) : p for p in args]
                return fz_primal(m, alt)
            end
            h = 1e-6
            fd = (with(h) - with(-h)) / (2h)
            a = adj[nm][e]
            abs(a - fd) <= 1e-6 + 1e-5 * abs(fd) || (bad += 1)
            worst = max(worst, abs(a - fd) / max(abs(fd), 1e-12))
        end
    end
    return (bad, worst)
end

function fz_baseline(src, args, dir)
    kernel = STADE.parse_kernel(STADE.io_read_corpus_entry(src))
    int_args = Dict{Symbol,Any}(); values = Dict{Symbol,Any}()
    for (nm, v) in args
        if get(kernel.sig.kinds, nm, :none) == :scalar_int
            int_args[nm] = Int(v)
        else
            values[nm] = v isa AbstractArray ? copy(v) : v
        end
    end
    yp = joinpath(dir, "fz.yaml")
    STADE.io_write_baseline_yaml(yp, :fz, int_args, values)
    return yp
end

# "ok", "FAIL", "REFUSED" or "ERR <first line>"
function fz_run(f)
    try
        r = f()
        return r === :refused ? "REFUSED" : (r ? "ok" : "FAIL")
    catch e
        msg = first(split(sprint(showerror, e), "\n"))
        return "ERR " * first(msg, 120)
    end
end

function fuzz_seed(seed::Int, outdir::String)
    src_text, argn = fz_kernel(seed)
    args = fz_args(seed, argn)
    dir = mktempdir(); src = joinpath(dir, "fz.jl")
    write(src, src_text)
    res = Dict{String,String}()
    for fuse in (false, true)
        tag = fuse ? "f" : "n"
        res["fd_$tag"] = fz_run(() -> (b = fz_fd(src, args, fuse)[1]; b == 0))
        yp = try fz_baseline(src, args, dir) catch e; nothing end
        if yp === nothing
            res["base_$tag"] = "ERR baseline"; continue
        end
        Random.seed!(seed)
        res["tan_$tag"] = fz_run(() -> STADE.stade_validate_tangent_file(src; yaml_path = yp, trials = 2, keep_push_pop = true, fuse_ii_loops = fuse).ok)
        res["dot_$tag"] = fz_run(() -> STADE.stade_validate_dotprod_file(src; yaml_path = yp, trials = 2, keep_push_pop = true, fuse_ii_loops = fuse).ok)
        res["hvp_$tag"] = fz_run(() -> STADE.stade_validate_hvp_file(src; yaml_path = yp, trials = 2, keep_push_pop = true, fuse_ii_loops = fuse).ok)
        res["nopp_$tag"] = fz_run() do
            try
                STADE.stade_adjoint_file(src, joinpath(dir, "nopp_b.jl"); keep_push_pop = false, fuse_ii_loops = fuse)
            catch e
                msg = sprint(showerror, e)
                (occursin("must be restored", msg) || occursin("no trip count", msg) || occursin("cannot size the stacks", msg)) && return :refused
                rethrow()
            end
            STADE.stade_validate_adjoint_file(src; yaml_path = yp, trials = 2, keep_push_pop = false, fuse_ii_loops = fuse).ok
        end
    end
    bad = [(k, v) for (k, v) in res if v != "ok" && v != "REFUSED"]
    if !isempty(bad)
        mkpath(outdir)
        open(joinpath(outdir, "fail_$seed.jl"), "w") do io
            println(io, src_text)
            println(io, "# input: ", join(["$(p.first) = $(p.second)" for p in args], "; "))
            for (k, v) in sort(bad); println(io, "# ", k, ": ", v); end
        end
    end
    return res
end

function fuzz_main(first_seed::Int, last_seed::Int, outdir::String)
    nbad = 0; nref = 0; nseed = 0
    for seed in first_seed:last_seed
        res = try
            fuzz_seed(seed, outdir)
        catch e
            Dict("harness" => "ERR " * first(split(sprint(showerror, e), "\n")))
        end
        nseed += 1
        bad = sort([(k, v) for (k, v) in res if v != "ok" && v != "REFUSED"])
        nref += count(v -> v == "REFUSED", values(res))
        if isempty(bad)
            say("seed ", seed, " ok")
        else
            nbad += 1
            say("seed ", seed, " FAIL ", join(["$k=$(first(v, 70))" for (k, v) in bad], " | "))
        end
    end
    say("\n", nseed, " seeds, ", nbad, " with a failure, ", nref, " refusals", nbad == 0 ? "   all fuzz checks passed" : "   *** $nbad SEEDS NOT OK ***")
    return nbad
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = ARGS
    fuzz_main(parse(Int, a[1]), parse(Int, a[2]), length(a) >= 3 ? a[3] : joinpath(@__DIR__, "fuzz-failures"))
end
