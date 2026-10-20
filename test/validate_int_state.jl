include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# Integer state in the reverse sweep.
#
# An integer local that an active statement reads as an index, and that depends on a counter or on an integer
# array rewritten in place (`j = idx[k]`, `j = n + 1`, `u = heap[1]`), used to be dropped from the reverse sweep.
# The adjoint was generated without error, then either crashed (`UndefVarError: j`) or, when a variable of the
# same name survived the loop, read the value of the last forward iteration and returned wrong gradients.
# `test/int-state-corpus/` holds the shapes that exposed it. See snap_itbr_analyze in STADE.jl.
#
# Five checks per kernel, with fuse_ii_loops off and on:
#   - the adjoint against central finite differences of the primal, at the entries listed below, with the
#     tolerance of the original bug report (1e-6 + 1e-5 * |fd|)
#   - the exact dot-product identity <Yb, J Xd> = <J' Yb, Xd> between the tangent and adjoint codes (1e-11)
#   - the tangent against finite differences, and the HVP against finite differences of the adjoint
# The finite-difference oracle of stade_validate_adjoint_file is not used on fmm2d: its T output holds a 1e30
# sentinel on the padding ring, which swallows every perturbation in a summed functional. The adjoint is checked
# on `out` alone instead.
#
# Then the refusals: keep_push_pop = false cannot size the stack of a restored integer.

say(args...) = (println(args...); flush(stdout))

const INT_DIR = joinpath(@__DIR__, "int-state-corpus")

const NX_FMM = 17
fmm_idx(i, j) = j * NX_FMM + (i + 1)
fmm_slowness() = [1.0 + 0.3 * sin(6.0 * ((k - 1) % NX_FMM) / NX_FMM) * cos(5.0 * div(k - 1, NX_FMM) / NX_FMM)
                  for k in 1:NX_FMM*NX_FMM]

function amr_args(p, tol)
    cap = 512
    xl = zeros(cap); xr = zeros(cap)
    for i in 1:4; xl[i] = (i - 1) / 4; xr[i] = i / 4; end
    return [:xl => xl, :xr => xr, :ncell => [4], :cap => cap, :p => p, :x0 => 0.37, :tol => tol, :out => zeros(1)]
end

# n is a float in the while kernels, compared in `k <= n`; the for kernels take an integer bound. 3.5 keeps it off the integer boundary, where a finite-difference step
# of 1e-6 would change the trip count and swamp the derivative
gather_args(n = 3.5) = [:x => [1.0, 2.0, 3.0, 4.0], :idx => [2, 3, 1], :n => n, :out => zeros(1)]
rewrite_args(n = 3.5) = [:x => [1.0, 2.0, 3.0, 4.0], :idx => [1], :n => n, :out => zeros(1)]
arith_args() = [:x => [1.1, 0.0], :y => [1.5, 0, 0, 0, 0], :n0 => 1, :nmax => 5, :out => zeros(1)]
fmm_args() = [:s => fmm_slowness(), :T => zeros(NX_FMM * NX_FMM), :state => zeros(Int, NX_FMM * NX_FMM),
              :heap => zeros(Int, NX_FMM * NX_FMM), :pos => zeros(Int, NX_FMM * NX_FMM), :off => [-1, 1, -NX_FMM, NX_FMM],
              :nx => NX_FMM, :ny => NX_FMM, :h => 1 / 16, :src => fmm_idx(8, 8), :rec => fmm_idx(13, 11), :out => zeros(1)]

# the AMR mesh edges are listed as targets: the gradient with respect to x0 and p alone never touches them,
# which is how a wrong index in the reverse sweep went unseen
const AMR_TARGETS = [(:x0,), (:xl, 1), (:xl, 2), (:xr, 2), (:xr, 4)]
const X4 = [(:x, 1), (:x, 2), (:x, 3), (:x, 4)]
const BC = [(:x, 1), (:x, 2), (:w, 2), (:w, 3), (:w, 4)]

# (label, kernel, args, targets)
const INT_CASES = [
    ("while_gather",       "while_gather",       gather_args,  [(:x, 1), (:x, 2), (:x, 3)]),
    ("while_wrapped_for",  "while_wrapped_for",  gather_args,  [(:x, 1), (:x, 2), (:x, 3)]),
    ("while_arith_index",  "while_arith_index",  arith_args,   [(:x, 1), (:y, 1)]),
    ("amr_tempj",          "amr_tempj",          () -> amr_args(50.0, 1e-5), AMR_TARGETS),
    ("amr_tempj_multi",    "amr_tempj",          () -> amr_args(8.0, 1e-4),  AMR_TARGETS),
    ("fmm2d",              "fmm2d",              fmm_args,     [(:s, fmm_idx(10, 9)), (:s, fmm_idx(11, 10)), (:s, fmm_idx(12, 10)), (:h,)]),
    ("for_int_rewrite",    "for_int_rewrite",    () -> rewrite_args(3), [(:x, 1), (:x, 2), (:x, 3)]),
    ("while_int_rewrite",  "while_int_rewrite",  rewrite_args, [(:x, 1), (:x, 2), (:x, 3)]),
    # integer arrays read and then written again, and control flow around integer writes
    ("two_phase_perm",     "two_phase_perm",     () -> [:x => [1.1, 2.2, 3.3, 4.4], :idx => [3, 1, 4, 2], :n => 4, :out => zeros(1)], X4),
    ("while_two_phase",    "while_two_phase",    () -> [:x => [1.1, 2.2, 3.3, 4.4], :idx => [3, 1, 4, 2], :n => 4.5, :out => zeros(1)], X4),
    ("partial_rewrite_m0", "partial_rewrite",    () -> [:x => [1.1, 2.2, 3.3, 4.4], :idx => [3, 1, 4, 2], :n => 4, :m => 0, :out => zeros(1)], X4),
    ("partial_rewrite_m2", "partial_rewrite",    () -> [:x => [1.1, 2.2, 3.3, 4.4], :idx => [3, 1, 4, 2], :n => 4, :m => 2, :out => zeros(1)], X4),
    ("rebuild_in_loop",    "rebuild_in_loop",    () -> [:x => [1.1, 2.2, 3.3, 4.4], :idx => [1, 1, 1, 1], :n => 4, :steps => 3, :out => zeros(1)], X4),
    ("cond_int_write",     "cond_int_write",     () -> [:x => [0.3, 0.7, 0.9, 0.2], :idx => [3, 1, 4, 2], :n => 4, :out => zeros(1)], X4),
    ("int_bound_array",    "int_bound_array",    () -> [:x => [1.0, 2.0, 3.0, 4.0, 5.0], :cnt => [1], :n => 3, :out => zeros(1)], [(:x, 1), (:x, 2), (:x, 3)]),
    ("int_bound_direct",   "int_bound_direct",   () -> [:x => [1.0, 2.0, 3.0, 4.0, 5.0], :cnt => [1], :n => 3, :out => zeros(1)], [(:x, 1), (:x, 2), (:x, 3)]),
    ("int_array_as_number", "int_array_as_number", () -> [:x => [1.1, 2.2, 3.3, 4.4], :cnt => [1, 2, 3, 4], :n => 4, :out => zeros(1)], X4),
    ("flag_two_phase",     "flag_two_phase",     () -> [:x => [1.1, 2.2, 3.3, 4.4], :flag => [1, 0, 1, 0], :n => 4, :out => zeros(1)], X4),
    ("branch_counter_for", "branch_counter_for", () -> [:x => [0.5, -0.3, 0.8, 0.9, -0.1], :w => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], :n => 5, :out => zeros(1)], BC),
    ("branch_counter_while", "branch_counter_while", () -> [:x => [0.5, -0.3, 0.8, 0.9, -0.1], :w => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], :n => 5.5, :out => zeros(1)], BC),
    # a value read after the loop that set it, with and without a pass; a trip count that jumps, checked on both sides of the jump
    ("after_loop_read",    "after_loop_read",    () -> [:x => [1.0, 2.0, 3.0, 4.0], :idx => [2, 3, 1], :n => 3.5, :out => zeros(1)], [(:x, 1), (:x, 2), (:x, 3)]),
    ("after_loop_zero",    "after_loop_read",    () -> [:x => [1.0, 2.0, 3.0, 4.0], :idx => [2, 3, 1], :n => 0.5, :out => zeros(1)], [(:x, 1), (:x, 2), (:x, 3)]),
    ("float_trip_below",   "float_trip",         () -> [:x => [0.6249], :lim => 10.0, :out => zeros(1)], [(:x, 1)]),
    ("float_trip_above",   "float_trip",         () -> [:x => [0.6251], :lim => 10.0, :out => zeros(1)], [(:x, 1)]),
    # controls: these passed before the change and must keep passing
    ("for_gather",         "for_gather",         () -> gather_args(3),  [(:x, 1), (:x, 2), (:x, 3)]),
    ("while_inline_index", "while_inline_index", arith_args,   [(:x, 1), (:y, 1)]),
    ("amr_inline",         "amr_inline",         () -> amr_args(50.0, 1e-5), AMR_TARGETS),
    ("amr_inline_multi",   "amr_inline",         () -> amr_args(8.0, 1e-4),  AMR_TARGETS),
    ("kepler",             "kepler",             () -> [:M => 2.0, :e => 0.7, :tol => 1e-12, :out => zeros(1)], [(:M,), (:e,)]),
]

dup_arg(v::AbstractArray) = copy(v)
dup_arg(v) = v

function int_state_primal(m, kname, args, outname)
    vals = Any[dup_arg(v) for (_, v) in args]
    Base.invokelatest(getfield(m, Symbol(kname)), vals...)
    return vals[findfirst(p -> p.first == outname, args)][1]
end

# calls the generated adjoint by reading its argument names; returns the adjoint buffers by primal name
function int_state_adjoint(m, kname, args, outname)
    fb = getfield(m, Symbol(kname * "_b"))
    init = getfield(m, Symbol("initstacks_" * kname * "_b"))
    names = [n for n in Base.method_argnames(first(methods(fb))) if !startswith(string(n), "#")]
    primal = Dict(args)
    stacks = Base.invokelatest(init)
    stacks = stacks === nothing ? () : (stacks isa Tuple ? stacks : (stacks,))
    si = 0
    vals = Any[]
    adj = Dict{Symbol,Any}()
    scalar_adj = Symbol[]
    for nm in names
        s = string(nm)
        if haskey(primal, nm)
            push!(vals, dup_arg(primal[nm]))
        elseif endswith(s, "_stack")
            si += 1
            push!(vals, stacks[si])
        elseif endswith(s, "b") && haskey(primal, Symbol(s[1:end-1]))
            base = Symbol(s[1:end-1])
            pv = primal[base]
            if pv isa AbstractArray
                buf = base == outname ? ones(size(pv)) : zeros(size(pv))
            else
                buf = 0.0
                push!(scalar_adj, base)
            end
            adj[base] = buf
            push!(vals, buf)
        else
            error("cannot map generated argument `$nm`")
        end
    end
    r = Base.invokelatest(fb, vals...)
    rs = r === nothing ? () : (r isa Tuple ? r : (r,))
    for (b, v) in zip(scalar_adj, rs)
        adj[b] = v
    end
    return adj
end

# adjoint against central finite differences of the primal; returns the number of mismatching targets
function int_state_fd(label, kname, mk, targets, fuse)
    dir = mktempdir()
    src = joinpath(INT_DIR, kname * ".jl")
    dst = joinpath(dir, kname * "_b.jl")
    STADE.stade_adjoint_file(src, dst; keep_push_pop = true, fuse_ii_loops = fuse)
    m = Module(:IntStateCase)
    Base.include(m, src)
    Base.include(m, dst)
    args = mk()
    adj = int_state_adjoint(m, kname, args, :out)
    bad = 0
    worst = 0.0
    for t in targets
        name = t[1]
        base = args[findfirst(p -> p.first == name, args)].second
        a = length(t) == 1 ? adj[name] : adj[name][t[2]]
        function with(delta)
            alt = [p.first == name ? (name => (length(t) == 1 ? base + delta :
                   (c = copy(base); c[t[2]] += delta; c))) : p for p in args]
            return int_state_primal(m, kname, alt, :out)
        end
        h = 1e-6 * max(1.0, abs(length(t) == 1 ? base : base[t[2]]))
        fd = (with(h) - with(-h)) / (2h)
        good = abs(a - fd) <= 1e-6 + 1e-5 * abs(fd)
        good || (bad += 1)
        worst = max(worst, abs(a - fd) / max(abs(fd), 1e-12))
    end
    return (bad, worst)
end

# the same arguments as a baseline file, so the four generic oracles run on them
function int_state_baseline(kname, args, dir)
    kernel = STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(INT_DIR, kname * ".jl")))
    int_args = Dict{Symbol,Any}()
    values = Dict{Symbol,Any}()
    for (nm, v) in args
        if get(kernel.sig.kinds, nm, :none) == :scalar_int
            int_args[nm] = Int(v)
        else
            values[nm] = v isa AbstractArray ? copy(v) : v
        end
    end
    yp = joinpath(dir, kname * ".yaml")
    STADE.io_write_baseline_yaml(yp, Symbol(kname), int_args, values)
    return yp
end

function validate_int_state(; trials::Int = 3)
    bad = 0
    for fuse in (false, true)
        say("--- adjoint vs finite differences, fuse_ii_loops = ", fuse)
        for (label, kname, mk, targets) in INT_CASES
            try
                (n_bad, worst) = int_state_fd(label, kname, mk, targets, fuse)
                n_bad == 0 || (bad += 1)
                say(rpad(label, 24), n_bad == 0 ? "ok    " : "FAIL  ", n_bad, " of ", length(targets),
                        " targets off, worst rel err ", round(worst, sigdigits = 3))
            catch e
                bad += 1
                say(rpad(label, 24), "ERR   ", first(sprint(showerror, e), 110))
            end
        end
    end

    for fuse in (false, true)
        say("--- tangent, dot-product identity and HVP, fuse_ii_loops = ", fuse)
        for (label, kname, mk, _) in INT_CASES
            # kepler's tol is an independent of the generic oracles, and a finite-difference step of 1e-6 makes the
            # tolerance 1e-12 negative, so the Newton loop never ends. Its adjoint is covered by the checks above.
            kname == "kepler" && continue
            src = joinpath(INT_DIR, kname * ".jl")
            yp = int_state_baseline(kname, mk(), mktempdir())
            Random.seed!(hash(label))
            for (what, f) in (("tangent", () -> STADE.stade_validate_tangent_file(src; yaml_path = yp, trials = trials,
                                    keep_push_pop = true, fuse_ii_loops = fuse)),
                              ("dotprod", () -> STADE.stade_validate_dotprod_file(src; yaml_path = yp, trials = trials,
                                    keep_push_pop = true, fuse_ii_loops = fuse)),
                              ("hvp",     () -> STADE.stade_validate_hvp_file(src; yaml_path = yp, trials = trials,
                                    keep_push_pop = true, fuse_ii_loops = fuse)))
                try
                    r = f()
                    r.ok || (bad += 1)
                    say(rpad("$label [$what]", 34), r.ok ? "ok    " : "FAIL  ", round(r.max_rel_err, sigdigits = 3))
                catch e
                    bad += 1
                    say(rpad("$label [$what]", 34), "ERR   ", first(sprint(showerror, e), 100))
                end
            end
        end
    end

    say("\n--- refusals")
    checks = 0
    function expect_refusal(label, fragment, f)
        checks += 1
        msg = try
            f(); ""
        catch e
            sprint(showerror, e)
        end
        if isempty(msg)
            bad += 1
            say(rpad(label, 52), " FAIL  accepted")
        elseif !occursin(fragment, msg)
            bad += 1
            say(rpad(label, 52), " FAIL  wrong reason: ", first(msg, 90))
        else
            say(rpad(label, 52), " ok")
        end
    end
    entry(name) = STADE.io_read_corpus_entry(joinpath(INT_DIR, name * ".jl"))

    # a restored integer needs a stack of unknown size
    expect_refusal("keep_push_pop = false with a restored integer", "must be restored",
                   () -> STADE.stade_adjoint(entry("for_int_rewrite"); keep_push_pop = false))
    # an integer array that is read and then written again needs the same stack
    expect_refusal("keep_push_pop = false with a restored integer array", "must be restored",
                   () -> STADE.stade_adjoint(entry("two_phase_perm"); keep_push_pop = false))
    expect_refusal("keep_push_pop = false, HVP", "must be restored",
                   () -> STADE.stade_hvp(entry("for_int_rewrite"); keep_push_pop = false))
    # a while kernel that needs a restored integer reports that; one that does not keeps the existing message
    expect_refusal("keep_push_pop = false, while with a restored integer", "must be restored",
                   () -> STADE.stade_adjoint(entry("while_gather"); keep_push_pop = false))
    expect_refusal("keep_push_pop = false, while without one", "no trip count",
                   () -> STADE.stade_adjoint(STADE.io_read_corpus_entry(joinpath(@__DIR__, "while-corpus", "while_count.jl"));
                                             keep_push_pop = false))
    # the HVP used to fail here with a KeyError on a stack key
    expect_refusal("keep_push_pop = false, while HVP", "no trip count",
                   () -> STADE.stade_hvp(STADE.io_read_corpus_entry(joinpath(@__DIR__, "while-corpus", "while_count.jl"));
                                         keep_push_pop = false))

    say("\n", checks, " refusal checks run",
            bad == 0 ? "   all integer-state checks passed" : "   *** $bad NOT OK ***")
    return bad
end

validate_int_state()
