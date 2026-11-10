# Battle test, family F9 (CPU emulation tier): run the generated CUDA plans on the CPU through the ownership shim
# and report any element that two threads plain-write, or that one thread reads while another writes or adds.
# Subjects: the sequential negatives (F2), the plain-write shapes (F9), the nests of depth 3 and 4 (F3), the S1 witness.
# No GPU.  Wraps validate_ownership.jl.
include(joinpath(@__DIR__, "validate_ownership.jl"))

# arrays of rank 3 and above cannot be written by io_write_baseline_yaml (finding S5), so the nests get their own draw
function bt_nest_baseline(kernel)
    rng = Random.Xoshiro(STADE.val_stable_seed(String(kernel.sig.name)))
    ints = Dict{Symbol,Int}(a => rand(rng, 2:4) for a in kernel.sig.args if kernel.sig.kinds[a] === :scalar_int)
    rank = count(a -> kernel.sig.kinds[a] === :scalar_int, kernel.sig.args)
    dims = Tuple(ints[Symbol("n", k)] for k in 1:rank)
    values = Dict{Symbol,Any}()
    for a in kernel.sig.args
        kd = kernel.sig.kinds[a]
        kd === :scalar_float && (values[a] = 0.5 + rand(rng))
        kd === :array_float && (values[a] = STADE.val_randn(rng, dims...))
    end
    return (int_args = ints, values = values)
end

# validate_ownership defines constants in Main, so it runs once per process. The standard subjects and the nests run in two processes.
function validate_battle_ownership(; names = nothing, int_lo = 3, int_hi = 5)
    dir = joinpath(@__DIR__, "battle-corpus")
    if names === nothing
        all = [splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl")]
        names = sort(filter(n -> (startswith(n, "f2_") && n != "f2_counter") || startswith(n, "f9_") || n == "f4_seq_inner_ind" ||
                                 (startswith(n, "f3_nest_") && !endswith(n, "_tri") && length(split(n, "_")[3]) <= 4), all))
    end
    nests = String[n for n in names if startswith(n, "f3_nest_")]
    rest = String[n for n in names if !(n in nests)]
    println("battle ownership: ", length(rest), " subjects with the standard draw, ", length(nests), " nests with their own draw")
    isempty(nests) && return validate_ownership(dir; only = rest, int_lo = int_lo, int_hi = int_hi, expected_flagged = String[])
    isempty(rest) && return validate_ownership(dir; only = nests, int_lo = int_lo, int_hi = int_hi, expected_flagged = String[], baseline_fn = bt_nest_baseline)
    return nothing
end
function bt_ownership_children()
    bad = 0
    dir = joinpath(@__DIR__, "battle-corpus")
    all = [splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl")]
    for grp in ("std", "nests")
        sel = sort(filter(n -> grp == "nests" ? (startswith(n, "f3_nest_") && !endswith(n, "_tri") && length(split(n, "_")[3]) <= 4) :
                               ((startswith(n, "f2_") && n != "f2_counter") || startswith(n, "f9_") || n == "f4_seq_inner_ind"), all))
        p = run(ignorestatus(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(@__FILE__) $sel`))
        p.exitcode == 0 || (bad += 1)
    end
    return bad
end
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        exit(bt_ownership_children() == 0 ? 0 : 1)
    end
    r = validate_battle_ownership(; names = ARGS)
    exit(r.bad == 0 ? 0 : 1)
end
