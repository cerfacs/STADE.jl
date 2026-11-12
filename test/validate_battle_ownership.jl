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

# f9_gather_write writes y[idx[i]] with a plain store. The tool trusts such an index to be a permutation (see cgen_rmw_hazard_arrays), so the
# standard draw, which may repeat a value, breaks the contract and gives a thread-order-dependent result on some Julia versions. This draw gives
# idx a rotation of 1:n. Every array has length n.
function bt_perm_baseline(kernel)
    rng = Random.Xoshiro(STADE.val_stable_seed(String(kernel.sig.name)))
    ints = Dict{Symbol,Int}(a => rand(rng, 3:5) for a in kernel.sig.args if kernel.sig.kinds[a] === :scalar_int)
    n = ints[Symbol("n")]
    values = Dict{Symbol,Any}()
    for a in kernel.sig.args
        kd = kernel.sig.kinds[a]
        kd === :scalar_float && (values[a] = 0.5 + rand(rng))
        kd === :array_float && (values[a] = STADE.val_randn(rng, n))
        kd === :array_int && (values[a] = Int[mod(k, n) + 1 for k in 1:n])
    end
    return (int_args = ints, values = values)
end

# validate_ownership defines constants in Main, so it runs once per process. The standard subjects, the nests and the permutation subject run in three processes.
function validate_battle_ownership(; names = nothing, int_lo = 3, int_hi = 5)
    dir = joinpath(@__DIR__, "battle-corpus")
    if names === nothing
        all = [splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl")]
        names = sort(filter(n -> (startswith(n, "f2_") && n != "f2_counter") || startswith(n, "f9_") || n == "f4_seq_inner_ind" ||
                                 (startswith(n, "f3_nest_") && !endswith(n, "_tri") && length(split(n, "_")[3]) <= 4), all))
    end
    nests = String[n for n in names if startswith(n, "f3_nest_")]
    perm = String[n for n in names if n == "f9_gather_write"]
    rest = String[n for n in names if !(n in nests) && !(n in perm)]
    println("battle ownership: ", length(rest), " subjects with the standard draw, ", length(nests), " nests with their own draw, ", length(perm), " with a permutation draw")
    isempty(nests) && isempty(perm) && return validate_ownership(dir; only = rest, int_lo = int_lo, int_hi = int_hi, expected_flagged = String[])
    isempty(rest) && isempty(perm) && return validate_ownership(dir; only = nests, int_lo = int_lo, int_hi = int_hi, expected_flagged = String[], baseline_fn = bt_nest_baseline)
    isempty(rest) && isempty(nests) && return validate_ownership(dir; only = perm, int_lo = int_lo, int_hi = int_hi, expected_flagged = String[], baseline_fn = bt_perm_baseline)
    return nothing
end
function bt_ownership_children()
    bad = 0
    dir = joinpath(@__DIR__, "battle-corpus")
    all = [splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl")]
    for grp in ("std", "nests", "perm")
        sel = sort(filter(n -> grp == "nests" ? (startswith(n, "f3_nest_") && !endswith(n, "_tri") && length(split(n, "_")[3]) <= 4) :
                               grp == "perm" ? n == "f9_gather_write" :
                               ((startswith(n, "f2_") && n != "f2_counter") || (startswith(n, "f9_") && n != "f9_gather_write") || n == "f4_seq_inner_ind"), all))
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
