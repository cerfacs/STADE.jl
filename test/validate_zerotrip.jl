include(joinpath(@__DIR__, "validate_corpus.jl"))

"""
    validate_zerotrip(dir="val-corpus")

Run the corpus with integer arguments drawn from a range that INCLUDES zero, so
loops whose bound is an integer argument actually execute zero times.

Why this exists. Six separate analyses in STADE.jl have treated a loop that may
not run as having run -- `norm_first_touch`, `cgen_last_assign_is_zero`,
`ii_kill_and_collect!`, `cgen_loop_convergent_constant`, and the post-loop
`known_consts` update in both `cgen_body` and `jgen_body`. Every one produced
silently wrong gradients. Not one was reachable from the corpus, because the
baseline generator drew integer arguments from [3, 5] and so no loop ever had a
zero trip count. Each was found only after a kernel was hand-written to retire a
bound past zero (`retire_empty`, `entry_empty`, `ii_kill`), and two of them
needed a GPU run on top of that.

This closes the gap from the other side: instead of relying on someone thinking
to write the kernel, draw the bound. A kernel that degenerates to running nothing
at all is redrawn rather than reported -- `val_generate_baseline` now rejects a
candidate whose observed primal output is identically zero, which is the case the
generator's old comment (wrongly) assumed was the only thing a zero draw could
produce.

Deterministic: validate_corpus seeds per kernel from its name, so a given kernel
draws the same integers on every run.

Expect FEWER kernels than the ordinary corpus run. A kernel whose arrays are
sized from the same integers may have no coherent draw containing a zero, and is
skipped rather than failed -- that is a gap in coverage, not a defect. What
matters is the kernels that DO get a zero-trip draw and still agree with finite
differences.

    cd test && julia validate_zerotrip.jl
"""
function validate_zerotrip(dir::String = joinpath(@__DIR__, "val-corpus");
                            keep_push_pop::Bool = false, fuse_ii_loops::Bool = true)
    println("corpus with int args drawn from [0, 2] -- loops whose bound is an int arg can be empty\n")
    results = validate_corpus(dir; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops,
                               int_lo = 0, int_hi = 2)

    # which kernels actually got a zero, i.e. where this run bought coverage the
    # ordinary one cannot
    zero_drawn = String[]
    for f in sort(filter(f -> endswith(f, ".yaml"), readdir(dir)))
        name = splitext(f)[1]
        b = try
            STADE.io_read_baseline_yaml(joinpath(dir, f))
        catch
            continue
        end
        any(v -> v == 0, values(b.int_args)) && push!(zero_drawn, name)
    end

    bad = count(r -> r.status != :ok, results)
    println("\n", length(zero_drawn), " kernels drew a zero-trip bound: ", join(zero_drawn, ", "))
    # These baselines hold integers from [0, 2] and `.yaml` is gitignored, so leaving
    # them behind hands the next script a corpus where nearly every loop is empty.
    # A GPU parity run inheriting one executes no device code and still reports a
    # pass. Deleting them keeps this script's deliberately degenerate draw local to
    # this script.
    for f in filter(f -> endswith(f, ".yaml"), readdir(dir))
        rm(joinpath(dir, f))
    end
    println(length(results) - bad, "/", length(results), " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_zerotrip()
