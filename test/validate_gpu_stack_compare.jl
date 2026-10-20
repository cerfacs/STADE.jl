include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

"""
    validate_gpu_stack_compare()

Check the stack comparison in the generated GPU parity script, with no GPU. The script compares the host stacks with the device stacks
and reports the worst difference (informational: it is not part of `ok`). A slot that nothing writes held allocator garbage, so host and
device differed there from run to run (a fused ttgc HVP read 1.0 in one run and 5e-16 in the next, on 0.4.7 and on 0.4.8 alike). The
script now zero-fills float stacks before each run, so only a written slot can differ.

The two helper functions are cut out of a real generated script and run on the CPU:

  1. an allocator that returns garbage gives all-zero float stacks, and integer stacks and non-array entries are left alone;
  2. identical written slots, with the unwritten slots left as they were, compare as 0.0;
  3. one written slot that differs is reported (the stale-scalar class this comparison exists for);
  4. the sabotage run, with the fill removed from the script, fails check 1.
"""
function validate_gpu_stack_compare()
    work = mktempdir()
    cp(joinpath(@__DIR__, "val-corpus", "asnap_ok.jl"), joinpath(work, "asnap_ok.jl"))
    Random.seed!(7)
    out = joinpath(work, "parity.jl")
    STADE.stade_validate_cuda_file(joinpath(work, "asnap_ok.jl"), out; mode = :adjoint, int_lo = 3, int_hi = 5)
    script = read(out, String)

    function helpers(text)
        m = Module(:StackCmp)
        Core.eval(m, :(_relerr_arr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1.0); init = 0.0)))
        for e in Meta.parseall(text).args
            e isa Expr && e.head == :function || continue
            nm = e.args[1] isa Expr && e.args[1].head == :call ? e.args[1].args[1] : nothing
            nm in (:_val_init_stacks_local, :_val_stack_relerr) && Core.eval(m, e)
        end
        return m
    end

    bad = 0
    check(label, ok) = (println(rpad(label, 72), ok ? " ok" : " FAIL"); ok || (bad += 1))
    garbage() = (fill(NaN, 6), Int[7, 8, 9], 42, fill(1.0e300, 3))
    allocator(_...) = garbage()

    m = helpers(script)
    r = Base.invokelatest(m._val_init_stacks_local, allocator, Any[])
    check("float stacks zero-filled", all(iszero, r[1]) && all(iszero, r[4]))
    check("integer stack and scalar entry left alone", r[2] == [7, 8, 9] && r[3] == 42)

    cpu = (zeros(6), [1, 2, 3], 5); gpu = (zeros(6), [9, 9, 9], 5)
    cpu[1][2] = 3.5; gpu[1][2] = 3.5
    (w, n) = Base.invokelatest(m._val_stack_relerr, cpu, gpu)
    check("identical written slots, unwritten slots zero: 0.0 over 1 stack", w == 0.0 && n == 1)
    gpu[1][2] = 7.0
    (w, n) = Base.invokelatest(m._val_stack_relerr, cpu, gpu)
    check("one written slot differs: reported", w > 0.4)

    # sabotage: no fill, so the garbage survives
    bare = replace(script, "(s isa AbstractArray && eltype(s) <: AbstractFloat) && fill!(s, 0.0)" => "nothing")
    check("sabotage input differs from the script", bare != script)
    r2 = Base.invokelatest(helpers(bare)._val_init_stacks_local, allocator, Any[])
    check("sabotage (fill removed): garbage survives, check 1 would fail", !all(iszero, r2[1]))

    println(bad == 0 ? "\nall stack-compare checks passed" : "\n$(bad) NOT OK")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_gpu_stack_compare() == 0 ? 0 : 1)
end
