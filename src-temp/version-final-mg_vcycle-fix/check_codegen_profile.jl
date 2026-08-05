# ============================================================
# check_codegen_profile.jl -- end-to-end correctness check for STADE's
# agen_* codegen: generates _b/initstacks_ (adjoint)
# Julia code from the real corpus primals, `eval`s it, and
# checks the ADJOINT against `val_finite_diff_check`
#
# This variant additionally runs the SAMPLING PROFILER on the
# `stade_adjoint(primal_expr)` call and writes the resulting
# call tree (with STADE.jl:LINE annotations) to
# "stade_profile.log".
#
# Run with:
#     julia check_codegen_profile.jl
# from a directory containing STADE.jl and all_b.jl.
#
# NOTE: sampling profilers can miss very short-lived calls.
# If stade_adjoint() runs too fast for useful samples to
# accumulate, N_PROFILE_REPEATS below is used to call it
# repeatedly inside the profiled region -- bump it up if
# stade_profile.log comes back mostly empty.
# ============================================================
isdefined(Main, :agen_emit) || include(joinpath(@__DIR__, "STADE.jl"))

using Profile

function grab_kernel_expr(path::String, name::Symbol)
    parsed = Meta.parseall(read(path, String))
    for e in parsed.args
        if e isa Expr && e.head == :function
            sig = e.args[1]
            if sig isa Expr && sig.head == :call && sig.args[1] == name
                return e
            end
        end
    end
    error("no `function $name(...)` found in $path")
end

const ALL_B_PATH = joinpath(@__DIR__, "all_b.jl")
const N_PROFILE_REPEATS = 1000  # bump up if profile output is too sparse

# generate adjoint + initstacks for `name`, eval them into
# Main alongside the primal itself, and return the generated Exprs
# (handy for eyeballing on failure);
# also write them as files on disk.
#
# The stade_adjoint(primal_expr) call itself is wrapped in
# Profile.@profile so we get a sampled call tree of everything
# it invokes inside STADE.jl.
function generate_and_eval(name::Symbol)
    primal_expr = grab_kernel_expr(ALL_B_PATH, name)

    Profile.init(n = 10^7, delay = 0.00001)  # fine sampling resolution
    Profile.clear()

    local adjoint_out
    Profile.@profile begin
        # repeat the call a bunch of times so the sampler has
        # enough samples to build a meaningful tree, since a
        # single call may be too fast to sample well
        for _ in 1:N_PROFILE_REPEATS
            adjoint_out = stade_adjoint(primal_expr)
        end
    end

    open(joinpath(@__DIR__, "stade_profile.log"), "w") do io
        println(io, "# sampled call tree for stade_adjoint(primal_expr)")
        println(io, "# ($(N_PROFILE_REPEATS) repeats profiled together)")
        println(io, "#" * "-"^60)
        Profile.print(io; format = :tree, C = false, mincount = 1)
    end
    println("wrote profile to ", joinpath(@__DIR__, "stade_profile.log"))

    name_str = String(name)
    io_write_kernel_file(name_str * "_b.jl", primal_expr, [adjoint_out.initstacks, adjoint_out.adjoint])
    Base.eval(Main, primal_expr)
    Base.eval(Main, adjoint_out.initstacks)
    Base.eval(Main, adjoint_out.adjoint)
    return (adjoint = adjoint_out.adjoint, initstacks = adjoint_out.initstacks)
end

function report(name, fx; trials = 10)
    r = val_check_fixture(fx; trials = trials)
    status = r.ok ? "ok  " : "FAIL"
    println(rpad(name, 22), status, "  max_rel_err=", round(r.max_rel_err, sigdigits = 3))
    return r
end

generate_and_eval(:calc)