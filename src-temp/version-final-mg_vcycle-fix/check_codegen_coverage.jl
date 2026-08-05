# ============================================================
# check_codegen_coverage.jl -- end-to-end correctness check for STADE's
# agen_* codegen: generates _b/initstacks_ (adjoint)
# Julia code from the real corpus primals, `eval`s it, and
# checks the ADJOINT against `val_finite_diff_check`
#
# This variant additionally tracks CODE COVERAGE of STADE.jl
# during the `stade_adjoint(primal_expr)` call.
#
# IMPORTANT: .cov files are only flushed to disk by Julia's
# atexit hook when the PROCESS EXITS -- there is no reliable,
# version-stable way to force a mid-run flush from inside the
# script. So this script does the coverage-instrumented work
# only, and exits cleanly; run "summarize_coverage_1.jl"
# (plain julia, no coverage flag needed) AFTERWARDS, as a
# separate process, to read the .cov file and produce
# "stade_calls.log".
#
# Run with (two separate commands, in order):
#     julia --code-coverage=user check_codegen_coverage.jl
#     julia summarize_coverage.jl
# from a directory containing STADE.jl and all_b.jl.
# ============================================================
isdefined(Main, :agen_emit) || include(joinpath(@__DIR__, "STADE.jl"))

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

# generate adjoint + initstacks for `name`, eval them into
# Main alongside the primal itself, and return the generated Exprs
# (handy for eyeballing on failure);
# also write them as files on disk.
function generate_and_eval(name::Symbol)
    primal_expr = grab_kernel_expr(ALL_B_PATH, name)
    adjoint_out = stade_adjoint(primal_expr)
    name_str    = String(name)
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

# Nothing else to do here -- let the process exit normally so
# Julia's atexit hook writes the STADE.jl.<pid>.cov file next to
# STADE.jl. Then run summarize_coverage.jl separately.
println("done -- now run: julia summarize_coverage.jl")