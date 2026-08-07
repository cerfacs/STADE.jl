# ============================================================
# check_codegen.jl -- end-to-end correctness check for STADE's
# tgen_*/agen_* codegen: generates _d (tangent) and _b/initstacks_
# (adjoint) Julia code from the real corpus primals, `eval`s it, and
# checks the ADJOINT against `val_finite_diff_check`
#
# Run with:
#     julia check_codegen.jl
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

# generate tangent + adjoint + initstacks for `name`, eval them into
# Main alongside the primal itself, and return the generated Exprs
# (handy for eyeballing on failure);
# also write them as files on disk.
function generate_and_eval(name::Symbol)
    primal_expr = grab_kernel_expr(ALL_B_PATH, name)
    tangent_expr = stade_tangent(primal_expr)
    adjoint_out = stade_adjoint(primal_expr)
    name_str    = String(name)
    io_write_kernel_file(name_str * "_d.jl", primal_expr, [tangent_expr])
    io_write_kernel_file(name_str * "_b.jl", primal_expr, [adjoint_out.initstacks, adjoint_out.adjoint])
    Base.eval(Main, primal_expr)
    Base.eval(Main, tangent_expr)
    Base.eval(Main, adjoint_out.initstacks)
    Base.eval(Main, adjoint_out.adjoint)
    return (tangent = tangent_expr, adjoint = adjoint_out.adjoint, initstacks = adjoint_out.initstacks)
end

function report(name, fx; trials = 10)
    r = val_check_fixture(fx; trials = trials)
    status = r.ok ? "ok  " : "FAIL"
    println(rpad(name, 22), status, "  max_rel_err=", round(r.max_rel_err, sigdigits = 3))
    return r
end

# tangent mode: directional derivative vs. raw central finite difference
function tangent_check(name, f_eval, f_tangent, x0; epsilon = 1e-6, trials = 5)
    worst = 0.0
    for _ in 1:trials
        d = randn(length(x0)); d = d ./ sqrt(sum(d .^ 2))
        fd = (f_eval(x0 .+ epsilon .* d) - f_eval(x0 .- epsilon .* d)) / (2epsilon)
        td = f_tangent(x0, d)
        denom = max(abs(fd), abs(td), 1e-12)
        worst = max(worst, abs(fd - td) / denom)
    end
    println(rpad(name, 22), worst <= 1e-3 ? "ok  " : "FAIL", "  max_rel_err=", round(worst, sigdigits = 3))
    return worst
end


# TODO: Add fixtures of validation golden-corpus here


println()
println("check_codegen.jl: all done")