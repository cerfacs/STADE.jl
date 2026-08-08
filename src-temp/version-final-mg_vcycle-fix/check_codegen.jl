# ============================================================
# check_codegen.jl -- end-to-end correctness check for STADE's
# agen_* codegen: generates _b/initstacks_ (adjoint) 
# Julia code from the real corpus primals, `eval`s it, and
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

let num_levels = 2, nfine = 5, nu1 = 2, nu2 = 2
    max_n = nfine - 1
    y_u = randn(max_n, num_levels); y_f = randn(max_n, num_levels); y_r = randn(max_n, num_levels)
    x0 = vcat(randn(max_n), randn(max_n), [1.0 + 0.1randn()])
    unpack = xv -> (xv[1:max_n], xv[max_n+1:2max_n], xv[2max_n+1])
    build_arrays = function (u0, f0)
        u = zeros(max_n, num_levels); u[:, 1] = u0
        f = zeros(max_n, num_levels); f[:, 1] = f0
        r = zeros(max_n, num_levels)
        return u, f, r
    end
    f_eval = function (xv)
        u0, f0, h1 = unpack(xv)
        u, f, r = build_arrays(u0, f0)
        calc(u, f, r, nfine, num_levels, h1, nu1, nu2, 0)
        return sum(y_u .* u) + sum(y_f .* f) + sum(y_r .* r)
    end
    f_grad = function (xv)
        u0, f0, h1 = unpack(xv)
        u, f, r = build_arrays(u0, f0)
        ub = copy(y_u); fb = copy(y_f); rb = copy(y_r)
        stacks = initstacks_calc_b()
        h1b = calc_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, 0.0, nu1, nu2, 0, stacks...)
        return vcat(ub[:, 1], fb[:, 1], [h1b])
    end
    report("calc", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
end

println()
println("check_codegen.jl: all done")