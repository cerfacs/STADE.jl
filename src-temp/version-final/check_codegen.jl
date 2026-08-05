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
# (handy for eyeballing on failure)
function generate_and_eval(name::Symbol)
    primal_expr = grab_kernel_expr(ALL_B_PATH, name)
    tangent_expr = stade_tangent(primal_expr)
    adjoint_out = stade_adjoint(primal_expr)
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

generate_and_eval(:func)
generate_and_eval(:calc)

let nnode = 5, nstep = 2
    y_u = randn(nnode); y_du = randn(nnode)
    x0 = vcat(randn(nnode), randn(nnode),
              [1.0 + 0.1randn(), 0.8 + 0.1randn(), 0.05 + 0.01randn()])
    unpack = xv -> (xv[1:nnode], xv[nnode+1:2nnode], xv[2nnode+1], xv[2nnode+2], xv[2nnode+3])
    f_eval = function (xv)
        u0, du0, c, dx, dt = unpack(xv)
        u = copy(u0); du = copy(du0)
        func(u, du, c, dx, dt, nstep, nnode)
        return sum(y_u .* u) + sum(y_du .* du)
    end
    f_grad = function (xv)
        u0, du0, c, dx, dt = unpack(xv)
        u = copy(u0); du = copy(du0)
        ub = copy(y_u); dub = copy(y_du)
        du_stack = initstacks_func_b()
        cb, dxb, dtb = func_b(u, ub, du, dub, c, 0.0, dx, 0.0, dt, 0.0, nstep, nnode, du_stack)
        return vcat(ub, dub, [cb, dxb, dtb])
    end
    report("advection (func)", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
end

let num_levels = 2, nfine = 5, nu1 = 2, nu2 = 2
    max_n = nfine - 1
    y_u = randn(max_n, num_levels); y_f = randn(max_n, num_levels)
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
        return sum(y_u .* u) + sum(y_f .* f)
    end
    f_grad = function (xv)
        u0, f0, h1 = unpack(xv)
        u, f, r = build_arrays(u0, f0)
        ub = copy(y_u); fb = copy(y_f); rb = zeros(max_n, num_levels)
        stacks = initstacks_calc_b()
        h1b = calc_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, 0.0, nu1, nu2, 0.0, 0, stacks...)
        return vcat(ub[:, 1], fb[:, 1], [h1b])
    end
    report("calc", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
end

let num_levels = 2, nfine = 5, nu1 = 2, nu2 = 2
    max_n = nfine - 1
    y_u = randn(max_n, num_levels); y_f = randn(max_n, num_levels)
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
        return sum(y_u .* u) + sum(y_f .* f)
    end
    f_tangent = function (xv, d)
        u0, f0, h1 = unpack(xv)
        ud0, fd0, h1d = unpack(d)
        u, f, r = build_arrays(u0, f0)
        ud, fd, rd = build_arrays(ud0, fd0)
        mg_vcycle_d(u, ud, f, fd, r, rd, nfine, num_levels, h1, h1d, nu1, nu2, 0)
        return sum(y_u .* ud) + sum(y_f .* fd)
    end
    tangent_check("calc", f_eval, f_tangent, x0)
end

println()
println("check_codegen.jl: all done")