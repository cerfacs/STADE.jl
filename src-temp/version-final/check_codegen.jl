# ============================================================
# check_codegen.jl -- end-to-end correctness check for STADE's
# tgen_*/agen_* codegen: generates _d (tangent) and _b/initstacks_
# (adjoint) Julia code from the real corpus primals, `eval`s it, and
# checks the ADJOINT against `val_finite_diff_check` the same way
# `val_run_all_tiers` does for the ground-truth `_b.jl` corpus --
# real max_rel_err numbers, not just "it ran". A separate, smaller
# group directly checks the TANGENT output against a raw central
# finite difference (val_finite_diff_check itself is built around a
# full-gradient f_grad, i.e. adjoint-shaped, so tangent gets its own
# comparison here rather than being forced through that framework).
#
# Coverage:
#   - M1 (straight-line): all 10 corpus kernels -- required minimum.
#   - M2 (conditionals): branchsel, clamped_sumsq, cond_field_choice,
#     relu_field -- attempted beyond the minimum.
#   - M3 (loop-carried): geomrecur, stencil_loss, advection (`func`
#     in all_b.jl) -- attempted beyond the minimum. advection's
#     fixture is VJP-seeded (no scalar loss, output is the full
#     u/du arrays -- see the comment at its fixture below).
#   - M4: mg_vcycle -- attempted. This is the first real execution of
#     the :tripcount machinery and the deepest control-flow nesting
#     in the whole corpus (3 levels of sequential loop, several
#     locals with an if/else-like default-then-conditionally-
#     overwritten pattern feeding a snapshot site each); everything
#     up to this point has been hand-traced but never actually run,
#     so this result carries the least prior confidence of anything
#     in this file.
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

# ============================================================
# M1: straight-line -- required minimum
# ============================================================

println("=== M1 (adjoint, via val_finite_diff_check) ===")
generate_and_eval(:dotprod)
generate_and_eval(:quadloss)
generate_and_eval(:bilinear)
generate_and_eval(:matvec_loss)
generate_and_eval(:affine_loss)
generate_and_eval(:normcomp)
generate_and_eval(:weightedsumsq)
generate_and_eval(:sumsq_shifted)
generate_and_eval(:two_field_loss)
generate_and_eval(:pipeline)

m1_worst = 0.0

let n = 5
    x0 = vcat(randn(n), randn(n))
    f_eval = xv -> (u = xv[1:n]; v = xv[n+1:2n]; loss = [0.0]; dotprod(loss, u, v, n); loss[1])
    f_grad = function (xv)
        u = xv[1:n]; v = xv[n+1:2n]
        ub = zeros(n); vb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        dotprod_b(loss, lossb, u, ub, v, vb, n)
        return vcat(ub, vb)
    end
    global m1_worst = max(m1_worst, report("dotprod", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let
    x0 = randn(3)
    f_eval = xv -> (loss = [0.0]; quadloss(loss, xv[1], xv[2], xv[3]); loss[1])
    f_grad = function (xv)
        loss = [0.0]; lossb = [1.0]
        xb, yb, zb = quadloss_b(loss, lossb, xv[1], 0.0, xv[2], 0.0, xv[3], 0.0)
        return [xb, yb, zb]
    end
    global m1_worst = max(m1_worst, report("quadloss", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let m = 4, n = 3
    x0 = vcat(randn(m), vec(randn(m, n)), randn(n))
    unpack = xv -> (xv[1:m], reshape(xv[m+1:m+m*n], m, n), xv[m+m*n+1:m+m*n+n])
    f_eval = function (xv)
        xx, aa, yy = unpack(xv); loss = [0.0]
        bilinear(loss, xx, aa, yy, m, n); loss[1]
    end
    f_grad = function (xv)
        xx, aa, yy = unpack(xv)
        xb = zeros(m); ab = zeros(m, n); yb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        bilinear_b(loss, lossb, xx, xb, aa, ab, yy, yb, m, n)
        return vcat(xb, vec(ab), yb)
    end
    global m1_worst = max(m1_worst, report("bilinear", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let m = 4, n = 3
    x0 = vcat(vec(randn(m, n)), randn(n), randn(m))
    unpack = xv -> (reshape(xv[1:m*n], m, n), xv[m*n+1:m*n+n], xv[m*n+n+1:m*n+n+m])
    f_eval = function (xv)
        aa, uu, v0 = unpack(xv); loss = [0.0]; v = copy(v0)
        matvec_loss(loss, aa, uu, v, m, n); loss[1]
    end
    f_grad = function (xv)
        aa, uu, v0 = unpack(xv)
        ab = zeros(m, n); ub = zeros(n); vb = zeros(m)
        v = copy(v0); loss = [0.0]; lossb = [1.0]
        matvec_loss_b(loss, lossb, aa, ab, uu, ub, v, vb, m, n)
        return vcat(vec(ab), ub, vb)
    end
    global m1_worst = max(m1_worst, report("matvec_loss", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let n = 5
    x0 = vcat(randn(n), randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n], xv[2n+1:3n])
    f_eval = function (xv)
        uu, aa, bb_ = unpack(xv); loss = [0.0]; v = zeros(n)
        affine_loss(loss, uu, aa, bb_, v, n); loss[1]
    end
    f_grad = function (xv)
        uu, aa, bb_ = unpack(xv)
        ub = zeros(n); ab = zeros(n); bbb = zeros(n)
        v = zeros(n); vb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        affine_loss_b(loss, lossb, uu, ub, aa, ab, bb_, bbb, v, vb, n)
        return vcat(ub, ab, bbb)
    end
    global m1_worst = max(m1_worst, report("affine_loss", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let n = 5
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv)
        uu, vv = unpack(xv); loss = [0.0]; w = zeros(n)
        normcomp(loss, uu, vv, w, n); loss[1]
    end
    f_grad = function (xv)
        uu, vv = unpack(xv)
        ub = zeros(n); vb = zeros(n); w = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        normcomp_b(loss, lossb, uu, ub, vv, vb, w, wb, n)
        return vcat(ub, vb)
    end
    global m1_worst = max(m1_worst, report("normcomp", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let n = 5
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv)
        uu, ww = unpack(xv); loss = [0.0]
        weightedsumsq(loss, uu, ww, n); loss[1]
    end
    f_grad = function (xv)
        uu, ww = unpack(xv)
        ub = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        weightedsumsq_b(loss, lossb, uu, ub, ww, wb, n)
        return vcat(ub, wb)
    end
    global m1_worst = max(m1_worst, report("weightedsumsq", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let n = 5
    x0 = vcat(randn(n), randn(2))
    unpack = xv -> (xv[1:n], xv[n+1], xv[n+2])
    f_eval = function (xv)
        uu, alpha, beta = unpack(xv); loss = [0.0]
        sumsq_shifted(loss, uu, alpha, beta, n); loss[1]
    end
    f_grad = function (xv)
        uu, alpha, beta = unpack(xv)
        ub = zeros(n); loss = [0.0]; lossb = [1.0]
        alphab, betab = sumsq_shifted_b(loss, lossb, uu, ub, alpha, 0.0, beta, 0.0, n)
        return vcat(ub, [alphab, betab])
    end
    global m1_worst = max(m1_worst, report("sumsq_shifted", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let n = 5
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv)
        uu, vv = unpack(xv); loss = [0.0]; p = zeros(n); q = zeros(n)
        two_field_loss(loss, uu, vv, p, q, n); loss[1]
    end
    f_grad = function (xv)
        uu, vv = unpack(xv)
        ub = zeros(n); vb = zeros(n); p = zeros(n); pb = zeros(n); q = zeros(n); qb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        two_field_loss_b(loss, lossb, uu, ub, vv, vb, p, pb, q, qb, n)
        return vcat(ub, vb)
    end
    global m1_worst = max(m1_worst, report("two_field_loss", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

let n = 5
    x0 = randn(n)
    f_eval = function (xv)
        loss = [0.0]; v = zeros(n); w = zeros(n)
        pipeline(loss, xv, v, w, n); loss[1]
    end
    f_grad = function (xv)
        ub = zeros(n); v = zeros(n); vb = zeros(n); w = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        pipeline_b(loss, lossb, xv, ub, v, vb, w, wb, n)
        return ub
    end
    global m1_worst = max(m1_worst, report("pipeline", (f_eval = f_eval, f_grad = f_grad, x0 = x0)).max_rel_err)
end

println("M1 worst max_rel_err=", round(m1_worst, sigdigits = 3))

# ============================================================
# tangent mode: directional derivative vs. raw central finite
# difference, for a couple of M1 kernels (val_finite_diff_check
# itself is built around a full-gradient f_grad, adjoint-shaped, so
# this is a separate, direct comparison instead of reusing it)
# ============================================================

println()
println("=== tangent (_d) vs. raw central finite difference ===")

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

let n = 5
    x0 = vcat(randn(n), randn(n))
    f_eval = xv -> (u = xv[1:n]; v = xv[n+1:2n]; loss = [0.0]; dotprod(loss, u, v, n); loss[1])
    f_tangent = function (xv, d)
        u = xv[1:n]; v = xv[n+1:2n]
        ud = d[1:n]; vd = d[n+1:2n]
        loss = [0.0]; lossd = [0.0]
        dotprod_d(loss, lossd, u, ud, v, vd, n)
        return lossd[1]
    end
    tangent_check("dotprod", f_eval, f_tangent, x0)
end

let
    x0 = randn(3)
    f_eval = xv -> (loss = [0.0]; quadloss(loss, xv[1], xv[2], xv[3]); loss[1])
    f_tangent = function (xv, d)
        loss = [0.0]; lossd = [0.0]
        quadloss_d(loss, lossd, xv[1], d[1], xv[2], d[2], xv[3], d[3])
        return lossd[1]
    end
    tangent_check("quadloss", f_eval, f_tangent, x0)
end

# ============================================================
# M2: conditionals -- attempted beyond the minimum
# ============================================================

println()
println("=== M2 (adjoint, attempted beyond the minimum) ===")
generate_and_eval(:branchsel)
generate_and_eval(:clamped_sumsq)
generate_and_eval(:cond_field_choice)
generate_and_eval(:relu_field)

let
    x0 = randn(2)
    f_eval = xv -> (loss = [0.0]; branchsel(loss, xv[1], xv[2]); loss[1])
    f_grad = function (xv)
        loss = [0.0]; lossb = [1.0]
        branch_stack = initstacks_branchsel_b()
        xb, yb = branchsel_b(loss, lossb, xv[1], 0.0, xv[2], 0.0, branch_stack)
        return [xb, yb]
    end
    report("branchsel", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
end

let n = 6
    x0 = randn(n) .+ 0.5 .* sign.(randn(n))
    f_eval = function (xv)
        u = copy(xv); loss = [0.0]
        clamped_sumsq(loss, u, n); loss[1]
    end
    f_grad = function (xv)
        u = copy(xv); ub = zeros(n)
        loss = [0.0]; lossb = [1.0]
        # clamped_sumsq's local `w` also gets a snapshot site (read by
        # a different, later statement while sitting inside a
        # sequential loop -- agen_'s conservative rule), so this
        # generated adjoint needs two stacks, not just branch_stack
        branch_stack, w_stack = initstacks_clamped_sumsq_b()
        clamped_sumsq_b(loss, lossb, u, ub, n, branch_stack, w_stack)
        return ub
    end
    report("clamped_sumsq", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
end

let n = 6, i_branch = 1
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv)
        uu, vv = unpack(xv); loss = [0.0]; w = zeros(n)
        cond_field_choice(loss, uu, vv, w, i_branch, n); loss[1]
    end
    f_grad = function (xv)
        uu, vv = unpack(xv)
        ub = zeros(n); vb = zeros(n); w = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        branch_stack = initstacks_cond_field_choice_b()
        cond_field_choice_b(loss, lossb, uu, ub, vv, vb, w, wb, i_branch, n, branch_stack)
        return vcat(ub, vb)
    end
    report("cond_field_choice", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
end

let n = 6
    x0 = randn(n) .+ 0.5 .* sign.(randn(n))
    f_eval = function (xv)
        u = copy(xv); loss = [0.0]; v = zeros(n)
        relu_field(loss, u, v, n); loss[1]
    end
    f_grad = function (xv)
        u = copy(xv); ub = zeros(n); v = zeros(n); vb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        branch_stack = initstacks_relu_field_b()
        relu_field_b(loss, lossb, u, ub, v, vb, n, branch_stack)
        return ub
    end
    report("relu_field", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
end

# ============================================================
# M3: loop-carried recurrences -- attempted beyond the minimum
# ============================================================

println()
println("=== M3 (adjoint, attempted beyond the minimum) ===")
generate_and_eval(:geomrecur)
generate_and_eval(:stencil_loss)

let n = 6
    x0 = vcat(randn(n), randn(1))
    unpack = xv -> (xv[1:n], xv[n+1])
    f_eval = function (xv)
        uu, c = unpack(xv); u = copy(uu); loss = [0.0]
        geomrecur(loss, u, c, n); loss[1]
    end
    f_grad = function (xv)
        uu, c = unpack(xv); u = copy(uu); ub = zeros(n)
        loss = [0.0]; lossb = [1.0]
        u_stack = initstacks_geomrecur_b()
        cb = geomrecur_b(loss, lossb, u, ub, c, 0.0, n, u_stack)
        return vcat(ub, [cb])
    end
    r = report("geomrecur", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
    println("  (expect ub[2:end] ~ 0 -- those inputs are overwritten before read)")
end

let n = 6
    x0 = randn(n)
    f_eval = function (xv)
        u = copy(xv); loss = [0.0]; w = zeros(n)
        stencil_loss(loss, u, w, n); loss[1]
    end
    f_grad = function (xv)
        u = copy(xv); ub = zeros(n); w = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        stencil_loss_b(loss, lossb, u, ub, w, wb, n)
        return ub
    end
    report("stencil_loss", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
end

# advection: the primal is named `func`, not `advection`, in
# all_b.jl. No scalar loss -- output is the full u/du arrays, so
# this is VJP-seeded exactly like val_fixture_advection: fix random
# y_u/y_du once, f_eval = sum(y_u.*u) + sum(y_du.*du) (turns the
# vector output into an effective scalar val_finite_diff_check can
# still use unchanged), and f_grad seeds ub/dub with y_u/y_du
# directly -- the seed a VJP needs. initstacks_func_b here takes NO
# arguments (unlike the corpus's own initstacks_func_b(du)): every
# :array/:value stack in this design holds popped Float64 scalars,
# never a whole-array copy, so there's no eltype/shape to derive
# from an argument.
generate_and_eval(:func)

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

println()
println("check_codegen.jl: M1/M2/M3 adjoint + M1 tangent spot check done")

# ============================================================
# tangent mode (_d), M2/M3: same raw central-finite-difference
# comparison as the M1 spot check above, extended to kernels with
# branches and a loop-carried recurrence. None of these kernels
# reassign a scalar-float ARGUMENT anywhere in the primal (only
# array-refs or purely-local scalars get reassigned in this corpus),
# so tgen_reassigned_scalar_args is empty for all six and every _d
# call below returns nothing -- no tuple destructuring needed.
# ============================================================

println()
println("=== tangent (_d) M2/M3 vs. raw central finite difference ===")

let
    x0 = randn(2)
    f_eval = xv -> (loss = [0.0]; branchsel(loss, xv[1], xv[2]); loss[1])
    f_tangent = function (xv, d)
        loss = [0.0]; lossd = [0.0]
        branchsel_d(loss, lossd, xv[1], d[1], xv[2], d[2])
        return lossd[1]
    end
    tangent_check("branchsel", f_eval, f_tangent, x0)
end

let n = 6
    x0 = randn(n) .+ 0.5 .* sign.(randn(n))
    f_eval = function (xv)
        u = copy(xv); loss = [0.0]
        clamped_sumsq(loss, u, n); loss[1]
    end
    f_tangent = function (xv, d)
        u = copy(xv); ud = copy(d)
        loss = [0.0]; lossd = [0.0]
        clamped_sumsq_d(loss, lossd, u, ud, n)
        return lossd[1]
    end
    tangent_check("clamped_sumsq", f_eval, f_tangent, x0)
end

let n = 6
    x0 = randn(n) .+ 0.5 .* sign.(randn(n))
    f_eval = function (xv)
        u = copy(xv); loss = [0.0]; v = zeros(n)
        relu_field(loss, u, v, n); loss[1]
    end
    f_tangent = function (xv, d)
        u = copy(xv); ud = copy(d)
        loss = [0.0]; lossd = [0.0]; v = zeros(n); vd = zeros(n)
        relu_field_d(loss, lossd, u, ud, v, vd, n)
        return lossd[1]
    end
    tangent_check("relu_field", f_eval, f_tangent, x0)
end

let n = 6, i_branch = 1
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv)
        uu, vv = unpack(xv); loss = [0.0]; w = zeros(n)
        cond_field_choice(loss, uu, vv, w, i_branch, n); loss[1]
    end
    f_tangent = function (xv, d)
        uu, vv = unpack(xv)
        ud, vd = unpack(d)
        loss = [0.0]; lossd = [0.0]; w = zeros(n); wd = zeros(n)
        cond_field_choice_d(loss, lossd, uu, ud, vv, vd, w, wd, i_branch, n)
        return lossd[1]
    end
    tangent_check("cond_field_choice", f_eval, f_tangent, x0)
end

let n = 6
    x0 = vcat(randn(n), randn(1))
    unpack = xv -> (xv[1:n], xv[n+1])
    f_eval = function (xv)
        uu, c = unpack(xv); u = copy(uu); loss = [0.0]
        geomrecur(loss, u, c, n); loss[1]
    end
    f_tangent = function (xv, d)
        uu, c = unpack(xv); u = copy(uu)
        ud0, cd = unpack(d); ud = copy(ud0)
        loss = [0.0]; lossd = [0.0]
        geomrecur_d(loss, lossd, u, ud, c, cd, n)
        return lossd[1]
    end
    tangent_check("geomrecur", f_eval, f_tangent, x0)
end

let n = 6
    x0 = randn(n)
    f_eval = function (xv)
        u = copy(xv); loss = [0.0]; w = zeros(n)
        stencil_loss(loss, u, w, n); loss[1]
    end
    f_tangent = function (xv, d)
        u = copy(xv); ud = copy(d)
        loss = [0.0]; lossd = [0.0]; w = zeros(n); wd = zeros(n)
        stencil_loss_d(loss, lossd, u, ud, w, wd, n)
        return lossd[1]
    end
    tangent_check("stencil_loss", f_eval, f_tangent, x0)
end

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
    f_tangent = function (xv, d)
        u0, du0, c, dx, dt = unpack(xv)
        ud0, dud0, cd, dxd, dtd = unpack(d)
        u = copy(u0); du = copy(du0); ud = copy(ud0); dud = copy(dud0)
        func_d(u, ud, du, dud, c, cd, dx, dxd, dt, dtd, nstep, nnode)
        return sum(y_u .* ud) + sum(y_du .* dud)
    end
    tangent_check("advection (func)", f_eval, f_tangent, x0)
end

println()
println("=== M4: mg_vcycle (VJP-seeded, deepest nesting -- riskiest kernel;")
println("        first real execution of the :tripcount machinery) ===")
generate_and_eval(:mg_vcycle)

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
        mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, 0)
        return sum(y_u .* u) + sum(y_f .* f)
    end
    f_grad = function (xv)
        u0, f0, h1 = unpack(xv)
        u, f, r = build_arrays(u0, f0)
        ub = copy(y_u); fb = copy(y_f); rb = zeros(max_n, num_levels)
        # unlike the corpus's own initstacks_mg_vcycle_b(f, u), mine
        # takes no arguments -- every stack here holds popped scalars,
        # never a shape/eltype-derived whole-array copy. Splatting
        # `stacks...` (not knowing the exact count/order by hand) is
        # deliberate -- mirrors val_fixtures.jl's own pattern exactly,
        # and sidesteps needing to have hand-verified how many
        # distinct stacks this kernel's :array/:value/:branch/
        # :tripcount sites collapse into
        stacks = initstacks_mg_vcycle_b()
        h1b = mg_vcycle_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, 0.0, nu1, nu2, 0, stacks...)
        return vcat(ub[:, 1], fb[:, 1], [h1b])
    end
    report("mg_vcycle", (f_eval = f_eval, f_grad = f_grad, x0 = x0))
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
        mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, 0)
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
    tangent_check("mg_vcycle", f_eval, f_tangent, x0)
end

println()
println("check_codegen.jl: all done")