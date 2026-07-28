# ============================================================
# val_fixtures.jl -- finite-difference oracle fixtures for the
# full M1-M4 corpus. Requires STADE.jl (val_finite_diff_check,
# val_check_fixture) and all_b.jl (the corpus functions) loaded
# first.
# ============================================================


# ==================== M1: straight-line ============================

function val_fixture_dotprod(n::Int)
    x0 = vcat(randn(n), randn(n))
    f_eval = function (xv::Vector{Float64})
        u = xv[1:n]; v = xv[n+1:2n]
        loss = [0.0]
        dotprod(loss, u, v, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        u = xv[1:n]; v = xv[n+1:2n]
        ub = zeros(n); vb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        dotprod_b(loss, lossb, u, ub, v, vb, n)
        return vcat(ub, vb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_quadloss()
    x0 = randn(3)
    f_eval = function (xv::Vector{Float64})
        loss = [0.0]
        quadloss(loss, xv[1], xv[2], xv[3])
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        loss = [0.0]; lossb = [1.0]
        xb, yb, zb = quadloss_b(loss, lossb, xv[1], 0.0, xv[2], 0.0, xv[3], 0.0)
        return [xb, yb, zb]
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_bilinear(m::Int, n::Int)
    x0 = vcat(randn(m), vec(randn(m, n)), randn(n))
    unpack = xv -> (xv[1:m], reshape(xv[m+1:m+m*n], m, n), xv[m+m*n+1:m+m*n+n])
    f_eval = function (xv::Vector{Float64})
        xx, aa, yy = unpack(xv)
        loss = [0.0]
        bilinear(loss, xx, aa, yy, m, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        xx, aa, yy = unpack(xv)
        xb = zeros(m); ab = zeros(m, n); yb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        bilinear_b(loss, lossb, xx, xb, aa, ab, yy, yb, m, n)
        return vcat(xb, vec(ab), yb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_matvec_loss(m::Int, n::Int)
    x0 = vcat(vec(randn(m, n)), randn(n), randn(m))
    unpack = xv -> (reshape(xv[1:m*n], m, n), xv[m*n+1:m*n+n], xv[m*n+n+1:m*n+n+m])
    f_eval = function (xv::Vector{Float64})
        aa, uu, v0 = unpack(xv)
        loss = [0.0]; v = copy(v0)
        matvec_loss(loss, aa, uu, v, m, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        aa, uu, v0 = unpack(xv)
        ab = zeros(m, n); ub = zeros(n); vb = zeros(m)
        v = copy(v0)
        loss = [0.0]; lossb = [1.0]
        matvec_loss_b(loss, lossb, aa, ab, uu, ub, v, vb, m, n)
        return vcat(vec(ab), ub, vb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_affine_loss(n::Int)
    x0 = vcat(randn(n), randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n], xv[2n+1:3n])
    f_eval = function (xv::Vector{Float64})
        uu, aa, bb_ = unpack(xv)
        loss = [0.0]; v = zeros(n)
        affine_loss(loss, uu, aa, bb_, v, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        uu, aa, bb_ = unpack(xv)
        ub = zeros(n); ab = zeros(n); bbb = zeros(n)
        v = zeros(n); vb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        affine_loss_b(loss, lossb, uu, ub, aa, ab, bb_, bbb, v, vb, n)
        return vcat(ub, ab, bbb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_normcomp(n::Int)
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv::Vector{Float64})
        uu, vv = unpack(xv)
        loss = [0.0]; w = zeros(n)
        normcomp(loss, uu, vv, w, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        uu, vv = unpack(xv)
        ub = zeros(n); vb = zeros(n)
        w = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        normcomp_b(loss, lossb, uu, ub, vv, vb, w, wb, n)
        return vcat(ub, vb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_weightedsumsq(n::Int)
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv::Vector{Float64})
        uu, ww = unpack(xv)
        loss = [0.0]
        weightedsumsq(loss, uu, ww, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        uu, ww = unpack(xv)
        ub = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        weightedsumsq_b(loss, lossb, uu, ub, ww, wb, n)
        return vcat(ub, wb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_sumsq_shifted(n::Int)
    x0 = vcat(randn(n), randn(2))
    unpack = xv -> (xv[1:n], xv[n+1], xv[n+2])
    f_eval = function (xv::Vector{Float64})
        uu, alpha, beta = unpack(xv)
        loss = [0.0]
        sumsq_shifted(loss, uu, alpha, beta, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        uu, alpha, beta = unpack(xv)
        ub = zeros(n)
        loss = [0.0]; lossb = [1.0]
        alphab, betab = sumsq_shifted_b(loss, lossb, uu, ub, alpha, 0.0, beta, 0.0, n)
        return vcat(ub, [alphab, betab])
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_two_field_loss(n::Int)
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv::Vector{Float64})
        uu, vv = unpack(xv)
        loss = [0.0]; p = zeros(n); q = zeros(n)
        two_field_loss(loss, uu, vv, p, q, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        uu, vv = unpack(xv)
        ub = zeros(n); vb = zeros(n)
        p = zeros(n); pb = zeros(n); q = zeros(n); qb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        two_field_loss_b(loss, lossb, uu, ub, vv, vb, p, pb, q, qb, n)
        return vcat(ub, vb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_pipeline(n::Int)
    x0 = randn(n)
    f_eval = function (xv::Vector{Float64})
        loss = [0.0]; v = zeros(n); w = zeros(n)
        pipeline(loss, xv, v, w, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        ub = zeros(n)
        v = zeros(n); vb = zeros(n)
        w = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        pipeline_b(loss, lossb, xv, ub, v, vb, w, wb, n)
        return ub
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end


# ==================== M2: conditionals ==============================

function val_fixture_branchsel()
    x0 = randn(2)   # kink at x==y is measure-zero; unguarded on purpose
    f_eval = function (xv::Vector{Float64})
        loss = [0.0]
        branchsel(loss, xv[1], xv[2])
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        loss = [0.0]; lossb = [1.0]
        xb, yb = branchsel_b(loss, lossb, xv[1], 0.0, xv[2], 0.0)
        return [xb, yb]
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_clamped_sumsq(n::Int)
    x0 = randn(n) .+ 0.5 .* sign.(randn(n))   # nudge off the u=0 kink
    f_eval = function (xv::Vector{Float64})
        u = copy(xv)
        loss = [0.0]
        clamped_sumsq(loss, u, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        u = copy(xv)
        ub = zeros(n)
        loss = [0.0]; lossb = [1.0]
        branch_stack = initstacks_clamped_sumsq_b()
        clamped_sumsq_b(loss, lossb, u, ub, n, branch_stack)
        return ub
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

# i_branch is structural (fixed at fixture-construction time), not
# part of x0 -- it's an Int64 selector, never differentiable.
function val_fixture_cond_field_choice(n::Int, i_branch::Int)
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv::Vector{Float64})
        uu, vv = unpack(xv)
        loss = [0.0]; w = zeros(n)
        cond_field_choice(loss, uu, vv, w, i_branch, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        uu, vv = unpack(xv)
        ub = zeros(n); vb = zeros(n)
        w = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        branch_stack = initstacks_cond_field_choice_b()
        cond_field_choice_b(loss, lossb, uu, ub, vv, vb, w, wb, i_branch, n, branch_stack)
        return vcat(ub, vb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_cond_loop_choice(n::Int, i_branch::Int)
    x0 = vcat(randn(n), randn(n))
    unpack = xv -> (xv[1:n], xv[n+1:2n])
    f_eval = function (xv::Vector{Float64})
        uu, vv = unpack(xv)
        loss = [0.0]
        cond_loop_choice(loss, uu, vv, i_branch, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        uu, vv = unpack(xv)
        ub = zeros(n); vb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        cond_loop_choice_b(loss, lossb, uu, ub, vv, vb, i_branch, n)
        return vcat(ub, vb)
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_relu_field(n::Int)
    x0 = randn(n) .+ 0.5 .* sign.(randn(n))   # nudge off the u=0 kink
    f_eval = function (xv::Vector{Float64})
        u = copy(xv)
        loss = [0.0]; v = zeros(n)
        relu_field(loss, u, v, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        u = copy(xv)
        ub = zeros(n)
        v = zeros(n); vb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        relu_field_b(loss, lossb, u, ub, v, vb, n)
        return ub
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end


# ==================== M3: loop-carried recurrences ==================

function val_fixture_geomrecur(n::Int)
    x0 = vcat(randn(n), randn(1))   # [u0...; c] -- u0[2:n] are dead
    # (overwritten before read) but included anyway: the adjoint
    # should independently show ~zero sensitivity there too.
    unpack = xv -> (xv[1:n], xv[n+1])
    f_eval = function (xv::Vector{Float64})
        uu, c = unpack(xv)
        u = copy(uu)
        loss = [0.0]
        geomrecur(loss, u, c, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        uu, c = unpack(xv)
        u = copy(uu)
        ub = zeros(n)
        loss = [0.0]; lossb = [1.0]
        u_stack = initstacks_geomrecur_b(u)
        cb = geomrecur_b(loss, lossb, u, ub, c, 0.0, n, u_stack)
        return vcat(ub, [cb])
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

function val_fixture_stencil_loss(n::Int)
    x0 = randn(n)   # n >= 3 required for a non-empty interior range
    f_eval = function (xv::Vector{Float64})
        u = copy(xv)
        loss = [0.0]; w = zeros(n)
        stencil_loss(loss, u, w, n)
        return loss[1]
    end
    f_grad = function (xv::Vector{Float64})
        u = copy(xv)
        ub = zeros(n)
        w = zeros(n); wb = zeros(n)
        loss = [0.0]; lossb = [1.0]
        stencil_loss_b(loss, lossb, u, ub, w, wb, n)
        return ub
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end

# No scalar loss -- func_b internally re-runs the forward pass to
# rebuild its stack, so f_grad must hand it fresh pre-forward copies
# of u0/du0, exactly like f_eval does. Output is VJP-seeded: y_u/y_du
# fixed once per fixture, not resampled per trial.
function val_fixture_advection(nnode::Int, nstep::Int)
    y_u = randn(nnode); y_du = randn(nnode)
    x0 = vcat(randn(nnode), randn(nnode),
              [1.0 + 0.1randn(), 0.8 + 0.1randn(), 0.05 + 0.01randn()])
    unpack = function (xv::Vector{Float64})
        return xv[1:nnode], xv[nnode+1:2nnode], xv[2nnode+1], xv[2nnode+2], xv[2nnode+3]
    end
    f_eval = function (xv::Vector{Float64})
        u0, du0, c, dx, dt = unpack(xv)
        u = copy(u0); du = copy(du0)
        func(u, du, c, dx, dt, nstep, nnode)
        return sum(y_u .* u) + sum(y_du .* du)
    end
    f_grad = function (xv::Vector{Float64})
        u0, du0, c, dx, dt = unpack(xv)
        u = copy(u0); du = copy(du0)
        ub = copy(y_u); dub = copy(y_du)
        du_stack = initstacks_func_b(du)
        cb, dxb, dtb = func_b(u, ub, du, dub, c, 0.0, dx, 0.0, dt, 0.0, nstep, nnode, du_stack)
        return vcat(ub, dub, [cb, dxb, dtb])
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end


# ==================== M4: full complexity ============================

# Live independents (traced from the primal): u at level 1 only
# (coarse-level u is zero-initialized by the algorithm itself,
# so u0 there is dead) and f at level 1 only (f at every other
# level is fully overwritten by restriction before being read,
# including the coarsest level -- traced the last coarsening
# iteration writes exactly the single coarsest-level entry the
# direct solve reads). r is dead everywhere: the primal only ever
# writes r, never reads the caller-supplied r0. All excluded
# entries are fixed at 0.0 rather than perturbed. Outputs are
# VJP-seeded on u and f (full arrays, all levels); r is seeded 0.
function val_fixture_mg_vcycle(; num_levels::Int = 2, nfine::Int = 5,
                                nu1::Int = 2, nu2::Int = 2)
    max_n = nfine - 1
    y_u = randn(max_n, num_levels)
    y_f = randn(max_n, num_levels)
    x0 = vcat(randn(max_n), randn(max_n), [1.0 + 0.1randn()])
    unpack = xv -> (xv[1:max_n], xv[max_n+1:2max_n], xv[2max_n+1])
    build_arrays = function (u0, f0)
        u = zeros(max_n, num_levels); u[:, 1] = u0
        f = zeros(max_n, num_levels); f[:, 1] = f0
        r = zeros(max_n, num_levels)
        return u, f, r
    end
    f_eval = function (xv::Vector{Float64})
        u0, f0, h1 = unpack(xv)
        u, f, r = build_arrays(u0, f0)
        mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, 0)
        return sum(y_u .* u) + sum(y_f .* f)
    end
    f_grad = function (xv::Vector{Float64})
        u0, f0, h1 = unpack(xv)
        u, f, r = build_arrays(u0, f0)
        ub = copy(y_u); fb = copy(y_f); rb = zeros(max_n, num_levels)
        stacks = initstacks_mg_vcycle_b(f, u)
        h1b = mg_vcycle_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, 0.0,
                           nu1, nu2, 0, stacks...)
        return vcat(ub[:, 1], fb[:, 1], [h1b])
    end
    return (f_eval = f_eval, f_grad = f_grad, x0 = x0)
end


# ==================== runner: all tiers ==============================

function val_run_all_tiers()
    m1 = [
        ("dotprod",        val_fixture_dotprod(5)),
        ("quadloss",       val_fixture_quadloss()),
        ("bilinear",       val_fixture_bilinear(4, 3)),
        ("matvec_loss",    val_fixture_matvec_loss(4, 3)),
        ("affine_loss",    val_fixture_affine_loss(5)),
        ("normcomp",       val_fixture_normcomp(5)),
        ("weightedsumsq",  val_fixture_weightedsumsq(5)),
        ("sumsq_shifted",  val_fixture_sumsq_shifted(5)),
        ("two_field_loss", val_fixture_two_field_loss(5)),
        ("pipeline",       val_fixture_pipeline(5)),
    ]
    m2 = [
        ("branchsel",              val_fixture_branchsel()),
        ("clamped_sumsq",          val_fixture_clamped_sumsq(6)),
        ("cond_field_choice(br1)", val_fixture_cond_field_choice(6, 1)),
        ("cond_field_choice(br0)", val_fixture_cond_field_choice(6, 0)),
        ("cond_loop_choice(br1)",  val_fixture_cond_loop_choice(6, 1)),
        ("cond_loop_choice(br0)",  val_fixture_cond_loop_choice(6, 0)),
        ("relu_field",             val_fixture_relu_field(6)),
    ]
    m3 = [
        ("geomrecur",     val_fixture_geomrecur(6)),
        ("stencil_loss",  val_fixture_stencil_loss(6)),
        ("advection",     val_fixture_advection(5, 2)),
    ]
    m4 = [
        ("mg_vcycle", val_fixture_mg_vcycle()),
    ]

    for (tier_name, fixtures) in [("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4)]
        println("--- ", tier_name, " ---")
        worst = 0.0
        for (name, fx) in fixtures
            r = val_check_fixture(fx)
            status = r.ok ? "ok  " : "FAIL"
            println(rpad(name, 26), status, "  max_rel_err=", round(r.max_rel_err, sigdigits = 3))
            worst = max(worst, r.max_rel_err)
        end
        println(tier_name, " worst max_rel_err=", round(worst, sigdigits = 3))
    end
end