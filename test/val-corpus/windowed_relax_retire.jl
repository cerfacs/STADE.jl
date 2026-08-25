# windowed_relax_retire(u, f, w0, num_passes, dx, n)
#
# Deliberately targeted structural test for Tier B detection's if-branch
# path (agen_tier_b_walk's `:if` recursion into agen_collect_reassigned's
# if-branch handling) -- not a named textbook algorithm. Relaxes a 1-D
# array over an active window of width w, where w is retired by one
# point every OTHER pass rather than every pass: w is reassigned inside
# an `if` nested directly in the ancestor sequential i_pass loop
# (mg_vcycle/mg_vcycle_multi/cascadic_mg_prolong/richardson_substep all
# reassign their ragged bound unconditionally on every outer iteration;
# none exercise the conditional case). w is then used as the bound of
# the nested relaxation loop, so this is a Tier B instance specific to
# the if-gated reassignment case. See
# agen_tier_b_offender/agen_tier_b_walk in STADE.jl.
#
# Caller responsibility: choose w0 and num_passes so w never drops below
# 1 (e.g. w0 > div(num_passes, 2)) -- this kernel does not clamp w.
#
# u: array of length n, relaxed in place
# f: source term array of length n, caller-filled
# w0: initial active window width (number of leading points relaxed)
# num_passes: number of relaxation passes
# dx: grid spacing
# n: length of u and f
function windowed_relax_retire(u, f, w0, num_passes, dx, n)
    w = w0
    dx2 = dx * dx
    for i_pass = 1:num_passes
        # retire the outermost active point every other pass
        if mod(i_pass, 2) == 0
            w = w - 1
        end
        # relax the currently-active leading window only; points beyond
        # w are treated as already retired/converged for this pass
        for i_j = 1:w
            left = 0.0
            if i_j > 1
                left = u[i_j - 1]
            end
            right = 0.0
            if i_j < n
                right = u[i_j + 1]
            end
            u[i_j] = 0.5 * (dx2 * f[i_j] + left + right)
        end
    end
    return nothing
end
