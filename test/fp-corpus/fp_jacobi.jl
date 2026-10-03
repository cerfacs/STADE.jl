# fp_jacobi(loss, u, f, a, i_n)
#
# Damped Jacobi on a 1-D Poisson problem, run to a tolerance. The minimal
# true fixed point: u converges to a solution of u = G(u, f, a), and its
# derivative depends only on that solution, not on the path taken to it.
#
# The loop is a `while` whose condition tests the CHANGE in the state, which
# is what distinguishes a fixed point from a counted loop. Tapenade's own
# tests all have this shape.
#
# The state coefficient is the constant 0.5, so the iteration contracts for
# EVERY parameter value. That is a requirement, not a detail: the validator
# draws parameters at random, and a kernel whose contraction depends on them
# would hang the suite instead of failing it.
#
# The tolerance is on a SUM OF SQUARES, so it is the SQUARE of the accuracy
# each component reaches. 1.0e-10 would converge components only to about
# 1.0e-5, which is coarser than the validator's finite-difference step, and
# the oracle then measures truncation rather than the derivative. 1.0e-24
# gives about 1.0e-12 per component, and costs about 40 iterations at a
# contraction of 0.5.
#
# loss: length-1 output array, accumulated in place
# u: the state, driven to a fixed point, length i_n
# f: source term, a parameter, length i_n
# a: diagonal weight, a parameter, length i_n
# i_n: length
function fp_jacobi(loss, u, f, a, i_n)
    d = 1.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            un = 0.5 * u[i_k] + 0.25 * f[i_k] * a[i_k]
            d = d + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
