# red_two_locals(loss, u, v, i_n)
#
# Two top-level reductions into two different local scalars, one after
# the other, both lowering. A single accumulator would not show that the
# pending-statement machinery keeps them apart, nor that the second reads
# the first's final value.
#
# Only the first reduction lowers. CSE rewrites the second body as
# `__cse_0 = v[i_y]; q = q + __cse_0 * __cse_0`, two statements, and
# cgen_idiomatic_scalar_reduction requires exactly one. A missed
# optimization rather than a defect: the loop stays on the host and is
# still correct. Inlining a single-use CSE temporary before matching
# would recover it.
#
# loss: length-1 output array, accumulated in place
# u, v: vectors of length i_n
# i_n: length
function red_two_locals(loss, u, v, i_n)
    p = 0.0
    for i_x = 1:i_n
        p = p + u[i_x] * v[i_x]
    end
    q = 0.0
    for i_y = 1:i_n
        q = q + v[i_y] * v[i_y]
    end
    loss[1] = loss[1] + p * q
    return nothing
end
