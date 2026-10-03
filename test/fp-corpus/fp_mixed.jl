# fp_mixed(loss, u, f, a, i_n)
#
# One `while` that steps to a time budget and one that solves to
# convergence, BOTH carrying the same state `u`.
#
# This is the subject that decides the selection rule of section 3.3. The
# first loop stops on `t`, a schedule that is never computed from `u`, so it
# is an ordinary loop and must be taped. The second stops on `d`, which is
# computed from the change in `u`, so it is a fixed point.
#
# Naming the state would select both, which is why the declaration is not
# keyed on the state array. The structural refusals separate neither: the
# budget loop reads and writes `u` exactly as the solver does, and its
# carried `t` is inactive, so nothing trips. The dependence of the CONDITION
# is the only signal that tells them apart.
#
# loss: length-1 output array, accumulated in place
# u: the state of both loops, length i_n
# f: source term, a parameter, length i_n
# a: diagonal weight, a parameter, length i_n
# i_n: length
function fp_mixed(loss, u, f, a, i_n)
    t = 0.0
    while t < 1.0
        for i_j = 1:i_n
            u[i_j] = u[i_j] + 0.1 * f[i_j] * a[i_j]
        end
        t = t + 0.25
    end
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
