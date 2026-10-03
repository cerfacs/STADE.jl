# fp_two_states(loss, u, v, f, a, i_n)
#
# Two coupled arrays driven to a JOINT fixed point in one loop. The
# declaration names two state arrays and one loop, which is not the same as
# fp_nested_fp: there the two states belong to two loops, one inside the
# other.
#
# tanh bounds the coupling, so the joint iteration matrix has spectral
# radius at most 0.6 whatever the parameters are. An unbounded coupling
# would let a random draw make the pair diverge.
#
# loss: length-1 output array, accumulated in place
# u, v: the coupled state, length i_n each
# f: source term, a parameter, length i_n
# a: coupling weight, a parameter, length i_n
# i_n: length
function fp_two_states(loss, u, v, f, a, i_n)
    d = 1.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            un = 0.4 * u[i_k] + 0.2 * v[i_k] + 0.25 * f[i_k]
            vn = 0.4 * v[i_k] + 0.2 * u[i_k] * tanh(a[i_k])
            d = d + (un - u[i_k]) * (un - u[i_k]) + (vn - v[i_k]) * (vn - v[i_k])
            u[i_k] = un
            v[i_k] = vn
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2 + v[i_x] ^ 2
    end
    return nothing
end
