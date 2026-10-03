# fp_param_written(loss, u, f, a, i_n)
#
# NEGATIVE. The declaration names `f` as a parameter, but the body writes
# it. An output cannot be a parameter of the relation: dG/dp has no meaning
# for a quantity the relation itself produces.
function fp_param_written(loss, u, f, a, i_n)
    d = 1.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            f[i_k] = 0.9 * f[i_k]
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
