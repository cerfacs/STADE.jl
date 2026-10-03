# fp_side_effect(loss, u, f, a, w, i_n)
#
# NEGATIVE. The body writes a second array that the declaration does not
# mention and that is not local to the loop. That write is a side effect
# outside the relation u = G(u, p), and the implicit adjoint would drop it.
function fp_side_effect(loss, u, f, a, w, i_n)
    d = 1.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            un = 0.5 * u[i_k] + 0.25 * f[i_k] * a[i_k]
            d = d + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
            w[i_k] = w[i_k] + un
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2 + w[i_x] ^ 2
    end
    return nothing
end
