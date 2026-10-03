# fp_not_read(loss, u, f, a, i_n)
#
# NEGATIVE. The loop writes the declared state without reading it, so
# nothing iterates towards anything. It is an assignment repeated until an
# unrelated condition is met, not a fixed point.
function fp_not_read(loss, u, f, a, i_n)
    d = 1.0
    while d > 1.0e-24
        d = 0.5 * d
        for i_k = 1:i_n
            u[i_k] = f[i_k] * a[i_k]
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
