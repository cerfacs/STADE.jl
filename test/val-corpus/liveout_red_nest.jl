function liveout_red_nest(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[t + ci - 1]
            a = a + xv2 * xv2 * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
