function liveout_red_top(x, w, loss, i_nc)
    a = 0.0
    for ci = 1:i_nc
        xv2 = x[ci]
        a = a + xv2 * xv2 * w[ci]
    end
    loss[1] = loss[1] + a * a
    return nothing
end
