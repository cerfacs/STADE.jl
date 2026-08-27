function entry_empty_d(x, xd, u, ud, i_npass, i_w0, out, outd)
    sd = x[1] * xd[1] + x[1] * xd[1]
    s = x[1] * x[1]
    wd = 0.0
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            sd = u[i_j] * xd[i_j] + x[i_j] * ud[i_j]
            s = x[i_j] * u[i_j]
        end
        outd[i_p] = outd[i_p] + (s * sd + s * sd)
        out[i_p] = out[i_p] + s * s
        wd = 0.0
        w = w - 3
    end
    return nothing
end

function entry_empty(x, u, i_npass, i_w0, out)
    s = x[1] * x[1]
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            s = x[i_j] * u[i_j]
        end
        out[i_p] = out[i_p] + s * s
        w = w - 3
    end
end
