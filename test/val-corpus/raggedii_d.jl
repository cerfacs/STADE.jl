function raggedii_d(out, outd, u, ud, x, xd, n, m0)
    for i_x = 1:n
        sd = 0.0
        s = 0.0
        for i_y = 1:2
            wd = 0.0
            w = m0 + i_y
            for i_j = 1:w
                sd = sd + (u[i_x] * xd[i_j] + x[i_j] * ud[i_x])
                s = s + x[i_j] * u[i_x]
            end
        end
        outd[i_x] = s * sd + s * sd
        out[i_x] = s * s
    end
    return nothing
end

function raggedii(out, u, x, n, m0)
    for i_x = 1:n
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            for i_j = 1:w
                s = s + x[i_j] * u[i_x]
            end
        end
        out[i_x] = s * s
    end
end
