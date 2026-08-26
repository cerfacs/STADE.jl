function bnd_branch_d(x, xd, flag, flagd, i_n, i_m, out, outd)
    sd = 0.0
    s = 0.0
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            sd = 0.0
            s = 0.0
            for i_j = 1:i_m
                sd = sd + (x[i_i] * xd[i_j] + x[i_j] * xd[i_i])
                s = s + x[i_j] * x[i_i]
            end
            outd[i_i] = outd[i_i] + (s * sd + s * sd)
            out[i_i] = out[i_i] + s * s
        end
    end
    return nothing
end

function bnd_branch(x, flag, i_n, i_m, out)
    s = 0.0
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            s = 0.0
            for i_j = 1:i_m
                s = s + x[i_j] * x[i_i]
            end
            out[i_i] = out[i_i] + s * s
        end
    end
end
