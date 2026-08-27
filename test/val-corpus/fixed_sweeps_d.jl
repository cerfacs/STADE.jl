function fixed_sweeps_d(y, yd, x, xd, i_n)
    for i_k = 1:3
        for i_i = 1:i_n
            yd[i_i] = 0.5 * yd[i_i] + xd[i_i]
            y[i_i] = 0.5 * y[i_i] + x[i_i]
        end
    end
    return nothing
end

function fixed_sweeps(y, x, i_n)
    for i_k = 1:3
        for i_i = 1:i_n
            y[i_i] = 0.5 * y[i_i] + x[i_i]
        end
    end
end
