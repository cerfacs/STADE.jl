function clamped_sumsq_d(loss, lossd, u, ud, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            wd = (2 * u[i_x]) * ud[i_x]
            w = u[i_x] ^ 2
        else
            wd = 0.0
            w = 0.0
        end
        lossd[1] = lossd[1] + wd
        loss[1] = loss[1] + w
    end
    return nothing
end

function clamped_sumsq(loss, u, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            w = u[i_x] ^ 2
        else
            w = 0.0
        end
        loss[1] = loss[1] + w
    end
end
