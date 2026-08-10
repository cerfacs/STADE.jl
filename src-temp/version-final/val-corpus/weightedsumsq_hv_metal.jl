import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function initstacks_weightedsumsq_b_metal()
    return nothing
end

function weightedsumsq_hv_metal(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (u[i_seq_x] ^ 2 * wd[i_seq_x] + w[i_seq_x] * ((2 * u[i_seq_x]) * ud[i_seq_x]))
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * ((2 * u[i_seq_x]) * ud[i_seq_x]) + u[i_seq_x] ^ 2 * lossbd[1])
        wb[i_seq_x] = wb[i_seq_x] + u[i_seq_x] ^ 2 * lossb[1]
        ubd[i_seq_x] = ubd[i_seq_x] + ((w[i_seq_x] * lossb[1]) * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * (lossb[1] * wd[i_seq_x] + w[i_seq_x] * lossbd[1]))
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * (w[i_seq_x] * lossb[1])
    end
    return nothing
end

function weightedsumsq_metal(loss, u, w, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
    return nothing
end
