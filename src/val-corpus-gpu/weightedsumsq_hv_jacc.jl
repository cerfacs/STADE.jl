import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_weightedsumsq_hv_1!(__jacc_i, i_n, loss, lossd, u, ud, w, wd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += u[i_seq_x] ^ 2 * wd[i_seq_x] + w[i_seq_x] * ((2 * u[i_seq_x]) * ud[i_seq_x])
    Atomix.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_weightedsumsq_hv_2!(__jacc_i, i_n, lossb, lossbd, u, ub, ubd, ud, w, wb, wbd, wd)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * ((2 * u[i_seq_x]) * ud[i_seq_x]) + u[i_seq_x] ^ 2 * lossbd[1])
    wb[i_seq_x] = wb[i_seq_x] + u[i_seq_x] ^ 2 * lossb[1]
    ubd[i_seq_x] = ubd[i_seq_x] + ((w[i_seq_x] * lossb[1]) * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * (lossb[1] * wd[i_seq_x] + w[i_seq_x] * lossbd[1]))
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * (w[i_seq_x] * lossb[1])
    return nothing
end

function jacc_kernel_weightedsumsq_1!(__jacc_i, i_n, loss, u, w)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function initstacks_weightedsumsq_b_jacc()
    return nothing
end

function weightedsumsq_hv_jacc(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_weightedsumsq_hv_1!(i_n, loss, lossd, u, ud, w, wd)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_weightedsumsq_hv_2!(i_n, lossb, lossbd, u, ub, ubd, ud, w, wb, wbd, wd)
    return nothing
end

function weightedsumsq_jacc(loss, u, w, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_weightedsumsq_1!(i_n, loss, u, w)
    return nothing
end
