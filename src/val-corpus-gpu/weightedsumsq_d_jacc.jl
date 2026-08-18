import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_weightedsumsq_d_1!(__jacc_i, i_n, loss, lossd, u, ud, w, wd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += u[i_seq_x] ^ 2 * wd[i_seq_x] + w[i_seq_x] * ((2 * u[i_seq_x]) * ud[i_seq_x])
    Atomix.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_weightedsumsq_1!(__jacc_i, i_n, loss, u, w)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function weightedsumsq_d_jacc(loss, lossd, u, ud, w, wd, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_weightedsumsq_d_1!(i_n, loss, lossd, u, ud, w, wd)
    return nothing
end

function weightedsumsq_jacc(loss, u, w, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_weightedsumsq_1!(i_n, loss, u, w)
    return nothing
end
