import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_clamped_sumsq_d_1!(__jacc_i, i_n, loss, lossd, u, ud)
    i_seq_x = 1 + (__jacc_i - 1)
    if u[i_seq_x] > 0.0
        wd = (2 * u[i_seq_x]) * ud[i_seq_x]
        w = u[i_seq_x] ^ 2
    else
        wd = 0.0
        w = 0.0
    end
    Atomix.@atomic lossd[1] += wd
    Atomix.@atomic loss[1] += w
    return nothing
end

function jacc_kernel_clamped_sumsq_1!(__jacc_i, i_n, loss, u)
    i_seq_x = 1 + (__jacc_i - 1)
    if u[i_seq_x] > 0.0
        w = u[i_seq_x] ^ 2
    else
        w = 0.0
    end
    Atomix.@atomic loss[1] += w
    return nothing
end

function clamped_sumsq_d_jacc(loss, lossd, u, ud, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_clamped_sumsq_d_1!(i_n, loss, lossd, u, ud)
    return nothing
end

function clamped_sumsq_jacc(loss, u, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_clamped_sumsq_1!(i_n, loss, u)
    return nothing
end
