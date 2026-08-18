import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_normcomp_d_1!(__jacc_i, i_n, u, ud, v, vd, w, wd)
    i_x = 1 + (__jacc_i - 1)
    wd[i_x] = ud[i_x] + -(vd[i_x])
    w[i_x] = u[i_x] - v[i_x]
    return nothing
end

function jacc_kernel_normcomp_d_2!(__jacc_i, i_n, loss, lossd, w, wd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * w[i_seq_x]) * wd[i_seq_x]
    Atomix.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_normcomp_1!(__jacc_i, i_n, u, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = u[i_x] - v[i_x]
    return nothing
end

function jacc_kernel_normcomp_2!(__jacc_i, i_n, loss, w)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function normcomp_d_jacc(loss, lossd, u, ud, v, vd, w, wd, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_normcomp_d_1!(i_n, u, ud, v, vd, w, wd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_normcomp_d_2!(i_n, loss, lossd, w, wd)
    return nothing
end

function normcomp_jacc(loss, u, v, w, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_normcomp_1!(i_n, u, v, w)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_normcomp_2!(i_n, loss, w)
    return nothing
end
