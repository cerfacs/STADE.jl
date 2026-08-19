import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_pipeline_d_1!(__jacc_i, i_n, u, ud, v, vd)
    i_x = 1 + (__jacc_i - 1)
    vd[i_x] = (2 * u[i_x]) * ud[i_x]
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function jacc_kernel_pipeline_d_2!(__jacc_i, i_n, u, ud, v, vd, w, wd)
    i_x = 1 + (__jacc_i - 1)
    wd[i_x] = u[i_x] * vd[i_x] + v[i_x] * ud[i_x]
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function jacc_kernel_pipeline_d_3!(__jacc_i, i_n, loss, lossd, w, wd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += wd[i_seq_x]
    Atomix.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function jacc_kernel_pipeline_1!(__jacc_i, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function jacc_kernel_pipeline_2!(__jacc_i, i_n, u, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function pipeline_d_jacc(loss, lossd, u, ud, v, vd, w, wd, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_d_1!(i_n, u, ud, v, vd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_d_2!(i_n, u, ud, v, vd, w, wd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_d_3!(i_n, loss, lossd, w, wd)
    return nothing
end

function pipeline_jacc(loss, u, v, w, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_1!(i_n, u, v)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_2!(i_n, u, v, w)
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, w)->w[i_seq_x]))(w))
    return nothing
end
