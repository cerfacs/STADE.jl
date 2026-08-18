import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_dotprod_d_1!(__jacc_i, i_n, loss, lossd, u, ud, v, vd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += v[i_seq_x] * ud[i_seq_x] + u[i_seq_x] * vd[i_seq_x]
    Atomix.@atomic loss[1] += u[i_seq_x] * v[i_seq_x]
    return nothing
end

function jacc_kernel_dotprod_1!(__jacc_i, i_n, loss, u, v)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += u[i_seq_x] * v[i_seq_x]
    return nothing
end

function dotprod_d_jacc(loss, lossd, u, ud, v, vd, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_dotprod_d_1!(i_n, loss, lossd, u, ud, v, vd)
    return nothing
end

function dotprod_jacc(loss, u, v, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_dotprod_1!(i_n, loss, u, v)
    return nothing
end
