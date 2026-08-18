import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_geomrecur_d_1!(__jacc_i, i_n, loss, lossd, u, ud)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * u[i_seq_x]) * ud[i_seq_x]
    Atomix.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_geomrecur_1!(__jacc_i, i_n, loss, u)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function geomrecur_d_jacc(loss, lossd, u, ud, c, cd, i_n)
    for i_seq_x = 2:i_n
        ud[i_seq_x] = u[i_seq_x - 1] * cd + c * ud[i_seq_x - 1]
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_geomrecur_d_1!(i_n, loss, lossd, u, ud)
    return nothing
end

function geomrecur_jacc(loss, u, c, i_n)
    for i_seq_x = 2:i_n
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_geomrecur_1!(i_n, loss, u)
    return nothing
end
