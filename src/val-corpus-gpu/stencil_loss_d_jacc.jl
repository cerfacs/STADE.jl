import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_stencil_loss_d_1!(__jacc_i, i_n, u, ud, w, wd)
    i_x = 2 + (__jacc_i - 1)
    wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function jacc_kernel_stencil_loss_d_2!(__jacc_i, i_n, loss, lossd, w, wd)
    i_seq_x = 2 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * w[i_seq_x]) * wd[i_seq_x]
    Atomix.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_stencil_loss_1!(__jacc_i, i_n, u, w)
    i_x = 2 + (__jacc_i - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function jacc_kernel_stencil_loss_2!(__jacc_i, i_n, loss, w)
    i_seq_x = 2 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function stencil_loss_d_jacc(loss, lossd, u, ud, w, wd, i_n)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_d_1!(i_n, u, ud, w, wd)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_d_2!(i_n, loss, lossd, w, wd)
    return nothing
end

function stencil_loss_jacc(loss, u, w, i_n)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_1!(i_n, u, w)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_2!(i_n, loss, w)
    return nothing
end
