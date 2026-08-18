import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_stencil_loss_b_1!(__jacc_i, i_n, u, w)
    i_x = 2 + (__jacc_i - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function jacc_kernel_stencil_loss_b_2!(__jacc_i, i_n, loss, w)
    i_seq_x = 2 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_stencil_loss_b_3!(__jacc_i, i_n, lossb, w, wb)
    i_seq_x = (i_n - 1) + (__jacc_i - 1) * -1
    wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    return nothing
end

function jacc_kernel_stencil_loss_b_4!(__jacc_i, i_n, ub, wb)
    i_x = 2 + (__jacc_i - 1)
    ub[i_x - 1] = ub[i_x - 1] + wb[i_x]
    ub[i_x] = ub[i_x] + 2.0 * -(wb[i_x])
    ub[i_x + 1] = ub[i_x + 1] + wb[i_x]
    wb[i_x] = 0.0
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

function initstacks_stencil_loss_b_jacc()
    return nothing
end

function stencil_loss_b_jacc(loss, lossb, u, ub, w, wb, i_n)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_b_1!(i_n, u, w)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_b_2!(i_n, loss, w)
    JACC.@parallel_for range = div(2 - (i_n - 1), -1) + 1 jacc_kernel_stencil_loss_b_3!(i_n, lossb, w, wb)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_b_4!(i_n, ub, wb)
    return nothing
end

function stencil_loss_jacc(loss, u, w, i_n)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_1!(i_n, u, w)
    JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_2!(i_n, loss, w)
    return nothing
end
