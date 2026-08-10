import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_stencil_loss_hv_1!(__jacc_i, i_n, u, ud, w, wd)
    i_x = 2 + (__jacc_i - 1)
    wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function jacc_kernel_stencil_loss_hv_2!(__jacc_i, i_n, ub, ubd, wb, wbd)
    i_x = 2 + (__jacc_i - 1)
    ubd[i_x - 1] = ubd[i_x - 1] + wbd[i_x]
    ub[i_x - 1] = ub[i_x - 1] + wb[i_x]
    ubd[i_x] = ubd[i_x] + 2.0 * -(wbd[i_x])
    ub[i_x] = ub[i_x] + 2.0 * -(wb[i_x])
    ubd[i_x + 1] = ubd[i_x + 1] + wbd[i_x]
    ub[i_x + 1] = ub[i_x + 1] + wb[i_x]
    wbd[i_x] = 0.0
    wb[i_x] = 0.0
    return nothing
end

function jacc_kernel_stencil_loss_1!(__jacc_i, i_n, u, w)
    i_x = 2 + (__jacc_i - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function initstacks_stencil_loss_b_jacc()
    return nothing
end

function stencil_loss_hv_jacc(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd)
    JACC.parallel_for(div((i_n - 1) - 2, 1) + 1, jacc_kernel_stencil_loss_hv_1!, i_n, u, ud, w, wd)
    for i_seq_x = 2:i_n - 1
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    for i_seq_x = i_n - 1:-1:2
        wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * (2 * wd[i_seq_x]) + (2 * w[i_seq_x]) * lossbd[1])
        wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    end
    JACC.parallel_for(div((i_n - 1) - 2, 1) + 1, jacc_kernel_stencil_loss_hv_2!, i_n, ub, ubd, wb, wbd)
    return nothing
end

function stencil_loss_jacc(loss, u, w, i_n)
    JACC.parallel_for(div((i_n - 1) - 2, 1) + 1, jacc_kernel_stencil_loss_1!, i_n, u, w)
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end
