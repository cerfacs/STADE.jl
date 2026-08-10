import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_stencil_loss_1!(__jacc_i, i_n, u, w)
    i_x = 2 + (__jacc_i - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function stencil_loss_jacc(loss, u, w, i_n)
    JACC.parallel_for(div((i_n - 1) - 2, 1) + 1, jacc_kernel_stencil_loss_1!, i_n, u, w)
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end
