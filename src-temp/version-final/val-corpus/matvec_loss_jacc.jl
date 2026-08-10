import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_matvec_loss_1!(__jacc_i, a, i_m, i_n, u, v)
    i_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function matvec_loss_jacc(loss, a, u, v, i_m, i_n)
    JACC.parallel_for(div(i_m - 1, 1) + 1, jacc_kernel_matvec_loss_1!, a, i_m, i_n, u, v)
    for i_seq_i = 1:i_m
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
    return nothing
end
