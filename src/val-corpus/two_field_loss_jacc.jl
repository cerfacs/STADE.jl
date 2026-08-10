import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_two_field_loss_1!(__jacc_i, i_n, p, u)
    i_x = 1 + (__jacc_i - 1)
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function jacc_kernel_two_field_loss_2!(__jacc_i, i_n, q, v)
    i_x = 1 + (__jacc_i - 1)
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function two_field_loss_jacc(loss, u, v, p, q, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_two_field_loss_1!, i_n, p, u)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_two_field_loss_2!, i_n, q, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    return nothing
end
