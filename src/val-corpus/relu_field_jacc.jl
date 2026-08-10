import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_relu_field_1!(__jacc_i, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    if u[i_x] > 0.0
        v[i_x] = u[i_x] ^ 2
    else
        v[i_x] = 0.0
    end
    return nothing
end

function relu_field_jacc(loss, u, v, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_relu_field_1!, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x]
    end
    return nothing
end
