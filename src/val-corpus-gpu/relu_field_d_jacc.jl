import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_relu_field_d_1!(__jacc_i, i_n, u, ud, v, vd)
    i_x = 1 + (__jacc_i - 1)
    if u[i_x] > 0.0
        vd[i_x] = (2 * u[i_x]) * ud[i_x]
        v[i_x] = u[i_x] ^ 2
    else
        vd[i_x] = 0.0
        v[i_x] = 0.0
    end
    return nothing
end

function jacc_kernel_relu_field_d_2!(__jacc_i, i_n, loss, lossd, v, vd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += vd[i_seq_x]
    Atomix.@atomic loss[1] += v[i_seq_x]
    return nothing
end

function jacc_kernel_relu_field_1!(__jacc_i, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    if u[i_x] > 0.0
        v[i_x] = u[i_x] ^ 2
    else
        v[i_x] = 0.0
    end
    return nothing
end

function jacc_kernel_relu_field_2!(__jacc_i, i_n, loss, v)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += v[i_seq_x]
    return nothing
end

function relu_field_d_jacc(loss, lossd, u, ud, v, vd, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_d_1!(i_n, u, ud, v, vd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_d_2!(i_n, loss, lossd, v, vd)
    return nothing
end

function relu_field_jacc(loss, u, v, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_1!(i_n, u, v)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_2!(i_n, loss, v)
    return nothing
end
