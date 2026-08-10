import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_affine_loss_d_1!(__jacc_i, a, ad, b, bd, i_n, u, ud, v, vd)
    i_x = 1 + (__jacc_i - 1)
    vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function jacc_kernel_affine_loss_1!(__jacc_i, a, b, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function affine_loss_d_jacc(loss, lossd, u, ud, a, ad, b, bd, v, vd, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_affine_loss_d_1!, a, ad, b, bd, i_n, u, ud, v, vd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * v[i_seq_x]) * vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    return nothing
end

function affine_loss_jacc(loss, u, a, b, v, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_affine_loss_1!, a, b, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    return nothing
end
