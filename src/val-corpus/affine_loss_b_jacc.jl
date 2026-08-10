import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_affine_loss_b_1!(__jacc_i, a, b, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function jacc_kernel_affine_loss_b_2!(__jacc_i, a, ab, bb, i_n, u, ub, vb)
    i_x = 1 + (__jacc_i - 1)
    ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
    ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
    bb[i_x] = bb[i_x] + vb[i_x]
    vb[i_x] = 0.0
    return nothing
end

function jacc_kernel_affine_loss_1!(__jacc_i, a, b, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function initstacks_affine_loss_b_jacc()
    return nothing
end

function affine_loss_b_jacc(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_affine_loss_b_1!, a, b, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    end
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_affine_loss_b_2!, a, ab, bb, i_n, u, ub, vb)
    return nothing
end

function affine_loss_jacc(loss, u, a, b, v, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_affine_loss_1!, a, b, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    return nothing
end
