import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_affine_loss_hv_1!(__jacc_i, a, ad, b, bd, i_n, u, ud, v, vd)
    i_x = 1 + (__jacc_i - 1)
    vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function jacc_kernel_affine_loss_hv_2!(__jacc_i, i_n, loss, lossd, v, vd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * v[i_seq_x]) * vd[i_seq_x]
    Atomix.@atomic loss[1] += v[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_affine_loss_hv_3!(__jacc_i, i_n, lossb, lossbd, v, vb, vbd, vd)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    vbd[i_seq_x] = vbd[i_seq_x] + (lossb[1] * (2 * vd[i_seq_x]) + (2 * v[i_seq_x]) * lossbd[1])
    vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    return nothing
end

function jacc_kernel_affine_loss_hv_4!(__jacc_i, a, ab, abd, ad, bb, bbd, i_n, u, ub, ubd, ud, vb, vbd)
    i_x = 1 + (__jacc_i - 1)
    abd[i_x] = abd[i_x] + (vb[i_x] * ud[i_x] + u[i_x] * vbd[i_x])
    ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
    ubd[i_x] = ubd[i_x] + (vb[i_x] * ad[i_x] + a[i_x] * vbd[i_x])
    ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
    bbd[i_x] = bbd[i_x] + vbd[i_x]
    bb[i_x] = bb[i_x] + vb[i_x]
    vbd[i_x] = 0.0
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

function affine_loss_hv_jacc(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, bd, bbd, vd, vbd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_affine_loss_hv_1!(a, ad, b, bd, i_n, u, ud, v, vd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_affine_loss_hv_2!(i_n, loss, lossd, v, vd)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_affine_loss_hv_3!(i_n, lossb, lossbd, v, vb, vbd, vd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_affine_loss_hv_4!(a, ab, abd, ad, bb, bbd, i_n, u, ub, ubd, ud, vb, vbd)
    return nothing
end

function affine_loss_jacc(loss, u, a, b, v, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_affine_loss_1!(a, b, i_n, u, v)
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, v)->v[i_seq_x] ^ 2))(v))
    return nothing
end
