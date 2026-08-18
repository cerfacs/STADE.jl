import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_matvec_loss_hv_1!(__jacc_i, a, ad, i_m, i_n, u, ud, v, vd)
    i_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        vd[i_i] = vd[i_i] + (u[i_seq_j] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * ud[i_seq_j])
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function jacc_kernel_matvec_loss_hv_2!(__jacc_i, i_m, loss, lossd, v, vd)
    i_seq_i = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * v[i_seq_i]) * vd[i_seq_i]
    Atomix.@atomic loss[1] += v[i_seq_i] ^ 2
    return nothing
end

function jacc_kernel_matvec_loss_hv_3!(__jacc_i, i_m, lossb, lossbd, v, vb, vbd, vd)
    i_seq_i = i_m + (__jacc_i - 1) * -1
    vbd[i_seq_i] = vbd[i_seq_i] + (lossb[1] * (2 * vd[i_seq_i]) + (2 * v[i_seq_i]) * lossbd[1])
    vb[i_seq_i] = vb[i_seq_i] + (2 * v[i_seq_i]) * lossb[1]
    return nothing
end

function jacc_kernel_matvec_loss_hv_4!(__jacc_i, a, ab, abd, ad, i_m, i_n, u, ub, ubd, ud, vb, vbd)
    i_i = 1 + (__jacc_i - 1)
    for i_seq_j = i_n:-1:1
        abd[i_i, i_seq_j] = abd[i_i, i_seq_j] + (vb[i_i] * ud[i_seq_j] + u[i_seq_j] * vbd[i_i])
        ab[i_i, i_seq_j] = ab[i_i, i_seq_j] + u[i_seq_j] * vb[i_i]
        Atomix.@atomic ubd[i_seq_j] += vb[i_i] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * vbd[i_i]
        Atomix.@atomic ub[i_seq_j] += a[i_i, i_seq_j] * vb[i_i]
    end
    return nothing
end

function jacc_kernel_matvec_loss_1!(__jacc_i, a, i_m, i_n, u, v)
    i_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function jacc_kernel_matvec_loss_2!(__jacc_i, i_m, loss, v)
    i_seq_i = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += v[i_seq_i] ^ 2
    return nothing
end

function initstacks_matvec_loss_b_jacc()
    return nothing
end

function matvec_loss_hv_jacc(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n, lossd, lossbd, ad, abd, ud, ubd, vd, vbd)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_hv_1!(a, ad, i_m, i_n, u, ud, v, vd)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_hv_2!(i_m, loss, lossd, v, vd)
    JACC.@parallel_for range = div(1 - i_m, -1) + 1 jacc_kernel_matvec_loss_hv_3!(i_m, lossb, lossbd, v, vb, vbd, vd)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_hv_4!(a, ab, abd, ad, i_m, i_n, u, ub, ubd, ud, vb, vbd)
    return nothing
end

function matvec_loss_jacc(loss, a, u, v, i_m, i_n)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_2!(i_m, loss, v)
    return nothing
end
