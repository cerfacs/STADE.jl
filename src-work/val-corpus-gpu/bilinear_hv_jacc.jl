import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_bilinear_hv_1!(__jacc_i, a, ad, i_m, i_n, loss, lossd, x, xd, y, yd)
    i_seq_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        Atomix.@atomic lossd[1] += (a[i_seq_i, i_seq_j] * y[i_seq_j]) * xd[i_seq_i] + (x[i_seq_i] * y[i_seq_j]) * ad[i_seq_i, i_seq_j] + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * yd[i_seq_j]
        Atomix.@atomic loss[1] += x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
    end
    return nothing
end

function jacc_kernel_bilinear_hv_2!(__jacc_i, a, ab, abd, ad, i_m, i_n, lossb, lossbd, x, xb, xbd, xd, y, yb, ybd, yd)
    i_seq_i = i_m + (__jacc_i - 1) * -1
    for i_seq_j = i_n:-1:1
        xbd[i_seq_i] = xbd[i_seq_i] + (lossb[1] * (y[i_seq_j] * ad[i_seq_i, i_seq_j] + a[i_seq_i, i_seq_j] * yd[i_seq_j]) + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossbd[1])
        xb[i_seq_i] = xb[i_seq_i] + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossb[1]
        abd[i_seq_i, i_seq_j] = abd[i_seq_i, i_seq_j] + (lossb[1] * (y[i_seq_j] * xd[i_seq_i] + x[i_seq_i] * yd[i_seq_j]) + (x[i_seq_i] * y[i_seq_j]) * lossbd[1])
        ab[i_seq_i, i_seq_j] = ab[i_seq_i, i_seq_j] + (x[i_seq_i] * y[i_seq_j]) * lossb[1]
        Atomix.@atomic ybd[i_seq_j] += lossb[1] * (a[i_seq_i, i_seq_j] * xd[i_seq_i] + x[i_seq_i] * ad[i_seq_i, i_seq_j]) + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * lossbd[1]
        Atomix.@atomic yb[i_seq_j] += (x[i_seq_i] * a[i_seq_i, i_seq_j]) * lossb[1]
    end
    return nothing
end

function jacc_kernel_bilinear_1!(__jacc_i, a, i_m, i_n, loss, x, y)
    i_seq_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        Atomix.@atomic loss[1] += x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
    end
    return nothing
end

function initstacks_bilinear_b_jacc()
    return nothing
end

function bilinear_hv_jacc(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n, lossd, lossbd, xd, xbd, ad, abd, yd, ybd)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_bilinear_hv_1!(a, ad, i_m, i_n, loss, lossd, x, xd, y, yd)
    JACC.@parallel_for range = div(1 - i_m, -1) + 1 jacc_kernel_bilinear_hv_2!(a, ab, abd, ad, i_m, i_n, lossb, lossbd, x, xb, xbd, xd, y, yb, ybd, yd)
    return nothing
end

function bilinear_jacc(loss, x, a, y, i_m, i_n)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_bilinear_1!(a, i_m, i_n, loss, x, y)
    return nothing
end
