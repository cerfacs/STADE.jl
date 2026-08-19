import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_bilinear_d_1!(__jacc_i, a, ad, i_m, i_n, loss, lossd, x, xd, y, yd)
    i_seq_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        Atomix.@atomic lossd[1] += (a[i_seq_i, i_seq_j] * y[i_seq_j]) * xd[i_seq_i] + (x[i_seq_i] * y[i_seq_j]) * ad[i_seq_i, i_seq_j] + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * yd[i_seq_j]
        Atomix.@atomic loss[1] += x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
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

function bilinear_d_jacc(loss, lossd, x, xd, a, ad, y, yd, i_m, i_n)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_bilinear_d_1!(a, ad, i_m, i_n, loss, lossd, x, xd, y, yd)
    return nothing
end

function bilinear_jacc(loss, x, a, y, i_m, i_n)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_bilinear_1!(a, i_m, i_n, loss, x, y)
    return nothing
end
