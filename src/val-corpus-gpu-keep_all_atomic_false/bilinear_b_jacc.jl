import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_bilinear_b_1!(__jacc_i, a, i_m, i_n, loss, x, y)
    i_seq_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        Atomix.@atomic loss[1] += x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
    end
    return nothing
end

function jacc_kernel_bilinear_b_2!(__jacc_i, a, ab, i_m, i_n, lossb, x, xb, y, yb)
    i_seq_i = i_m + (__jacc_i - 1) * -1
    for i_seq_j = i_n:-1:1
        xb[i_seq_i] = xb[i_seq_i] + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossb[1]
        ab[i_seq_i, i_seq_j] = ab[i_seq_i, i_seq_j] + (x[i_seq_i] * y[i_seq_j]) * lossb[1]
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

function bilinear_b_jacc(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_bilinear_b_1!(a, i_m, i_n, loss, x, y)
    JACC.@parallel_for range = div(1 - i_m, -1) + 1 jacc_kernel_bilinear_b_2!(a, ab, i_m, i_n, lossb, x, xb, y, yb)
    return nothing
end

function bilinear_jacc(loss, x, a, y, i_m, i_n)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_bilinear_1!(a, i_m, i_n, loss, x, y)
    return nothing
end
