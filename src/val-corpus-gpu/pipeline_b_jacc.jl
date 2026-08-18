import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_pipeline_b_1!(__jacc_i, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function jacc_kernel_pipeline_b_2!(__jacc_i, i_n, u, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function jacc_kernel_pipeline_b_3!(__jacc_i, i_n, loss, w)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function jacc_kernel_pipeline_b_4!(__jacc_i, i_n, lossb, wb)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    return nothing
end

function jacc_kernel_pipeline_b_5!(__jacc_i, i_n, u, ub, v, vb, wb)
    i_x = 1 + (__jacc_i - 1)
    vb[i_x] = vb[i_x] + u[i_x] * wb[i_x]
    ub[i_x] = ub[i_x] + v[i_x] * wb[i_x]
    wb[i_x] = 0.0
    return nothing
end

function jacc_kernel_pipeline_b_6!(__jacc_i, i_n, u, ub, vb)
    i_x = 1 + (__jacc_i - 1)
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
    vb[i_x] = 0.0
    return nothing
end

function jacc_kernel_pipeline_1!(__jacc_i, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function jacc_kernel_pipeline_2!(__jacc_i, i_n, u, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function jacc_kernel_pipeline_3!(__jacc_i, i_n, loss, w)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function initstacks_pipeline_b_jacc()
    return nothing
end

function pipeline_b_jacc(loss, lossb, u, ub, v, vb, w, wb, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_b_1!(i_n, u, v)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_b_2!(i_n, u, v, w)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_b_3!(i_n, loss, w)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_pipeline_b_4!(i_n, lossb, wb)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_b_5!(i_n, u, ub, v, vb, wb)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_b_6!(i_n, u, ub, vb)
    return nothing
end

function pipeline_jacc(loss, u, v, w, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_1!(i_n, u, v)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_2!(i_n, u, v, w)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_pipeline_3!(i_n, loss, w)
    return nothing
end
