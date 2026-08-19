import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_clamped_sumsq_b_1!(__jacc_i, branch_stack, i_n, loss, u)
    i_seq_x = 1 + (__jacc_i - 1)
    if u[i_seq_x] > 0.0
        branch_stack[(i_seq_x - 1) + 1] = 1
        w = u[i_seq_x] ^ 2
    else
        branch_stack[(i_seq_x - 1) + 1] = 0
        w = 0.0
    end
    Atomix.@atomic loss[1] += w
    return nothing
end

function jacc_kernel_clamped_sumsq_b_2!(__jacc_i, branch_stack, i_n, lossb, u, ub)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    wb = 0.0
    wb = wb + lossb[1]
    __branch = branch_stack[(i_seq_x - 1) + 1]
    if __branch == 1
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * wb
        wb = 0.0
    else
        wb = 0.0
    end
    return nothing
end

function jacc_kernel_clamped_sumsq_1!(__jacc_i, i_n, loss, u)
    i_seq_x = 1 + (__jacc_i - 1)
    if u[i_seq_x] > 0.0
        w = u[i_seq_x] ^ 2
    else
        w = 0.0
    end
    Atomix.@atomic loss[1] += w
    return nothing
end

function initstacks_clamped_sumsq_b_jacc(i_n)
    branch_stack = JACC.zeros(Int64, div(i_n - 1, 1) + 1)
    return branch_stack
end

function clamped_sumsq_b_jacc(loss, lossb, u, ub, i_n, branch_stack)
    w = 0.0
    wb = 0.0
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_clamped_sumsq_b_1!(branch_stack, i_n, loss, u)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_clamped_sumsq_b_2!(branch_stack, i_n, lossb, u, ub)
    return nothing
end

function clamped_sumsq_jacc(loss, u, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_clamped_sumsq_1!(i_n, loss, u)
    return nothing
end
