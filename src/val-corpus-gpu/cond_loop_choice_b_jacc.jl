import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_cond_loop_choice_b_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_cond_loop_choice_b_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_cond_loop_choice_b_3!(__jacc_i, i_n, lossb, u, ub)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    return nothing
end

function jacc_kernel_cond_loop_choice_b_4!(__jacc_i, i_n, lossb, v, vb)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    return nothing
end

function jacc_kernel_cond_loop_choice_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_cond_loop_choice_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function initstacks_cond_loop_choice_b_jacc()
    branch_stack = JACC.zeros(Int64, 1)
    return branch_stack
end

function cond_loop_choice_b_jacc(loss, lossb, u, ub, v, vb, i_branch, i_n, branch_stack)
    if i_branch == 1
        branch_stack[1] = 1
        __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u)->u[i_seq_x] ^ 2))(u))
        JACC.@parallel_for range = 1 jacc_kernel_cond_loop_choice_b_1!(loss, __jgen_redval_1)
    else
        branch_stack[1] = 0
        __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, v)->v[i_seq_x] ^ 2))(v))
        JACC.@parallel_for range = 1 jacc_kernel_cond_loop_choice_b_2!(loss, __jgen_redval_2)
    end
    __branch = branch_stack[1]
    if __branch == 1
        JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_cond_loop_choice_b_3!(i_n, lossb, u, ub)
    else
        JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_cond_loop_choice_b_4!(i_n, lossb, v, vb)
    end
    return nothing
end

function cond_loop_choice_jacc(loss, u, v, i_branch, i_n)
    if i_branch == 1
        __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u)->u[i_seq_x] ^ 2))(u))
        JACC.@parallel_for range = 1 jacc_kernel_cond_loop_choice_1!(loss, __jgen_redval_1)
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, v)->v[i_seq_x] ^ 2))(v))
        JACC.@parallel_for range = 1 jacc_kernel_cond_loop_choice_2!(loss, __jgen_redval_2)
    end
    return nothing
end
