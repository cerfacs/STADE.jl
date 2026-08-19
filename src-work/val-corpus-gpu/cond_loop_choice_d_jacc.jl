import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_cond_loop_choice_d_1!(__jacc_i, i_n, loss, lossd, u, ud)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * u[i_seq_x]) * ud[i_seq_x]
    Atomix.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_cond_loop_choice_d_2!(__jacc_i, i_n, loss, lossd, v, vd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * v[i_seq_x]) * vd[i_seq_x]
    Atomix.@atomic loss[1] += v[i_seq_x] ^ 2
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

function cond_loop_choice_d_jacc(loss, lossd, u, ud, v, vd, i_branch, i_n)
    if i_branch == 1
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_loop_choice_d_1!(i_n, loss, lossd, u, ud)
    else
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_loop_choice_d_2!(i_n, loss, lossd, v, vd)
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
