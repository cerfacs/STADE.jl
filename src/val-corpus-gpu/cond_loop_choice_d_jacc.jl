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

function jacc_kernel_cond_loop_choice_1!(__jacc_i, i_n, loss, u)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_cond_loop_choice_2!(__jacc_i, i_n, loss, v)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += v[i_seq_x] ^ 2
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
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_loop_choice_1!(i_n, loss, u)
    else
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_loop_choice_2!(i_n, loss, v)
    end
    return nothing
end
