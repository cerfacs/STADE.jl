import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_cond_field_choice_d_1!(__jacc_i, i_n, u, ud, w, wd)
    i_x = 1 + (__jacc_i - 1)
    wd[i_x] = (2 * u[i_x]) * ud[i_x]
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function jacc_kernel_cond_field_choice_d_2!(__jacc_i, i_n, v, vd, w, wd)
    i_x = 1 + (__jacc_i - 1)
    wd[i_x] = (2 * v[i_x]) * vd[i_x]
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function jacc_kernel_cond_field_choice_1!(__jacc_i, i_n, u, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function jacc_kernel_cond_field_choice_2!(__jacc_i, i_n, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function cond_field_choice_d_jacc(loss, lossd, u, ud, v, vd, w, wd, i_branch, i_n)
    if i_branch == 1
        JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_cond_field_choice_d_1!, i_n, u, ud, w, wd)
    else
        JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_cond_field_choice_d_2!, i_n, v, vd, w, wd)
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end

function cond_field_choice_jacc(loss, u, v, w, i_branch, i_n)
    if i_branch == 1
        JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_cond_field_choice_1!, i_n, u, w)
    else
        JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_cond_field_choice_2!, i_n, v, w)
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end
