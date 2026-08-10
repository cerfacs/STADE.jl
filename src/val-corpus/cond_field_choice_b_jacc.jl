import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_cond_field_choice_b_1!(__jacc_i, i_n, u, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function jacc_kernel_cond_field_choice_b_2!(__jacc_i, i_n, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function jacc_kernel_cond_field_choice_b_3!(__jacc_i, i_n, u, ub, wb)
    i_x = 1 + (__jacc_i - 1)
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * wb[i_x]
    wb[i_x] = 0.0
    return nothing
end

function jacc_kernel_cond_field_choice_b_4!(__jacc_i, i_n, v, vb, wb)
    i_x = 1 + (__jacc_i - 1)
    vb[i_x] = vb[i_x] + (2 * v[i_x]) * wb[i_x]
    wb[i_x] = 0.0
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

function initstacks_cond_field_choice_b_jacc()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function cond_field_choice_b_jacc(loss, lossb, u, ub, v, vb, w, wb, i_branch, i_n, branch_stack)
    if i_branch == 1
        push!(branch_stack, 1)
        JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_cond_field_choice_b_1!, i_n, u, w)
    else
        push!(branch_stack, 0)
        JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_cond_field_choice_b_2!, i_n, v, w)
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    __branch = pop!(branch_stack)
    if __branch == 1
        JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_cond_field_choice_b_3!, i_n, u, ub, wb)
    else
        JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_cond_field_choice_b_4!, i_n, v, vb, wb)
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
