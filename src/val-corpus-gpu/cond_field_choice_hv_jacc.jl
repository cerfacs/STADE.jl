import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_cond_field_choice_hv_1!(__jacc_i, i_n, u, ud, w, wd)
    i_x = 1 + (__jacc_i - 1)
    wd[i_x] = (2 * u[i_x]) * ud[i_x]
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function jacc_kernel_cond_field_choice_hv_2!(__jacc_i, i_n, v, vd, w, wd)
    i_x = 1 + (__jacc_i - 1)
    wd[i_x] = (2 * v[i_x]) * vd[i_x]
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function jacc_kernel_cond_field_choice_hv_3!(__jacc_i, i_n, loss, lossd, w, wd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += wd[i_seq_x]
    Atomix.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function jacc_kernel_cond_field_choice_hv_4!(__jacc_i, i_n, lossb, lossbd, wb, wbd)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    wbd[i_seq_x] = wbd[i_seq_x] + lossbd[1]
    wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    return nothing
end

function jacc_kernel_cond_field_choice_hv_5!(__jacc_i, i_n, u, ub, ubd, ud, wb, wbd)
    i_x = 1 + (__jacc_i - 1)
    ubd[i_x] = ubd[i_x] + (wb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * wbd[i_x])
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * wb[i_x]
    wbd[i_x] = 0.0
    wb[i_x] = 0.0
    return nothing
end

function jacc_kernel_cond_field_choice_hv_6!(__jacc_i, i_n, v, vb, vbd, vd, wb, wbd)
    i_x = 1 + (__jacc_i - 1)
    vbd[i_x] = vbd[i_x] + (wb[i_x] * (2 * vd[i_x]) + (2 * v[i_x]) * wbd[i_x])
    vb[i_x] = vb[i_x] + (2 * v[i_x]) * wb[i_x]
    wbd[i_x] = 0.0
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

function jacc_kernel_cond_field_choice_3!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function initstacks_cond_field_choice_b_jacc()
    branch_stack = JACC.zeros(Int64, 1)
    return branch_stack
end

function cond_field_choice_hv_jacc(loss, lossb, u, ub, v, vb, w, wb, i_branch, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd, branch_stack)
    if i_branch == 1
        branch_stack[1] = 1
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_field_choice_hv_1!(i_n, u, ud, w, wd)
    else
        branch_stack[1] = 0
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_field_choice_hv_2!(i_n, v, vd, w, wd)
    end
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_field_choice_hv_3!(i_n, loss, lossd, w, wd)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_cond_field_choice_hv_4!(i_n, lossb, lossbd, wb, wbd)
    __branch = branch_stack[1]
    if __branch == 1
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_field_choice_hv_5!(i_n, u, ub, ubd, ud, wb, wbd)
    else
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_field_choice_hv_6!(i_n, v, vb, vbd, vd, wb, wbd)
    end
    return nothing
end

function cond_field_choice_jacc(loss, u, v, w, i_branch, i_n)
    if i_branch == 1
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_field_choice_1!(i_n, u, w)
    else
        JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_cond_field_choice_2!(i_n, v, w)
    end
    __jgen_redval_3 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, w)->w[i_seq_x]))(w))
    JACC.@parallel_for range = 1 jacc_kernel_cond_field_choice_3!(loss, __jgen_redval_3)
    return nothing
end
