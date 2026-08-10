import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_cond_field_choice_hv_1!(i_n, u, ud, w, wd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = (2 * u[i_x]) * ud[i_x]
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function metal_kernel_cond_field_choice_hv_2!(i_n, v, vd, w, wd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = (2 * v[i_x]) * vd[i_x]
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function metal_kernel_cond_field_choice_hv_3!(i_n, u, ub, ubd, ud, wb, wbd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ubd[i_x] = ubd[i_x] + (wb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * wbd[i_x])
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * wb[i_x]
    wbd[i_x] = 0.0f0
    wb[i_x] = 0.0f0
    return nothing
end

function metal_kernel_cond_field_choice_hv_4!(i_n, v, vb, vbd, vd, wb, wbd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vbd[i_x] = vbd[i_x] + (wb[i_x] * (2 * vd[i_x]) + (2 * v[i_x]) * wbd[i_x])
    vb[i_x] = vb[i_x] + (2 * v[i_x]) * wb[i_x]
    wbd[i_x] = 0.0f0
    wb[i_x] = 0.0f0
    return nothing
end

function metal_kernel_cond_field_choice_1!(i_n, u, w)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function metal_kernel_cond_field_choice_2!(i_n, v, w)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function initstacks_cond_field_choice_b_metal()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function cond_field_choice_hv_metal(loss, lossb, u, ub, v, vb, w, wb, i_branch, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd, branch_stack)
    nthread_per_block = 256
    if i_branch == 1
        push!(branch_stack, 1)
        @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_cond_field_choice_hv_1!(i_n, u, ud, w, wd)
    else
        push!(branch_stack, 0)
        @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_cond_field_choice_hv_2!(i_n, v, vd, w, wd)
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        wbd[i_seq_x] = wbd[i_seq_x] + lossbd[1]
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    __branch = pop!(branch_stack)
    if __branch == 1
        @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_cond_field_choice_hv_3!(i_n, u, ub, ubd, ud, wb, wbd)
    else
        @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_cond_field_choice_hv_4!(i_n, v, vb, vbd, vd, wb, wbd)
    end
    return nothing
end

function cond_field_choice_metal(loss, u, v, w, i_branch, i_n)
    nthread_per_block = 256
    if i_branch == 1
        @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_cond_field_choice_1!(i_n, u, w)
    else
        @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_cond_field_choice_2!(i_n, v, w)
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end
