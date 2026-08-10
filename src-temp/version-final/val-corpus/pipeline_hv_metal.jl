import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_pipeline_hv_1!(i_n, u, ud, v, vd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vd[i_x] = (2 * u[i_x]) * ud[i_x]
    v[i_x] = u[i_x] ^ 2 + 1.0f0
    return nothing
end

function metal_kernel_pipeline_hv_2!(i_n, u, ud, v, vd, w, wd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = u[i_x] * vd[i_x] + v[i_x] * ud[i_x]
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function metal_kernel_pipeline_hv_3!(i_n, u, ub, ubd, ud, v, vb, vbd, vd, wb, wbd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vbd[i_x] = vbd[i_x] + (wb[i_x] * ud[i_x] + u[i_x] * wbd[i_x])
    vb[i_x] = vb[i_x] + u[i_x] * wb[i_x]
    ubd[i_x] = ubd[i_x] + (wb[i_x] * vd[i_x] + v[i_x] * wbd[i_x])
    ub[i_x] = ub[i_x] + v[i_x] * wb[i_x]
    wbd[i_x] = 0.0f0
    wb[i_x] = 0.0f0
    return nothing
end

function metal_kernel_pipeline_hv_4!(i_n, u, ub, ubd, ud, vb, vbd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ubd[i_x] = ubd[i_x] + (vb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * vbd[i_x])
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
    vbd[i_x] = 0.0f0
    vb[i_x] = 0.0f0
    return nothing
end

function metal_kernel_pipeline_1!(i_n, u, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = u[i_x] ^ 2 + 1.0f0
    return nothing
end

function metal_kernel_pipeline_2!(i_n, u, v, w)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function initstacks_pipeline_b_metal()
    return nothing
end

function pipeline_hv_metal(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_pipeline_hv_1!(i_n, u, ud, v, vd)
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_pipeline_hv_2!(i_n, u, ud, v, vd, w, wd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        wbd[i_seq_x] = wbd[i_seq_x] + lossbd[1]
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_pipeline_hv_3!(i_n, u, ub, ubd, ud, v, vb, vbd, vd, wb, wbd)
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_pipeline_hv_4!(i_n, u, ub, ubd, ud, vb, vbd)
    return nothing
end

function pipeline_metal(loss, u, v, w, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_pipeline_1!(i_n, u, v)
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_pipeline_2!(i_n, u, v, w)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end
