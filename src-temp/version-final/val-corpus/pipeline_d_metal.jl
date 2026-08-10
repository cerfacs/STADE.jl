import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_pipeline_d_1!(i_n, u, ud, v, vd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vd[i_x] = (2 * u[i_x]) * ud[i_x]
    v[i_x] = u[i_x] ^ 2 + 1.0f0
    return nothing
end

function metal_kernel_pipeline_d_2!(i_n, u, ud, v, vd, w, wd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = u[i_x] * vd[i_x] + v[i_x] * ud[i_x]
    w[i_x] = v[i_x] * u[i_x]
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

function pipeline_d_metal(loss, lossd, u, ud, v, vd, w, wd, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_pipeline_d_1!(i_n, u, ud, v, vd)
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_pipeline_d_2!(i_n, u, ud, v, vd, w, wd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
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
