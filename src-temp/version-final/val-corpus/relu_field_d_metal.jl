import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_relu_field_d_1!(i_n, u, ud, v, vd)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    if u[i_x] > 0.0f0
        vd[i_x] = (2 * u[i_x]) * ud[i_x]
        v[i_x] = u[i_x] ^ 2
    else
        vd[i_x] = 0.0f0
        v[i_x] = 0.0f0
    end
    return nothing
end

function metal_kernel_relu_field_1!(i_n, u, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    if u[i_x] > 0.0f0
        v[i_x] = u[i_x] ^ 2
    else
        v[i_x] = 0.0f0
    end
    return nothing
end

function relu_field_d_metal(loss, lossd, u, ud, v, vd, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_relu_field_d_1!(i_n, u, ud, v, vd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x]
    end
    return nothing
end

function relu_field_metal(loss, u, v, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_relu_field_1!(i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x]
    end
    return nothing
end
