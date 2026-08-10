import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_stencil_loss_d_1!(i_n, u, ud, w, wd)
    __tid = (thread_position_in_grid()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    wd[i_x] = (ud[i_x - 1] + -(2.0f0 * ud[i_x])) + ud[i_x + 1]
    w[i_x] = (u[i_x - 1] - 2.0f0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function metal_kernel_stencil_loss_1!(i_n, u, w)
    __tid = (thread_position_in_grid()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0f0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function stencil_loss_d_metal(loss, lossd, u, ud, w, wd, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) metal_kernel_stencil_loss_d_1!(i_n, u, ud, w, wd)
    for i_seq_x = 2:i_n - 1
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end

function stencil_loss_metal(loss, u, w, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) metal_kernel_stencil_loss_1!(i_n, u, w)
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end
