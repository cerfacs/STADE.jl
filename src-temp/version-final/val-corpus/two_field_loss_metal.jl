import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_two_field_loss_1!(i_n, p, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function metal_kernel_two_field_loss_2!(i_n, q, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function two_field_loss_metal(loss, u, v, p, q, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_two_field_loss_1!(i_n, p, u)
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_two_field_loss_2!(i_n, q, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    return nothing
end
