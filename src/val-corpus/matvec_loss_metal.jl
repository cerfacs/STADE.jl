import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function matvec_loss_metal(loss, a, u, v, i_m, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_m - 1, 1) + 1, nthread_per_block) metal_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    for i_seq_i = 1:i_m
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
    return nothing
end
