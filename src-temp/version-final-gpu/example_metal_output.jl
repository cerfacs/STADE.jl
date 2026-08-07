import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_bump_1!(n, u, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = v[i_x] + 2.0f0 * u[i_x]
    Metal.@atomic v[1] += u[i_x]
    return nothing
end

function bump_metal(u, v, n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(n - 1, 1) + 1, nthread_per_block) metal_kernel_bump_1!(n, u, v)
    acc = 0.0f0
    for i_seq_t = 1:n
        acc = acc + u[i_seq_t]
    end
    v[2] = acc
    return nothing
end
