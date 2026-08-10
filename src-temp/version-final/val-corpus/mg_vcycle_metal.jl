import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_mg_vcycle_1!(f, hl2, i_seq_level, n, r, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    left = 0.0f0
    if j > 1
        left = u[j - 1, i_seq_level]
    end
    right = 0.0f0
    if j < n
        right = u[j + 1, i_seq_level]
    end
    r[j, i_seq_level] = f[j, i_seq_level] - Float32((2.0f0 * u[j, i_seq_level] - left) - right) / Float32(hl2)
    return nothing
end

function metal_kernel_mg_vcycle_2!(f, i_seq_level, nc, r)
    __tid = (thread_position_in_grid()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jf = j * 2
    f[j, i_seq_level + 1] = 0.25f0 * r[jf - 1, i_seq_level] + 0.5f0 * r[jf, i_seq_level] + 0.25f0 * r[jf + 1, i_seq_level]
    return nothing
end

function metal_kernel_mg_vcycle_3!(i_seq_level, nc, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    u[j, i_seq_level + 1] = 0.0f0
    return nothing
end

function metal_kernel_mg_vcycle_4!(i_seq_level, nc, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jf = j * 2
    Metal.@atomic u[jf, i_seq_level] += u[j, i_seq_level + 1]
    return nothing
end

function metal_kernel_mg_vcycle_5!(i_seq_level, nc, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div((nc + 1) - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jf = j * 2 - 1
    cl = 0.0f0
    if j > 1
        cl = u[j - 1, i_seq_level + 1]
    end
    cr = 0.0f0
    if j <= nc
        cr = u[j, i_seq_level + 1]
    end
    Metal.@atomic u[jf, i_seq_level] += 0.5f0 * (cl + cr)
    return nothing
end

function mg_vcycle_metal(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    nthread_per_block = 256
    n = n * 2
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                left = 0.0f0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0f0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5f0 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
        @metal threads = nthread_per_block groups = cld(div(n - 1, 1) + 1, nthread_per_block) metal_kernel_mg_vcycle_1!(f, hl2, i_seq_level, n, r, u)
        ncg = div(nl, 2)
        nc = ncg - 1
        @metal threads = nthread_per_block groups = cld(div(nc - 1, 1) + 1, nthread_per_block) metal_kernel_mg_vcycle_2!(f, i_seq_level, nc, r)
        @metal threads = nthread_per_block groups = cld(div(nc - 1, 1) + 1, nthread_per_block) metal_kernel_mg_vcycle_3!(i_seq_level, nc, u)
        nl = ncg
        hl = hl * 2.0f0
    end
    hl2 = hl * hl
    u[1, num_levels] = 0.5f0 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        hl = Float32(hl) / Float32(2.0f0)
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        @metal threads = nthread_per_block groups = cld(div(nc - 1, 1) + 1, nthread_per_block) metal_kernel_mg_vcycle_4!(i_seq_level, nc, u)
        @metal threads = nthread_per_block groups = cld(div((nc + 1) - 1, 1) + 1, nthread_per_block) metal_kernel_mg_vcycle_5!(i_seq_level, nc, u)
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                left = 0.0f0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0f0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5f0 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
    return nothing
end
