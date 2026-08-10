import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_mg_vcycle_1!(f, hl2, i_seq_level, n, r, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    left = 0.0
    if j > 1
        left = u[j - 1, i_seq_level]
    end
    right = 0.0
    if j < n
        right = u[j + 1, i_seq_level]
    end
    r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
    return nothing
end

function cuda_kernel_mg_vcycle_2!(f, i_seq_level, nc, r)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jf = j * 2
    f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
    return nothing
end

function cuda_kernel_mg_vcycle_3!(i_seq_level, nc, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    u[j, i_seq_level + 1] = 0.0
    return nothing
end

function cuda_kernel_mg_vcycle_4!(i_seq_level, nc, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jf = j * 2
    CUDA.@atomic u[jf, i_seq_level] += u[j, i_seq_level + 1]
    return nothing
end

function cuda_kernel_mg_vcycle_5!(i_seq_level, nc, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((nc + 1) - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jf = j * 2 - 1
    cl = 0.0
    if j > 1
        cl = u[j - 1, i_seq_level + 1]
    end
    cr = 0.0
    if j <= nc
        cr = u[j, i_seq_level + 1]
    end
    CUDA.@atomic u[jf, i_seq_level] += 0.5 * (cl + cr)
    return nothing
end

function mg_vcycle_cuda(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    nthread_per_block = 256
    n = n * 2
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
        @cuda threads = nthread_per_block blocks = cld(div(n - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_1!(f, hl2, i_seq_level, n, r, u)
        ncg = div(nl, 2)
        nc = ncg - 1
        @cuda threads = nthread_per_block blocks = cld(div(nc - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_2!(f, i_seq_level, nc, r)
        @cuda threads = nthread_per_block blocks = cld(div(nc - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_3!(i_seq_level, nc, u)
        nl = ncg
        hl = hl * 2.0
    end
    hl2 = hl * hl
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        @cuda threads = nthread_per_block blocks = cld(div(nc - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_4!(i_seq_level, nc, u)
        @cuda threads = nthread_per_block blocks = cld(div((nc + 1) - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_5!(i_seq_level, nc, u)
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
    return nothing
end
