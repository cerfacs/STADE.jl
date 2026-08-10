import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_mg_vcycle_d_1!(f, fd, hl2, hl2d, i_seq_level, n, r, rd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    leftd = 0.0
    left = 0.0
    if j > 1
        leftd = ud[j - 1, i_seq_level]
        left = u[j - 1, i_seq_level]
    end
    rightd = 0.0
    right = 0.0
    if j < n
        rightd = ud[j + 1, i_seq_level]
        right = u[j + 1, i_seq_level]
    end
    rd[j, i_seq_level] = fd[j, i_seq_level] + -(((1.0 / hl2) * ((2.0 * ud[j, i_seq_level] + -leftd) + -rightd) + -(((2.0 * u[j, i_seq_level] - left) - right) / hl2 ^ 2) * hl2d))
    r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
    return nothing
end

function cuda_kernel_mg_vcycle_d_2!(f, fd, i_seq_level, nc, r, rd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jfd = 0.0
    jf = j * 2
    fd[j, i_seq_level + 1] = (0.25 * rd[jf - 1, i_seq_level] + 0.5 * rd[jf, i_seq_level]) + 0.25 * rd[jf + 1, i_seq_level]
    f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
    return nothing
end

function cuda_kernel_mg_vcycle_d_3!(i_seq_level, nc, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    ud[j, i_seq_level + 1] = 0.0
    u[j, i_seq_level + 1] = 0.0
    return nothing
end

function cuda_kernel_mg_vcycle_d_4!(i_seq_level, nc, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(nc - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jfd = 0.0
    jf = j * 2
    CUDA.@atomic ud[jf, i_seq_level] += ud[j, i_seq_level + 1]
    CUDA.@atomic u[jf, i_seq_level] += u[j, i_seq_level + 1]
    return nothing
end

function cuda_kernel_mg_vcycle_d_5!(i_seq_level, nc, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((nc + 1) - 1, 1) + 1
        return nothing
    end
    j = 1 + (__tid - 1)
    jfd = 0.0
    jf = j * 2 - 1
    cld = 0.0
    cl = 0.0
    if j > 1
        cld = ud[j - 1, i_seq_level + 1]
        cl = u[j - 1, i_seq_level + 1]
    end
    crd = 0.0
    cr = 0.0
    if j <= nc
        crd = ud[j, i_seq_level + 1]
        cr = u[j, i_seq_level + 1]
    end
    CUDA.@atomic ud[jf, i_seq_level] += 0.5 * (cld + crd)
    CUDA.@atomic u[jf, i_seq_level] += 0.5 * (cl + cr)
    return nothing
end

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

function mg_vcycle_d_cuda(u, ud, f, fd, r, rd, nfine, num_levels, h1, h1d, nu1, nu2, n)
    nthread_per_block = 256
    nd = 0.0
    n = n * 2
    nld = 0.0
    nl = nfine
    hld = h1d
    hl = h1
    for i_seq_level = 1:num_levels - 1
        nd = 0.0
        n = nl - 1
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_seq_j > 1
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                end
                rightd = 0.0
                right = 0.0
                if i_seq_j < n
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                end
                ud[i_seq_j, i_seq_level] = 0.5 * (((f[i_seq_j, i_seq_level] * hl2d + hl2 * fd[i_seq_j, i_seq_level]) + leftd) + rightd)
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
        @cuda threads = nthread_per_block blocks = cld(div(n - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_d_1!(f, fd, hl2, hl2d, i_seq_level, n, r, rd, u, ud)
        ncgd = 0.0
        ncg = div(nl, 2)
        ncd = 0.0
        nc = ncg - 1
        @cuda threads = nthread_per_block blocks = cld(div(nc - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_d_2!(f, fd, i_seq_level, nc, r, rd)
        @cuda threads = nthread_per_block blocks = cld(div(nc - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_d_3!(i_seq_level, nc, u, ud)
        nld = 0.0
        nl = ncg
        hld = 2.0hld
        hl = hl * 2.0
    end
    hl2d = hl * hld + hl * hld
    hl2 = hl * hl
    ud[1, num_levels] = (0.5 * f[1, num_levels]) * hl2d + (0.5hl2) * fd[1, num_levels]
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nld = 0.0
        nl = nl * 2
        hld = 0.5hld
        hl = hl / 2.0
        nd = 0.0
        n = nl - 1
        ncgd = 0.0
        ncg = div(nl, 2)
        ncd = 0.0
        nc = ncg - 1
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        @cuda threads = nthread_per_block blocks = cld(div(nc - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_d_4!(i_seq_level, nc, u, ud)
        @cuda threads = nthread_per_block blocks = cld(div((nc + 1) - 1, 1) + 1, nthread_per_block) cuda_kernel_mg_vcycle_d_5!(i_seq_level, nc, u, ud)
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_seq_j > 1
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                end
                rightd = 0.0
                right = 0.0
                if i_seq_j < n
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                end
                ud[i_seq_j, i_seq_level] = 0.5 * (((f[i_seq_j, i_seq_level] * hl2d + hl2 * fd[i_seq_j, i_seq_level]) + leftd) + rightd)
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
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
