import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_relax_1!(i_seq_inner, n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = v[i_x] + u[i_x] / (1.0 + i_seq_inner)
    return nothing
end

function cuda_kernel_relax_2!(i_seq_outer, n, v, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = w[i_x] + v[i_x] * i_seq_outer
    return nothing
end

function relax_cuda(u, v, w, n, i_nouter, i_ninner)
    nthread_per_block = 256
    for i_seq_outer = 1:i_nouter
        for i_seq_inner = 1:i_ninner
            @cuda threads = nthread_per_block blocks = cld(div(n - 1, 1) + 1, nthread_per_block) cuda_kernel_relax_1!(i_seq_inner, n, u, v)
        end
        @cuda threads = nthread_per_block blocks = cld(div(n - 1, 1) + 1, nthread_per_block) cuda_kernel_relax_2!(i_seq_outer, n, v, w)
    end
    return nothing
end

function relax(u, v, w, n, i_nouter, i_ninner)
    #= none:15 =#
    #= none:16 =#
    for i_seq_outer = 1:i_nouter
        #= none:17 =#
        for i_seq_inner = 1:i_ninner
            #= none:18 =#
            for i_x = 1:n
                #= none:19 =#
                v[i_x] = v[i_x] + u[i_x] / (1.0 + i_seq_inner)
                #= none:20 =#
            end
            #= none:21 =#
        end
        #= none:22 =#
        for i_x = 1:n
            #= none:23 =#
            w[i_x] = w[i_x] + v[i_x] * i_seq_outer
            #= none:24 =#
        end
        #= none:25 =#
    end
    #= none:26 =#
    return nothing
end