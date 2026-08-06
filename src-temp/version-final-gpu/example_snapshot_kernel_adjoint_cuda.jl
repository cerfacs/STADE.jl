import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_sq_test_1!(n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = u[i_x]
    v[i_x] = v[i_x] * v[i_x]
    return nothing
end

function initstacks_sq_test_b_cuda()
    v_stack = Vector{Float64}()
    return v_stack
end

function sq_test_b_cuda(u, ub, v, vb, n, v_stack)
    for i_x = 1:n
        v[i_x] = u[i_x]
        push!(v_stack, v[i_x])
        v[i_x] = v[i_x] * v[i_x]
    end
    for i_x = n:-1:1
        v[i_x] = pop!(v_stack)
        vb[i_x] = v[i_x] * vb[i_x] + v[i_x] * vb[i_x]
        ub[i_x] = ub[i_x] + vb[i_x]
        vb[i_x] = 0.0
    end
    return nothing
end

function sq_test_cuda(u, v, n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(n - 1, 1) + 1, nthread_per_block) cuda_kernel_sq_test_1!(n, u, v)
    return nothing
end
