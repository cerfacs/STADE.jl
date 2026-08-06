import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_sq_test_1!(n, u, v)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = u[i_x]
    v[i_x] = v[i_x] * v[i_x]
    return nothing
end

function initstacks_sq_test_b_amdgpu()
    v_stack = Vector{Float64}()
    return v_stack
end

function sq_test_b_amdgpu(u, ub, v, vb, n, v_stack)
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

function sq_test_amdgpu(u, v, n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div(n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_sq_test_1!(n, u, v)
    return nothing
end
