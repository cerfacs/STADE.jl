import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_bump_1!(n, u, v)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = v[i_x] + 2.0 * u[i_x]
    AMDGPU.@atomic v[1] += u[i_x]
    return nothing
end

function bump_amdgpu(u, v, n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div(n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_bump_1!(n, u, v)
    acc = 0.0
    for i_seq_t = 1:n
        acc = acc + u[i_seq_t]
    end
    v[2] = acc
    return nothing
end
