import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function dotprod_amdgpu(loss, u, v, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    return nothing
end
