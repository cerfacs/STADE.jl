import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function initstacks_dotprod_b_amdgpu()
    return nothing
end

function dotprod_b_amdgpu(loss, lossb, u, ub, v, vb, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
        vb[i_seq_x] = vb[i_seq_x] + u[i_seq_x] * lossb[1]
    end
    return nothing
end

function dotprod_amdgpu(loss, u, v, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    return nothing
end
