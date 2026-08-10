import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function initstacks_weightedsumsq_b_cuda()
    return nothing
end

function weightedsumsq_b_cuda(loss, lossb, u, ub, w, wb, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + u[i_seq_x] ^ 2 * lossb[1]
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * (w[i_seq_x] * lossb[1])
    end
    return nothing
end

function weightedsumsq_cuda(loss, u, w, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
    return nothing
end
