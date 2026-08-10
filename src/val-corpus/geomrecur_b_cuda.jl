import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function initstacks_geomrecur_b_cuda()
    u_stack = Vector{Float64}()
    return u_stack
end

function geomrecur_b_cuda(loss, lossb, u, ub, c, cb, i_n, u_stack)
    for i_seq_x = 2:i_n
        push!(u_stack, u[i_seq_x])
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    end
    for i_seq_x = i_n:-1:2
        u[i_seq_x] = pop!(u_stack)
        cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
        ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
        ub[i_seq_x] = 0.0
    end
    return cb
end

function geomrecur_cuda(loss, u, c, i_n)
    for i_seq_x = 2:i_n
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
    return nothing
end
