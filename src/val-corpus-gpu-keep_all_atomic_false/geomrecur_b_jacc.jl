import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_geomrecur_b_1!(__jacc_i, i_n, lossb, u, ub)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    return nothing
end

function initstacks_geomrecur_b_jacc(i_n)
    u_stack = JACC.zeros(Float64, div(i_n - 2, 1) + 1)
    return u_stack
end

function geomrecur_b_jacc(loss, lossb, u, ub, c, cb, i_n, u_stack)
    for i_seq_x = 2:i_n
        u_stack[(i_seq_x - 2) + 1] = u[i_seq_x]
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u)->u[i_seq_x] ^ 2))(u))
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_geomrecur_b_1!(i_n, lossb, u, ub)
    for i_seq_x = i_n:-1:2
        u[i_seq_x] = u_stack[(i_seq_x - 2) + 1]
        cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
        ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
        ub[i_seq_x] = 0.0
    end
    return cb
end

function geomrecur_jacc(loss, u, c, i_n)
    for i_seq_x = 2:i_n
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u)->u[i_seq_x] ^ 2))(u))
    return nothing
end
