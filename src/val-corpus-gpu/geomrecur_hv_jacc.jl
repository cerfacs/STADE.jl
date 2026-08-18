import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_geomrecur_hv_1!(__jacc_i, i_n, loss, lossd, u, ud)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * u[i_seq_x]) * ud[i_seq_x]
    Atomix.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_geomrecur_hv_2!(__jacc_i, i_n, lossb, lossbd, u, ub, ubd, ud)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    ubd[i_seq_x] = ubd[i_seq_x] + (lossb[1] * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * lossbd[1])
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    return nothing
end

function jacc_kernel_geomrecur_1!(__jacc_i, i_n, loss, u)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function initstacks_geomrecur_b_jacc(i_n)
    u_stack = JACC.zeros(Float64, div(i_n - 2, 1) + 1)
    return u_stack
end

function geomrecur_hv_jacc(loss, lossb, u, ub, c, cb, i_n, lossd, lossbd, ud, ubd, cd, cbd, u_stack)
    u_stack_d = JACC.zeros(Float64, div(i_n - 2, 1) + 1)
    for i_seq_x = 2:i_n
        u_stack_d[(i_seq_x - 2) + 1] = ud[i_seq_x]
        u_stack[(i_seq_x - 2) + 1] = u[i_seq_x]
        ud[i_seq_x] = u[i_seq_x - 1] * cd + c * ud[i_seq_x - 1]
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_geomrecur_hv_1!(i_n, loss, lossd, u, ud)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_geomrecur_hv_2!(i_n, lossb, lossbd, u, ub, ubd, ud)
    for i_seq_x = i_n:-1:2
        ud[i_seq_x] = u_stack_d[(i_seq_x - 2) + 1]
        u[i_seq_x] = u_stack[(i_seq_x - 2) + 1]
        cbd = cbd + (ub[i_seq_x] * ud[i_seq_x - 1] + u[i_seq_x - 1] * ubd[i_seq_x])
        cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
        ubd[i_seq_x - 1] = ubd[i_seq_x - 1] + (ub[i_seq_x] * cd + c * ubd[i_seq_x])
        ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
        ubd[i_seq_x] = 0.0
        ub[i_seq_x] = 0.0
    end
    return (cb, cbd)
end

function geomrecur_jacc(loss, u, c, i_n)
    for i_seq_x = 2:i_n
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_geomrecur_1!(i_n, loss, u)
    return nothing
end
