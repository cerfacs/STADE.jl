import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_dotprod_hv_1!(__jacc_i, i_n, loss, lossd, u, ud, v, vd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += v[i_seq_x] * ud[i_seq_x] + u[i_seq_x] * vd[i_seq_x]
    Atomix.@atomic loss[1] += u[i_seq_x] * v[i_seq_x]
    return nothing
end

function jacc_kernel_dotprod_hv_2!(__jacc_i, i_n, lossb, lossbd, u, ub, ubd, ud, v, vb, vbd, vd)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    ubd[i_seq_x] = ubd[i_seq_x] + (lossb[1] * vd[i_seq_x] + v[i_seq_x] * lossbd[1])
    ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
    vbd[i_seq_x] = vbd[i_seq_x] + (lossb[1] * ud[i_seq_x] + u[i_seq_x] * lossbd[1])
    vb[i_seq_x] = vb[i_seq_x] + u[i_seq_x] * lossb[1]
    return nothing
end

function initstacks_dotprod_b_jacc()
    return nothing
end

function dotprod_hv_jacc(loss, lossb, u, ub, v, vb, i_n, lossd, lossbd, ud, ubd, vd, vbd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_dotprod_hv_1!(i_n, loss, lossd, u, ud, v, vd)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_dotprod_hv_2!(i_n, lossb, lossbd, u, ub, ubd, ud, v, vb, vbd, vd)
    return nothing
end

function dotprod_jacc(loss, u, v, i_n)
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u, v)->u[i_seq_x] * v[i_seq_x]))(u, v))
    return nothing
end
