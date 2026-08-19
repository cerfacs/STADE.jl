import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_dotprod_b_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_dotprod_b_2!(__jacc_i, i_n, lossb, u, ub, v, vb)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
    vb[i_seq_x] = vb[i_seq_x] + u[i_seq_x] * lossb[1]
    return nothing
end

function jacc_kernel_dotprod_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function initstacks_dotprod_b_jacc()
    return nothing
end

function dotprod_b_jacc(loss, lossb, u, ub, v, vb, i_n)
    __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u, v)->u[i_seq_x] * v[i_seq_x]))(u, v))
    JACC.@parallel_for range = 1 jacc_kernel_dotprod_b_1!(loss, __jgen_redval_1)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_dotprod_b_2!(i_n, lossb, u, ub, v, vb)
    return nothing
end

function dotprod_jacc(loss, u, v, i_n)
    __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u, v)->u[i_seq_x] * v[i_seq_x]))(u, v))
    JACC.@parallel_for range = 1 jacc_kernel_dotprod_1!(loss, __jgen_redval_1)
    return nothing
end
