import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_relu_field_hv_1!(__jacc_i, branch_stack, i_n, u, ud, v, vd)
    i_x = 1 + (__jacc_i - 1)
    if u[i_x] > 0.0
        branch_stack[(i_x - 1) + 1] = 1
        vd[i_x] = (2 * u[i_x]) * ud[i_x]
        v[i_x] = u[i_x] ^ 2
    else
        branch_stack[(i_x - 1) + 1] = 0
        vd[i_x] = 0.0
        v[i_x] = 0.0
    end
    return nothing
end

function jacc_kernel_relu_field_hv_2!(__jacc_i, i_n, loss, lossd, v, vd)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += vd[i_seq_x]
    Atomix.@atomic loss[1] += v[i_seq_x]
    return nothing
end

function jacc_kernel_relu_field_hv_3!(__jacc_i, i_n, lossb, lossbd, vb, vbd)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    vbd[i_seq_x] = vbd[i_seq_x] + lossbd[1]
    vb[i_seq_x] = vb[i_seq_x] + lossb[1]
    return nothing
end

function jacc_kernel_relu_field_hv_4!(__jacc_i, branch_stack, i_n, u, ub, ubd, ud, vb, vbd)
    i_x = i_n + (__jacc_i - 1) * -1
    __branch = branch_stack[(i_x - 1) + 1]
    if __branch == 1
        ubd[i_x] = ubd[i_x] + (vb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * vbd[i_x])
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
    else
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
    end
    return nothing
end

function jacc_kernel_relu_field_1!(__jacc_i, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    if u[i_x] > 0.0
        v[i_x] = u[i_x] ^ 2
    else
        v[i_x] = 0.0
    end
    return nothing
end

function jacc_kernel_relu_field_2!(__jacc_i, i_n, loss, v)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += v[i_seq_x]
    return nothing
end

function initstacks_relu_field_b_jacc(i_n)
    branch_stack = JACC.zeros(Int64, div(i_n - 1, 1) + 1)
    return branch_stack
end

function relu_field_hv_jacc(loss, lossb, u, ub, v, vb, i_n, lossd, lossbd, ud, ubd, vd, vbd, branch_stack)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_hv_1!(branch_stack, i_n, u, ud, v, vd)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_hv_2!(i_n, loss, lossd, v, vd)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_relu_field_hv_3!(i_n, lossb, lossbd, vb, vbd)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_relu_field_hv_4!(branch_stack, i_n, u, ub, ubd, ud, vb, vbd)
    return nothing
end

function relu_field_jacc(loss, u, v, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_1!(i_n, u, v)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_2!(i_n, loss, v)
    return nothing
end
