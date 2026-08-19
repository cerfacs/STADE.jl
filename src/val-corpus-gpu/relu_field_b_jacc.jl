import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_relu_field_b_1!(__jacc_i, branch_stack, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    if u[i_x] > 0.0
        branch_stack[(i_x - 1) + 1] = 1
        v[i_x] = u[i_x] ^ 2
    else
        branch_stack[(i_x - 1) + 1] = 0
        v[i_x] = 0.0
    end
    return nothing
end

function jacc_kernel_relu_field_b_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_relu_field_b_3!(__jacc_i, i_n, lossb, vb)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    vb[i_seq_x] = vb[i_seq_x] + lossb[1]
    return nothing
end

function jacc_kernel_relu_field_b_4!(__jacc_i, branch_stack, i_n, u, ub, vb)
    i_x = i_n + (__jacc_i - 1) * -1
    __branch = branch_stack[(i_x - 1) + 1]
    if __branch == 1
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
        vb[i_x] = 0.0
    else
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

function jacc_kernel_relu_field_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function initstacks_relu_field_b_jacc(i_n)
    branch_stack = JACC.zeros(Int64, div(i_n - 1, 1) + 1)
    return branch_stack
end

function relu_field_b_jacc(loss, lossb, u, ub, v, vb, i_n, branch_stack)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_b_1!(branch_stack, i_n, u, v)
    __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, v)->v[i_seq_x]))(v))
    JACC.@parallel_for range = 1 jacc_kernel_relu_field_b_2!(loss, __jgen_redval_2)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_relu_field_b_3!(i_n, lossb, vb)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_relu_field_b_4!(branch_stack, i_n, u, ub, vb)
    return nothing
end

function relu_field_jacc(loss, u, v, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_relu_field_1!(i_n, u, v)
    __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, v)->v[i_seq_x]))(v))
    JACC.@parallel_for range = 1 jacc_kernel_relu_field_2!(loss, __jgen_redval_2)
    return nothing
end
