import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_relu_field_1!(__jacc_i, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    if u[i_x] > 0.0
        v[i_x] = u[i_x] ^ 2
    else
        v[i_x] = 0.0
    end
    return nothing
end

function initstacks_relu_field_b_jacc()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function relu_field_hv_jacc(loss, lossb, u, ub, v, vb, i_n, lossd, lossbd, ud, ubd, vd, vbd, branch_stack)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            push!(branch_stack, 1)
            vd[i_x] = (2 * u[i_x]) * ud[i_x]
            v[i_x] = u[i_x] ^ 2
        else
            push!(branch_stack, 0)
            vd[i_x] = 0.0
            v[i_x] = 0.0
        end
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        vbd[i_seq_x] = vbd[i_seq_x] + lossbd[1]
        vb[i_seq_x] = vb[i_seq_x] + lossb[1]
    end
    for i_x = i_n:-1:1
        __branch = pop!(branch_stack)
        if __branch == 1
            ubd[i_x] = ubd[i_x] + (vb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * vbd[i_x])
            ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
            vbd[i_x] = 0.0
            vb[i_x] = 0.0
        else
            vbd[i_x] = 0.0
            vb[i_x] = 0.0
        end
    end
    return nothing
end

function relu_field_jacc(loss, u, v, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_relu_field_1!, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x]
    end
    return nothing
end
