import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function initstacks_clamped_sumsq_b_metal()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function clamped_sumsq_b_metal(loss, lossb, u, ub, i_n, branch_stack)
    w = 0.0f0
    wb = 0.0f0
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0f0
            push!(branch_stack, 1)
            w = u[i_seq_x] ^ 2
        else
            push!(branch_stack, 0)
            w = 0.0f0
        end
        loss[1] = loss[1] + w
    end
    for i_seq_x = i_n:-1:1
        wb = wb + lossb[1]
        __branch = pop!(branch_stack)
        if __branch == 1
            ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * wb
            wb = 0.0f0
        else
            wb = 0.0f0
        end
    end
    return nothing
end

function clamped_sumsq_metal(loss, u, i_n)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0f0
            w = u[i_seq_x] ^ 2
        else
            w = 0.0f0
        end
        loss[1] = loss[1] + w
    end
    return nothing
end
