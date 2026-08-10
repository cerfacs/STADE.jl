import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function initstacks_cond_loop_choice_b_amdgpu()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function cond_loop_choice_hv_amdgpu(loss, lossb, u, ub, v, vb, i_branch, i_n, lossd, lossbd, ud, ubd, vd, vbd, branch_stack)
    if i_branch == 1
        push!(branch_stack, 1)
        for i_seq_x = 1:i_n
            lossd[1] = lossd[1] + (2 * u[i_seq_x]) * ud[i_seq_x]
            loss[1] = loss[1] + u[i_seq_x] ^ 2
        end
    else
        push!(branch_stack, 0)
        for i_seq_x = 1:i_n
            lossd[1] = lossd[1] + (2 * v[i_seq_x]) * vd[i_seq_x]
            loss[1] = loss[1] + v[i_seq_x] ^ 2
        end
    end
    __branch = pop!(branch_stack)
    if __branch == 1
        for i_seq_x = i_n:-1:1
            ubd[i_seq_x] = ubd[i_seq_x] + (lossb[1] * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * lossbd[1])
            ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
        end
    else
        for i_seq_x = i_n:-1:1
            vbd[i_seq_x] = vbd[i_seq_x] + (lossb[1] * (2 * vd[i_seq_x]) + (2 * v[i_seq_x]) * lossbd[1])
            vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
        end
    end
    return nothing
end

function cond_loop_choice_amdgpu(loss, u, v, i_branch, i_n)
    if i_branch == 1
        for i_seq_x = 1:i_n
            loss[1] = loss[1] + u[i_seq_x] ^ 2
        end
    else
        for i_seq_x = 1:i_n
            loss[1] = loss[1] + v[i_seq_x] ^ 2
        end
    end
    return nothing
end
