import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_sq_test_1!(n, u, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = u[i_x]
    v[i_x] = v[i_x] * v[i_x]
    return nothing
end

function initstacks_sq_test_b_metal()
    v_stack = Vector{Float64}()
    return v_stack
end

function sq_test_b_metal(u, ub, v, vb, n, v_stack)
    for i_x = 1:n
        v[i_x] = u[i_x]
        push!(v_stack, v[i_x])
        v[i_x] = v[i_x] * v[i_x]
    end
    for i_x = n:-1:1
        v[i_x] = pop!(v_stack)
        vb[i_x] = v[i_x] * vb[i_x] + v[i_x] * vb[i_x]
        ub[i_x] = ub[i_x] + vb[i_x]
        vb[i_x] = 0.0f0
    end
    return nothing
end

function sq_test_metal(u, v, n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(n - 1, 1) + 1, nthread_per_block) metal_kernel_sq_test_1!(n, u, v)
    return nothing
end
