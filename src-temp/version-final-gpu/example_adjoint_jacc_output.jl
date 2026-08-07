import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_sq_test_1!(__jacc_i, n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = u[i_x]
    v[i_x] = v[i_x] * v[i_x]
    return nothing
end

function initstacks_sq_test_b_jacc()
    v_stack = Vector{Float64}()
    return v_stack
end

function sq_test_b_jacc(u, ub, v, vb, n, v_stack)
    for i_x = 1:n
        v[i_x] = u[i_x]
        push!(v_stack, v[i_x])
        v[i_x] = v[i_x] * v[i_x]
    end
    for i_x = n:-1:1
        v[i_x] = pop!(v_stack)
        vb[i_x] = v[i_x] * vb[i_x] + v[i_x] * vb[i_x]
        ub[i_x] = ub[i_x] + vb[i_x]
        vb[i_x] = 0.0
    end
    return nothing
end

function sq_test_jacc(u, v, n)
    JACC.parallel_for(div(n - 1, 1) + 1, jacc_kernel_sq_test_1!, n, u, v)
    return nothing
end
