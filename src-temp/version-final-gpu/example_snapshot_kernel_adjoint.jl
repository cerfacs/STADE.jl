function initstacks_sq_test_b()
    v_stack = Vector{Float64}()
    return v_stack
end

function sq_test_b(u, ub, v, vb, n, v_stack)
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

function sq_test(u, v, n)
    #= none:10 =#
    #= none:11 =#
    for i_x = 1:n
        #= none:12 =#
        v[i_x] = u[i_x]
        #= none:13 =#
        v[i_x] = v[i_x] * v[i_x]
        #= none:14 =#
    end
    #= none:15 =#
    return nothing
end
