function initstacks_entry_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return (branch_stack, s_stack)
end

function entry_branch_b(x, xb, y, yb, flag, flagb, i_n, out, outb, branch_stack, s_stack)
    s = 0.0
    sb = 0.0
    s = x[1] * y[1]
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            branch_stack[(i_i - 1) + 1] = 1
            s_stack[(i_i - 1) + 1] = s
            s = x[i_i] * y[i_i]
        else
            branch_stack[(i_i - 1) + 1] = 0
        end
        out[i_i] = out[i_i] + s * s
        s_stack[max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)] = s
    end
    s_stack[(max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1] = s
    s = s_stack[(max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1]
    for i_i = i_n:-1:1
        s = s_stack[max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)]
        sb = sb + s * outb[i_i]
        sb = sb + s * outb[i_i]
        __branch = branch_stack[(i_i - 1) + 1]
        if __branch == 1
            s = s_stack[(i_i - 1) + 1]
            xb[i_i] = xb[i_i] + y[i_i] * sb
            yb[i_i] = yb[i_i] + x[i_i] * sb
            sb = 0.0
        end
    end
    xb[1] = xb[1] + y[1] * sb
    yb[1] = yb[1] + x[1] * sb
    sb = 0.0
    return nothing
end

function entry_branch(x, y, flag, i_n, out)
    s = x[1] * y[1]
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            s = x[i_i] * y[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
end
