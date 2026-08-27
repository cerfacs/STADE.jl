function initstacks_entry_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return (branch_stack, s_stack)
end

function entry_branch_hv(x, xb, y, yb, flag, flagb, i_n, out, outb, xd, xbd, yd, ybd, flagd, flagbd, outd, outbd, branch_stack, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    sd = y[1] * xd[1] + x[1] * yd[1]
    s = x[1] * y[1]
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            branch_stack[(i_i - 1) + 1] = 1
            s_stack_d[(i_i - 1) + 1] = sd
            s_stack[(i_i - 1) + 1] = s
            sd = y[i_i] * xd[i_i] + x[i_i] * yd[i_i]
            s = x[i_i] * y[i_i]
        else
            branch_stack[(i_i - 1) + 1] = 0
        end
        outd[i_i] = outd[i_i] + (s * sd + s * sd)
        out[i_i] = out[i_i] + s * s
        s_stack_d[max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)] = sd
        s_stack[max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)] = s
    end
    s_stack_d[(max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1] = sd
    s_stack[(max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1] = s
    sd = s_stack_d[(max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1]
    s = s_stack[(max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1]
    for i_i = i_n:-1:1
        sd = s_stack_d[max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)]
        s = s_stack[max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)]
        sbd = sbd + (outb[i_i] * sd + s * outbd[i_i])
        sb = sb + s * outb[i_i]
        sbd = sbd + (outb[i_i] * sd + s * outbd[i_i])
        sb = sb + s * outb[i_i]
        __branch = branch_stack[(i_i - 1) + 1]
        if __branch == 1
            sd = s_stack_d[(i_i - 1) + 1]
            s = s_stack[(i_i - 1) + 1]
            xbd[i_i] = xbd[i_i] + (sb * yd[i_i] + y[i_i] * sbd)
            xb[i_i] = xb[i_i] + y[i_i] * sb
            ybd[i_i] = ybd[i_i] + (sb * xd[i_i] + x[i_i] * sbd)
            yb[i_i] = yb[i_i] + x[i_i] * sb
            sbd = 0.0
            sb = 0.0
        end
    end
    xbd[1] = xbd[1] + (sb * yd[1] + y[1] * sbd)
    xb[1] = xb[1] + y[1] * sb
    ybd[1] = ybd[1] + (sb * xd[1] + x[1] * sbd)
    yb[1] = yb[1] + x[1] * sb
    sbd = 0.0
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
