# nest_if(x, y, i_n)
#
# An `if` inside an `if`, inside a loop. With keep_push_pop = true (the default) the branch flags share one stack. Through 0.4.9 the
# flag of the outer `if` was pushed before the flag of the inner one, while the backward sweep pops the outer flag first, so it read
# the inner flag (wrong gradient, relative error about 1.9 on the default draw). No corpus kernel had an `if` inside an `if`.
# The conditions depend on the loop index, so every draw runs both arms of the inner `if` and skips the outer one once. Each arm is
# nonlinear, so the Hessian check has a signal.
#
# x: input array, length i_n
# y: output accumulator, length i_n, updated in place
function nest_if(x, y, i_n)
    for i = 1:i_n
        if i > 1
            if i > 2
                y[i] = y[i] + x[i] * x[i]
            else
                y[i] = y[i] + x[i] * x[i] * x[i]
            end
        end
    end
    return nothing
end
