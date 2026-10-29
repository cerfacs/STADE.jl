# while_carried_cond(y, z, x, i_n)
#
# A float carried from one pass of the for loop to the next is read by the condition of a while loop and then assigned in both arms of a branch.
# The while may not run, so it proves no write; and its condition reads the carried value, so the entry value of s is live and must not be reset.
#
# y: output, length i_n, accumulates the carried value once per while pass
# z: output, length i_n, receives the square of the new carried value
# x: input, length i_n
# i_n: length
function while_carried_cond(y, z, x, i_n)
    s = 1.0
    for i_i = 1:i_n
        i_k = 0
        # the condition reads the carried value
        while s > 0.3 && i_k < 3
            y[i_i] = y[i_i] + s
            s = s * 0.5
            i_k = i_k + 1
        end
        if x[i_i] > 0.0
            s = x[i_i]
        else
            s = 2.0 * x[i_i]
        end
        z[i_i] = s * s
    end
    return nothing
end
