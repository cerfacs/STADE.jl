# remat_shapes(x, w, y, z, u, loss, s, i_n, i_nc)
#
# Six tile loops reassigning the operand scalar `wv`, each shaped so that a different part of the rematerialized-restore rule decides
# whether the pre-write save of `wv` can be replaced by recomputing an earlier load. Loop 1 is the only one that may: its old `wv`
# is a pure load of a read-only array through an int local defined once. The others each break one condition and must keep the save:
# 2 a self-referential write between loads (the load after it has no pure predecessor), 3 an array the kernel writes, 4 a nested loop rewriting `wv` between the loads, 5 an
# earlier load that sits inside an `if`, 6 a float local standing in the load's operands. Every loop reads `wv` nonlinearly, so a
# restored value that is wrong shows up in the gradient. Loop 7 reads a float scalar argument, a read-only operand, and may remat. Loop 8 rewrites the array it loads from between the
# two loads (kept: re-reading would see the new value). Loop 9 writes it BEFORE the loads but inside the loop around them, so later
# iterations change what a backward re-read sees (kept).
#
# x: input, length i_n + i_nc + 1
# w: weights, length 2 * i_nc + 2
# y: loop 1 output, read by loop 3; length i_n
# u: read and rewritten by loops 8 and 9 and used nowhere else, so its old values are never snapshotted; length i_n + 1
# z: output of loops 2..9, length 8 * i_n
# loss: length-1 output array, accumulated in place
# s: scalar argument
# i_n, i_nc: tile count and tap-block count
function remat_shapes(x, w, y, z, u, loss, s, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            base = (ci - 1) * 2
            wv = w[base + 1]
            a = a + wv * wv * x[t + ci - 1]
            wv = w[base + 2]
            a = a + wv * wv * x[t + ci]
        end
        y[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci]
            a = a + wv * wv * x[t + ci - 1]
            wv = wv * 0.5
            a = a + wv * wv * x[t + ci]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = y[t]
            a = a + wv * wv * x[t + ci - 1]
            wv = y[t] * 0.5
            a = a + wv * wv * x[t + ci]
        end
        z[i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci]
            for j = 1:2
                wv = wv * w[ci + j]
            end
            a = a + wv * wv * x[t + ci - 1]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[2 * i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = 0.0
            if x[t + ci - 1] > 0.0
                wv = w[ci]
            end
            a = a + wv * wv * x[t + ci - 1]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[3 * i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            sc = x[ci] * 2.0
            wv = w[ci] * sc
            a = a + wv * wv * x[t + ci - 1]
            wv = w[ci + 1] * sc
            a = a + wv * wv * x[t + ci]
        end
        z[4 * i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci] * s
            a = a + wv * wv * x[t + ci - 1]
            wv = w[ci + 1] * s
            a = a + wv * wv * x[t + ci]
        end
        z[5 * i_n + t] = a
    end
    for t = 1:i_n
        wv = u[t]
        u[t] = u[t] * 0.5 + x[t]
        a = wv * wv * x[t]
        wv = u[t]
        a = a + wv * wv * x[t + 1]
        z[6 * i_n + t] = a
    end
    for t = 1:i_n
        u[1] = x[t] * 0.5 + x[t + 1]
        wv = u[1]
        a = wv * wv * x[t]
        wv = u[1] * 2.0
        a = a + wv * wv * x[t + 1]
        z[7 * i_n + t] = a
    end
    for m = 1:8 * i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
