# remat_cross(x, w, y, u, z, loss, i_n, i_nc)
#
# Register-tiled loops whose operand scalars cross the iteration boundary: the first write of a scalar in a `ci` iteration saves the value the
# previous iteration left behind. When every read of the scalar sits in the same body after that write, nobody can observe the old value, and
# the backward sweep instead rebuilds the END-of-iteration value from the pure rhs of the last write at the head of each reverse iteration. Each
# loop uses scalars of its own, because the read rule looks at every read of a scalar in the kernel; each one is a variation on the rule:
#
#   loop 1  one load per iteration (xv) and two (wv, wu)         -> cross-iteration remat, the second wv by the same-block rule
#   loop 2  a chain of body ints (b1 = ci * 2, b2 = b1 + 1)       -> remat, the ints recomputed up front by the sweep
#   loop 3  step 2 from 2, and a reversed loop from i_nc to 1     -> remat
#   loop 4  a read of the scalar before its first write           -> kept (the previous iteration's value is read)
#   loop 5  the scalar is read before the loop (live entry value) -> kept (the entry restore is needed)
#   loop 6  a nested loop rewrites the scalar after its first write -> kept
#   loop 7  the loaded array is written in the loop               -> kept
#
# cross_readbefore, cross_entry, cross_after, cross_ifwrite and cross_nested isolate one refusal each, so that sabotaging one guard cannot be
# masked by another.
#
# x: input, length i_n + i_nc + 2
# w: weights, length 4 * i_nc + 4
# y: output of loop 1, length i_n
# u: read and written by loop 7 only, length i_nc + 2
# z: output of loops 2..7, length 6 * i_n
# loss: length-1 output array, accumulated in place
# i_n, i_nc: tile count and tap-block count
function remat_cross(x, w, y, u, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[t + ci - 1]
            wv = w[2 * ci - 1]
            a = a + xv * wv * wv
            wv = w[2 * ci]
            a = a + xv * wv * wv
        end
        y[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            b1 = ci * 2
            b2 = b1 + 1
            xv2 = x[t + ci - 1] * 2.0
            wv2 = w[b2]
            a = a + xv2 * wv2 * wv2
        end
        z[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 2:2:i_nc
            xv3 = x[t + ci - 1]
            a = a + xv3 * xv3 * w[ci]
        end
        for cj = i_nc:-1:1
            xv4 = x[t + cj]
            a = a + xv4 * xv4 * w[cj + 1]
        end
        z[i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        xv5 = x[t] * 0.5
        for ci = 1:i_nc
            a = a + xv5 * xv5
            xv5 = x[t + ci - 1] * w[ci]
            a = a + xv5 * w[ci + 1]
        end
        z[2 * i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        xv6 = x[t] * w[1]
        a = xv6 * xv6
        for ci = 1:i_nc
            xv6 = x[t + ci] * w[ci]
            a = a + xv6 * xv6 * xv6
        end
        z[3 * i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv7 = x[t + ci - 1]
            a = a + xv7 * xv7
            for cj = 1:2
                xv7 = xv7 * w[ci + cj]
            end
            a = a + xv7 * xv7
        end
        z[4 * i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv8 = u[ci]
            a = a + xv8 * xv8 * x[t + ci - 1]
            u[ci + 1] = x[t + ci] * 0.5
        end
        z[5 * i_n + t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + y[m] * y[m]
    end
    for m = 1:6 * i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
