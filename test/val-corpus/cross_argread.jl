# cross_argread(x, w, loss, s, i_nc)
#
# The scalar argument `s` is read before the loop and then overwritten by the loop's first write each iteration. Nothing inside the loop reads
# it before that write, so only the coverage rule -- every read follows a write in the same loop body -- sees the read outside.
#
# x: length i_nc, w: length i_nc, loss: length 1, s: scalar argument
function cross_argread(x, w, loss, s, i_nc)
    a = s * s
    for ci = 1:i_nc
        s = x[ci] * 2.0
        a = a + s * s * w[ci]
    end
    loss[1] = loss[1] + a * a
    return nothing
end
