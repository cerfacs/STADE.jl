# cross_mixed(x, w, loss, i_nc)
#
# Two top-level loops write the scalar `xv` first thing in the iteration and read it afterwards. In the first its rhs goes through a float local,
# so the end-of-iteration value cannot be rebuilt and its first-write save has to stay; the second could drop its own. The reverse sweep runs the
# second loop first and would leave `xv` altered for the first, whose last iteration relies on the value live after the forward sweep. All or
# nothing per scalar. The loops sit at the top level because inside an enclosing loop a block-boundary restore supplies that value anyway.
#
# x: length i_nc, w: length i_nc, loss: length 1
function cross_mixed(x, w, loss, i_nc)
    a = 0.0
    for ci = 1:i_nc
        tmp = x[ci] * w[ci]
        xv = tmp * 0.5
        a = a + xv * xv * w[ci]
    end
    b = 0.0
    for ci = 1:i_nc
        xv = x[ci]
        b = b + xv * xv * w[ci]
    end
    loss[1] = loss[1] + a * a + b * b
    return nothing
end
