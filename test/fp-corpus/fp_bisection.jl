# fp_bisection(loss, u, f, a, i_n)
#
# A halving search for the root of u = tanh(f)*tanh(a), rather than a
# contraction.
#
# Taping this gives a gradient of ZERO. The parameters enter ONLY through
# the branch condition, never through the arithmetic, so the program is
# piecewise constant: for nearby inputs it returns bit-identical values and
# the derivative OF THE PROGRAM is rightly 0. The derivative of the MAP it
# approximates is not, since u* = tanh(f)*tanh(a).
#
# This is the witness that the implicit method is sometimes the only correct
# route rather than a speed optimization, and it is the one subject where
# the taped comparison must NOT agree.
#
# Two earlier versions of this kernel failed to witness anything, and both
# failures are worth recording.
#
# The first used a fixed starting width of 1.0 and a root of f/a. With
# random baselines the root fell outside the bracket, the search ran to an
# endpoint, and finite differences returned 0 as well. The two agreed and
# the kernel proved nothing.
#
# The second derived the width from the inputs to bracket the root. That put
# the parameters back into the ARITHMETIC, so the taped gradient became
# nonzero and merely wrong, which is a different failure and no better.
#
# tanh bounds the root inside (-1, 1), so a fixed width of 2.0 from a fixed
# start always brackets it, and the parameters stay out of the arithmetic.
#
# loss: length-1 output array, accumulated in place
# u: the state, length i_n
# f: target value, a parameter, length i_n
# a: scale, a parameter, length i_n
# i_n: length
function fp_bisection(loss, u, f, a, i_n)
    w = 2.0
    for i_b = 1:i_n
        u[i_b] = 0.0
    end
    while w > 1.0e-12
        for i_k = 1:i_n
            if u[i_k] > tanh(f[i_k]) * tanh(a[i_k])
                u[i_k] = u[i_k] - w
            else
                u[i_k] = u[i_k] + w
            end
        end
        w = 0.5 * w
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
