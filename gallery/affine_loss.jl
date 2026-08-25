# affine_loss(loss, u, a, b, v, i_n)
#
# An elementwise affine map v = a*u + b, followed by a sum-of-squares
# loss over v.
#
# loss: length-1 output array; loss[1] receives the sum of v[i]^2
# u: input field, length i_n
# a: elementwise scale, length i_n
# b: elementwise shift, length i_n
# v: scratch/output array for a*u + b, length i_n
# i_n: number of elements
function affine_loss(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    return nothing
end