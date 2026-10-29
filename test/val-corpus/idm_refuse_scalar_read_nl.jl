# idm_refuse_scalar_read_nl(v, z, x, i_n)
#
# Must stay on the host, and the scalar gate must be the reason. Iteration i_t accumulates into v[i_t] and
# v[i_t + 1], then reads v[i_t] into a scalar and squares it. The neighbouring
# iteration also adds to v[i_t], so the scalar depends on the iteration
# order.
#
# v: accumulator array of length i_n + 1, updated in place
# z: output array of length i_n
# x: input array of length i_n
# i_n: number of iterations
function idm_refuse_scalar_read_nl(v, z, x, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t]
        v[i_t + 1] = v[i_t + 1] + x[i_t] * 2.0
        # a read of the accumulator, used nonlinearly so its value is saved
        s = v[i_t]
        z[i_t] = s * s * 3.0
    end
    return nothing
end
