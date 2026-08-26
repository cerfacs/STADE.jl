# bnd_readfirst(x, i_n, i_m, out)
#
# Guard kernel: a scalar with a block-boundary snapshot whose value is
# read at the TOP of the loop body, before the reset that kills it. The
# boundary snapshot supplies the end-of-body value, never the entering
# one, so the reset's site push is genuinely needed here and the
# end-of-body kill must not reach it. Program order is what separates
# this from cellscatter; a rule keyed on the variable rather than the
# site retires the wrong push (caught by stade_site_level_tbr_check).
#
# x: input array of length max(i_n, i_m)
# i_n: number of outer passes
# i_m: inner segment length
# out: output array of length i_n, updated in place
function bnd_readfirst(x, i_n, i_m, out)
    s = 1.0
    for i_i = 1:i_n
        # reads the value s carries in from the previous pass
        out[i_i] = out[i_i] + s * s
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_i] * x[i_j]
        end
    end
    return nothing
end
