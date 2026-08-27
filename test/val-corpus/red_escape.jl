# red_escape(x, y, i_n, i_m, out, acc)
#
# A running total accumulated across BOTH loops of a nest, then consumed
# nonlinearly by a statement sitting after the inner loop.
#
# Guard kernel for two things at once. First, fuse_ii_loops' reduction
# gate: `s` is the inner loop's accumulator, excluded from push, and
# every kind carrying a reduction defers its adjoint to the backward
# position -- so the `out[i_i] + s * s` read, which sits after the loop,
# is reversed BEFORE the loop's backward code with nothing having
# restored s's primal. Second, the top-level-write condition in
# snap_boundary_kill_vars: that read WOULD be served by the enclosing
# body's block-boundary snapshot, and s not being written at the top
# level of that body is precisely why no such snapshot exists here.
# cellscatter is the same shape with the top-level `auxu = 0.0` present,
# and is safe for exactly that reason -- the two kernels differ by one
# statement and must classify differently.
#
# Initializing s outside the outer loop is the point, not an oversight:
# moving it inside makes the snapshot exist and the hazard disappear.
#
# x: input array of length i_m
# y: input array of length i_n
# i_n: number of outer passes
# i_m: inner segment length
# out: output array of length i_n, updated in place
# acc: length-1 output array holding the trailing term
function red_escape(x, y, i_n, i_m, out, acc)
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            s = s + x[i_j] * y[i_i]
        end
        # nonlinear in s, and reversed before the inner loop's backward code
        out[i_i] = out[i_i] + s * s
    end
    s = x[1] * y[1]
    acc[1] = acc[1] + s * s
    return nothing
end
