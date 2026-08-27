# fixed_sweeps(y, x, i_n)
#
# Three damped Jacobi passes over a vector, sweep count fixed at 3.
#
# The nest is deliberately NOT fused: the outer loop carries a value, as
# each pass reads the y the previous pass wrote (rule 1). It is the
# corpus witness for cgen_'s split fallback. cgen_reduction_only_loop
# calls the outer loop eligible, because y is read and written at the
# same index and that is what a commutative accumulation looks like to
# its array test. But `y[i] = 0.5 * y[i] + x[i]` is not additive, so
# three threads running it concurrently race, and cgen_device_assign
# refuses to emit the kernel. Splitting the INNER loop is safe and is
# what the fallback in cgen_body/jgen_body picks instead. Without it,
# converting this kernel fails outright.
#
# y: array of length i_n, relaxed in place
# x: source term array of length i_n
# i_n: length of y and x
function fixed_sweeps(y, x, i_n)
    for i_k = 1:3
        for i_i = 1:i_n
            y[i_i] = 0.5 * y[i_i] + x[i_i]
        end
    end
    return nothing
end
