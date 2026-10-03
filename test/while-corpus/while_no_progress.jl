# while_no_progress(loss, u, v, i_n)
#
# NEGATIVE. The condition reads only `i_n`, which the body never writes, so
# the loop either runs forever or not at all. STADE cannot prove
# termination and does not try. It checks the far weaker property that the
# condition reads at least one variable the body writes, which catches this
# mistake at generation time rather than hanging a job.
function while_no_progress(loss, u, v, i_n)
    i_k = 1
    while i_n > 0
        v[i_k] = u[i_k]
    end
    loss[1] = loss[1] + v[1]
    return nothing
end
