# relax(u, v, w, n, i_nouter, i_ninner)
#
# Two nested sequential loops (an outer time-step sweep, an inner
# relaxation sub-iteration sweep) wrapping a single iteration-
# independent spatial loop. The inner loop's own scan index
# (i_seq_inner) is read inside the independent loop's body without
# being assigned there, so it must come through as a free variable
# passed into the generated kernel/launch call, not be treated as a
# local temporary.
#
# u, v, w: arrays of length n
# n: length of u/v/w
# i_nouter: number of outer sequential steps
# i_ninner: number of inner sequential sub-iterations per outer step
function relax(u, v, w, n, i_nouter, i_ninner)
    for i_seq_outer = 1:i_nouter
        for i_seq_inner = 1:i_ninner
            for i_x = 1:n
                v[i_x] = v[i_x] + u[i_x] / (1.0 + i_seq_inner)
            end
        end
        for i_x = 1:n
            w[i_x] = w[i_x] + v[i_x] * i_seq_outer
        end
    end
    return nothing
end