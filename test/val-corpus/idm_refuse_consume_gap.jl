# idm_refuse_consume_gap(u, x, i_n)
#
# Must stay on the host. Iteration i_t reads u[i_t], writes the next
# element u[i_t + 1], and then overwrites u[i_t]. The read and the final
# write look like a consume pair, but a write to another element sits
# between them. Iteration i_t + 1 reads the element that iteration i_t
# wrote, so the loop carries a value between its iterations.
#
# u: state array of length i_n + 1, updated in place
# x: input array of length i_n
# i_n: number of iterations
function idm_refuse_consume_gap(u, x, i_n)
    for i_t = 1:i_n
        s = u[i_t]
        u[i_t + 1] = s * x[i_t] + 1.0
        u[i_t] = 0.0
    end
    return nothing
end
