# entry_empty(x, u, i_npass, i_w0, out)
#
# Guard kernel for norm_first_touch's literal-bounds requirement. The
# inner loop retires to a non-positive width, so on later passes it runs
# zero times and `s` keeps the value it carried in -- which the statement
# below then reads. A nested write only kills the entry value when the
# loop is PROVEN to run, which a runtime bound never is; accepting it
# lets norm_insert_dead_entry_resets zero a live value.
#
# Note this needs the window to actually reach zero: with a bound the
# baseline generator can only draw positive, the inner loop always runs
# and the bug hides. Retiring by three per pass reaches it on every draw.
#
# x: input array of length >= i_w0
# u: input array of the same length
# i_npass: number of passes
# i_w0: initial window width
# out: output array of length i_npass, updated in place
function entry_empty(x, u, i_npass, i_w0, out)
    s = x[1] * x[1]
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            s = x[i_j] * u[i_j]
        end
        # reads the carried value on any pass where the window was empty
        out[i_p] = out[i_p] + s * s
        w = w - 3
    end
end
