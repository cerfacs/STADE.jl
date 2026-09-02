# cse_branch(out, u, v, i_flag, i_n)
#
# Coverage kernel for emit_cse_block's run boundary at an `:if`.
#
# Both arms compute the same subexpression from the same inputs, and the
# statement after the branch computes it a third time. A temporary bound
# inside one arm must not be reused in the sibling arm or after the branch:
# emit_cse_block ends the run at the `:if` and processes each arm on its
# own, so each gets its own name and the code after the branch gets a
# third. Reusing one across arms would read a name that was never assigned
# on the path actually taken.
#
# Each arm also repeats its expression twice internally, so there is a real
# binding to make inside each arm rather than an empty one.
#
# out: output array of length i_n, filled in place
# u: input array of length i_n
# v: input array of length i_n
# i_flag: selector array of length i_n
# i_n: array length
function cse_branch(out, u, v, i_flag, i_n)
    for i_r = 1:i_n
        s = 0.0
        if i_flag[i_r] > 0
            # u*v appears twice in this arm
            s = u[i_r] * v[i_r] + u[i_r] * v[i_r] * v[i_r]
        else
            # the same expression, but on the path not taken above
            s = u[i_r] * v[i_r] - u[i_r] * v[i_r] * u[i_r]
        end
        # and a third time, outside both arms
        out[i_r] = s * (u[i_r] * v[i_r])
    end
    return nothing
end
