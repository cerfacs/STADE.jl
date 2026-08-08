function initstacks_cond_loop_choice_b()
    return
end
function cond_loop_choice_b(loss, lossb, u, ub, v, vb, i_branch, i_n)
    if i_branch == 1
        for i_seq_x = i_n:-1:1
            ub[i_seq_x] = ub[i_seq_x] + 2 * u[i_seq_x] * lossb[1]
        end
    else
        for i_seq_x = i_n:-1:1
            vb[i_seq_x] = vb[i_seq_x] + 2 * v[i_seq_x] * lossb[1]
        end
    end
    return 
end
function cond_loop_choice(loss, u, v, i_branch, i_n)
    if i_branch == 1
        for i_seq_x = 1:i_n
            loss[1] = loss[1] + u[i_seq_x] ^ 2
        end
    else
        for i_seq_x = 1:i_n
            loss[1] = loss[1] + v[i_seq_x] ^ 2
        end
    end
end
