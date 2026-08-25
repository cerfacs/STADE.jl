function richardson_substep_d(y_init, y_initd, out, outd, a_coef, a_coefd, dt_stage, dt_staged, num_stages)
    nsubd = 0.0
    nsub = 1
    for i_stage = 1:num_stages
        hd = (1.0 / nsub) * dt_staged
        h = dt_stage / nsub
        yd = y_initd
        y = y_init
        for i_sub = 1:nsub
            yd = yd + -((((a_coef * y) * hd + (h * y) * a_coefd) + (h * a_coef) * yd))
            y = y - h * a_coef * y
        end
        outd[i_stage] = yd
        out[i_stage] = y
        nsubd = 0.0
        nsub = nsub * 2
    end
    return nothing
end

function richardson_substep(y_init, out, a_coef, dt_stage, num_stages)
    nsub = 1
    for i_stage = 1:num_stages
        h = dt_stage / nsub
        y = y_init
        for i_sub = 1:nsub
            y = y - h * a_coef * y
        end
        out[i_stage] = y
        nsub = nsub * 2
    end
end
