function initstacks_unet_b()
    tripcount_stack = Vector{Int64}()
    return tripcount_stack
end

function unet_hv(h, w, c_in, c1, b_e1a, b_e1ab, t_e1, t_e1b, b_e1ad, b_e1abd, t_e1d, t_e1bd, tripcount_stack)
    sd = 0.0
    sbd = 0.0
    kh = 3
    kw = 3
    khkw = kh * kw
    hw = h * w
    push!(tripcount_stack, hw)
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        sd = 0.0
        s = 0.0
        push!(tripcount_stack, khkw)
        for i_seq_k = 1:c_in * khkw
        end
        t_e1d[idx] = sd + b_e1ad[co]
        t_e1[idx] = s + b_e1a[co]
    end
    kh = 3
    kw = 3
    khkw = kh * kw
    hw = h * w
    hw = pop!(tripcount_stack)
    for idx = c1 * hw:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        b_e1abd[co] = b_e1abd[co] + t_e1bd[idx]
        b_e1ab[co] = b_e1ab[co] + t_e1b[idx]
        t_e1bd[idx] = 0.0
        t_e1b[idx] = 0.0
        khkw = pop!(tripcount_stack)
        for i_seq_k = c_in * khkw:-1:1
        end
        sbd = 0.0
        sb = 0.0
    end
    return nothing
end

function unet(h, w, c_in, c1, b_e1a, t_e1)
    kh = 3
    kw = 3
    khkw = kh * kw
    hw = h * w
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        s = 0.0
        for i_seq_k = 1:c_in * khkw
        end
        t_e1[idx] = s + b_e1a[co]
    end
end
