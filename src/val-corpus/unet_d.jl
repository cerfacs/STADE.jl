function unet_d(h, w, c_in, c1, b_e1a, b_e1ad, t_e1, t_e1d)
    khd = 0.0
    kh = 3
    kwd = 0.0
    kw = 3
    khkwd = 0.0
    khkw = kh * kw
    hwd = 0.0
    hw = h * w
    for idx = 1:c1 * hw
        idxm1d = 0.0
        idxm1 = idx - 1
        cod = 0.0
        co = div(idxm1, hw) + 1
        sd = 0.0
        s = 0.0
        for i_seq_k = 1:c_in * khkw
        end
        t_e1d[idx] = b_e1ad[co]
        t_e1[idx] = s + b_e1a[co]
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
