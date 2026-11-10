# Battle test, self-test: the oracle must see faults that are planted in fake plans (plan section 9).
# A test that cannot fail proves nothing.  No STADE generation runs here.
include(joinpath(@__DIR__, "validate_battle_guards.jl"))

function validate_battle_selftest()
    nb = Ref(0)
    chk(name, cond) = (cond || (nb[] += 1; println("NOT OK ", name)); cond)
    mk(host, kernels...) = (host = host, kernels = collect(kernels))
    vars = [:i, :j, :k]
    # 1 a thread loop and a serial inner loop
    p1 = mk(:(function h() end), :(function kk(); __tid = 1; i = 1 + (__tid - 1); for j = 1:m; y[i] = 1; end; end))
    o, _ = bt_observe(p1, vars, Symbol[])
    chk("K outer", o[:i] == Dict(:K => 1)); chk("T inner", o[:j] == Dict(:T => 1)); chk("X absent", o[:k] == Dict(:X => 1))
    # 2 a host loop
    p2 = mk(:(function h(); for i = 1:n; CUDA.@allowscalar y[i] = 1; end; end))
    o, al = bt_observe(p2, vars, Symbol[])
    chk("H host", o[:i] == Dict(:H => 1)); chk("allowscalar counted", al == 1)
    # 3 reduction twin
    p3 = mk(:(function h(); if n < 32768; for i = 1:n; s = s + x[i]; end; else; s = s + mapreduce(init = 0.0, (__mr_1,) -> __mr_1, +, view(x, 1:n)); end; end))
    o, _ = bt_observe(p3, vars, Symbol[]); chk("R twin", o[:i] == Dict(:R => 1))
    # 4 lane-strided inner loop counts as thread-parallel
    p4 = mk(:(function h() end), :(function kk(); i = 1 + (__tid - 1); for j = 1 + __ly:32:m; y[i] = 1; end; end))
    o, _ = bt_observe(p4, vars, Symbol[]); chk("lane loop L", o[:j] == Dict(:L => 1))
    chk("L beside K is a match", bt_verdict(:K, Dict(:K => 1, :L => 1), :cuda) === :ok)
    # 5 whiles
    p5 = mk(:(function h(); while r > tol; r = r - 1; end; end), :(function kk(); i = 1 + (__tid - 1); while q > tol; q = q - 1; end; end))
    o, _ = bt_observe(p5, [:i], [:r, :q]); chk("while H", o[:r] == Dict(:H => 1)); chk("while T", o[:q] == Dict(:T => 1))
    # 6 planted faults, as verdicts
    chk("false offload seen", bt_verdict(:H, Dict(:K => 1), :cuda) === :false_offload)
    chk("missed offload seen (H)", bt_verdict(:K, Dict(:H => 1), :jacc) === :missed_offload)
    chk("missed offload seen (T)", bt_verdict(:K, Dict(:T => 1), :cuda) === :missed_offload)
    chk("R is a match for K on both backends (v0.5.6)", bt_verdict(:K, Dict(:R => 1), :cuda) === :ok && bt_verdict(:K, Dict(:R => 1), :jacc) === :ok)
    chk("R with K is a match for K", bt_verdict(:K, Dict(:R => 1, :K => 1), :jacc) === :ok)
    chk("R is not a match for H", bt_verdict(:H, Dict(:R => 1), :cuda) !== :ok)
    chk("R with H is a missed offload for K", bt_verdict(:K, Dict(:R => 1, :H => 1), :jacc) === :missed_offload)
    chk("extra offload seen", bt_verdict(:T, Dict(:K => 1), :cuda) === :extra_offload)
    chk("absent seen", bt_verdict(:K, Dict(:X => 1), :cuda) === :absent)
    chk("plain match", bt_verdict(:T, Dict(:T => 2), :jacc) === :ok)
    # 7 a mixed observation is not a match
    chk("mixed is not ok", bt_verdict(:K, Dict(:K => 1, :H => 1), :cuda) !== :ok)
    # 8 the syntactic classifier on planted shapes
    cls(src) = bt_classify_subject(Meta.parseall(src).args |> a -> first(x for x in a if x isa Expr && x.head === :function))
    chk("recurrence is S", cls("function f(u, n)\n for a1 = 2:n\n u[a1] = 0.5 * u[a1 - 1]\n end\n return nothing\n end")[:a1] === :S)
    chk("own index is I", cls("function f(u, x, n)\n for a1 = 1:n\n u[a1] = x[a1]\n end\n return nothing\n end")[:a1] === :I)
    chk("scatter accumulate is I", cls("function f(v, x, idx, n)\n for a1 = 1:n\n j1 = idx[a1]\n v[j1] = v[j1] + x[a1]\n end\n return nothing\n end")[:a1] === :I)
    chk("carried scalar is S", cls("function f(x, y, n)\n s = 0.0\n for a1 = 1:n\n s = 0.5 * s + x[a1]\n y[a1] = s\n end\n return nothing\n end")[:a1] === :S)
    chk("sum scalar is I", cls("function f(x, o, n)\n s = 0.0\n for a1 = 1:n\n s = s + x[a1]\n end\n o[1] = s\n return nothing\n end")[:a1] === :I)
    chk("read of accumulated array is S", cls("function f(u, v, n, m)\n for a1 = 1:n\n for a2 = 1:m\n v[a2] = v[a2] + u[a2]\n end\n for a3 = 1:m\n u[a3] = u[a3] + v[a3]\n end\n end\n return nothing\n end")[:a1] === :S)
    chk("same element each iteration is S", cls("function f(u, x, n)\n for a1 = 1:n\n u[1] = x[a1]\n end\n return nothing\n end")[:a1] === :S)
    # 9 sabotage of the guard checks (plan section 9): each guard check must see its planted fault
    good_k = :(function kk(n); __tid = 1; if __tid > n; return nothing; end; y[__tid] = 1; return nothing; end)
    bad_k  = :(function kk(n); __tid = 1; y[__tid] = 1; return nothing; end)
    chk("G1 cuda guard seen", bt_cuda_guarded(good_k)); chk("G1 cuda missing guard seen", !bt_cuda_guarded(bad_k))
    good_h = :(function h(n); if n > 0; JACC.@parallel_for range = n kern!(a); end; end)
    bad_h  = :(function h(n); JACC.@parallel_for range = n kern!(a); end)
    chk("G1 jacc guarded launch accepted", isempty(bt_jacc_unguarded(good_h))); chk("G1 jacc unguarded launch seen", !isempty(bt_jacc_unguarded(bad_h)))
    chk("G2 banned call seen", :dot in bt_calls(:(function h(); s = dot(x, y); end)))
    chk("G2 clean code accepted", isempty(intersect(Set(bt_calls(:(function h(); s = x + y; end))), BT_BANNED)))
    chk("G3 hash differs for different text", bt_hash((host = :(a + 1), kernels = Expr[])) != bt_hash((host = :(a + 2), kernels = Expr[])))
    println(nb[] == 0 ? "battle selftest: all checks pass" : "battle selftest: $(nb[]) NOT OK")
    return nb[]
end
if abspath(PROGRAM_FILE) == @__FILE__
    validate_battle_selftest() == 0 || exit(1)
end
