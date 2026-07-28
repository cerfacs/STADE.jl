# ============================================================
# check_parse_m1.jl -- quick local verification for parse_kernel /
# shape_infer against the M1 ("straight-line") corpus primals.
#
# Not part of STADE.jl itself -- a throwaway harness. Run with:
#     julia check_parse_m1.jl
# from a directory containing STADE.jl and all_b.jl.
#
# For each M1 kernel it re-parses the *exact* primal Expr (pulled
# out of all_b.jl by function name, via lowering) through
# parse_kernel and prints the inferred kinds + independents/
# dependents, so you can eyeball them against the corpus by hand
# before wiring up the rest of the pipeline.
# ============================================================

include("STADE.jl")

# pull `function name(...) ... end` back out of all_b.jl as a raw
# Expr, the same way io_read_kernel would from a single-function file
function grab_kernel_expr(path::String, name::Symbol)
    parsed = Meta.parseall(read(path, String))
    for e in parsed.args
        if e isa Expr && e.head == :function
            sig = e.args[1]
            if sig isa Expr && sig.head == :call && sig.args[1] == name
                return e
            end
        end
    end
    error("no `function $name(...)` found in $path")
end

m1_names = [
    :dotprod, :quadloss, :bilinear, :matvec_loss, :affine_loss,
    :normcomp, :weightedsumsq, :sumsq_shifted, :two_field_loss, :pipeline,
]

println("--- M1 parse_kernel/shape_infer check ---")
for name in m1_names
    expr = grab_kernel_expr("all_b.jl", name)
    kernel = parse_kernel(expr)
    println(rpad(String(name), 16), " args=", kernel.sig.args)
    println(" "^16, " kinds=", kernel.sig.kinds)
    println(" "^16, " independents=", kernel.sig.independents,
            " dependents=", kernel.sig.dependents)
    println(" "^16, " #statements(top level)=", length(kernel.body))
end
println("--- all M1 kernels parsed without a hard error ---")

# mg_vcycle (M4) needs the stepped-range (`hi:-1:lo`) support -- check
# the up-sweep loop actually picked up a non-default step.
println()
println("--- mg_vcycle (M4) parse_kernel/shape_infer check ---")
mg_expr = grab_kernel_expr("all_b.jl", :mg_vcycle)
mg_kernel = parse_kernel(mg_expr)
println("args=", mg_kernel.sig.args)
println("kinds=", mg_kernel.sig.kinds)
println("independents=", mg_kernel.sig.independents,
        " dependents=", mg_kernel.sig.dependents)
up_sweep = mg_kernel.body[end]
println("last top-level statement kind=", up_sweep.kind,
        " var=", up_sweep.var, " lo=", up_sweep.lo,
        " step=", up_sweep.step, " hi=", up_sweep.hi)
println("--- mg_vcycle parsed without a hard error ---")