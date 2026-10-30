# House rules 1 to 4 of claude/skill-stade-dev.md and the placement of public functions, checked on the text of src/STADE.jl.
#   rule 1: one `module STADE ... end` that wraps the file and no other module
#   rule 2 and 3: no struct, @enum or other type definition and no global const or global at the top level
#   rule 4: every top-level function carries the prefix of a stage
#   placement: every `stade_` function sits under a `# ==== stade_*` banner, and every `norm_`, `bgen_`, `inl_` and `jgen_` function under the banner of its own stage
#   julia test/validate_house_rules.jl

hr_prefixes() = ("parse_", "shape_", "der_", "emit_", "act_", "snap_", "lin_", "tgen_", "agen_", "hvp_", "cgen_", "jgen_", "ii_", "val_", "io_", "stade_", "inl_", "norm_", "bgen_")
hr_strict() = ("stade_", "norm_", "bgen_", "inl_", "jgen_")

# the name a top-level definition defines: `function f(`, `f(args) = ` or nothing
function hr_defname(l)
    m = match(r"^function\s+([A-Za-z_][A-Za-z0-9_!]*)", l)
    m === nothing && (m = match(r"^([A-Za-z_][A-Za-z0-9_!]*)\(.*\)\s*=[^=]", l))
    return m === nothing ? nothing : String(m.captures[1])
end

function hr_violations(text::AbstractString)
    lines = split(text, '\n')
    out = Tuple{Symbol,Int,String}[]
    pat = r"^(const|global|struct|mutable struct|abstract type|primitive type|@enum)\b"
    modules = [i for (i, l) in enumerate(lines) if occursin(r"^\s*module\s", l)]
    ends = [i for (i, l) in enumerate(lines) if l == "end # module STADE"]
    (length(modules) == 1 && occursin(r"^module STADE\s*$", lines[modules[1]]) && length(ends) == 1) || push!(out, (:rule1, isempty(modules) ? 0 : modules[1], "expected one `module STADE` and one `end # module STADE`, found $(length(modules)) modules"))
    banner = ""
    for (i, l) in enumerate(lines)
        m = match(r"^# =+ ([a-z]+)_\*", l)
        m !== nothing && (banner = String(m.captures[1]); continue)
        # a section with its own heading line: `# ii_* : ...`
        m = match(r"^# ([a-z]+)_\* :", l)
        m !== nothing && (banner = String(m.captures[1]); continue)
        occursin(pat, l) && push!(out, (:rule23, i, strip(l)))
        n = hr_defname(l)
        n === nothing && continue
        occursin(r"^[a-z]+_", n) || continue
        st = String(match(r"^([a-z]+_)", n).captures[1])
        st in hr_prefixes() || push!(out, (:rule4, i, n))
        if st in hr_strict() && st != "jgen_" && st != "stade_"
            banner == chop(st) || push!(out, (:place, i, "$(n) is under the banner `$(banner)_*`"))
        elseif st == "jgen_"
            banner == "jgen" || push!(out, (:place, i, "$(n) is under the banner `$(banner)_*`"))
        elseif st == "stade_"
            banner == "stade" || push!(out, (:place, i, "$(n) is under the banner `$(banner)_*`"))
        end
    end
    return out
end

function validate_house_rules()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    src = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
    v = hr_violations(src)
    for (k, label) in ((:rule1, "rule 1: one module that wraps the file"), (:rule23, "rules 2 and 3: no top-level const, global, struct or @enum"),
                       (:rule4, "rule 4: every function carries a stage prefix"), (:place, "placement: stade_, norm_, bgen_, inl_ and jgen_ functions under their own banner"))
        vk = filter(x -> x[1] == k, v)
        check(isempty(vk), "src/STADE.jl, " * label * (isempty(vk) ? "" : " (first: line $(vk[1][2]): $(first(vk[1][3], 70)))"))
    end
    # sabotage: the scan must see each kind of fault when it is injected; the text is the source with one change
    seen(k, text) = any(x -> x[1] == k, hr_violations(text))
    cases = [
        ("a global const", :rule23, src * "\nconst X_ = 1\n"),
        ("a struct", :rule23, src * "\nstruct S_\n    a::Int\nend\n"),
        ("a mutable struct", :rule23, src * "\nmutable struct M_\n    a::Int\nend\n"),
        ("an @enum", :rule23, src * "\n@enum E_ a_ b_\n"),
        ("a global", :rule23, src * "\nglobal G_ = 1\n"),
        ("an abstract type", :rule23, src * "\nabstract type A_ end\n"),
        ("a second module", :rule1, src * "\nmodule Other_\nend\n"),
        ("no module line", :rule1, replace(src, "module STADE\n" => "\n"; count = 1)),
        ("a function with another prefix", :rule4, src * "\nfunction zzz_f(x)\n    return x\nend\n"),
        ("a one-line function with another prefix", :rule4, src * "\nzzz_g(x) = x\n"),
        ("a stade_ function under a cgen_ banner", :place, replace(src, "# ==================== stade_* (public API)" => "# ==================== cgen_* (public API)"; count = 1)),
        ("a norm_ function under a stade_ banner", :place, replace(src, "# ==================== norm_*" => "# ==================== stade_*"; count = 1)),
        ("a jgen_ function under a cgen_ banner", :place, replace(src, "# ==================== jgen_*" => "# ==================== cgen_*"; count = 1)),
    ]
    for (label, k, text) in cases
        check(text != src && seen(k, text), "sabotage: the scan finds $(label)")
    end
    check(!any(x -> x[1] == :rule23, hr_violations("function f()\n    const_x = 1\n    return const_x\nend\n")), "a local named const_x is not reported")
    println("\n", bad == 0 ? "all house-rule checks passed" : "*** $bad house-rule check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_house_rules() == 0 ? 0 : 1)
end
