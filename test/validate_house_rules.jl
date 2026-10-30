# House rules 2 and 3 of claude/skill-stade-dev.md, checked on the text of src/STADE.jl:
# no struct or @enum (records are NamedTuples) and no global const or other top-level state.
# A line at column 1 that starts a const, global, struct, mutable struct, abstract type, primitive type or @enum is a violation.
# `module STADE` and `end`, `using`/`import` and function definitions are not covered here.
#   julia test/validate_house_rules.jl

function hr_violations(text::AbstractString)
    pat = r"^(const|global|struct|mutable struct|abstract type|primitive type|@enum)\b"
    return [(i, strip(l)) for (i, l) in enumerate(split(text, '\n')) if occursin(pat, l)]
end

function validate_house_rules()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    src = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
    v = hr_violations(src)
    check(isempty(v), "src/STADE.jl has no top-level const, global, struct or @enum" * (isempty(v) ? "" : " (first: line $(v[1][1]): $(first(v[1][2], 60)))"))
    # sabotage: the scan must see each kind of line when it is injected
    for (label, line) in (("a global const", "const X_ = 1"), ("a struct", "struct S_\n    a::Int\nend"), ("a mutable struct", "mutable struct M_\n    a::Int\nend"),
                          ("an @enum", "@enum E_ a_ b_"), ("a global", "global G_ = 1"), ("an abstract type", "abstract type A_ end"))
        hit = hr_violations(src * "\n" * line * "\n")
        check(!isempty(hit), "sabotage: the scan finds $(label) injected at the top level")
    end
    # an indented const (a local) is not a top-level one: the scan leaves it alone
    check(isempty(hr_violations("function f()\n    const_x = 1\n    return const_x\nend\n")), "a local named const_x is not reported")
    println("\n", bad == 0 ? "all house-rule checks passed" : "*** $bad house-rule check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_house_rules() == 0 ? 0 : 1)
end
