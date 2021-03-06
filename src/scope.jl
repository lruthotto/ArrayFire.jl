export @afgc

macro afgc(expr)
    @assert expr.head == :function || expr.head == :(=) "Only works on functions or assignments"
    sc = (:(ArrayFire.scope() do ; end))
    sc.args[2].args[2] = expr.args[2]
    expr.args[2] = sc
    esc(expr)
end

global const scopes = Vector{Vector{AFArray}}()

function enter_scope()
    push!(scopes, Vector{AFArray}())
end

matches(k, except::Void) = false
matches(k, except::AFArray) = k === except
matches(k, except) = any(x -> x === k, except)

function leave_scope(except)
    scope = pop!(scopes)
    for k in scope
        if matches(k, except)
            if !isempty(scopes)
                push!(scopes[end], k)
            end
        else
            finalize(k)
        end
    end
end

function scope(f)
    enter_scope()
    r = f()
    leave_scope(r)
    return r
end
