# lab_spline.jl
using LinearAlgebra
using Printf
using Statistics
using Random
# Для распараллеливания:
Threads = Base.Threads

# Для воспроизводимости (особенно для sin_noise)
Random.seed!(1234)

# -----------------------
# Утилиты
# -----------------------
function make_open_uniform_knots(a::Float64, b::Float64, h::Float64, d::Int)
    n_intervals = Int(round((b - a)/h))
    interior = [a + i*h for i in 1:(n_intervals-1)]
    knots = vcat(fill(a, d+1), interior, fill(b, d+1))
    return knots
end

"""
Вычисляет значения всех B_{j,d}(x) для j=1..n (где n = length(knots)-d-1)
Реализация: табличный Cox–de Boor (DP), возвращает вектор длины n.
"""
function bspline_basis_all(x::Float64, d::Int, knots::Vector{Float64})
    K = length(knots)
    n = K - d - 1  # число B-сплайнов
    # N[i, r] -- i=1..n, r=1..(d+1) (r=1 соответствует r=0 в формуле)
    N = zeros(Float64, n, d+1)
    # r = 0 (в таблице rindex = 1)
    for i in 1:n
        left = knots[i]
        right = knots[i+1]
        if (left <= x < right) || (x == knots[end] && right == knots[end])
            N[i, 1] = 1.0
        else
            N[i, 1] = 0.0
        end
    end
    # повышаем порядок r = 1..d (в таблице rindex = 2..d+1)
    for rindex in 2:(d+1)
        r = rindex - 1
        for i in 1:n
            # term1: ((x - t_i)/(t_{i+r} - t_i)) * N[i, r-1]
            denom1 = knots[i + r] - knots[i]
            t1 = denom1 == 0.0 ? 0.0 : ((x - knots[i]) / denom1) * N[i, rindex-1]
            # term2: ((t_{i+r+1}-x)/(t_{i+r+1} - t_{i+1})) * N[i+1, r-1]
            if i+1 <= n
                denom2 = knots[i + r + 1] - knots[i + 1]
                t2 = denom2 == 0.0 ? 0.0 : ((knots[i + r + 1] - x) / denom2) * N[i+1, rindex-1]
            else
                t2 = 0.0
            end
            N[i, rindex] = t1 + t2
        end
    end
    # вернуть столбец степени d (rindex = d+1)
    return N[:, d+1]
end

"""
Оценивает сплайн s(x) = sum_j c[j]*B_j,d(x) в точках xs.
"""
function eval_spline(c::Vector{Float64}, d::Int, knots::Vector{Float64}, xs::Vector{Float64})
    ys = zeros(Float64, length(xs))
    @inbounds for (k,x) in enumerate(xs)
        B = bspline_basis_all(x, d, knots)
        ys[k] = dot(c, B)
    end
    return ys
end

"""
Коэффициенты c_j = f(tau*_j), где tau*_j = (tau_{j+1}+...+tau_{j+d})/d.
"""
function coeffs_variation_diminishing(f::Function, knots::Vector{Float64}, d::Int)
    K = length(knots)
    n = K - d - 1
    c = zeros(Float64, n)
    for j in 1:n
        tau_star = sum(knots[j+1 : j+d]) / d
        c[j] = f(tau_star)
    end
    return c
end

"""
3-point quadratic quasi-interpolant (d == 2).
  c_j = (-f(t_{j+1}) + 4 f((t_{j+1}+t_{j+2})/2) - f(t_{j+2})) / 2
"""
function coeffs_three_point_quasi(f::Function, knots::Vector{Float64}, d::Int)
    @assert d == 2
    K = length(knots)
    n = K - d - 1
    c = zeros(Float64, n)
    # Greville points τ_j = (t_{j+1} + t_{j+2})/2 for d=2
    # Граничные значения: c[1] = f(τ_1), c[n] = f(τ_n)
    c[1] = f((knots[2] + knots[3]) / 2)
    c[end] = f((knots[n+1] + knots[n+2]) / 2)
    for j in 2:(n-1)
        a = knots[j+1]
        cpt = (knots[j+1] + knots[j+2]) / 2
        cc = knots[j+2]
        c[j] = (- f(a) + 4.0 * f(cpt) - f(cc)) / 2.0
    end
    return c
end

# -----------------------
# Ошибка и эксперимент
# -----------------------
function max_abs_error(f::Function, c::Vector{Float64}, d::Int, knots::Vector{Float64}; factor_refine::Int=10)
    a = knots[d+1]
    b = knots[end-d]
    N = 200 * factor_refine
    xs = range(a, b, length = N)
    ys_true = [f(x) for x in xs]
    ys_approx = eval_spline(c, d, knots, collect(xs))
    return maximum(abs.(ys_true .- ys_approx)), xs, ys_true, ys_approx
end

function run_experiment(f::Function; a=0.0, b=1.0, h=0.05, d::Int=2, factor_refine::Int=10, parallel::Bool=false)
    knots = make_open_uniform_knots(a, b, h, d)
    if !parallel
        c_vd = coeffs_variation_diminishing(f, knots, d)
        c_q3 = coeffs_three_point_quasi(f, knots, d)
    else
        K = length(knots); n = K - d - 1
        c_vd = zeros(Float64, n)
        c_q3 = zeros(Float64, n)
        Threads.@threads for j in 1:n
            tau_star = sum(knots[j+1 : j+d]) / d
            c_vd[j] = f(tau_star)
            if j == 1
                c_q3[j] = f((knots[2] + knots[3]) / 2)
            elseif j == n
                c_q3[j] = f((knots[n+1] + knots[n+2]) / 2)
            else
                a_ = knots[j+1]; mid = (knots[j+1] + knots[j+2]) / 2; cc = knots[j+2]
                c_q3[j] = (-f(a_) + 4.0*f(mid) - f(cc)) / 2.0
            end
        end
    end
    err_vd, xs, ys_true, ys_vd = max_abs_error(f, c_vd, d, knots; factor_refine=factor_refine)
    err_q3, _, _, ys_q3 = max_abs_error(f, c_q3, d, knots; factor_refine=factor_refine)
    return Dict(
        "knots"=>knots,
        "c_vd"=>c_vd, "c_q3"=>c_q3,
        "err_vd"=>err_vd, "err_q3"=>err_q3,
        "xs"=>xs, "ys_true"=>ys_true, "ys_vd"=>ys_vd, "ys_q3"=>ys_q3
    )
end

# Доп. метрики
function compute_l2_tv(ys_true::AbstractVector{T}, ys_approx::AbstractVector{T}) where T<:Real
    m = length(ys_true)
    err_L2 = sqrt(sum((ys_true .- ys_approx).^2) / m)
    tv = sum(abs.(diff(ys_approx)))
    return err_L2, tv
end

# -----------------------
# Прогон набора тестов: 6 функций
# -----------------------
function run_all_tests(; a=0.0, b=1.0, d::Int=2, hs=[1/10, 1/20, 1/40, 1/80], factor_refine::Int=10, outcsv::String="results_fixed.csv")
    funcs = Dict(
        "x^2" => x->x^2,
        "sin_2pi" => x->sin(2π*x),
        "sin_20pi" => x->sin(20π*x),
        "abs" => x->abs(x-0.5),
        "book" => x->(x <= 0.5 ? 1.0 : exp(-10*(x-0.5))),
        "sin_noise" => x->sin(2π*x) + 0.01*(2rand() - 1)
    )

    open(outcsv, "w") do io
        println(io, join(["func","h","method","err_inf","err_L2","tv"], ","))
    end

    for (fname,f) in funcs
        println("\n=== Function: $fname ===")
        for h in hs
            res = run_experiment(f; a=a, b=b, h=h, d=d, factor_refine=factor_refine, parallel=false)
            xs = collect(res["xs"])
            ys_true = res["ys_true"]
            ys_vd = res["ys_vd"]
            ys_q3 = res["ys_q3"]

            err_inf_vd = res["err_vd"]
            err_inf_q3 = res["err_q3"]
            err_L2_vd, tv_vd = compute_l2_tv(ys_true, ys_vd)
            err_L2_q3, tv_q3 = compute_l2_tv(ys_true, ys_q3)

            # print short summary
            @printf("h=%.5f  VD: err_inf=%.5e  err_L2=%.5e  tv=%.5e\n", h, err_inf_vd, err_L2_vd, tv_vd)
            @printf("           Q3: err_inf=%.5e  err_L2=%.5e  tv=%.5e\n\n", err_inf_q3, err_L2_q3, tv_q3)

            open(outcsv, "a") do io
                println(io, join([fname, string(h), "VD", string(err_inf_vd), string(err_L2_vd), string(tv_vd)], ","))
                println(io, join([fname, string(h), "Q3", string(err_inf_q3), string(err_L2_q3), string(tv_q3)], ","))
            end
        end
    end
    println("Saved CSV: $outcsv")
    return outcsv
end

# Если запускается как скрипт — выполнить тесты
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running full test suite (fixed implementation)...")
    csv = run_all_tests()
    println("Done. Results saved to: ", csv)
end
