using Test
using HierarchicalEOM
using CUDA

@testset "CUDA (Input-Output HEOM)" begin
    tier = 12
    Δ = 2 * π
    Γ = 0.1 * Δ
    λ = 0.1 * Δ
    ω0 = 0.2 * Δ
    Z = sigmaz()
    X = sigmax()
    Hsys = 0.5 * Δ * Z
    ρ0 = ket2dm(basis(2, 0))
    bath = BosonBath(
        X,
        [0.5 * λ^2, 0.5 * λ^2],
        [-1.0im * ω0 + Γ, 1.0im * ω0 + Γ],
        [0.5im * λ^2, -0.5im * λ^2],
        [-1.0im * ω0 + Γ, 1.0im * ω0 + Γ],
    )

    # compare result with mesolve
    a = qeye(2) ⊗ destroy(tier)
    H = Hsys ⊗ qeye(tier) + λ * tensor(X, qeye(tier)) * (a + a') + ω0 * a' * a
    tlist = LinRange(0, 20 / Δ, 100)
    sol_me = mesolve(H, ρ0 ⊗ ket2dm(basis(tier, 1)), tlist, [sqrt(Γ * 2) * a]; e_ops = [a' * a], progress_bar = Val(false))

    # dynamical field
    input1(p, t) = λ * exp(-1.0im * ω0 * t - Γ * t)
    bath_input1 = BosonDynamicalField(X, η_in = input1)

    input2(p, t) = λ * exp(1.0im * ω0 * t - Γ * t)
    bath_input2 = BosonDynamicalField(X, η_in = input2)

    output1R(p, t) = λ * exp(1.0im * ω0 * (p.tout - t) - Γ * (p.tout - t))
    bath_output_1R = BosonDynamicalField(X, η_out_fn_R = output1R)

    output2L(p, t) = λ * exp(-1.0im * ω0 * (p.tout - t) - Γ * (p.tout - t))
    bath_output_2L = BosonDynamicalField(X, η_out_fn_L = output2L)

    baths = [bath, bath_output_1R, bath_output_2L, bath_input1, bath_input2]

    M_full_cpu = M_Boson(Hsys, tier, baths; assemble = Val(:full), verbose = false)
    M_lazy_cpu = M_Boson(Hsys, tier, baths; assemble = Val(:combine), verbose = false)
    M_full_gpu = cu(M_full_cpu)
    M_lazy_gpu = cu(M_lazy_cpu)
    HDict = M_full_cpu.hierarchy
    e_ops = [
        TrADO(M_full_cpu, 1),
        TrADO(M_full_cpu, HDict.nvec2idx[Nvec([0, 0, 0, 1, 1, 0])]),
        TrADO(M_full_cpu, HDict.nvec2idx[Nvec([0, 0, 1, 0, 0, 1])]),
        TrADO(M_full_cpu, HDict.nvec2idx[Nvec([0, 0, 1, 1, 0, 0])]),
        TrADO(M_full_cpu, HDict.nvec2idx[Nvec([0, 0, 1, 1, 1, 1])]),
    ]

    result_full = ComplexF64[1.0 + 0.0im]
    result_lazy = ComplexF64[1.0 + 0.0im]
    for tout in tlist
        (tout == 0) && continue
        p = (tout = tout,)

        # assemble = Val(:full)
        sol_heom_full = HEOMsolve(M_full_gpu, ρ0, [0, tout], params = p, e_ops = e_ops, progress_bar = Val(false))
        exp_vals_full = sol_heom_full.expect[:, end]
        push!(
            result_full,
            exp(-2 * Γ * tout) * exp_vals_full[1] - exp(1im * ω0 * tout - Γ * tout) * exp_vals_full[2] -
                exp(-1im * ω0 * tout - Γ * tout) * exp_vals_full[3] - exp_vals_full[4] + exp_vals_full[5],
        )

        # assemble = Val(:combine)
        sol_heom_lazy = HEOMsolve(M_lazy_gpu, ρ0, [0, tout], params = p, e_ops = e_ops, progress_bar = Val(false))
        exp_vals_lazy = sol_heom_lazy.expect[:, end]
        push!(
            result_lazy,
            exp(-2 * Γ * tout) * exp_vals_lazy[1] - exp(1im * ω0 * tout - Γ * tout) * exp_vals_lazy[2] -
                exp(-1im * ω0 * tout - Γ * tout) * exp_vals_lazy[3] - exp_vals_lazy[4] + exp_vals_lazy[5],
        )
    end
    @test all(isapprox.(result_full, sol_me.expect[1, :], atol = 1.0e-6))
    @test all(isapprox.(result_lazy, sol_me.expect[1, :], atol = 1.0e-6))
end