# [Hierarchical Equations of Motion Liouvillian Superoperator (HEOMLS) Matrix](@id doc-HEOMLS-Matrix)

## Overview
The hierarchical equations of motion Liouvillian superoperator (HEOMLS) ``\hat{\mathcal{M}}`` characterizes the dynamics in the full [auxiliary density operators (ADOs)](@ref doc-ADOs) space, namely 
```math
\partial_{t}\rho^{(m,n,p)}_{\textbf{j} \vert \textbf{q}}(t)=\hat{\mathcal{M}}\rho^{(m,n,p)}_{\textbf{j} \vert \textbf{q}}(t)
```
and it can, numerically, be expressed as a matrix. 

In `HierarchicalEOM.jl`, all different types of HEOMLS ``\hat{\mathcal{M}}`` are subtype of `AbstractHEOMLSMatrix`.

The HEOMLS ``\hat{\mathcal{M}}`` not only characterizes the bare system dynamics (based on system Hamiltonian), but it also encodes the system-and-multiple-bosonic-baths and system-and-multiple-fermionic-baths interactions based on [Bosonic Bath](@ref doc-Bosonic-Bath) and [Fermionic Bath](@ref doc-Fermionic-Bath), respectively. For a specific ``m``th-level-bosonic-and-``n``th-level-fermionic auxiliary density operator ``\rho^{(m,n,p)}_{\textbf{j} \vert \textbf{q}}``, it will be coupled to the following ADOs through ``\hat{\mathcal{M}}``:
 - ``(m+1)``th-level-bosonic-and-``n``th-level-fermionic ADOs
 - ``(m-1)``th-level-bosonic-and-``n``th-level-fermionic ADOs
 - ``m``th-level-bosonic-and-``(n+1)``th-level-fermionic ADOs
 - ``m``th-level-bosonic-and-``(n-1)``th-level-fermionic ADOs
and thus forms the two-fold hierarchy relations. See our paper for more details.

In practice, the size of the matrix ``\hat{\mathcal{M}}`` must be finite and the hierarchical equations must be truncated at a suitable bosonic-level (``m_\textrm{max}``) and fermionic-level (``n_\textrm{max}``). These truncation levels (tiers) must be given when constructing ``\hat{\mathcal{M}}``.
```julia
Hs::QuantumObject   # system Hamiltonian
Bbath::BosonBath    # bosonic   bath object
Fbath::FermionBath  # fermionic bath object
Btier::Int          # bosonic   truncation level 
Ftier::Int          # fermionic truncation level 

M = M_Boson(Hs, Btier, Bbath)
M = M_Fermion(Hs, Ftier, Fbath)
M = M_Boson_Fermion(Hs, Btier, Ftier, Bbath, Fbath)
```

## [Importance Value and Threshold](@id doc-Importance-Value-and-Threshold)
The main computational complexity can be quantified by the total number of [auxiliary density operators (ADOs)](@ref doc-ADOs) because it directly affects the size of ``\hat{\mathcal{M}}``. 

The computational effort can be further optimized by associating an **importance value** ``\mathcal{I}`` to each ADO and then discarding all the ADOs (in the second and higher levels) whose importance value is smaller than a threshold value ``\mathcal{I}_\textrm{th}``. The importance value for a given ADO : ``\mathcal{I}\left(\rho^{(m,n,p)}_{\textbf{j} \vert \textbf{q}}\right)`` is determined by its corresponding exponential terms of bath correlation function [see [Phys. Rev. B 88, 235426 (2013)](https://doi.org/10.1103/PhysRevB.88.235426) and [Phys. Rev. B 103, 235413 (2021)](https://doi.org/10.1103/PhysRevB.103.235413)]. This allows us to only consider the ADOs which affects the dynamics more, and thus, reduce the size of ``\hat{\mathcal{M}}``. Also see our paper [ [Communications Physics 6, 313 (2023)](https://doi.org/10.1038/s42005-023-01427-2)  ] for more details.

When you specify a threshold value ``\mathcal{I}_\textrm{th}`` with the parameter `threshold` to construct ``\hat{\mathcal{M}}``, we will remain all the ADOs where their hierarchy levels ``(m,n)\in\{(0,0), (0,1), (1,0), (1,1)\}``, and all the other high-level ADOs may be neglected if ``\mathcal{I}\left(\rho^{(m,n,p)}_{\textbf{j} \vert \textbf{q}}\right) < \mathcal{I}_\textrm{th}``. 
```julia
Hs::QuantumObject   # system Hamiltonian
Bbath::BosonBath    # bosonic   bath object
Fbath::FermionBath  # fermionic bath object
Btier::Int          # bosonic   truncation level 
Ftier::Int          # fermionic truncation level 

M = M_Boson(Hs, Btier, Bbath; threshold=1e-7)
M = M_Fermion(Hs, Ftier, Fbath; threshold=1e-7)
M = M_Boson_Fermion(Hs, Btier, Ftier, Bbath, Fbath; threshold=1e-7)
```
!!! note "Default value of importance threshold"
    The full hierarchical equations can be recovered in the limiting case ``\mathcal{I}_\textrm{th}\rightarrow 0``, which is the default value of the parameter : `threshold=0.0`. This means that all of the ADOs will be taken into account by default.

## Methods
All of the HEOMLS matrices supports the following two `Base` Functions :
 - `size(M::AbstractHEOMLSMatrix)` : Returns the size of the HEOMLS matrix.
 - Bracket operator `[i,j]` : Returns the `(i, j)`-element(s) in the HEOMLS matrix.
```julia
M::AbstractHEOMLSMatrix

m, n = size(M)
M[10, 12]
M[2:4, 2:4]
M[1,:]
M[:,2]
```