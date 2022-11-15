@doc raw"""
    bond_correlation!(BB::AbstractArray{C}, b″::Bond, b′::Bond,
                      unit_cell::UnitCell, lattice::Lattice,
                      Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},
                      Gττup::AbstractArray{T,3}, Gττdn::AbstractArray{T,3}) where {C<:Complex, T<:Number}

Calculate the uneqaul-time bond correlation function
```math
\mathcal{B}_{\mathbf{r}}^{(a,b,\mathbf{r}''),(c,d,\mathbf{r'})}(\tau)=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{B}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{(a,b,\mathbf{r}''),(c,d,\mathbf{r'})}(\tau,0)
=\frac{1}{N}\sum_{\mathbf{i},\sigma,\sigma'}\langle\hat{B}_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{r}''}^{a,b}(\tau)\hat{B}_{\sigma',\mathbf{i},\mathbf{r'}}^{c,d}(0)\rangle,
```
where bond `b″` defines the bond operators
```math
\hat{B}_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{r}''}^{a,b}=\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}+\mathbf{r}''}^{\dagger}\hat{b}_{\sigma,\mathbf{i}+\mathbf{r}}^{\phantom{\dagger}}
```
and bond `b′` defines the operators
```math
\hat{B}_{\sigma',\mathbf{i},\mathbf{r}'}^{c,d}=\hat{c}_{\sigma',\mathbf{i}+\mathbf{r}'}^{\dagger}\hat{d}_{\sigma',\mathbf{i}}^{\phantom{\dagger}}.
```
"""
function bond_correlation!(BB::AbstractArray{C}, b″::Bond, b′::Bond,
                           unit_cell::UnitCell, lattice::Lattice,
                           Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},
                           Gττup::AbstractArray{T,3}, Gττdn::AbstractArray{T,3}) where {C<:Complex, T<:Number}

    # get dimension of system
    D = unit_cell.D

    # length of imaginary time axis
    Lτ = size(BB,D+1) - 1

    # get τ=0 Green's function matrices
    Gup_00 = @view Gττup[:,:,1] # G₊(0,0)
    Gdn_00 = @view Gττdn[:,:,1] # G₋(0,0)

    # unrwap bond b″ = (r′ + r_a - r_b) info
    b, a = b″.orbitals
    r″ = b″.displacement

    # unrwap bond b′ = (r′ + r_c - r_d) info
    d, c = b′.orbitals
    r′ = b′.displacement

    # zero vector
    z = zeros(Int, D)

    # get τ=0 Green's function matrices
    Gup_00 = @view Gττup[:,:,1] # G₊(0,0)
    Gdn_00 = @view Gττdn[:,:,1] # G₋(0,0)

    # iterate over imaginary time
    for l in 0:Lτ
        # get the bond correlations for τ = Δτ⋅l
        BB_τ = selectdim(BB, ndims(BB), l+1)
        # get relevant Green's function matrices
        Gup_ττ = @view Gττup[:,:,l+1] # G₊(τ,τ)
        Gdn_ττ = @view Gττdn[:,:,l+1] # G₋(τ,τ)
        Gup_τ0 = @view Gτ0up[:,:,l+1] # G₊(τ,0)
        Gdn_τ0 = @view Gτ0dn[:,:,l+1] # G₋(τ,0)
        Gup_βmτ0 = @view Gτ0up[:,:,Lτ-l+1] # G₊(β-τ,0)
        Gdn_βmτ0 = @view Gτ0dn[:,:,Lτ-l+1] # G₋(β-τ,0)
        # BB(τ,r) = BB(τ,r) + 1/N sum_i G₊(b,i+r,τ|a,i+r+r″,τ)⋅G₊(d,i,0|c,i+r′,0)
        contract_Grr_G00!(BB_τ, Gup_ττ, Gup_00, b, a, d, c, z, r″, z, r′, 1, unit_cell, lattice)
        # BB(τ,r) = BB(τ,r) + 1/N sum_i G₋(b,i+r,τ|a,i+r+r″,τ)⋅G₋(d,i,0|c,i+r′,0)
        contract_Grr_G00!(BB_τ, Gdn_ττ, Gdn_00, b, a, d, c, z, r″, z, r′, 1, unit_cell, lattice)
        # BB(τ,r) = BB(τ,r) + 1/N sum_i G₊(b,i+r,τ|a,i+r+r″,τ)⋅G₋(d,i,0|c,i+r′,0)
        contract_Grr_G00!(BB_τ, Gup_ττ, Gdn_00, b, a, d, c, z, r″, z, r′, 1, unit_cell, lattice)
        # BB(τ,r) = BB(τ,r) + 1/N sum_i G₋(b,i+r,τ|a,i+r+r″,τ)⋅G₊(d,i,0|c,i+r′,0)
        contract_Grr_G00!(BB_τ, Gdn_ττ, Gup_00, b, a, d, c, z, r″, z, r′, 1, unit_cell, lattice)
        # BB(τ,r) = BB(τ,r) + 1/N sum_i G₊(d,i,β-τ|a,i+r+r″,0)⋅G₊(b,i+r,τ|c,i+r′,0)
        contract_G0r_Gr0!(BB_τ, Gup_βmτ0, Gup_τ0, d, a, b, c, z, r″, z, r′, 1, unit_cell, lattice)
        # BB(τ,r) = BB(τ,r) + 1/N sum_i G₋(d,i,β-τ|a,i+r+r″,0)⋅G₋(b,i+r,τ|c,i+r′,0)
        contract_G0r_Gr0!(BB_τ, Gdn_βmτ0, Gdn_τ0, d, a, b, c, z, r″, z, r′, 1, unit_cell, lattice)
    end

    return nothing
end

@doc raw"""
    bond_correlation!(BB::AbstractArray{C}, b″::Bond, b′::Bond,
                      unit_cell::UnitCell, lattice::Lattice,
                      Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {C<:Complex, T<:Number}

Calculate the eqaul-time bond correlation function
```math
\mathcal{B}_{\mathbf{r}}^{(a,b,\mathbf{r}''),(c,d,\mathbf{r'})}=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{B}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{(a,b,\mathbf{r}''),(c,d,\mathbf{r'})}
=\frac{1}{N}\sum_{\mathbf{i},\sigma,\sigma'}\langle\hat{B}_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{r}''}^{a,b}\hat{B}_{\sigma',\mathbf{i},\mathbf{r'}}^{c,d}\rangle,
```
where bond `b″` defines the bond operators
```math
\hat{B}_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{r}''}^{a,b}=\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}+\mathbf{r}''}^{\dagger}\hat{b}_{\sigma,\mathbf{i}+\mathbf{r}}^{\phantom{\dagger}}
```
and bond `b′` defines the operators
```math
\hat{B}_{\sigma',\mathbf{i},\mathbf{r}'}^{c,d}=\hat{c}_{\sigma',\mathbf{i}+\mathbf{r}'}^{\dagger}\hat{d}_{\sigma',\mathbf{i}}^{\phantom{\dagger}}.
```
"""
function bond_correlation!(BB::AbstractArray{C}, b″::Bond, b′::Bond,
                           unit_cell::UnitCell, lattice::Lattice,
                           Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {C<:Complex, T<:Number}

    # get dimension of system
    D = unit_cell.D

    # unrwap bond b″ = (r′ + r_a - r_b) info
    b, a = b″.orbitals
    r″ = b″.displacement

    # unrwap bond b′ = (r′ + r_c - r_d) info
    d, c = b′.orbitals
    r′ = b′.displacement

    # zero vector
    z = zeros(Int, D)

    # define relevant bonds
    b_da_nr′ = Bond((a,d), -r′) # -r′ + (r_d - r_a)
    b_bc_nr″ = Bond((c,b), -r″) # -r″ + (r_b - r_c)
    
    # BB(r) = BB(r) + 1/N sum_i G₊(b,i+r|a,i+r+r″)⋅G₊(d,i|c,i+r′)
    contract_Grr_G00!(BB, Gup, Gup, b, a, d, c, z, r″, z, r′, 1, unit_cell, lattice)
    # BB(r) = BB(r) + 1/N sum_i G₋(b,i+r|a,i+r+r″)⋅G₋(d,i|c,i+r′)
    contract_Grr_G00!(BB, Gdn, Gdn, b, a, d, c, z, r″, z, r′, 1, unit_cell, lattice)
    # BB(r) = BB(r) + 1/N sum_i G₊(b,i+r|a,i+r+r″)⋅G₋(d,i|c,i+r′)
    contract_Grr_G00!(BB, Gup, Gdn, b, a, d, c, z, r″, z, r′, 1, unit_cell, lattice)
    # BB(r) = BB(r) + 1/N sum_i G₋(b,i+r|a,i+r+r″)⋅G₊(d,i|c,i+r′)
    contract_Grr_G00!(BB, Gdn, Gup, b, a, d, c, z, r″, z, r′, 1, unit_cell, lattice)
    # BB(r) = BB(r) - 1/N sum_i G₊(d,i|a,i+r+r″)⋅G₊(b,i+r|c,i+r′)
    contract_G0r_Gr0!(BB, Gup, Gup, d, a, b, c, z, r″, z, r′, -1, unit_cell, lattice)
    # BB(r) = BB(r) - 1/N sum_i G₋(d,i|a,i+r+r″)⋅G₋(b,i+r|c,i+r′)
    contract_G0r_Gr0!(BB, Gdn, Gdn, d, a, b, c, z, r″, z, r′, -1, unit_cell, lattice)
    # BB(r) = BB(r) + 1/N sum_i δ(d,a)⋅δ(-r′,r)⋅G₊(b,i+r-r″|c,i)
    contract_δGr0!(BB, Gup, b_da_nr′, b_bc_nr″, 1, unit_cell, lattice)
    # BB(r) = BB(r) + 1/N sum_i δ(d,a)⋅δ(-r′,r)⋅G₋(b,i+r-r″|c,i)
    contract_δGr0!(BB, Gdn, b_da_nr′, b_bc_nr″, 1, unit_cell, lattice)

    return nothing
end