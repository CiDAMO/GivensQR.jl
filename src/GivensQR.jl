module GivensQR

export givensqr_op

using SparseArrays, LinearAlgebra, LinearOperators

mutable struct GivensQR_model{T}
    nrows ::Int
    ncols ::Int
    rows ::Array{Int}
    cols ::Array{Int}
    cosines ::Array{T}
    sines ::Array{T}
    col_perm ::Array{Int}
    rank ::Int
end

function qr(R :: SparseMatrixCSC{T,Ti}) where {Ti <: Integer, T}
    m = R.m
    n = R.n
    rows = Int[]
    cols = Int[]
    cosines = T[]
    sines = T[]
    k = min(m,n)
    no_null_col = n
    col_perm = AbstractArray{Int}(1:n)
    rank = 0

    for j = 1:min(k, no_null_col)

        while (no_null_col > j && R[j:m,j].nzind == [])

            aux = col_perm[j]
            col_perm[j] = col_perm[no_null_col]
            col_perm[no_null_col] = aux

            no_null_col -= 1
        end

        for i in R[:,col_perm[j]].nzind
            if i > j
                c = R[col_perm[j],col_perm[j]]
                s = R[i,col_perm[j]]
                r = -1 * sqrt(c*c + s*s)
                c = c/r
                s = s/r
                row_i = R[i,:]
                row_j = R[col_perm[j],:]
                R[col_perm[j],:] = c*row_j + s*row_i
                R[i,:] = -s*row_j + c*row_i
                R[i,j] = 0

                push!(rows, i)
                push!(cols, j)
                push!(cosines, c)
                push!(sines, s)
            end
        end
    end

    return GivensQR_model(m, n, rows, cols, cosines, sines, col_perm, min(k, no_null_col))
end

function qr(R :: Matrix{T}) where  T
    m, n = size(R)
    rows = Int[]
    cols = Int[]
    cosines = T[]
    sines = T[]
    k = min(m,n)
    no_null_col = n
    col_perm = AbstractArray{Int}(1:n)
    rank = 0

    for j = 1:min(k, no_null_col)

        while (no_null_col > j && norm(R[j:m,j], Inf) < 1e-14)

            aux = col_perm[j]
            col_perm[j] = col_perm[no_null_col]
            col_perm[no_null_col] = aux

            no_null_col -= 1
        end

        for i = 1:m
            if i > j && R[i, col_perm[j]] >= 1e-14
                c = R[col_perm[j],col_perm[j]]
                s = R[i,col_perm[j]]
                r = -1 * sqrt(c*c + s*s)
                c = c/r
                s = s/r
                row_i = R[i,:]
                row_j = R[col_perm[j],:]
                R[col_perm[j],:] = c*row_j + s*row_i
                R[i,:] = -s*row_j + c*row_i
                R[i,j] = 0

                push!(rows, i)
                push!(cols, j)
                push!(cosines, c)
                push!(sines, s)
            end
        end
    end

    return GivensQR_model(m, n, rows, cols, cosines, sines, col_perm, min(k, no_null_col))
end

function givensqr_op(G :: GivensQR_model{T}, x :: AbstractVector{S}) where {T,S}
    m = G.nrows
    n = G.ncols
    rank = G.rank
    cols = G.cols
    rows = G.rows
    sines = G.sines
    cosines = G.cosines
    p = length(cols)
    y = similar(x, promote_type(T, S))
    y .= x

    for k = p:-1:1
        i = rows[k]
        j = cols[k]
        c = cosines[k]
        s = sines[k]
        yi = y[i]
        yj = y[j]
        y[i] = c*yi  + s*yj
        y[j] = -s*yi + c*yj
    end

    return y
end

function givensqr_op(G :: GivensQR_model{T}) where T
    return LinearOperator(T, G.nrows, G.nrows, false, false, x -> givensqr_op(G, x))
end

end # module
