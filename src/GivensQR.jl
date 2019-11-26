module GivensQR


export givensqr, Q_op


using SparseArrays, LinearAlgebra, LinearOperators


mutable struct GivensQR_model{T}
    m ::Int
    n ::Int
    rows ::Array{Int}
    cols ::Array{Int}
    cosines ::Array{T}
    sines ::Array{T}
    col_perm ::Array{Int}
    rank ::Int
end


function givensqr(R ::SparseMatrixCSC)
    m = R.m
    n = R.n
    rows = []
    cols = []
    cosines = []
    sines = []
    k = min(m,n)
    no_null_col = n
    col_perm = AbstractArray{Int}(1:n)
    rank = 0

    for j = 1:min(k, no_null_col)

        while (no_null_col > j && R[j:m,j].nzind == [])
            aux1 = R[:,j]
            R[:,j] = R[:,no_null_col]
            R[:,no_null_col] = aux1

            aux2 = col_perm[j]
            col_perm[j] = col_perm[no_null_col]
            col_perm[no_null_col] = aux2

            no_null_col -= 1
        end

        for i in R[:,j].nzind
            if i > j && R[i,j] != 0
                c = R[j,j]
                s = R[i,j]
                r = sqrt(c*c + s*s)
                c = c/r
                s = s/r
                row_i = R[i,:]
                row_j = R[j,:]
                R[j,:] = c*row_j + s*row_i
                R[i,:] = -s*row_j + c*row_i
                R[i,j] = 0

                rows = vcat(rows, [i])
                cols = vcat(cols, [j])
                cosines = vcat(cosines, [c])
                sines = vcat(sines, [s])
            end
        end
    end

    i = 1
    while !(i > k || R[i,i] == 0)
        i += 1
        rank += 1
    end

    rows = convert(Array{Int}, rows)
    cols = convert(Array{Int}, cols)
    cosines = convert(Array{Float64}, cosines)
    sines = convert(Array{Float64}, sines)

    return GivensQR_model(m, n, rows, cols, cosines, sines, col_perm, rank)
end


function Q_op(G ::GivensQR_model, x)
    m = G.m
    n = G.n
    rank = G.rank
    cols = G.cols
    rows = G.rows
    sines = G.sines
    cosines = G.cosines
    p = length(cols)

    for k = p:-1:1
        i = rows[k]
        j = cols[k]
        c = cosines[k]
        s = sines[k]
        xi = x[i]
        xj = x[j]
        x[i] = c*xi  + s*xj
        x[j] = -s*xi + c*xj
    end

    return x
end


# i will change Q_op to nullspace_op and t_nullspace_op
function Q_op(A ::SparseMatrixCSC)
    G = givensqr(A)
    Q = LinearOperator(Float64, A.m, A.m, false, false, v -> Q_op(G, v))

    return Q, G.col_perm
end


end # module
