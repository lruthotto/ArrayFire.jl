A = AFArray(rand(10,10));
@test !issparse(A)

# build an identity matrix
n = 10;

Ie = AFArray(eye(10))
Aid = sparse(Ie)

@test issparse(Aid)

Ie2 = full(Aid)

@test !issparse(Ie2)

@test all(Ie2 == Ie)

e = sparse(eye(10))
a1 = AFArray(e)
@test issparse(a1)
e2 = SparseMatrixCSC(a1)
@test e2 == e

I = AFArray(vec(collect(1:n)))
J = AFArray(vec(collect(1:n)))
V = AFArray(randn(n))
Aid2 = create_sparse_array(n, n, V, I, J, AF_STORAGE_CSR)
@test issparse(Aid2)


Ah = sprandn(2*n,n,.1)
AH = sparse(AFArray(full(Ah)))
@test nnz(Ah)==sparse_get_nnz(AH)
@test norm(Ah.nzval - Array(sparse_get_values(AH)),Inf) < 1e-7
@test all(Ah .== SparseMatrixCSC(AH))
(V,R,C,t) = sparse_get_info(AH)
@test norm(Array(V) - Ah.nzval,Inf) < 1e-7
@test all(R.==Ah.rowval)
@test all(C.==Ah.colptr)
xh = randn(n)
xd = AFArray(xh)
t1 = Ah*xh
t2 = AH*xd
@test norm(t1-Array(t2),Inf) < 1e-7

