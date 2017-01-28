var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#A-Julia-Linear-Operator-Package-1",
    "page": "Home",
    "title": "A Julia Linear Operator Package",
    "category": "section",
    "text": "Operators behave like matrices (with exceptions) but are defined by their effect when applied to a vector. They can be transposed, conjugated, or combined with other operators cheaply. The costly operation is deferred until multiplied with a vector."
},

{
    "location": "index.html#Compatibility-1",
    "page": "Home",
    "title": "Compatibility",
    "category": "section",
    "text": "Julia 0.4 and up."
},

{
    "location": "index.html#How-to-Install-1",
    "page": "Home",
    "title": "How to Install",
    "category": "section",
    "text": "Pkg.add(\"LinearOperators\")"
},

{
    "location": "index.html#Operators-Available-1",
    "page": "Home",
    "title": "Operators Available",
    "category": "section",
    "text": "Operator Description\nLinearOperator Base class. Useful to define operators from functions\nopEye Identity operator\nopOnes All ones operator\nopZeros All zeros operator\nopDiagonal Square (equivalent to diagm()) or rectangular diagonal operator\nopInverse Equivalent to \\\nopCholesky More efficient than opInverse for symmetric positive definite matrices\nopHouseholder Apply a Householder transformation I-2hh'\nopHermitian Represent a symmetric/hermitian operator based on the diagonal and strict lower triangle\nopRestriction Represent a selection of \"rows\" when composed on the left with an existing operator\nopExtension Represent a selection of \"columns\" when composed on the right with an existing operator\nLBFGSOperator Limited-memory BFGS approximation in operator form (damped or not)\nInverseLBFGSOperator Inverse of a limited-memory BFGS approximation in operator form (damped or not)\nLSR1Operator Limited-memory SR1 approximation in operator form"
},

{
    "location": "index.html#Utility-Functions-1",
    "page": "Home",
    "title": "Utility Functions",
    "category": "section",
    "text": "Function Description\ncheck_ctranspose Cheap check that A' is correctly implemented\ncheck_hermitian Cheap check that A = A'\ncheck_positive_definite Cheap check that an operator is positive (semi-)definite\ndiag Extract the diagonal of an operator\nfull Convert an abstract operator to a dense array\nhermitian Determine whether the operator is Hermitian\npush! For L-BFGS or L-SR1 operators, push a new pair {s,y}\nreset! For L-BFGS or L-SR1 operators, reset the data\nshape Return the size of a linear operator\nshow Display basic information about an operator\nsize Return the size of a linear operator\nsymmetric Determine whether the operator is symmetric"
},

{
    "location": "index.html#Other-Operations-on-Operators-1",
    "page": "Home",
    "title": "Other Operations on Operators",
    "category": "section",
    "text": "Operators can be transposed (A.'), conjugated (conj(A)) and conjugate-transposed (A'). Operators can be sliced (A[:,3], A[2:4,1:5], A[1,1]), but unlike matrices, slices always return operators (see differences)."
},

{
    "location": "index.html#Differences-1",
    "page": "Home",
    "title": "Differences",
    "category": "section",
    "text": "Unlike matrices, an operator never reduces to a vector or a number.using LinearOperators #hide\nA = rand(5,5)\nopA = LinearOperator(A)\nA[:,1] * 3 # VectoropA[:,1] * 3 # LinearOperator# A[:,1] * [3] # ERRORopA[:,1] * [3] # VectorThis is also true for A[i,J], which returns vectors on Julia 0.5, and for the scalar A[i,j]. Similarly, opA[1,1] is an operator of size (1,1):\"(opA[1,1] * [3])[1] - A[1,1] * 3In the same spirit, the operator full always returns a matrix.full(opA[:,1])"
},

{
    "location": "index.html#Other-Operators-1",
    "page": "Home",
    "title": "Other Operators",
    "category": "section",
    "text": "LLDL features a limited-memory LDLᵀ factorization operator that may be used as preconditioner in iterative methods\nMUMPS.jl features a full distributed-memory factorization operator that may be used to represent the preconditioner in, e.g., constraint-preconditioned Krylov methods."
},

{
    "location": "index.html#Testing-1",
    "page": "Home",
    "title": "Testing",
    "category": "section",
    "text": "julia> Pkg.test(\"LinearOperators\")(Image: GPLv3)"
},

{
    "location": "tutorial.html#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial.html#Using-matrices-1",
    "page": "Tutorial",
    "title": "Using matrices",
    "category": "section",
    "text": "Operators may be defined from matrices and combined using the usual operations, but the result is deferred until the operator is applied.using LinearOperators\nA1 = rand(5,7)\nA2 = sprand(7,3,.3)\nop1 = LinearOperator(A1)\nop2 = LinearOperator(A2)\nop = op1 * op2  # Does not form A1 * A2\nx = rand(3)\ny = op * x"
},

{
    "location": "tutorial.html#Inverse-1",
    "page": "Tutorial",
    "title": "Inverse",
    "category": "section",
    "text": "Operators may be defined to represent (approximate) inverses.A = rand(5,5)\nA = A' * A\nop = opCholesky(A)  # Use, e.g., as a preconditioner\nv = rand(5)\nnorm(A \\ v - op * v) / norm(v)In this example, the Cholesky factor is computed only once and can be used many times transparently."
},

{
    "location": "tutorial.html#Using-functions-1",
    "page": "Tutorial",
    "title": "Using functions",
    "category": "section",
    "text": "Operators may be defined from functions. In the example below, the transposed isn't defined, but it may be inferred from the conjugate transposed. Missing operations are represented as nullable functions. Nullable types were introduced in Julia 0.4.dft = LinearOperator(10, 10, false, false,\n                     v -> fft(v),\n                     Nullable{Function}(),  # will be inferred\n                     w -> ifft(w))\nx = rand(10)\ny = dft * x\nnorm(dft' * y - x)  # DFT is an orthogonal operatordft.' * yBy default a linear operator defined by functions and that is neither symmetric nor hermitian will have element type Complex128. This behavior may be overridden by specifying the type explicitly, e.g.,dft = LinearOperator{Float64}(10, 10, false, false,\n                              v -> fft(v),\n                              Nullable{Function}(),\n                              w -> ifft(w))"
},

{
    "location": "tutorial.html#Limited-memory-BFGS-1",
    "page": "Tutorial",
    "title": "Limited memory BFGS",
    "category": "section",
    "text": "Two other useful operators are the Limited-Memory BFGS in forward and inverse form.B = LBFGSOperator(20)\nH = InverseLBFGSOperator(20)\nr = 0.0\nfor i = 1:100\n  s = rand(20)\n  y = rand(20)\n  push!(B, s, y)\n  push!(H, s, y)\n  r += norm(B * H * s - s)\nend\nrThere is also a LSR1 operator that behaves similarly to these ones."
},

{
    "location": "tutorial.html#Restriction,-extension-and-slices-1",
    "page": "Tutorial",
    "title": "Restriction, extension and slices",
    "category": "section",
    "text": "The restriction operator restricts a vector to a set of indices.v = collect(1:5)\nR = opRestriction([2;5], 5)\nR * vNotice that it corresponds to a matrix with rows of the identity given by the indices.full(R)The extension operator is the transpose of the restriction. It extends a vector with zeros.v = collect(1:2)\nE = opExtension([2;5], 5)\nE * vWith these operators, we define the slices of an operator op.A = rand(5,5)\nopA = LinearOperator(A)\nI = [1;3;5]\nJ = 2:4\nA[I,J] * ones(3)opRestriction(I, 5) * opA * opExtension(J, 5) * ones(3)A main difference with matrices, is that slices do not return vectors nor numbers.opA[1,:] * ones(5)opA[:,1] * ones(1)opA[1,1] * ones(1)"
},

{
    "location": "reference.html#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "reference.html#LinearOperators.LinearOperator",
    "page": "Reference",
    "title": "LinearOperators.LinearOperator",
    "category": "Type",
    "text": "Base type to represent a linear operator. The usual arithmetic operations may be applied to operators to combine or otherwise alter them. They can be combined with other operators, with matrices and with scalars. Operators may be transposed and conjugate-transposed using the usual Julia syntax.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opEye",
    "page": "Reference",
    "title": "LinearOperators.opEye",
    "category": "Function",
    "text": "opEye(T, n)\nopEye(n)\n\nIdentity operator of order n and of data type T (defaults to Float64).\n\n\n\nopEye(T, nrow, ncol)\nopEye(nrow, ncol)\n\nRectangular identity operator of size nrowxncol and of data type T (defaults to Float64).\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opOnes",
    "page": "Reference",
    "title": "LinearOperators.opOnes",
    "category": "Function",
    "text": "opOnes(T, nrow, ncol)\nopOnes(nrow, ncol)\n\nOperator of all ones of size nrow-by-ncol and of data type T (defaults to Float64).\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opZeros",
    "page": "Reference",
    "title": "LinearOperators.opZeros",
    "category": "Function",
    "text": "opZeros(T, nrow, ncol)\nopZeros(nrow, ncol)\n\nZero operator of size nrow-by-ncol and of data type T (defaults to Float64).\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opDiagonal",
    "page": "Reference",
    "title": "LinearOperators.opDiagonal",
    "category": "Function",
    "text": "opDiagonal(d)\n\nDiagonal operator with the vector d on its main diagonal.\n\n\n\nopDiagonal(nrow, ncol, d)\n\nRectangular diagonal operator of size nrow-by-ncol with the vector d on its main diagonal.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opInverse",
    "page": "Reference",
    "title": "LinearOperators.opInverse",
    "category": "Function",
    "text": "opInverse(M; symmetric=false, hermitian=false)\n\nInverse of a matrix as a linear operator using \\. Useful for triangular matrices. Note that each application of this operator applies \\.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opCholesky",
    "page": "Reference",
    "title": "LinearOperators.opCholesky",
    "category": "Function",
    "text": "opCholesky(M, [check=false])\n\nInverse of a Hermitian and positive definite matrix as a linear operator using its Cholesky factorization. The factorization is computed only once. The optional check argument will perform cheap hermicity and definiteness checks.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opHouseholder",
    "page": "Reference",
    "title": "LinearOperators.opHouseholder",
    "category": "Function",
    "text": "opHouseholder(h)\n\nApply a Householder transformation defined by the vector h. The result is x -> (I - 2 h h') x.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opHermitian",
    "page": "Reference",
    "title": "LinearOperators.opHermitian",
    "category": "Function",
    "text": "opHermitian(d, A)\n\nA symmetric/hermitian operator based on the diagonal d and lower triangle of A.\n\n\n\nopHermitian(A)\n\nA symmetric/hermitian operator based on a matrix.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opRestriction",
    "page": "Reference",
    "title": "LinearOperators.opRestriction",
    "category": "Function",
    "text": "Z = opRestriction(I, ncol)\nZ = opRestriction(:, ncol)\n\nCreates a LinearOperator restricting a ncol-sized vector to indices I. The operation Z * v is equivalent to v[I]. I can be :.\n\nZ = opRestriction(k, ncol)\n\nAlias for opRestriction([k], ncol).\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.opExtension",
    "page": "Reference",
    "title": "LinearOperators.opExtension",
    "category": "Function",
    "text": "Z = opExtension(I, ncol)\nZ = opExtension(:, ncol)\n\nCreates a LinearOperator extending a vector of size length(I) to size ncol, where the position of the elements on the new vector are given by the indices I. The operation w = Z * v is equivalent to w = zeros(ncol); w[I] = v.\n\nZ = opExtension(k, ncol)\n\nAlias for opExtension([k], ncol).\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.LBFGSOperator",
    "page": "Reference",
    "title": "LinearOperators.LBFGSOperator",
    "category": "Type",
    "text": "A type for limited-memory BFGS approximations.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.InverseLBFGSOperator",
    "page": "Reference",
    "title": "LinearOperators.InverseLBFGSOperator",
    "category": "Function",
    "text": "InverseLBFGSOperator(T, n, [mem=5; scaling=false])\nInverseLBFGSOperator(n, [mem=5; scaling=false])\n\nConstruct a limited-memory BFGS approximation in inverse form. If the type T is omitted, then Float64 is used.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.LSR1Operator",
    "page": "Reference",
    "title": "LinearOperators.LSR1Operator",
    "category": "Type",
    "text": "A type for limited-memory SR1 approximations.\n\n\n\n"
},

{
    "location": "reference.html#Operators-1",
    "page": "Reference",
    "title": "Operators",
    "category": "section",
    "text": "LinearOperator\nopEye\nopOnes\nopZeros\nopDiagonal\nopInverse\nopCholesky\nopHouseholder\nopHermitian\nopRestriction\nopExtension\nLBFGSOperator\nInverseLBFGSOperator\nLSR1Operator"
},

{
    "location": "reference.html#LinearOperators.check_ctranspose",
    "page": "Reference",
    "title": "LinearOperators.check_ctranspose",
    "category": "Function",
    "text": "check_ctranspose(op)\n\nCheap check that the operator and its conjugate transposed are related.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.check_hermitian",
    "page": "Reference",
    "title": "LinearOperators.check_hermitian",
    "category": "Function",
    "text": "check_hermitian(op)\n\nCheap check that the operator is Hermitian.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.check_positive_definite",
    "page": "Reference",
    "title": "LinearOperators.check_positive_definite",
    "category": "Function",
    "text": "check_positive_definite(op; semi=false)\n\nCheap check that the operator is positive (semi-)definite.\n\n\n\n"
},

{
    "location": "reference.html#Base.LinAlg.diag",
    "page": "Reference",
    "title": "Base.LinAlg.diag",
    "category": "Function",
    "text": "diag(op)\n\nExtract the diagonal of a L-BFGS operator in forward mode.\n\n\n\ndiag(op)\n\nExtract the diagonal of a L-SR1 operator in forward mode.\n\n\n\n"
},

{
    "location": "reference.html#Base.full",
    "page": "Reference",
    "title": "Base.full",
    "category": "Function",
    "text": "A = full(op)\n\nMaterialize an operator as a dense array using op.ncol products.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.hermitian",
    "page": "Reference",
    "title": "LinearOperators.hermitian",
    "category": "Function",
    "text": "hermitian(op)\nishermitian(op)\n\nDetermine whether the operator is Hermitian.\n\n\n\n"
},

{
    "location": "reference.html#Base.push!",
    "page": "Reference",
    "title": "Base.push!",
    "category": "Function",
    "text": "push!(op, s, y)\n\nPush a new {s,y} pair into a L-BFGS operator.\n\n\n\npush!(op, s, y)\n\nPush a new {s,y} pair into a L-SR1 operator.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.reset!",
    "page": "Reference",
    "title": "LinearOperators.reset!",
    "category": "Function",
    "text": "reset!(data)\n\nResets the given LBFGS data.\n\n\n\nreset!(op)\n\nResets the LBFGS data of the given operator.\n\n\n\nreset!(data)\n\nReset the given LSR1 data.\n\n\n\nreset!(op)\n\nResets the LSR1 data of the given operator.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.shape",
    "page": "Reference",
    "title": "LinearOperators.shape",
    "category": "Function",
    "text": "m, n = shape(op)\n\nAn alias for size.\n\n\n\n"
},

{
    "location": "reference.html#Base.show",
    "page": "Reference",
    "title": "Base.show",
    "category": "Function",
    "text": "show(io, op)\n\nDisplay basic information about a linear operator.\n\n\n\n"
},

{
    "location": "reference.html#Base.size",
    "page": "Reference",
    "title": "Base.size",
    "category": "Function",
    "text": "m, n = size(op)\n\nReturn the size of a linear operator as a tuple.\n\n\n\nm = size(op, d)\n\nReturn the size of a linear operator along dimension d.\n\n\n\n"
},

{
    "location": "reference.html#LinearOperators.symmetric",
    "page": "Reference",
    "title": "LinearOperators.symmetric",
    "category": "Function",
    "text": "symmetric(op)\nissymmetric(op)\n\nDetermine whether the operator is symmetric.\n\n\n\n"
},

{
    "location": "reference.html#Utility-functions-1",
    "page": "Reference",
    "title": "Utility functions",
    "category": "section",
    "text": "check_ctranspose\ncheck_hermitian\ncheck_positive_definite\ndiag\nfull\nhermitian\npush!\nreset!\nshape\nshow\nsize\nsymmetric"
},

]}
