# Deprecated use of positional argument mem
@deprecate LBFGSOperator(T :: DataType, n :: Int, mem :: Int; kwargs...) LBFGSOperator(T, n; mem=mem, kwargs...) false
@deprecate LBFGSOperator(n :: Int, mem :: Int; kwargs...) LBFGSOperator(n; mem=mem, kwargs...) false
@deprecate InverseLBFGSOperator(T :: DataType, n :: Int, mem :: Int; kwargs...) InverseLBFGSOperator(T, n; mem=mem, kwargs...) false
@deprecate InverseLBFGSOperator(n :: Int, mem :: Int; kwargs...) InverseLBFGSOperator(n; mem=mem, kwargs...) false
@deprecate LSR1Operator(T :: DataType, n :: Int, mem :: Int; kwargs...) LSR1Operator(T, n; mem=mem, kwargs...) false
@deprecate LSR1Operator(n :: Int, mem :: Int; kwargs...) LSR1Operator(n; mem=mem, kwargs...) false

