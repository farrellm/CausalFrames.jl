# What each built-in state reads from a row, for the window transforms'
# projection (`inputcolumns`). A dependent summarizer's state reads nothing:
# its dependencies declare the columns. A reversed `DotProduct` is dependent
# too, but declaring its columns anyway costs nothing, since its canonical
# dependency reads the same two.
inputcolumns(::Count) = ()
inputcolumns(::CountDistinct{C}) where {C} = (C,)
inputcolumns(::Sum{C}) where {C} = (C,)
inputcolumns(::SumPower{C}) where {C} = (C,)
inputcolumns(::Product{C}) where {C} = (C,)
inputcolumns(::DotProduct{A,B}) where {A,B} = (A, B)
inputcolumns(::CrossProduct{A,B}) where {A,B} = (A, B)
inputcolumns(::CoMoment{A,B}) where {A,B} = (A, B)
inputcolumns(::CentralMoments{C}) where {C} = (C,)
inputcolumns(::AgeWeightedSum{C}) where {C} = (C,)
inputcolumns(::SortedValues{C}) where {C} = (C,)
inputcolumns(::WindowValues{C}) where {C} = (C,)
inputcolumns(
    ::Union{Min{C},Max{C},First{C},Last{C},MinIndex{C},MaxIndex{C},
        MinWithIndex{C},MaxWithIndex{C}},
) where {C} = (C,)
inputcolumns(::FitModel{N,P,Y}) where {N,P,Y} = (P..., Y)
inputcolumns(
    ::Union{Moment,Mean,Variance,Std,Skewness,Kurtosis,Covariance,Correlation,
        LinearRegression,
        Quantile,Median,PercentRank,MeanAbsDev},
) = ()
# A renamed dependency reads what its inner summarizer reads, its moved hoisted
# terms under the names they moved to.
function inputcolumns(s::Scoped)
    cols = inputcolumns(s.inner)
    _, _, _, _, TD, TA = scopeplan(s)
    (cols === nothing || isempty(TD)) && return cols
    return map(c -> (j = findfirst(==(c), TD); j === nothing ? c : TA[j]), cols)
end
