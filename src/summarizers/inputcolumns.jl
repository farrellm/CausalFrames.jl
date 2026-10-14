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
inputcolumns(::CoMoment{A,B}) where {A,B} = (A, B)
inputcolumns(::AgeWeightedSum{C}) where {C} = (C,)
inputcolumns(::SortedValues{C}) where {C} = (C,)
inputcolumns(::WindowValues{C}) where {C} = (C,)
inputcolumns(
    ::Union{Min{C},Max{C},First{C},Last{C},MinIndex{C},MaxIndex{C},
        MinWithIndex{C},MaxWithIndex{C}},
) where {C} = (C,)
inputcolumns(::FitModel{N,P,Y}) where {N,P,Y} = (P..., Y)
inputcolumns(
    ::Union{Moment,Mean,Variance,Std,Covariance,Correlation,LinearRegression,
        Quantile,Median,PercentRank,MeanAbsDev},
) = ()
