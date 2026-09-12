# API reference

Every operator and summarizer has its own section on one of the pages below;
the links here go straight to it.

| Category | Operators |
|---|---|
| [Frames and pipelines](core.md) | [`Context`](@ref), [`CausalFrame`](@ref), [`CausalPipeline`](@ref), [`load`](@ref), [`stream`](@ref), [`scan`](@ref) |
| [Sources](sources.md) | [`emptyframe`](@ref "emptyframe"), [`concatenate`](@ref "concatenate"), [`merge`](@ref "merge"), [`clock`](@ref "clock") |
| [File I/O](io.md) | [`readcsv`](@ref "readcsv"), [`readparquet`](@ref "readparquet"), [`readjls`](@ref "readjls"), [`writecsv`](@ref "writecsv"), [`writeparquet`](@ref "writeparquet"), [`writejls`](@ref "writejls") |
| [Row transformations](rows.md) | [`filterrows`](@ref "filterrows"), [`addcolumns`](@ref "addcolumns"), [`head`](@ref "head"), [`lastrow`](@ref "lastrow"), [`lag`](@ref "lag"), [`Acausal.lead`](@ref "Acausal.lead"), [`settime`](@ref "settime"), [`Acausal.settime`](@ref "Acausal.settime") |
| [Column transformations](columns.md) | [`selectcolumns`](@ref "selectcolumns"), [`dropcolumns`](@ref "dropcolumns"), [`reordercolumns`](@ref "reordercolumns"), [`forwardfill`](@ref "forwardfill"), [`fillmissing`](@ref "fillmissing"), [`asofjoin`](@ref "asofjoin"), [`Acausal.futurejoin`](@ref "Acausal.futurejoin") |
| [Summarizing transforms](summarizing.md) | [`summarize`](@ref "summarize"), [`summarizecycles`](@ref "summarizecycles"), [`intervalize`](@ref "intervalize"), [`summarizewindows`](@ref "summarizewindows"), [`addsummarycolumns`](@ref "addsummarycolumns"), [`addrollingcolumns`](@ref "addrollingcolumns") |
| [Model fitting (MLJ)](models.md) | [`applymodels`](@ref "applymodels"), [`addpredictions`](@ref "addpredictions"), [`modelreports`](@ref "modelreports") |

The [summarizers](summarizers.md) the summarizing transforms take:

| Category | Summarizers |
|---|---|
| Counting | [`Count`](@ref "Count"), [`CountDistinct`](@ref "CountDistinct") |
| Accumulating | [`Sum`](@ref "Sum"), [`SumPower`](@ref "SumPower"), [`Product`](@ref "Product"), [`DotProduct`](@ref "DotProduct") |
| Moments | [`Moment`](@ref "Moment"), [`Mean`](@ref "Mean"), [`Variance`](@ref "Variance"), [`Std`](@ref "Std") |
| Relating two columns | [`Covariance`](@ref "Covariance"), [`Correlation`](@ref "Correlation"), [`LinearRegression`](@ref "LinearRegression") |
| Tracking a single row | [`Min`](@ref "Min"), [`Max`](@ref "Max"), [`First`](@ref "First"), [`Last`](@ref "Last") |
| Model fitting | [`FitModel`](@ref "FitModel") |
