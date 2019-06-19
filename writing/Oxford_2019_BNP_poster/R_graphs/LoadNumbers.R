# From the parametric sensitivy environment
for (numbername in names(data_env$numbers_list)) {
    if (numbername %in% names(data_env$number_precision_list)) {
        digits <- data_env$number_precision_list[[numbername]]
    } else {
        digits <- 0
    }
    DefineMacro(
        numbername,
        data_env$numbers_list[[numbername]],
        digits=digits)
}
