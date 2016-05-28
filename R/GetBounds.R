## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Obtaining confidence intervals after performing diagnostic checking
################################################################################

GetBounds <- function(duality, alpha, position) {
    n_path <- dim(duality$primal)[3]
    primal <- mean(duality$primal[1, position,]) 
    primal_error <- qnorm(1 - alpha/2) * sd(duality$primal[1, position,])/sqrt(n_path)
    dual <- mean(duality$dual[1, position,])
    dual_error <- qnorm(1 - alpha/2) * sd(duality$dual[1, position,])/sqrt(n_path)
    return(c(primal - primal_error, dual + dual_error))
}
