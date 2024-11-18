library(tidyverse)
library(nloptr)



#' Numerically stable log(plogis(x))
#'
#' This is a helper function to make logistic regression loss numerically stable.
#' Based on [this blog](https://fa.bianp.net/blog/2019/evaluate_logistic/) by Fabian Pedregosa
#' and [this note](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
#' by Martin Mächler
#'
#' @param x Vector of numbers
#'
#' @return Vector of results
#' @export
#'
#' @examples
logsig <- function(x) {
  ifelse(
    test = x < -33.3,
    yes = x,
    no = ifelse(
      test = x <= -18,
      yes = x - exp(x),
      no = ifelse(
        test = x <= 37,
        yes = -log1p(exp(-x)),
        no = -exp(-x)
      )
    )
  )
}



#' Numerically stable `plogis(t) - y`
#'
#' This is a helper function to make the gradient of the logistic regression loss
#' numerically stable.
#' Based on [this blog](https://fa.bianp.net/blog/2019/evaluate_logistic/) by Fabian Pedregosa.
#'
#' @param t Vector of inputs to the logistic/sigmoid function
#' @param y Dependent variable: [0, 1]. Should be the same length as t.
#'
#' @return Vector of outputs
#' @export
#'
#' @examples
sigm_b <- function(t, y) {
  ifelse(
    test = t < 0,
    yes = {
      exp_t <- exp(t)
      ((1 - y) * exp_t - y) / (1 + exp_t)
    },
    no = {
      exp_nt <- exp(-t)
      ((1 - y) - y * exp_nt) / (1 + exp_nt)
    }
  )
}



logNpexp <- function(z, x) {
  ex <- exp(-x)
  ifelse(z > ex, log(z) + log1p(ex/z), -x + log1p(z*exp(x)))
}


#' Simple Logistic Regression objective and gradient
#'
#' This is a helper function that defines SLR's objective and gradient in order
#' to fit the parameters via NLopt.
#'
#' @param x NxP matrix of independent variables. First column should be 1 - for intercept
#' @param y Vector of observations. 0 or 1.
#' @param w Vector of model weight. This parameter will be optimised by NLopt.
#'
#' @return List of functions: objective and gradient
#' @export
#'
#' @examples
g_slr_list <- function(x, y, w) {
  t <- x %*% w
  s <- sigm_b(t, y)
  list(
    "objective" = sum(-logsig(t) + (1 - y) * (t)),
    "gradient" = c(t(x) %*% s)
  )
}


# g_mlr_list1 <- function(x, s, w) {
#   w_lr <- w[2:length(w)]
#   b <- w[1]
#   t <- x %*% w_lr
#   list(
#     "objective" = sum(-s*log(1 + b^2 + exp(-t)) + (1-s)*(log(b^2 + exp(-t))-log(1+ b^2 + exp(-t)))),
#     "gradient" = c(
#       (2*b*exp(t)*(exp(t)*(b^2*s + s + 1) + s))/((b^2*exp(t) + 1)*((b^2+1)*exp(t)+1)),
#       # (exp(-t)*(s*(x+b^2 + t(x)%*%exp(-t))))/()
#       # s%*%x/((b^2 + 1)*exp(t)) + ((s-1) %*% x)/((b^2 + 1)*exp(t))
#       )
#   )
# }


g_mlr1<- function(x, s, w) {
  w_lr <- w[2:length(w)]
  b <- w[1]
  t <- x %*% w_lr
  sum(-s*logNpexp(1 + b^2, t) + (1-s)*(logNpexp(b^2, t)-logNpexp(1+ b^2, t)))
}


g_mlr2<- function(x, s, w, c_hat) {
  t <- x %*% w
  logCpexpT <- logNpexp(1-c_hat,t)
  # sum(s*(log(c_hat) - logsig(-t)) + (1-s)*(logNpexp(1-c_hat,t) - logsig(-t)))
  sum(s*(log(c_hat) - logCpexpT) + logCpexpT - logsig(-t))
  # sum(s*(log(c_hat) - logsig(-t)) + (1-s)*(logNpexp(1-c_hat,t) - logsig(-t)))
}
g_mlr2_grad <- function(x, s, w, c_hat) {
  t <- x %*% w
  sum(s*(log(c_hat) - logsig(-t)) + (1-s)*(logNpexp(1-c_hat,t) - logsig(-t)))
}
# eval_f_mlr2 <- function(w) {
#   -g_mlr2(x = xm, s = tr_data$s, w = w, c_hat = c_hat)
# }


# slr(x = tr_data %>% select(x1, x2), y = tr_data$y)


# tr_data %>% select(x1, x2)
#
# tr_data
#
# xm <- cbind(intercept = 1, as.matrix(tr_data %>% select(x1, x2)))
# x0 <- rep(.1, ncol(xm))



# opts <- list("algorithm" = "NLOPT_LD_LBFGS",
# opts <- list("algorithm" = "NLOPT_LN_COBYLA",
#   "xtol_rel" = 1.0e-12, "maxeval" = 1e4
# )

# res <- nloptr(
#   x0 = x0,
#   eval_f = eval_f_mlr2,
#   opts = opts
# )
# res
#

mlr <- function(x, s) {
  xm <- cbind(intercept = 1, as.matrix(x))
  x0 <- rep(0.1, ncol(xm)+1)

  opts <- list("algorithm" = "NLOPT_LN_COBYLA",
    "xtol_rel" = 1.0e-12, "maxeval" = 1e4
  )

  eval_f_mlr1 <- function(w) {
    -g_mlr1(x = xm, s = s, w = w)
  }

  res1 <- nloptr(
    x0 = x0,
    eval_f = eval_f_mlr1,
    opts = opts
  )

  if (res1$status < 0) {
    stop(simpleError(paste("NLopt returned an error code:", res1$status, res1$message)))
  }

  c_hat <- 1/(1+res1$solution[1]^2)

  x0 <- rep(.1, ncol(xm))

  eval_f_mlr2 <- function(w) {
    -g_mlr2(x = xm, s = s, w = w, c_hat = c_hat)
  }

  res2 <- nloptr(
    x0 = x0,
    eval_f = eval_f_mlr2,
    opts = opts
  )

  if (res2$status < 0) {
    stop(simpleError(paste("NLopt returned an error code:", res2$status, res2$message)))
  }

  coefs <- res2$solution
  names(coefs) <- colnames(xm)
  return(coefs)
}


# mlr(x = tr_data %>% select(x1,x1), s = tr_data$s)



#' Simple Logistic Regression
#'
#' @param x NxP matrix-compatible table with independent variables.
#' @param y Vector of binary ([0, 1]) class assignments.
#'
#' @return Vector of coefficients after fitting the model.
#' @export
#'
#' @examples
slr <- function(x, y) {
  xm <- cbind(intercept = 1, as.matrix(x))
  x0 <- rep(0, ncol(xm))

  eval_f_list <- function(w) {
    g_slr_list(x = xm, y = y, w = w)
  }

  opts <- list(
    "algorithm" = "NLOPT_LD_LBFGS",
    # opts <- list("algorithm" = "NLOPT_LN_COBYLA",
    "xtol_rel" = 1.0e-12, "maxeval" = 1e4
  )

  res <- nloptr(
    x0 = x0,
    eval_f = eval_f_list,
    opts = opts
  )

  if (res$status < 0) {
    stop(simpleError(paste("NLopt returned an error code:", res$status, res$message)))
  }

  coefs <- res$solution
  names(coefs) <- colnames(xm)
  return(coefs)
}


