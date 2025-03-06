#' Numerically stable log(plogis(x)) := -log1p(exp(-x))
#'
#' This is a helper function to make logistic regression loss numerically stable.
#' Based on [this blog](https://fa.bianp.net/blog/2019/evaluate_logistic/) by Fabian Pedregosa
#' and [this note](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
#' by Martin Mächler
#'
#' @param x Vector of numbers
#'
#' @return Vector of results
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
#'
#' @examples
sigm_b <- function(t, y) {
  ifelse(
    test = t < 0,
    yes = { # maybe only this branch is needed
      exp_t <- exp(t)
      ((1 - y) * exp_t - y) / (1 + exp_t)
    },
    no = { # this is to prevent exp overflow, which seems to be more of a python problem.
      exp_nt <- exp(-t)
      ((1 - y) - y * exp_nt) / (1 + exp_nt)
    }
  )
}



#' Numerically stable log(n + exp(x))
#'
#' @param n Number
#' @param x Number
#'
#' @return log(1/(n + exp(-x)))
#' @examples
logNpexp <- function(n, x) {
  n <- c(n)
  x <- c(x)
  if (n == 0) {
    x
  } else {
    pm <- pmax.int(x, log(n))
    pm + log(n * exp(-pm)+exp(x - pm))
  }
}


#' Numerically stable exp(x)/(n + exp(x))
#'
#' @param n Number
#' @param x Number
#'
#' @return
#'
#' @examples
exp_by_npexp <- function(n, x) {
  res <- double(length = length(x))
  cnd <-  x > 0
  res[cnd] <- 1/(1+n*exp(-x[cnd]))
  res[!cnd] <- 1/(n/exp(x[!cnd]) + 1)
  res
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
#'
#' @examples
g_slr_list <- function(x, y, w) {
  t <- x %*% w
  s <- sigm_b(t, y) # might be unnecessary in R. exp doesn't overflow
  list(
    "objective" = sum(-logsig(t) + (1 - y) * (t)),
    "gradient" = c(crossprod(x = x, y = s))
  )
}


g_mlr1_list <- function(x, s, w) {
  w_lr <- w[2:length(w)]
  b <- w[1]
  t <- -1* x %*% w_lr
  expt <- exp(t)
  lNexp1pBsqT <- logNpexp(1 + b^2, t)
  list(
    "objective" = -sum(-s * lNexp1pBsqT + (1 - s) * (logNpexp(b^2, t) - lNexp1pBsqT)),
    "gradient" = -c(
      sum((1 - s) * (2 * b) / (b^2 + expt) - 2 * b / (1 + b^2 + expt)),
      crossprod(x = x, y = (exp_by_npexp(1 + b^2, t) + (s - 1) * exp_by_npexp(b^2, t)))
    )
  )
}


g_mlr2_list <- function(x, s, w, c_hat) {
  t <- -1* x %*% w
  logCpexpT <- logNpexp(1 - c_hat, t)
  list(
    "objective" = -sum(s * (log(c_hat) - logCpexpT) + logCpexpT + logsig(-t)),
    "gradient" = -c(crossprod(x = x, y = ((s - 1) * exp_by_npexp(1 - c_hat, t) + exp_by_npexp(1, t))))
  )
}


#' Simple Logistic Regression
#'
#' @param x NxP matrix-compatible table with independent variables.
#' @param y Vector of binary ([0, 1]) class assignments.
#'
#' @return Vector of coefficients after fitting the model.
#' @import nloptr
#' @export
#'
#' @examples
slr <- function(x, y, print_level = NULL) {
  xm <- cbind(intercept = 1, as.matrix(x))
  x0 <- rep(0, ncol(xm))

  eval_f_list <- function(w) {
    g_slr_list(x = xm, y = y, w = w)
  }

  opts <- list(
    "algorithm" = "NLOPT_LD_LBFGS",
    "xtol_rel" = 1.0e-12,
    "print_level" = print_level
  )

  res <- nloptr::nloptr(
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


#' Simple Logistic Regression
#'
#' Implementation based on Jaskie et al., 2019.
#'
#' Jaskie, Kristen, Charles Elkan, and Andreas Spanias. "A modified logistic regression for positive and unlabeled learning." 2019 53rd Asilomar Conference on Signals, Systems, and Computers. IEEE, 2019.
#'
#' @param x NxP matrix-compatible table with independent variables.
#' @param y Vector of binary ([0, 1]) class assignments with 1 corresponding to positive-unlabeled and 0 to unlabelled observations. Denoted as `s` in Jaskie et al.
#' @param ret_c Logical. Whether or not to return the estimated c - probability of positive sample to be unlabeled
#'
#' @return Named vector of coefficients. If `ret_c = TRUE` first element is `c_hat`.
#' @import nloptr
#' @export
#'
#' @examples
mlr <- function(x, y, ret_c = FALSE, print_level = NULL) {
  xm <- cbind(intercept = 1, as.matrix(x))
  x0 <- c(0.5, rep(0, ncol(xm)))

  if (sum(y) == 0) {
    stop(simpleError(paste("No positive observations in data")))
  }
  opts <- list(
    "algorithm" = "NLOPT_LD_LBFGS",
    "xtol_rel" = 1.0e-12,
    "print_level" = print_level,
    "maxeval" = -1 # objective based on `b` can bounce around for a while.
  )

  eval_f_mlr1 <- function(w) {
    g_mlr1_list(x = xm, s = y, w = w)
  }

  res1 <- nloptr::nloptr(
    x0 = x0,
    eval_f = eval_f_mlr1,
    lb = c(.Machine$double.eps, rep(-Inf, ncol(xm))),
    opts = opts
  )

  if (res1$status < 0) {
    tb <- table(y)
    warning("Numbers of observations per class: 0: ", tb["0"], " 1: ", tb["1"], call. = F)
    stop(simpleError(paste("NLopt returned an error code from step1:", res1$status, res1$message)))
  }

  c_hat <- 1 / (1 + res1$solution[1]^2)

  x0 <- rep(0, ncol(xm))

  eval_f_mlr2 <- function(w) {
    g_mlr2_list(x = xm, s = y, w = w, c_hat = c_hat)
  }

  res2 <- nloptr::nloptr(
    x0 = x0,
    eval_f = eval_f_mlr2,
    opts = opts
  )

  if (res2$status < 0) {
    tb <- table(y)
    warning("Numbers of observations per class: 0: ", tb["0"], " 1: ", tb["1"], call. = F)
    stop(simpleError(paste("NLopt returned an error code from step2:", res2$status, res2$message)))
  }

  coefs <- res2$solution
  names(coefs) <- colnames(xm)
  if (ret_c) coefs <- c(c_hat = c_hat, coefs)
  return(coefs)
}

mlr_ref <- function(x, y, ret_c = TRUE, print_level = NULL) {
  xm <- cbind(intercept = 1, as.matrix(x))
  x0 <- c(0.5, rep(0, ncol(xm)))

  if (sum(y) == 0) {
    stop(simpleError(paste("No positive observations in data")))
  }
  opts <- list(
    "algorithm" = "NLOPT_LD_LBFGS",
    "xtol_rel" = 1.0e-12,
    "print_level" = print_level,
    "maxeval" = -1 # objective based on `b` can bounce around for a while.
  )

  eval_f_mlr1 <- function(w) {
    g_mlr1_list(x = xm, s = y, w = w)
  }

  res1 <- nloptr::nloptr(
    x0 = x0,
    eval_f = eval_f_mlr1,
    lb = c(.Machine$double.eps, rep(-Inf, ncol(xm))),
    opts = opts
  )

  if (res1$status < 0) {
    tb <- table(y)
    warning("Numbers of observations per class: 0: ", tb["0"], " 1: ", tb["1"], call. = F)
    stop(simpleError(paste("NLopt returned an error code from step1:", res1$status, res1$message)))
  }

  c_hat <- 1 / (1 + res1$solution[1]^2)

  # x0 <- rep(0, ncol(xm))
  #
  # eval_f_mlr2 <- function(w) {
  #   g_mlr2_list(x = xm, s = y, w = w, c_hat = c_hat)
  # }
  #
  # res2 <- nloptr::nloptr(
  #   x0 = x0,
  #   eval_f = eval_f_mlr2,
  #   opts = opts
  # )
  #
  # if (res2$status < 0) {
  #   tb <- table(y)
  #   warning("Numbers of observations per class: 0: ", tb["0"], " 1: ", tb["1"], call. = F)
  #   stop(simpleError(paste("NLopt returned an error code from step2:", res2$status, res2$message)))
  # }

  coefs <- res1$solution[2:length(res1$solution)]
  names(coefs) <- colnames(xm)
  if (ret_c) coefs <- c(c_hat = c_hat, coefs)
  return(coefs)
}
