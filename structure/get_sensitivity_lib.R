
# This is identical to same function in the previous get_sensitivity_lib.R
ReadStructureData <- function(input.basename, output.basename) {
  # Read the structure data and return a list of R data structures.
  # Args:
  #   *.basename: The base filename of the files missing only the file extensions
  #
  # input.basename is not used.

  `%.%` <- function(x, y) {
    paste(x, y, sep=".")
  }

  genome.matrix <- read.delim(output.basename %.% "genome", sep="", header=F)
  n <- nrow(genome.matrix)
  l <- ncol(genome.matrix)
  genome.matrix <- matrix(as.numeric(as.matrix(genome.matrix)), ncol=l, nrow=n)

  # Load the outputs of the VB analysis.
  mean.p <- read.delim(output.basename %.% "meanP", sep="", header=F)
  k <- ncol(mean.p)
  names(mean.p) <- paste("p", 1:k, sep=".")
  mean.q <- read.delim(output.basename %.% "meanQ", sep="", header=F)
  names(mean.q) <- paste("q", 1:k, sep=".")

  # The "var" files contain the actual variational parameters.
  # The varP file has two copies of variational parameters. -- the first
  # k columns are pi.var_beta, and the next k columns are pi.var_gamma.
  var.p <- as.matrix(read.delim(output.basename %.% "varP", sep="", header=F))
  colnames(var.p) <- paste(rep(c("p.beta", "p.gamma"), each=k), rep(1:k, 2), sep=".")
  var.q <- as.matrix(read.delim(output.basename %.% "varQ", sep="", header=F))
  colnames(var.q) <- paste("q", 1:k, sep=".")

  return(list(genome.matrix=genome.matrix, mean.p=mean.p, mean.q=mean.q,
              var.p=var.p, var.q=var.q))
}


SetMomentParams <- function(data) {
  pop_params <- list()

  pop_params$n_ind <- nrow(data$var.q) # Number of individuals
  pop_params$n_gene <- nrow(data$var.p) # Number of genes
  pop_params$n_pop <- ncol(data$var.q) # Number of populations

  pop_params$p_vec <- list()
  pop_params$omp_vec <- list()
  for (gene_ind in 1:pop_params$n_gene) {
    pop_params$p_vec[[gene_ind]] <- rep(NA, pop_params$n_pop)
    pop_params$omp_vec[[gene_ind]] <- rep(NA, pop_params$n_pop)
    e_log_p_vec <- c()
    e_log_omp_vec <- c()
    for (pop_ind in 1:pop_params$n_pop) {
      # If p ~ Beta(alpha, beta), then a row of data$var.p has the alpha parameters
      # for all the populations in the first n_pop columns, then the beta parameters
      # in the next n_pop columns.
      p_vec <- c(data$var.p[gene_ind, pop_ind], data$var.p[gene_ind, pop_params$n_pop + pop_ind])
      e_log_p <- DirichletELogX(p_vec)
      e_log_p_vec[pop_ind] <- e_log_p[1]
      e_log_omp_vec[pop_ind] <- e_log_p[2]
    }
    pop_params$p_vec[[gene_ind]] <- e_log_p_vec
    pop_params$omp_vec[[gene_ind]] <- e_log_omp_vec
  }

  pop_params$q_vec <- list()
  for (ind in 1:pop_params$n_ind) {
    e_log_q <- DirichletELogX(data$var.q[ind, ])
    pop_params$q_vec[[ind]] <- e_log_q
  }
  return(pop_params)
}

SetNaturalParams <- function(data) {
  nat_params <- list()

  nat_params$n_ind <- nrow(data$var.q) # Number of individuals
  nat_params$n_gene <- nrow(data$var.p) # Number of genes
  nat_params$n_pop <- ncol(data$var.q) # Number of populations

  # Actually these will be the natural parameters not the moment parameters.
  nat_params$p_vec <- list()
  nat_params$omp_vec <- list()
  for (gene_ind in 1:nat_params$n_gene) {
    nat_params$p_vec[[gene_ind]] <- rep(NA, nat_params$n_pop)
    nat_params$omp_vec[[gene_ind]] <- rep(NA, nat_params$n_pop)
    alpha_p_vec <- c()
    alpha_omp_vec <- c()
    for (pop_ind in 1:nat_params$n_pop) {
      # If p ~ Beta(alpha, beta), then a row of data$var.p has the alpha parameters
      # for all the populations in the first n_pop columns, then the beta parameters
      # in the next n_pop columns.
      p_vec <- c(data$var.p[gene_ind, pop_ind], data$var.p[gene_ind, nat_params$n_pop + pop_ind])
      alpha_p_vec[pop_ind] <- p_vec[1]
      alpha_omp_vec[pop_ind] <- p_vec[2]
    }
    nat_params$p_vec[[gene_ind]] <- alpha_p_vec
    nat_params$omp_vec[[gene_ind]] <- alpha_omp_vec
  }

  nat_params$q_vec <- list()
  for (ind in 1:nat_params$n_ind) {
    alpha_q <- data$var.q[ind, ]
    nat_params$q_vec[[ind]] <- alpha_q
  }
  return(nat_params)
}


SetPriors <- function(n_ind, n_gene, n_pop) {
  pp <- list()
  pp$n_ind <- n_ind
  pp$n_gene <- n_gene
  pp$n_pop <- n_pop
  pp$p_alpha <- 1
  pp$p_beta <- 1
  pp$q_alpha <- 1 / n_pop
  return(pp)
}


GetGenomeIntegerMatrix <- function(data) {
  return(matrix(as.integer(data$genome.matrix), nrow(data$genome.matrix), ncol(data$genome.matrix)))
}

ConvertParameterListToDataFrame <- function(par_list) {
  df_list <- list()
  for (n in 1:length(par_list)) {
    val <- par_list[[n]]
    df_list[[n]] <- data.frame(n=n, k=1:length(val), value=val)
  }
  return(do.call(rbind, df_list))
}

GetELBO <- function(genome_matrix_int, loc_natural_params, prior_params, use_kahan_sum=TRUE) {
  loc_moment_params <- GetMomentParameters(loc_natural_params)
  log_lik <- GetLogLikelihood(genome_matrix_int, loc_moment_params, prior_params, use_kahan_sum)
  entropy <- GetEntropy(genome_matrix_int, loc_natural_params, use_kahan_sum)
  cat("Log Lik: ", log_lik, "\n")
  cat("Entropy: ", entropy, "\n")
  return(log_lik + entropy)
}
