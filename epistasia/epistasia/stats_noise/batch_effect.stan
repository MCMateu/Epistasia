data {
  int<lower=1> M;               // number of configurations (states x)
  int<lower=1> R;               // number of replicates (batches r)
  matrix[M, R] F_obs;           // observed landscape values \tilde{F}_r(x)
}

parameters {
  // True biological landscape \bar{F}(x)
  vector[M] F_bar;

  // Additive batch effects (free parameters for first R-1 batches),
  // last one is constrained by sum(a) = 0 in transformed parameters.
  vector[R-1] a_raw;

  // Multiplicative batch effects (log-scale),
  // will be shifted to enforce geom mean(b) = 1.
  vector<lower=-3, upper=3>[R] log_b_raw;

  // Single baseline noise (log-scale), bounded to avoid underflow
  real<lower=-5, upper=1> log_sigma;
}

transformed parameters {
  vector[R] a;                  // additive batch effects with sum(a) = 0
  vector[R] b;                  // multiplicative batch effects with geom mean(b) = 1
  real sigma;
  real gm_log_b;

  // Enforce sum_r a_r = 0
  for (r in 1:(R - 1)) {
    a[r] = a_raw[r];
  }
  a[R] = -sum(a_raw);           // last element set so that sum(a) = 0

  // Enforce geometric mean(b) = 1 via log_b_raw
  gm_log_b = mean(log_b_raw);   // mean of log(b) corresponds to log geom mean
  for (r in 1:R) {
    b[r] = exp(log_b_raw[r] - gm_log_b);
  }

  sigma = exp(log_sigma);
}

model {
  // Priors on batch effects
  a_raw      ~ normal(0, 0.5);
  log_b_raw  ~ normal(0, 0.3);

  // Priors on F_bar and sigma
  F_bar      ~ normal(0, 5);       // weakly informative prior for the landscape
  log_sigma  ~ normal(log(0.05), 0.5);

  // Likelihood
  for (m in 1:M) {
    for (r in 1:R) {
      F_obs[m, r] ~ normal(F_bar[m] + a[r], sigma * b[r]);
    }
  }
}
