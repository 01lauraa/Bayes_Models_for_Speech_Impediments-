data {
  int<lower = 1> N_trials;
  int<lower =1> seq_length;
  array[N_trials*seq_length] int<lower = 1, upper = 5> resp;
  vector[N_trials*seq_length] intensity;
}

parameters {
  real<lower = 0, upper = 1> a;
  real<lower = 0, upper = 1> t;
  real<lower = 0, upper = 1> f;
  real alpha_c;
  real beta_c;
}

transformed parameters {
  array[N_trials*seq_length] simplex[5] theta;
  vector[N_trials*seq_length] c;
  c = inv_logit(alpha_c + intensity*beta_c);
  
  for(n in 1:N_trials*seq_length){
        theta[n,1] = 1 - a; // PR_OM
        theta[n,2] = a * (1 - t); // PR_IE
        theta[n,3] = a * t * (1 - f); // PR_PME
        theta[n,4] = a * t * f * (1 - c[n]); // PR_FR
        theta[n,5] = a * t * f * c[n]; // PR_SR
 }
}

model {
  target += beta_lpdf(a | 4,2);
  target += beta_lpdf(t | 2,2);
  target += beta_lpdf(f | 1,2);
  target += normal_lpdf(alpha_c | .3,.5);
  target += normal_lpdf(beta_c | 0,1) ;
  
  for(n in 1:N_trials*seq_length)
        target += categorical_lpmf(resp[n] | theta[n]);
}

generated quantities {
  array[N_trials*seq_length] int pred_resp;
  for(n in 1:N_trials*seq_length)
        pred_resp[n] = categorical_rng(theta[n]);
}

