## Copyright 2018 Eugenio Gianniti
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

clear all;
close all hidden;
clc;

ernest_txt = "/Users/eugenio/Dottorato/Experiment Results/Ernest/query26-NNLS.txt";
ml_root = "/Users/eugenio/Dottorato/Experiment Results/TPCDS500-D_processed_logs/ml/Q26";
initial_idx = 1;
deadline = 360e3;
cores_per_vm = 4;

%% End configuration

ernest = load (ernest_txt);

ml_txt = fullfile (ml_root, "model.txt");
ml = load (ml_txt);

smallest = min (ml.configuration.runs);
smallest_name = sprintf ("%d.csv", smallest);
filename = fullfile (ml_root, smallest_name);
data = read_data (filename);

data(:, 1) -= data(:, 2);
data(:, end) = 1 ./ data(:, end);
relevant = data(initial_idx, ml.useful_columns);
datasize = relevant(end - 1);

mu = ml.working_mu';
sigma = ml.working_sigma';
scaled = (relevant - mu) ./ sigma;

%% Initial guess with ML
X_0 = scaled(:, 2:end - 1);
chi_c = sigma(1) / sigma(end) * ml.w(end);
chi_0 = mu(1) + ml.b * sigma(1) - chi_c * mu(end) + ...
        sigma(1) * X_0 * ml.w(1:end - 1);

c_0 = chi_c / (deadline - chi_0);
nu_0 = ceil (c_0 / cores_per_vm);
nu1 = max (nu_0, 1);

%% First feasibility assessment
c = cores_per_vm * nu1;
X = build_ernest_matrix (datasize, c);
t1 = X * ernest.theta;
feasible = (t1 <= deadline);

if (feasible)
  bounds.upper = nu1;
  bounds.lower = 0;
  nu = nu1 - 1;
else
  bounds.lower = nu1;
  bounds.upper = +Inf;
  nu = nu1 + 1;
endif

%% Auxiliary functions
function next = check_bounds (tentative, bounds)

  rounded_tentative = round (tentative);

  if (isfinite (bounds.upper))
    if (rounded_tentative > bounds.lower && rounded_tentative < bounds.upper)
      next = rounded_tentative;
    else
      next = round ((bounds.upper + bounds.lower) / 2);
    endif
  else % ! isfinite (bounds.upper)
    if (rounded_tentative > bounds.lower)
      next = rounded_tentative;
    else
      next = bounds.lower + 1;
    endif
  endif

endfunction

function nu = hyperbola (nu1, t1, nu2, t2, deadline)

  alpha = (t2 - t1) / (nu1 - nu2) * nu1 * nu2;
  beta = (t1 * nu1 - t2 * nu2) / (nu1 - nu2);
  nu = alpha / (deadline - beta);

endfunction

%% Local search with hyperbola
do
  c = cores_per_vm * nu;

  nu2 = nu1;
  nu1 = nu;
  t2 = t1;

  X = build_ernest_matrix (datasize, c);
  t1 = X * ernest.theta;
  feasible = (t1 <= deadline);

  if (feasible)
    bounds.upper = nu;
  else
    bounds.lower = nu;
  endif

  tentative = hyperbola (nu1, t1, nu2, t2, deadline);
  nu = check_bounds (tentative, bounds);
until (bounds.upper - bounds.lower == 1);

%% Final solution
nu_opt = bounds.upper
c_opt = cores_per_vm * nu_opt
X_opt = build_ernest_matrix (datasize, c_opt);
t_opt = X_opt * ernest.theta
perc_err = 100 * (deadline - t_opt) / deadline
