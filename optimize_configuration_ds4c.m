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

ernest_txt = "/Users/eugenio/Dottorato/Experiment Results/POWER8/full/ml/query40/ernest.txt";
ml_root = "/Users/eugenio/Dottorato/Experiment Results/POWER8/1000/ml/query40";
initial_idx = 1;
deadline = 1337868;
cores_per_vm = 2;
stringent_bound_for_derivative = true;

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
datasize = data(initial_idx, end - 1);

mu = ml.working_mu';
sigma = ml.working_sigma';
scaled = (relevant - mu) ./ sigma;

labeled_data = read_csv_table (filename, 1);
labels = fieldnames (labeled_data);
idx = strncmp (labels, "nTask", 5);
tasks = max (cellfun (@(l) max (labeled_data.(l)), labels(idx)));
max_vms = ceil (tasks / cores_per_vm);

if (stringent_bound_for_derivative)
  theta_der = ernest.theta(2:end);

  if (theta_der(3) > 0)
    crossing = (sqrt (theta_der(2) ^ 2 + ...
                      4 * theta_der(1) * theta_der(3) * datasize) - ...
                theta_der(2)) / 4 / theta_der(3);
  elseif (theta_der(2) > 0)
    crossing = theta_der(1) * datasize / theta_der(2);
  endif

  max_upper = min (round (crossing), max_vms);
else
  max_upper = max_vms;
endif

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
  bounds.upper = max_upper;
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
iterations = 0;

do
  ++iterations;

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
query = ernest.query
datasize
iterations
nu_opt = bounds.upper
c_opt = cores_per_vm * nu_opt
X_opt = build_ernest_matrix (datasize, c_opt);
t_opt = X_opt * ernest.theta
deadline
perc_err = 100 * (deadline - t_opt) / deadline
