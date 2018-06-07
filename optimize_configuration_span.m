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
datasize = 1000;
deadline = 1337868;
cores_per_vm = 2;
lowest_vms = 3;
highest_vms = 100;
save_plots = false;

%% End configuration

ernest = load (ernest_txt);
query = ernest.query;
features = ernest.features;

vms = (lowest_vms:highest_vms)';
cores = cores_per_vm * vms;
datasizes = datasize * ones (size (cores));
full_X = build_ernest_matrix (datasizes, cores);
X = full_X(:, features);
t = X * ernest.theta;

deviations = deadline - t;
infeasible = (deviations < 0);
feasible = ! infeasible;
c_infeasible = cores (infeasible);
t_infeasible = t (infeasible);
c_feasible = cores (feasible);
t_feasible = t (feasible);

%% Visualization
figure;
plot (c_feasible, t_feasible, ".b", "LineWidth", 2, "DisplayName", "Feasible");
hold all;
plot (c_infeasible, t_infeasible, ".r", "LineWidth", 2, "DisplayName", "Infeasible");
axis auto;
limits = xlim ();
plot (limits, [deadline deadline], "-r", "LineWidth", 2, "DisplayName", "Deadline");
grid on;
axis auto;
xlabel Cores;
ylabel ("Execution time [ms]");
legend location NorthEast;

if (save_plots)
  filename = sprintf ("%s-datasize%d-deadline%.0f-cores%d", ...
                      query, datasize, deadline, cores_per_vm);
  print ("-depsc2", filename);
endif

pause (1);

%% Final solution
[c_opt, idx] = min (c_feasible);
t_opt = t_feasible(idx);
nu_opt = c_opt / cores_per_vm;

%% Output
query
features
datasize
nu_opt
c_opt
t_opt
deadline
perc_err = 100 * (deadline - t_opt) / deadline
