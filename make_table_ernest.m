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

comparison_file = "/Users/eugenio/Downloads/real_vs_dagsim.csv";
ml_dir = "/Users/eugenio/Downloads/ernest";

%% End of configuration

function [cores, datasize] = parse_experiment (experiment)
  tokens = strsplit (experiment, "_");
  machines = str2num (tokens{1});
  cores_per_vm = str2num (tokens{2});
  datasize = str2num (tokens{4});
  cores = machines * cores_per_vm;
endfunction

dagsim = read_csv_table (comparison_file);
queries = unique (dagsim.Query);

n_r = numel (dagsim.Query);
all_cores = all_times = all_dagsim = all_ernest = NaN (n_r, 1);
all_queries = cell (n_r, 1);

for (ii = 1:numel (queries))
  query = queries{ii};
  filename = fullfile (ml_dir, query, "ernest.txt");
  current_ernest = load (filename);

  idx = (strcmp (dagsim.Query, query));
  current_experiments = dagsim.Experiment(idx);
  all_queries(idx) = dagsim.Query(idx);
  all_times(idx) = current_times = dagsim.Measured(idx);
  all_dagsim(idx) = 100 * dagsim.("Error[1]")(idx);

  [cores, datasizes] = cellfun (@parse_experiment, current_experiments);
  all_cores(idx) = cores;
  X = build_ernest_matrix (datasizes, cores);
  y_hat = X * current_ernest.theta;
  all_ernest(idx) = 100 * (y_hat - current_times) ./ current_times;
endfor

for (ii = 1:n_r)
  printf ("\\taburow{%s}{%d}{%.0f}{%.2f}{%.2f}\n", all_queries{ii}, ...
          all_cores(ii), all_times(ii), all_dagsim(ii), all_ernest(ii));
endfor

dagsim_mape = mean (abs (all_dagsim))
ernest_mape = mean (abs (all_ernest))

dagsim_max = max (abs (all_dagsim))
ernest_max = max (abs (all_ernest))
