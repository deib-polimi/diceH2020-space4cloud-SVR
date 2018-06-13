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

real_vs_dagsim = "/Users/eugenio/Dottorato/Experiment Results/Ernest/D12v2-real_vs_dagsim.csv";
deadline = 300e3;
save_plots = false;

%% End configuration

function [cores, datasize] = parse_experiment (experiment)

  pieces = strsplit (experiment, "_");
  cores = str2double (pieces{1}) * str2double (pieces{2});
  datasize = str2double (pieces{4});

endfunction

%% Real work
data = read_csv_table (real_vs_dagsim);

[cores, datasizes] = cellfun (@parse_experiment, data.Experiment);
t = data.Measured;
queries = data.Query;
sample = [t, datasizes, cores];

distinct_queries = unique (queries);
for (ii = 1:numel (distinct_queries))
  current_query = distinct_queries{ii};
  query_idx = strcmp (queries, current_query);
  query_sample = sample(query_idx, :);
  results.(current_query) = [];

  distinct_datasizes = unique (query_sample(:, 2));
  for (jj = 1:numel (distinct_datasizes))
    current_datasize = distinct_datasizes(jj);
    datasize_idx = (query_sample(:, 2) == current_datasize);
    datasize_sample = query_sample(datasize_idx, [1 3]);

    distinct_cores = unique (datasize_sample(:, end));
    configurations = numel (distinct_cores);
    proper_t = NaN (configurations, 1);

    for (cc = 1:configurations)
      cores_idx = (datasize_sample(:, end) == distinct_cores(cc));
      proper_t(cc) = mean (datasize_sample(cores_idx, 1));
    endfor

    deviations = deadline - proper_t;
    infeasible = (deviations < 0);
    feasible = ! infeasible;
    c_infeasible = distinct_cores (infeasible);
    t_infeasible = proper_t (infeasible);
    c_feasible = distinct_cores (feasible);
    t_feasible = proper_t (feasible);

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
    title_string = sprintf ("%s - Dataset %d", current_query, current_datasize);
    title (title_string);

    if (save_plots)
      filename = sprintf ("%s-deadline%.0f-datasize%d", ...
                          current_query, deadline, current_datasize);
      print ("-depsc2", filename);
    endif

    pause (1);

    %% Final solution
    [c_opt, idx] = min (c_feasible);
    t_opt = t_feasible(idx);
    perc_err = 100 * (deadline - t_opt) / deadline;

    if (all (infeasible))
      c_opt = t_opt = perc_err = NaN;
    endif

    results.(current_query) = [results.(current_query); ...
                               current_datasize, configurations, ...
                               c_opt, t_opt, perc_err];
  endfor
endfor

%% Output
for ([matrix, query] = results)
  for (ii = 1:rows (matrix))
    query
    datasize = matrix(ii, 1)
    configurations = matrix(ii, 2)
    c_opt = matrix(ii, 3)
    t_opt = matrix(ii, 4)
    deadline
    perc_err = matrix(ii, 5)
    printf ("\n");
  endfor
endfor
