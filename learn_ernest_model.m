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

base_directory = "/Users/eugenio/Dottorato/Experiment Results/POWER8/full/ml/query40";

use_nnls = true;
features = true (1, 6);
features(end - 1:end) = false;

configuration.runs = [6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44];
configuration.missing_runs = [];

configuration.seed = 17;
configuration.train_fraction = 0.8;

%% End of configuration

experimental_data = cell (size (configuration.runs));
for (ii = 1:numel (configuration.runs))
  name = sprintf ("%d.csv", configuration.runs(ii));
  filename = fullfile (base_directory, name);
  data = read_csv_table (filename, 1);
  t = data.applicationCompletionTime - data.applicationDeltaBeforeComputing;
  datasize = data.dataSize;

  if (isfield (data, "nContainers"))
    cores = data.nContainers;
  else
    cores = data.nCores;
  endif

  experimental_data{ii} = [t, datasize, cores];
endfor

clean_experimental_data = cellfun (@(A) nthargout (1, @clear_outliers, A),
                                   experimental_data, "UniformOutput", false);

[available_idx, missing_idx] = find_configurations (configuration.runs, ...
                                                    configuration.missing_runs);

sample = vertcat (clean_experimental_data{available_idx});
missing_sample = vertcat (clean_experimental_data{missing_idx});
n_miss = rows (missing_sample);

rand ("seed", configuration.seed);
n = rows (sample);
idx = randperm (n);
n_train = round (configuration.train_fraction * n);
idx_tr = idx(1:n_train);
idx_tst = idx(n_train + 1:end);
n_test = numel (idx_tst);

y_tr = sample(idx_tr, 1);
datasize_tr = sample(idx_tr, 2);
cores_tr = sample(idx_tr, 3);
full_X_tr = build_ernest_matrix (datasize_tr, cores_tr);
X_tr = full_X_tr(:, features);

y_tst = sample(idx_tst, 1);
datasize_tst = sample(idx_tst, 2);
cores_tst = sample(idx_tst, 3);
full_X_tst = build_ernest_matrix (datasize_tst, cores_tst);
X_tst = full_X_tst(:, features);

if (n_miss > 0)
  y_miss = missing_sample(:, 1);
  datasize_miss = missing_sample(:, 2);
  cores_miss = missing_sample(:, 3);
  full_X_miss = build_ernest_matrix (datasize_miss, cores_miss);
  X_miss = full_X_miss(:, features);
else
  y_miss = NaN (0, 1);
  X_miss = NaN (0, sum (features));
endif

if (use_nnls)
  theta = lsqnonneg (X_tr, y_tr);
else
  pkg load statistics;
  theta = regress (y_tr, X_tr);
endif

y_hat_tr = X_tr * theta;
y_hat_tst = X_tst * theta;
y_hat_miss = X_miss * theta;

train_mape = 100 * mean (abs ((y_tr - y_hat_tr) ./ y_tr));
test_mape = 100 * mean (abs ((y_tst - y_hat_tst) ./ y_tst));
missing_mape = 100 * mean (abs ((y_miss - y_hat_miss) ./ y_miss));

name = sprintf ("%d.csv", configuration.runs(1));
one_table = fullfile (base_directory, name);
fid = fopen (one_table, "r");
first_line = fgetl (fid);
[~] = fclose (fid);

query = strtrim (strrep (first_line, "Application class:", ""));

bases = {"ernest_ols.txt", "ernest_nnls.txt"};
outfilename = fullfile (base_directory, bases{1 + use_nnls});
save (outfilename, "query", "features", "theta", "n_train", "train_mape", ...
      "n_test", "test_mape", "n_miss", "missing_mape", "configuration");
