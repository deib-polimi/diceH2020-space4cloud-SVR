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

infile = "/Users/eugenio/Downloads/ernest/Q52-Azure-D12v2.csv";
outdir = "/Users/eugenio/Downloads/ernest";

use_nnls = true;
features = true(1, 5);
features(end) = false;

missing_cores = 32:4:52;
missing_datasizes = [750 1000];

seed = 17;
train_fraction = 0.8;

%% End of configuration

data = read_csv_table (infile, 1);
t = data.applicationCompletionTime - data.applicationDeltaBeforeComputing;
datasize = data.dataSize;
cores = data.nContainers;
full_sample = [t, datasize, cores];

[~, clean_idx] = clear_outliers (t);
sample = full_sample(clean_idx, :);
clean_cores = sample(:, end);
clean_sizes = sample(:, end - 1);

if (! all (ismember (missing_cores, clean_cores)))
  error ("the configured 'missing_cores' are not compatible with available data");
endif

if (! all (ismember (missing_datasizes, clean_sizes)))
  error ("the configured 'missing_datasizes' are not compatible with available data");
endif

cores_idx = ismember (clean_cores, missing_cores);
datasize_idx = ismember (clean_sizes, missing_datasizes);

missing_idx = cores_idx | datasize_idx;
available_sample = sample(! missing_idx, :);
missing_sample = sample(missing_idx, :);
n_missing = sum (missing_idx);
available_cores = unique (available_sample(:, end));
available_datasizes = unique (available_sample(:, end - 1));

rand ("seed", seed);
n = rows (available_sample);
idx = randperm (n);
n_train = round (train_fraction * n);
idx_tr = idx(1:n_train);
idx_tst = idx(n_train + 1:end);
n_test = numel (idx_tst);

y_tr = available_sample(idx_tr, 1);
datasize_tr = available_sample(idx_tr, 2);
cores_tr = available_sample(idx_tr, 3);
full_X_tr = build_ernest_matrix (datasize_tr, cores_tr);
X_tr = full_X_tr(:, features);

y_tst = available_sample(idx_tst, 1);
datasize_tst = available_sample(idx_tst, 2);
cores_tst = available_sample(idx_tst, 3);
full_X_tst = build_ernest_matrix (datasize_tst, cores_tst);
X_tst = full_X_tst(:, features);

if (n_missing > 0)
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

abs_residuals_train = abs ((y_tr - y_hat_tr) ./ y_tr);
abs_residuals_test = abs ((y_tst - y_hat_tst) ./ y_tst);
abs_residuals_miss = abs ((y_miss - y_hat_miss) ./ y_miss);

train_mape = 100 * mean (abs_residuals_train);
test_mape = 100 * mean (abs_residuals_test);
missing_mape = 100 * mean (abs_residuals_miss);
max_train_pe = 100 * max (abs_residuals_train);
max_test_pe = 100 * max (abs_residuals_test);
max_missing_pe = 100 * max (abs_residuals_miss);

fid = fopen (infile, "r");
first_line = fgetl (fid);
[~] = fclose (fid);

query = strtrim (strrep (first_line, "Application class:", ""));

techniques = {"OLS", "NNLS"};
technique = techniques{1 + use_nnls};
outbase = sprintf ("%s-%s.txt", query, technique);
outfilename = fullfile (outdir, outbase);
save (outfilename, "query", "technique", "features", "seed", "theta", ...
      "available_cores", "missing_cores", ...
      "available_datasizes", "missing_datasizes", ...
      "n_train", "train_mape", "max_train_pe", ...
      "n_test", "test_mape", "max_test_pe", ...
      "n_missing", "missing_mape", "max_missing_pe");
