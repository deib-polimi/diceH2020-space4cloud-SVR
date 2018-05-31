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

clear all
close all hidden
clc

base_directory = "/Users/eugenio/Dottorato/Experiment Results/Azure-merged-data-A-D/ml/Q52";

configuration.runs = [6 8 10 12 14 16 18 20 22 24 26 28 32 34 36 38 40 42 44 46 48 52];

train_fraction = 0.8;

%% End of configuration

experimental_data = cell (size (configuration.runs));
for (ii = 1:numel (configuration.runs))
  name = sprintf ("%d.csv", configuration.runs(ii));
  filename = fullfile (base_directory, name);
  data = read_csv_table (filename, 1);
  t = data.applicationCompletionTime - data.applicationDeltaBeforeComputing;
  datasize = data.dataSize;
  cores = data.nContainers;
  experimental_data{ii} = [t, datasize, cores];
endfor

clean_experimental_data = cellfun (@(A) nthargout (1, @clear_outliers, A),
                                   experimental_data, "UniformOutput", false);

sample = vertcat (clean_experimental_data{:});

n = rows (sample);
idx = randperm (n);
n_train = round (train_fraction * n);
idx_tr = idx(1:n_train);
idx_tst = idx(n_train + 1:end);
n_test = numel (idx_tst);

y_tr = sample(idx_tr, 1);
datasize_tr = sample(idx_tr, 2);
cores_tr = sample(idx_tr, 3);
one = ones (size (y_tr));
sm = datasize_tr ./ cores_tr;
lm = log2 (cores_tr);
X_tr = [one, sm, lm, cores_tr];

y_tst = sample(idx_tst, 1);
datasize_tst = sample(idx_tst, 2);
cores_tst = sample(idx_tst, 3);
one = ones (size (y_tst));
sm = datasize_tst ./ cores_tst;
lm = log2 (cores_tst);
X_tst = [one, sm, lm, cores_tst];

theta = lsqnonneg (X_tr, y_tr);

y_hat_tr = X_tr * theta;
y_hat_tst = X_tst * theta;

train_mape = 100 * mean (abs ((y_tr - y_hat_tr) ./ y_tr));
test_mape = 100 * mean (abs ((y_tst - y_hat_tst) ./ y_tst));

one_table = sprintf ("%s/%d.csv", base_directory, configuration.runs(1));
fid = fopen (one_table, "r");
first_line = fgetl (fid);
fclose (fid);

query = strtrim (strrep (first_line, "Application class:", ""));

outfilename = fullfile (base_directory, "ernest.txt");
save (outfilename, "query", "theta", "n_train", "train_mape", ...
      "n_test", "test_mape");
