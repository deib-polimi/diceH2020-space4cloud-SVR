## Copyright 2017-2018 Eugenio Gianniti
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

base_directory = "/Users/eugenio/Dottorato/Experiment Results/TPCDS500-D_processed_logs/ml/Q26";

only_containers = false;

configuration.runs = [12 16 20 24 28 32 36 40 44 48 52];
configuration.missing_runs = [];

configuration.seed = 17;
configuration.train_fraction = 0.6;
configuration.test_fraction = 0.2;

configuration.options = "-s 3 -t 0 -q -h 0 ";
configuration.C_range = linspace (1e-4, 1, 20);
configuration.epsilon_range = linspace (1e-4, 1, 20);

%% End of configuration

experimental_data = cell (size (configuration.runs));
for (ii = 1:numel (configuration.runs))
  inbase = sprintf ("%d.csv", configuration.runs(ii));
  filename = fullfile (base_directory, inbase);
  experimental_data{ii} = read_data (filename);
endfor

clean_experimental_data = cellfun (@(A) nthargout (1, @clear_outliers, A),
                                   experimental_data, "UniformOutput", false);

avg_times = cellfun (@(A) mean (A(:, 1) - A(:, 2)), clean_experimental_data);

[available_idx, missing_idx] = find_configurations (configuration.runs, ...
                                                    configuration.missing_runs);

sample = vertcat (clean_experimental_data{available_idx});
sample(:, 1) -= sample(:, 2);
sample(:, end) = 1 ./ sample(:, end);

rand ("seed", configuration.seed);
idx = randperm (rows (sample));
shuffled = sample(idx, :);

[~, mu, sigma] = zscore (shuffled);

constant_columns = find (sigma == 0);
cols = 1:columns (shuffled);
useful_columns = setdiff (cols, constant_columns);
useful_columns = setdiff (useful_columns, [2]);
working_sample = shuffled(:, useful_columns);
working_mu = mu(useful_columns);
working_sigma = sigma(useful_columns);

if (only_containers)
  useful_columns = [useful_columns(1); useful_columns(end)];
  working_sample = [working_sample(:, 1), working_sample(:, end)];
  working_mu = [working_mu(1); working_mu(end)];
  working_sigma = [working_sigma(1); working_sigma(end)];
endif

weights = ones (rows (working_sample), 1);

results = model_selection_with_thresholds (working_sample, weights, avg_times,
                                           configuration);

model = results.model;
b = - model.rho;
w = model.SVs' * model.sv_coef;
useful_columns = useful_columns(:);
working_mu = working_mu(:);
working_sigma = working_sigma(:);

C = results.C;
epsilon = results.epsilon;
train_error = results.train_error;
test_error = results.test_error;
cv_error = results.cv_error;

inbase = sprintf ("%d.csv", configuration.runs(1));
one_table = fullfile (base_directory, inbase);
fid = fopen (one_table, "r");
first_line = fgetl (fid);
second_line = strtrim (fgetl (fid));
fclose (fid);

query = strtrim (strrep (first_line, "Application class:", ""));
headers = strsplit (second_line, ",");
% +1 to discard the applicationId, 2:end to avoid the predicted time
useful_headers = { headers{useful_columns(2:end) + 1} }';

outfilename = fullfile (base_directory, "model.txt");
save (outfilename, "b", "w", "useful_headers", "useful_columns", ...
      "working_mu", "working_sigma", "C", "epsilon", ...
      "train_error", "test_error", "cv_error", "configuration");

data.b = b;
data.mu_t = working_mu(1);
data.sigma_t = working_sigma(1);

for (ii = 1:numel (w))
  feature.w = w(ii);
  feature.mu = working_mu(ii + 1);
  feature.sigma = working_sigma(ii + 1);
  name = useful_headers{ii};

  if (strcmp (name, "nContainers"))
    name = "x";
  endif

  if (strcmp (name, "users"))
    name = "h";
  endif

  features.(name) = feature;
endfor

if (! isfield (features, "h"))
  feature.w = 0;
  feature.mu = 0;
  feature.sigma = 1;
  features.h = feature;
endif

data.mlFeatures = features;
full_data.(query) = data;

pkg load io
json_content = object2json (full_data);
json_filename = fullfile (base_directory, "model.json");
fid = fopen (json_filename, "w");
fdisp (fid, json_content);
fclose (fid);
