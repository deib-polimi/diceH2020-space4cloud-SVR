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

clear all
close all hidden
clc

interesting_index = 4;
directory = "/Users/eugenio/Dottorato/Experiment Results/Power8/500/ml/query55";
csv_name = "44.csv";
model_name = "model.txt";

%% End of the configurations

csv_file = [directory, "/", csv_name];
model_file = [directory, "/", model_name];

load (model_file);

mu_t = working_mu(1);
mu_c = working_mu(end);
mu_all = working_mu(2:end - 1);

sigma_t = working_sigma(1);
sigma_c = working_sigma(end);
sigma_all = working_sigma(2:end - 1);

w_c = w(end);
w_all = w(1:end - 1);

data = read_data (csv_file);

interesting_example = data(interesting_index, useful_columns);
interesting_example(:, end) = 1 ./ interesting_example(:, end);

t = interesting_example(1);
c_inv = interesting_example(end);
features = interesting_example(2:end - 1);

chi_c = w_c * sigma_t / sigma_c;

scaled_features = (features(:) - mu_all) ./ sigma_all;
chi_0 = mu_t + sigma_t * (b - mu_c / sigma_c + w_all' * scaled_features);

chi_c
chi_0

prediction = chi_c * c_inv + chi_0
t
err = 100 * abs (t - prediction) / t
