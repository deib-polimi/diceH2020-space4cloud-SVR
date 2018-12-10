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

root = "/Users/eugenio/Dottorato/Experiment Results/Ernest/POWER8";
job = "kmeans";

use_nnls = true;
features = true (1, 5);
features(end) = false;

% Positive: training, negative: test
if (ismember (job, {"query26"}))
  datasizes = { [250 -250]; ...
                [750 -750]; ...
                [1000 -1000]; ...
                [250 750 -1000]; ...
                [250 1000 -750]; ...
                [750 1000 -250]; ...
              };

  cases = { [6 10 14 18 24 28 32 36 40 44]; ...
            [6 12 18 26 32 38 44]; ...
            [6 14 24 32 40 44]; ...
            [6 16 30 42 44]; ...
            [6 8 18 32 44]; ...
            [6 10 16 26 44]; ...
          };
elseif (ismember (job, {"kmeans"}))
  datasizes = { [5 -5]; ...
                [10 -10]; ...
                [15 -15]; ...
                [20 -20]; ...
                [5 10 15 -20]; ...
                [5 10 20 -15]; ...
                [5 15 20 -10]; ...
                [10 15 20 -5]; ...
              };

  cases = { [6 10 14 18 22 26 30 34 38 42 46]; ...
            [6 12 18 24 30 36 42 48]; ...
            [6 14 22 30 38 48]; ...
            [6 16 26 36 42 48]; ...
            [6 8 18 30 40 48]; ...
            [6 10 16 24 34 48]; ...
            [6 16 26 36 46]; ...
          };
else
  error ("unrecognized job '%s'", job);
endif

%% End of configuration

infile = fullfile (root, [job, ".csv"]);
data = read_csv_table (infile, 0);
t = data.applicationCompletionTime;
datasize = data.dataSize;
cores = data.nContainers;
full_ernest_matrix = build_ernest_matrix (datasize, cores);

n_d = numel (datasizes);
n_c = numel (cases);
train_mape = NaN (n_c, n_d);
test_mape = NaN (n_c, n_d);
max_train_pe = NaN (n_c, n_d);
max_test_pe = NaN (n_c, n_d);
n_train = NaN (n_c, n_d);
n_test = NaN (n_c, n_d);
exception = cell (n_c, n_d);

state_nearly_singular = warning ("error", "Octave:nearly-singular-matrix");
state_singular = warning ("error", "Octave:singular-matrix");

for (dd = 1:n_d)
  current_datasizes = datasizes{dd};

  if (! all (ismember (abs (current_datasizes), datasize)))
    error ("datasizes n. %d are not compatible with available data", dd);
  endif

  for_training = current_datasizes > 0;
  datasize_train_idx = ismember (datasize, current_datasizes(for_training));
  datasize_test_idx = ismember (datasize, - current_datasizes(! for_training));
  extrapolation = ! any (datasize_train_idx & datasize_test_idx);

  for (cc = 1:n_c)
    current_cores = cases{cc};

    if (! all (ismember (current_cores, cores)))
      error ("case n. %d is not compatible with available data", cc);
    endif

    core_idx = ismember (cores, current_cores);
    train_idx = datasize_train_idx & core_idx;

    if (extrapolation)
      test_idx = datasize_test_idx;
    else
      test_idx = datasize_test_idx & ! core_idx;
    endif

    n_train(cc, dd) = sum (train_idx);
    n_test(cc, dd) = sum (test_idx);
    y_tr = t(train_idx);
    y_tst = t(test_idx);
    X_tr = full_ernest_matrix(train_idx, features);
    X_tst = full_ernest_matrix(test_idx, features);


    try
      if (use_nnls)
        theta = lsqnonneg (X_tr, y_tr);
      else
        pkg load statistics;
        theta = regress (y_tr, X_tr);
      endif
    catch e
      exception(cc, dd) = e;
      warning ("datasize n. %d, case n. %d skipped due to an exception in training", ...
               dd, cc);
      continue;
    end_try_catch

    y_hat_tr = X_tr * theta;
    y_hat_tst = X_tst * theta;

    abs_residuals_train = abs ((y_tr - y_hat_tr) ./ y_tr);
    abs_residuals_test = abs ((y_tst - y_hat_tst) ./ y_tst);

    train_mape(cc, dd) = 100 * mean (abs_residuals_train);
    test_mape(cc, dd) = 100 * mean (abs_residuals_test);
    max_train_pe(cc, dd) = 100 * max (abs_residuals_train);
    max_test_pe(cc, dd) = 100 * max (abs_residuals_test);
  endfor
endfor

warning (state_nearly_singular, "Octave:nearly-singular-matrix");
warning (state_singular, "Octave:singular-matrix");
