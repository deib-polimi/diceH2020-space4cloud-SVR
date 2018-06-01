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

root = pwd;

data_root = fullfile (root, "data");
sim_root = fullfile (root, "..", "dagsim-processed");
ml_root = fullfile (root, "ml");

queries = {"Q26", "Q52"};
vms = {"D12v2"};

plot_dagsim = true;
plot_chi = false;
plot_ml = true;
plot_ernest = true;
save_results = false;

n_queries = numel (queries);
n_vms = numel (vms);

for (ii = 1:n_queries)
  query = queries{ii};

  for (jj = 1:n_vms)
    vm = vms{jj};
    template = fullfile (data_root, query, vm, "*.csv");
    files = glob (template);

    for (kk = 1:numel (files))
      aux = read_data (files{kk});
      containers = num2str (mean (aux(:, end)));
      real_data.(query).(vm).(containers) = aux;
    endfor

    sim_file = fullfile (sim_root, sprintf ("%s-%s.csv", query, vm));
    aux = csvread (sim_file, 1, 1);
    sim_data.(query).(vm).times = aux(:, 3);
    sim_data.(query).(vm).cores = aux(:, 1);
    clear aux;
  endfor

  ml_file = fullfile (ml_root, query, "model.txt");
  ml.(query) = load (ml_file);

  ernest_file = fullfile (ml_root, query, "ernest.txt");
  ernest.(query) = load (ernest_file);
endfor

for ([query_data, query] = real_data)
  for ([vm_data, vm] = query_data)
    avg.(query).(vm) = structfun (@mean, vm_data, "UniformOutput", false);
  endfor
endfor

figure;
hold all;

colmap = lines (n_queries * n_vms);
ii = 0;

for ([query_data, query] = avg)
  model = ml.(query);
  current_ernest = ernest.(query);

  for ([vm_data, vm] = query_data)
    color = colmap(++ii, :);

    exe_times = structfun (@(x) x(1) - x(2), vm_data);
    datasizes = structfun (@(x) x(end - 1), vm_data);
    cores = structfun (@(x) x(end), vm_data);

    [cores, idx] = sort (cores);
    datasizes = datasizes(idx);
    exe_times = exe_times(idx);

    exe_times /= 1000;

    name = ["Real ", query, " ", vm];
    plot (cores, exe_times, "o", "color", color, "DisplayName", name);

    sim_times = sim_data.(query).(vm).times;
    sim_cores = sim_data.(query).(vm).cores;

    [sim_cores, idx] = sort (sim_cores);
    sim_times = sim_times(idx);
    sim_times /= 1000;

    if (plot_dagsim)
      name = ["Dagsim ", query, " ", vm];
      plot (sim_cores, sim_times, ":", "color", color, "DisplayName", name);
    endif

    idx = ismember (sim_cores, cores);
    relevant_times = sim_times(idx);
    err = 100 * (relevant_times - exe_times) ./ exe_times;

    names{end + 1} = [query, " ", vm];
    real_cores{end + 1} = cores;
    sign_err{end + 1} = err;
    abs_err{end + 1} = abs (err);

    predictions = pred_cores = [];
    for ([x, _] = vm_data)
      sample = x(model.useful_columns)';

      features = sample(2:end);
      pred_cores(end + 1) = features(end);

      features(end) = 1 ./ features(end);
      scaled = (features - model.working_mu(2:end)) ./ model.working_sigma(2:end);
      scaled_prediction = model.b + model.w' * scaled;
      predictions(end + 1) = scaled_prediction * model.working_sigma(1) + model.working_mu(1);

      chi(end + 1).zero = model.working_mu(1) + ...
                          model.working_sigma(1) * (model.b - ...
                                                    model.working_mu(end) / model.working_sigma(end) + ...
                                                    model.w(1:end - 1)' * scaled(1:end - 1));
      chi(end).c = model.w(end) * model.working_sigma(1) / model.working_sigma(end);
    endfor

    all_chi_0 = [chi.zero]';
    all_chi_c = [chi.c]';

    predictions /= 1000;
    [pred_cores, idx] = sort (pred_cores);
    predictions = predictions(idx);
    all_chi_0 = all_chi_0(idx);
    all_chi_c = all_chi_c(idx);

    if (plot_ml)
      name = ["ML ", query, " ", vm];
      plot (pred_cores, predictions, "-", "color", color, "DisplayName", name);
    endif

    chi_0 = all_chi_0(1);
    chi_c = all_chi_c(1);
    chi_predictions = chi_0 + chi_c ./ pred_cores;
    chi_predictions /= 1000;

    if (plot_chi)
      name = ["\chi ", query, " ", vm];
      plot (pred_cores, chi_predictions, "--", "color", color, "DisplayName", name);
    endif

    err = 100 * (predictions(:) - exe_times(:)) ./ exe_times(:);
    ml_sign_err{end + 1} = err;
    ml_abs_err{end + 1} = abs (err);

    chi_err = 100 * (chi_predictions(:) - exe_times(:)) ./ exe_times(:);
    chi_sign_err{end + 1} = chi_err;
    chi_abs_err{end + 1} = abs (chi_err);

    one = ones (size (datasizes));
    sm = datasizes ./ cores;
    lm = log2 (cores);
    X = [one, sm, lm, cores];

    ernest_predictions = X * current_ernest.theta;
    ernest_predictions /= 1000;

    if (plot_ernest)
      name = ["Ernest ", query, " ", vm];
      plot (cores, ernest_predictions, "-.", "color", color, "DisplayName", name);
    endif

    ernest_err = 100 * (ernest_predictions(:) - exe_times(:)) ./ exe_times(:);
    ernest_sign_err{end + 1} = ernest_err;
    ernest_abs_err{end + 1} = abs (ernest_err);

    if (save_results)
      filename = [query, "-", vm, ".txt"];
      save (filename, "cores", "exe_times", "sim_cores", "sim_times",
            "pred_cores", "predictions", "chi_predictions");
    endif
  endfor
endfor

grid on;
axis auto;
legend Location EastOutside;
xlabel Cores;
ylabel ("Execution time [s]");

if (plot_dagsim)
  figure;
  hold all;

  for (ii = 1:numel (sign_err))
    cores = real_cores{ii};
    err = sign_err{ii};
    name = names{ii};
    plot (cores, err, "DisplayName", name);
  endfor

  grid on;
  axis auto;
  legend Location EastOutside;
  title ("Dagsim vs real, signed");
  xlabel Cores;
  ylabel ("Signed error [%]");

  figure;
  hold all;
  disp ("Dagsim");

  for (ii = 1:numel (abs_err))
    cores = real_cores{ii};
    err = abs_err{ii};
    name = names{ii};
    plot (cores, err, "DisplayName", name);

    disp (name);
    average = mean (err)
  endfor

  grid on;
  axis auto;
  legend Location EastOutside;
  title ("Dagsim vs real");
  xlabel Cores;
  ylabel ("Relative error [%]");
endif

if (plot_ml)
  figure;
  hold all;

  for (ii = 1:numel (sign_err))
    cores = real_cores{ii};
    err = ml_sign_err{ii};
    name = names{ii};
    plot (cores, err, "DisplayName", name);
  endfor

  grid on;
  axis auto;
  legend Location EastOutside;
  title ("ML vs real, signed");
  xlabel Cores;
  ylabel ("Signed error [%]");

  figure;
  hold all;
  disp ("ML");

  for (ii = 1:numel (abs_err))
    cores = real_cores{ii};
    err = ml_abs_err{ii};
    name = names{ii};
    plot (cores, err, "DisplayName", name);

    disp (name);
    average = mean (err)
  endfor

  grid on;
  axis auto;
  legend Location EastOutside;
  title ("ML vs real");
  xlabel Cores;
  ylabel ("Relative error [%]");
endif

if (plot_chi)
  figure;
  hold all;

  for (ii = 1:numel (sign_err))
    cores = real_cores{ii};
    err = chi_sign_err{ii};
    name = names{ii};
    plot (cores, err, "DisplayName", name);
  endfor

  grid on;
  axis auto;
  legend Location EastOutside;
  title ("\chi vs real, signed");
  xlabel Cores;
  ylabel ("Signed error [%]");

  figure;
  hold all;
  disp ("\chi");

  for (ii = 1:numel (abs_err))
    cores = real_cores{ii};
    err = chi_abs_err{ii};
    name = names{ii};
    plot (cores, err, "DisplayName", name);

    disp (name);
    average = mean (err)
  endfor

  grid on;
  axis auto;
  legend Location EastOutside;
  title ("\chi vs real");
  xlabel Cores;
  ylabel ("Relative error [%]");
endif

if (plot_ernest)
  figure;
  hold all;

  for (ii = 1:numel (sign_err))
    cores = real_cores{ii};
    err = ernest_sign_err{ii};
    name = names{ii};
    plot (cores, err, "DisplayName", name);
  endfor

  grid on;
  axis auto;
  legend Location EastOutside;
  title ("Ernest vs real, signed");
  xlabel Cores;
  ylabel ("Signed error [%]");

  figure;
  hold all;
  disp ("Ernest");

  for (ii = 1:numel (abs_err))
    cores = real_cores{ii};
    err = ernest_abs_err{ii};
    name = names{ii};
    plot (cores, err, "DisplayName", name);

    disp (name);
    average = mean (err)
  endfor

  grid on;
  axis auto;
  legend Location EastOutside;
  title ("Ernest vs real");
  xlabel Cores;
  ylabel ("Relative error [%]");
endif
