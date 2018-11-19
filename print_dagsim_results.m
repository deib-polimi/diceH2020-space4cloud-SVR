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

%% Setup
filename = "/Users/eugenio/Dottorato/Experiment Results/POWER8/dagsim/results/merged_cases.csv";
save_plots = true;

% Empty arrays mean "use every alternative in the input file"
queries = {};
datasizes = [];

%% Load data
data = read_csv_table (filename);

if (isempty (queries))
  queries = unique (data.Query);
endif

if (isempty (datasizes))
  datasizes = unique (data.Datasize);
endif

%% Do stuff
for (qq = 1:numel (queries))
  current_query = queries{qq};
  q_idx = strcmp (current_query, data.Query);

  for (dd = 1:numel (datasizes))
    current_datasize = datasizes(dd);
    d_idx = data.Datasize == current_datasize;
    idx = q_idx & d_idx;

    if (sum (idx) == 0)
      warning ("the combination %s/datasize %d does not yield any results", ...
               current_query, current_datasize);
    else
      cases = data.Case(idx);
      training = data.("Training MAPE")(idx);
      testing = data.("Test MAPE")(idx);

      figure;
      plot (cases, training, "LineWidth", 2, "DisplayName", "Training");
      set (gca, "FontSize", 14);
      hold all;
      plot (cases, testing, "LineWidth", 2, "DisplayName", "Test");
      grid on;
      xticks (cases);
      axis auto;
      legend ({}, "location", "eastoutside", "interpreter", "none");
      xlabel Case;
      ylabel MAPE;
      name = sprintf ("%s datasize %d", current_query, current_datasize);
      title (name, "interpreter", "none");
      drawnow expose;
      pause (1);

      if (save_plots)
        filename = strrep (name, " ", "_");
        print (filename, "-depsc", "-tiff");
        pause (1);
      endif
    endif
  endfor
endfor
