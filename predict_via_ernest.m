## Copyright 2019 Eugenio Gianniti
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

modelfile = "~/Dottorato/Experiment Results/D12v2/ernest/query52/ernest_nnls.txt";
datasizes = 250:250:1000;
cores = 2:2:100;

%% End of configurations

model = load (modelfile);
[dd, cc] = meshgrid (datasizes, cores);
matrix = build_ernest_matrix (dd(:), cc(:));
A = matrix(:, model.features);
time = A * model.theta;

printf ("application,dataSize,nCores,predicted time\n");

for (ii = 1:numel (time))
  printf ("%s,%g,%d,%f\n", model.query, dd(ii), cc(ii), time(ii));
endfor
