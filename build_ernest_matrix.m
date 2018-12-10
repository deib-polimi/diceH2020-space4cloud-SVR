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

## -*- texinfo -*-
## @deftypefn {Function File} {@var{X} =} build_ernest_matrix (@var{datasizes}, @var{machines})
##
## Given a vector of @var{datasizes} and one of @var{machines},
## create the design matrix @var{X} to learn the Ernest model.
## The last two columns are squared @var{datasizes} divided by
## @var{machines}, which is one of the additional features discussed
## in their paper, and square root of @var{datasizes} divided by
## @var{machines}.
##
## @end deftypefn

function X = build_ernest_matrix (datasizes, machines)

  narginchk (2, 2);
  nargoutchk (1, 1);

  validateattributes (datasizes, {"numeric"}, {"column"}, ...
                      "build_ernest_matrix", "datasizes", 1);
  validateattributes (machines, {"numeric"}, {"column"}, ...
                      "build_ernest_matrix", "machines", 2);

  if (rows (datasizes) != rows (machines))
    error ("DATASIZES and MACHINES must have the same number of elements");
  endif

  sm = datasizes ./ machines;
  s2m = datasizes .^ 2 ./ machines;
  lm = log2 (machines);
  one = ones (size (machines));
  rootm = sqrt (datasizes) ./ machines;
  X = [one, sm, lm, machines, s2m, rootm];

endfunction
