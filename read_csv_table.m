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

## -*- texinfo -*-
## @deftypefn {Function File} {@var{data} =} read_csv_table (@var{filename})
## @deftypefnx {Function File} {@var{data} =} read_csv_table (@var{filename}, @var{skip})
##
## Read the @var{data} stored in a CSV called @var{filename} and
## return it as a structure whose field names are taken from the
## first line in the file.
## Optionally, you can configure a number of lines to @var{skip}
## at the beginning of the file.
## Numeric columns are turned into column vectors, other general
## data types are cell arrays.
##
## This function depends on 'csv2cell' from the io package.
##
## @seealso{csv2cell}
## @end deftypefn

function data = read_csv_table (filename, skip = 0)

  if (ischar (filename))
    pkg load io;

    raw = csv2cell (filename, skip);

    data = struct ();
    names = raw(1, :);

    for (ii = 1:numel (names))
      field = names{ii};

      if (isnumeric (raw{2, ii}))
        data.(field) = cell2mat (raw(2:end, ii));
      else
        data.(field) = raw(2:end, ii);
      endif
    endfor
  else
    error ("FILENAME must be a string");
  endif

endfunction
