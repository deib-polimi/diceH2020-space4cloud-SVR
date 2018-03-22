#!/bin/sh

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

root="${1?error: missing traces directory}"

find "$root" -name summary.csv -type f | while IFS= read -r filename; do
    relname="${filename#$root/}"

    cores="$(echo "$relname" | cut -d / -f 1 | awk -F _ '{ print $1 * $2 }')"
    query="$(echo "$relname" | cut -d / -f 2)"
    vm="$(echo "$relname" | cut -d / -f 3)"

    destdir="$query/$vm"
    mkdir -p "$destdir"

    destname="$cores.csv"
    destfile="$destdir/$destname"

    if ls "$destdir" | grep -q "^$destname$"; then
        tmp="$destdir/aux.csv"
        {
            cat "$destfile"
            tail -n +3 "$filename"
        } > "$tmp"
        mv "$tmp" "$destfile"
    else
        cp "$filename" "$destfile"
    fi
done
