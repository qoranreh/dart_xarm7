#!/usr/bin/env bash
# θ₀ (gs://openpi-assets/checkpoints/pi05_libero, public) + θ* (SNUMPR/dart-pi05-viewpoint-medium) 받기
set -uo pipefail
D=/var/tmp/pbs.128706.ECE-util1/dart/ckpts
get() {  # url out size
  local url=$1 out=$2 size=$3
  mkdir -p "$(dirname "$out")"
  [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ] && return 0
  for i in 1 2 3 4; do curl -sfL --retry 3 -o "$out" "$url" && [ "$(stat -c%s "$out")" = "$size" ] && return 0; sleep $((2**i)); done
  echo "FAIL $url"; return 1
}
curl -s "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/&fields=items(name,size)&maxResults=1000" \
 | /usr/bin/python3 -c "import sys,json;[print(x['name'],x['size']) for x in json.load(sys.stdin)['items']]" > $D/pi05_libero.list
while read name size; do get "https://storage.googleapis.com/openpi-assets/$name" "$D/${name#checkpoints/}" $size; done < $D/pi05_libero.list
echo "[pi05_libero done] $(date)"
R=SNUMPR/dart-pi05-viewpoint-medium
curl -s "https://huggingface.co/api/models/$R/tree/main?recursive=true" \
 | /usr/bin/python3 -c "import sys,json;[print(x['path'],x.get('size',0)) for x in json.load(sys.stdin) if x['type']=='file']" > $D/medium.list
while read name size; do get "https://huggingface.co/$R/resolve/main/$name" "$D/dart-pi05-viewpoint-medium/$name" $size; done < $D/medium.list
echo "[medium done] $(date)"
du -sh $D/*/
