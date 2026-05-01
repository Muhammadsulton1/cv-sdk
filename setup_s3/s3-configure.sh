set -e

DEFAULT_TTLS='{"video-frames":"3m","inference":35m", "hls-streams": "1d"}'
TTL_JSON=${BUCKET_TTLS_JSON:-$DEFAULT_TTLS}

echo "Configuring fs settings..."
echo "Buckets TTL map: ${TTL_JSON}"

cleaned=$(echo "${TTL_JSON}" | tr -d ' "' | sed 's/[{}]//g')
OLD_IFS="$IFS"
IFS='
'
for pair in $(echo "${cleaned}" | tr ',' '\n'); do
  bucket=$(echo "${pair}" | cut -d: -f1)
  ttl=$(echo "${pair}" | cut -d: -f2-)
  [ -z "${bucket}" ] && continue
  if [ -z "${ttl}" ]; then
    echo "Skipping ${bucket}: ttl missing" >&2
    continue
  fi

  echo "Setting TTL ${ttl} for bucket ${bucket}..."
  weed shell -master=master:9333 <<EOF
fs.configure -locationPrefix=/buckets/${bucket}/ -ttl=${ttl} -worm=true -apply
EOF
done
IFS="$OLD_IFS"

echo "Configuration complete"
