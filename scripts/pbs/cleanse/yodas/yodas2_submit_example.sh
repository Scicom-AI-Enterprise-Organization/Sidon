#!/bin/bash
# Example helper that submits one Yodas2 cleansing PBS job per language.
set -eux

# Update these defaults before submitting.
PROJECT_ROOT="/work/gj18/e43001/github.com/Sidon"
SCRIPT_PATH="$PROJECT_ROOT/scripts/pbs/cleanse/yodas/yodas2.sh"
OUTPUT_ROOT="/work/gj18/e43001/yodas2_sidon"
S3_UPLOAD_URI="s3://yodas2_sidon/"

# Space-separated languages; one PBS job is submitted for each entry.
LANGUAGES="aa000 ab000 af000 ak000 am000 ar000 as000 ay000 az000 ba000 be000 bg000 bh000 bi000 bm000 bn000 bo000 br000 bs000 ca000 co000 cr000 cs000 cy000 da000 de000 de100 de101 de102 dz000 ee000 el000 en000 en001 en002 en003 en004 en005 en100 en101 en102 en103 en104 en105 en106 en107 en108 en109 en110 en111 en112 en113 en114 en115 en116 en117 en118 en119 en120 en121 en122 en123 en124 en125 en126 en127 eo000 es000 es100 es101 es102 es103 es104 es105 es106 et000 eu000 fa000 ff000 fi000 fj000 fo000 fr000 fr100 fr101 fr102 fr103 fy000 ga000 gd000 gl000 gn000 gu000 ha000 hi000 hi100 ho000 hr000 ht000 hu000 hy000 ia000 id000 id100 id101 ie000 ig000 ik000 is000 it000 it100 it101 iu000 iw000 ja000 ja100 jv000 ka000 ki000 kk000 kl000 km000 kn000 ko000 ko100 ko101 ko102 ko103 ks000 ku000 ky000 la000 lb000 lg000 ln000 lo000 lt000 lv000 mg000 mi000 mk000 ml000 mn000 mr000 ms000 my000 na000 nd000 ne000 nl000 nl100 no000 nv000 oc000 om000 or000 pa000 pl000 ps000 pt000 pt100 pt101 pt102 pt103 qu000 rm000 rn000 ro000 ru000 ru001 ru100 ru101 ru102 ru103 ru104 ru105 ru106 rw000 sa000 sc000 sd000 sg000 sh000 si000 sk000 sl000 sm000 sn000 so000 sq000 sr000 st000 su000 sv000 sw000 ta000 te000 tg000 th000 th100 ti000 tk000 tn000 to000 tr000 tr100 ts000 tt000 ug000 uk000 uk100 ur000 uz000 ve000 vi000 vi100 vi101 vo000 wo000 xh000 yi000 yo000 zh000 zu000"

# Use commas to avoid quoting issues when the value is passed via qsub -v.
SPLITS="train"

# Optional extras.
S3_UPLOAD_EXTRA_ARGS="--endpoint-url https://s3ds.mdx.jp"
REMOVE_HF_CACHE=true

# PBS options common to every job.
JOB_NAME_PREFIX="yodas2-cleanse"
QUEUE="regular-g"
WALLTIME="24:00:00"
SELECT="1"
MAX_QUEUE_JOBS=16
QUEUE_POLL_INTERVAL=30

wait_for_queue_slot() {
  if [ "${MAX_QUEUE_JOBS:-0}" -le 0 ]; then
    return
  fi

  while true; do
    queue_count=$(qstat -u "$USER" 2>/dev/null | awk -v prefix="$JOB_NAME_PREFIX" 'NR>2 && index($2, prefix)==1 {count++} END{print count+0}')
    if [ -z "$queue_count" ]; then
      echo "Unable to determine current queue depth; retrying in $QUEUE_POLL_INTERVAL seconds." >&2
      sleep "$QUEUE_POLL_INTERVAL"
      continue
    fi

    if [ "$queue_count" -lt "$MAX_QUEUE_JOBS" ]; then
      break
    fi

    echo "Queue currently has $queue_count jobs with prefix $JOB_NAME_PREFIX; waiting for a free slot."
    sleep "$QUEUE_POLL_INTERVAL"
  done
}

if [ ! -x "$SCRIPT_PATH" ]; then
  echo "Could not find executable yodas2.sh at $SCRIPT_PATH" >&2
  exit 1
fi

read -r -a LANGUAGE_ARRAY <<< "$LANGUAGES"
if [ ${#LANGUAGE_ARRAY[@]} -eq 0 ]; then
  echo "LANGUAGES is empty; nothing to submit." >&2
  exit 1
fi

for language in "${LANGUAGE_ARRAY[@]}"; do
  job_name="${JOB_NAME_PREFIX}-${language}"
  echo "Submitting $job_name for language $language"
  s3url=$S3_UPLOAD_URI$language
  if aws $S3_UPLOAD_EXTRA_ARGS s3api head-object --bucket yodas2_sidon --key "${language}/completed_train.txt" &>/dev/null; then
    echo $language "is finished skipping..."
  else
    wait_for_queue_slot
    qsub_cmd=(
      qsub
      -N "$job_name"
      -q "$QUEUE"
      -l "select=$SELECT"
      -l "walltime=$WALLTIME"
      -v  LANGUAGES=$language,SPLITS=$SPLITS,OUTPUT_ROOT=$OUTPUT_ROOT,S3_UPLOAD_URI=$s3url,REMOVE_HF_CACHE=$REMOVE_HF_CACHE,S3_UPLOAD_EXTRA_ARGS="$S3_UPLOAD_EXTRA_ARGS"
    )

    qsub_cmd+=("$SCRIPT_PATH")
    "${qsub_cmd[@]}"
  fi
done
