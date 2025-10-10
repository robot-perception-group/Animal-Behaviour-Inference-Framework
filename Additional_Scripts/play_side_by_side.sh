#!/usr/bin/env bash
# side_by_side_pairs_ffmpeg_pipe_mpv.sh
# Usage:
#   ./side_by_side_pairs_ffmpeg_pipe_mpv.sh /path/left /path/right [left_delay_ms] [right_delay_ms]
#
# Keys in mpv:
#   space=Pause • ←/→=Seek • [ / ]=Speed • { }=Half/Double • n=Next • p=Previous • q=Quit
#
# Env (optional):
#   SXS_HEIGHT=720   SXS_FPS=30   SXS_FONTSIZE=26   CRF=18   PRESET=veryfast   FFLOGLEVEL=fatal
#   WRAP=1           # 1=wrap playlist (default), 0=end when reaching ends

: "${SXS_HEIGHT:=720}"
: "${SXS_FPS:=30}"
: "${SXS_FONTSIZE:=26}"
: "${CRF:=18}"
: "${PRESET:=veryfast}"
: "${FFLOGLEVEL:=fatal}"
: "${WRAP:=1}"

set +o pipefail 2>/dev/null || true  # avoid pipefail killing us on skips

# ---- args ----
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 left_folder right_folder [left_delay_ms] [right_delay_ms]" >&2
  exit 1
fi
L_DIR="$1"; R_DIR="$2"
L_DELAY_MS="${3:-0}"
R_DELAY_MS="${4:-0}"

[[ -d "$L_DIR" && -d "$R_DIR" ]] || { echo "Folders not found."; exit 1; }
[[ "$L_DELAY_MS" =~ ^[0-9]+$ ]] || { echo "left_delay_ms must be non-negative integer (ms)"; exit 1; }
[[ "$R_DELAY_MS" =~ ^[0-9]+$ ]] || { echo "right_delay_ms must be non-negative integer (ms)"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg required"; exit 1; }
command -v mpv    >/dev/null || { echo "mpv required"; exit 1; }

# ms → sec
L_DELAY_SEC="$(awk "BEGIN{printf \"%.6f\", $L_DELAY_MS/1000}")"
R_DELAY_SEC="$(awk "BEGIN{printf \"%.6f\", $R_DELAY_MS/1000}")"

# ---- collect files (natural sort) ----
if sort -V </dev/null >/dev/null 2>&1; then SORT=(sort -V); else SORT=(sort); fi
list_vids(){ find "$1" -maxdepth 1 -type f \
  \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.m4v' -o -iname '*.webm' \) \
  -exec basename {} \; | "${SORT[@]}"; }
mapfile -t L_FILES < <(list_vids "$L_DIR")
mapfile -t R_FILES < <(list_vids "$R_DIR")
(( ${#L_FILES[@]} > 0 && ${#R_FILES[@]} > 0 )) || { echo "No videos found."; exit 1; }
PAIRS=$(( ${#L_FILES[@]} < ${#R_FILES[@]} ? ${#L_FILES[@]} : ${#R_FILES[@]} ))

# ---- mpv keys ----
INPUTCONF="$(mktemp -t sxs_inputconf.XXXXXX)"
cat >"$INPUTCONF" <<'EOF'
n quit 4
p quit 5
q quit 200
EOF
LABEL_L="$(mktemp -t sxs_labelL.XXXXXX)"
LABEL_R="$(mktemp -t sxs_labelR.XXXXXX)"
cleanup(){ rm -f "$INPUTCONF" "$LABEL_L" "$LABEL_R"; }
trap cleanup EXIT

echo "Pairs: $PAIRS"
echo "Left delay:  ${L_DELAY_MS} ms (${L_DELAY_SEC}s)"
echo "Right delay: ${R_DELAY_MS} ms (${R_DELAY_SEC}s)"
echo "Wrap mode: $WRAP (1=on, 0=off)"
echo

i=0
while (( PAIRS > 0 )); do
  (( i = (i + PAIRS) % PAIRS ))  # clamp/wrap index
  L="$L_DIR/${L_FILES[$i]}"; R="$R_DIR/${R_FILES[$i]}"
  printf '%s\n' "${L_FILES[$i]}" > "$LABEL_L"
  printf '%s\n' "${R_FILES[$i]}" > "$LABEL_R"

  echo "▶ Pair $((i+1))/$PAIRS"
  echo "   L: $L"
  echo "   R: $R"

  FILTER="
    [0:v:0]fps=${SXS_FPS},setsar=1,scale=-2:${SXS_HEIGHT}:flags=lanczos,format=yuv420p,
            drawtext=textfile='${LABEL_L}':x=(w-text_w)/2:y=10:fontcolor=white:fontsize=${SXS_FONTSIZE}:box=1:boxcolor=black@0.5:boxborderw=8,
            tpad=start_duration=${L_DELAY_SEC}[L];
    [1:v:0]fps=${SXS_FPS},setsar=1,scale=-2:${SXS_HEIGHT}:flags=lanczos,format=yuv420p,
            drawtext=textfile='${LABEL_R}':x=(w-text_w)/2:y=10:fontcolor=white:fontsize=${SXS_FONTSIZE}:box=1:boxcolor=black@0.5:boxborderw=8,
            tpad=start_duration=${R_DELAY_SEC}[R];
    [L][R]hstack=inputs=2,setsar=1,format=yuv420p[vout]
  "

  ffmpeg -hide_banner -loglevel "$FFLOGLEVEL" \
    -thread_queue_size 1024 -i "$L" \
    -thread_queue_size 1024 -i "$R" \
    -filter_complex "$FILTER" \
    -map "[vout]" -c:v libx264 -preset "$PRESET" -crf "$CRF" -an -f matroska - 2>/dev/null \
    | mpv - --no-audio --force-window=yes --no-terminal --keep-open=no --idle=no --input-conf="$INPUTCONF"

  mpv_exit=${PIPESTATUS[1]:-0}
  # echo "(debug) mpv exit: $mpv_exit"

  case "$mpv_exit" in
    0|4|104)  (( i++ )) ;;                        # finish or 'n' → next
    5|105)    (( i-- )) ;;                        # 'p' → prev
    200)      echo "Quitting."; break ;;
    *)        echo "(info) mpv exit $mpv_exit → next…" >&2; (( i++ )) ;;
  esac

  if [[ "$WRAP" == "0" ]]; then
    (( i >= PAIRS )) && break
    (( i < 0 )) && break
  fi
  echo
done

echo "All pairs done."
