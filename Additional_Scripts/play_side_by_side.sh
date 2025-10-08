#!/usr/bin/env bash
# side_by_side_pairs_mpv_nopipe_labels.sh
# Usage:
#   ./side_by_side_pairs_mpv_nopipe_labels.sh /path/left /path/right [delay_ms_for_left]
#
# Optional env:
#   SXS_HEIGHT=720       # per-side target height (keeps aspect)
#   SXS_FPS=30           # normalize fps for both sides to avoid hstack stalls
#   SXS_FONTSIZE=26      # label font size
#   MPV_LOG=warn         # mpv log level: trace|debug|info|warn|error|fatal

set -u
: "${SXS_HEIGHT:=720}"
: "${SXS_FPS:=30}"
: "${SXS_FONTSIZE:=26}"
: "${MPV_LOG:=warn}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 left_folder right_folder [delay_ms_for_left]" >&2
  exit 1
fi

L_DIR="$1"
R_DIR="$2"
DELAY_MS="${3:-0}"

[[ -d "$L_DIR" ]] || { echo "Not a directory: $L_DIR" >&2; exit 1; }
[[ -d "$R_DIR" ]] || { echo "Not a directory: $R_DIR" >&2; exit 1; }
[[ "$DELAY_MS" =~ ^[0-9]+$ ]] || { echo "delay_ms must be a non-negative integer"; exit 1; }
command -v mpv >/dev/null 2>&1 || { echo "mpv is required (sudo apt install mpv)"; exit 1; }

# ms → sec string for tpad
DELAY_SEC="$(awk "BEGIN{printf \"%.6f\", $DELAY_MS/1000}")"

# Natural sort if available
if sort -V </dev/null >/dev/null 2>&1; then SORT_CMD=(sort -V); else SORT_CMD=(sort); fi

list_videos() {
  local d="$1"
  find "$d" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.m4v' -o -iname '*.webm' \) \
    -exec basename {} \; | "${SORT_CMD[@]}"
}

mapfile -t L_FILES < <(list_videos "$L_DIR")
mapfile -t R_FILES < <(list_videos "$R_DIR")
LEN_L=${#L_FILES[@]}
LEN_R=${#R_FILES[@]}
(( LEN_L>0 && LEN_R>0 )) || { echo "No video files found in one or both folders."; exit 1; }
PAIRS=$(( LEN_L < LEN_R ? LEN_L : LEN_R ))

echo "Found $LEN_L files in: $L_DIR"
echo "Found $LEN_R files in: $R_DIR"
echo "Playing $PAIRS pair(s)…  Left-delay: ${DELAY_MS} ms (${DELAY_SEC}s)"
echo "Keys: space pause • ←/→ seek • [/] speed • { } half/double • n next • p previous • q quit"
echo

# Key bindings (guarantee our n/p/q)
INPUTCONF="$(mktemp -t sxs_inputconf.XXXXXX)"
cat >"$INPUTCONF" <<'EOF'
n quit 4
p quit 5
q quit 200
EOF

# temp files for labels (avoids escaping issues)
LABEL_L="$(mktemp -t sxs_labelL.XXXXXX)"
LABEL_R="$(mktemp -t sxs_labelR.XXXXXX)"

cleanup(){ rm -f "$INPUTCONF" "$LABEL_L" "$LABEL_R"; }
trap cleanup EXIT

i=0
while (( i < PAIRS )); do
  L="$L_DIR/${L_FILES[$i]}"
  R="$R_DIR/${R_FILES[$i]}"
  BN_L="${L_FILES[$i]}"
  BN_R="${R_FILES[$i]}"

  echo "▶ Pair $((i+1))/$PAIRS"
  echo "   L: $L"
  echo "   R: $R"

  printf '%s\n' "$BN_L" > "$LABEL_L"
  printf '%s\n' "$BN_R" > "$LABEL_R"

  # Robust lavfi graph inside mpv:
  # - fps=SXS_FPS on both sides to avoid timebase/fps mismatches
  # - setsar=1 to kill anamorphic "squish"
  # - scale both to the same height, keep aspect (-2 makes width even)
  # - format=yuv420p for broad compatibility
  # - drawtext labels top-center with translucent box
  # - left delay with tpad=start_duration
  # - hstack the two streams
  GRAPH="[vid1]fps=${SXS_FPS},setsar=1,scale=-2:${SXS_HEIGHT}:flags=lanczos,format=yuv420p,"\
"drawtext=textfile=${LABEL_L}:x=(w-text_w)/2:y=10:fontcolor=white:fontsize=${SXS_FONTSIZE}:box=1:boxcolor=black@0.5:boxborderw=8,"\
"tpad=start_duration=${DELAY_SEC}[L];"\
"[vid2]fps=${SXS_FPS},setsar=1,scale=-2:${SXS_HEIGHT}:flags=lanczos,format=yuv420p,"\
"drawtext=textfile=${LABEL_R}:x=(w-text_w)/2:y=10:fontcolor=white:fontsize=${SXS_FONTSIZE}:box=1:boxcolor=black@0.5:boxborderw=8[R];"\
"[L][R]hstack=inputs=2,setsar=1,format=yuv420p[vo]"

  mpv --no-audio --no-terminal --force-window=yes --keep-open=no --idle=no --hwdec=no \
      --msg-level=all=${MPV_LOG} \
      --input-conf="$INPUTCONF" \
      --external-file="$R" \
      --lavfi-complex="$GRAPH" \
      "$L"
  code=$?

  case "$code" in
    0)      (( i++ )) ;;                          # natural end → next
    4|104)  (( i++ )) ;;                          # n → next
    5|105)  (( i>0 )) && (( i-- )) || i=0 ;;      # p → prev
    200)    echo "Quitting."; break ;;            # q → quit script
    *)      echo "(Info) mpv exit $code; continuing to next…" >&2; (( i++ )) ;;
  esac
  echo
done

echo "All pairs done."
