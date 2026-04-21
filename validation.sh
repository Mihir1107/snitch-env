#!/usr/bin/env bash
# validation.sh — validate The Snitch OpenEnv submission.
# Usage: ./validation.sh <hf_space_url> [repo_dir]

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

PING_URL="${1:-}"
REPO_DIR="${2:-.}"
if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <hf_space_url> [repo_dir]"
  exit 1
fi
REPO_DIR="$(cd "$REPO_DIR" && pwd)"
PING_URL="${PING_URL%/}"

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }
stop() { printf "\n${RED}${BOLD}Stopped at %s.${NC}\n" "$1"; exit 1; }

printf "\n${BOLD}==== Snitch Validator ====${NC}\n"
log "Repo: $REPO_DIR"
log "Ping: $PING_URL"
printf "\n"

log "${BOLD}Step 1/3: Ping HF Space${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{"task_id":"easy"}' \
  "$PING_URL/reset" --max-time 30 || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space live"
else
  fail "HTTP $HTTP_CODE"; stop "Step 1"
fi

log "${BOLD}Step 2/3: Docker build${NC}"
if ! docker build -q "$REPO_DIR" > /dev/null 2>&1; then
  fail "docker build"; stop "Step 2"
fi
pass "docker build"

log "${BOLD}Step 3/3: openenv validate${NC}"
if ! command -v openenv &>/dev/null; then
  fail "install: pip install openenv-core"; stop "Step 3"
fi
if ( cd "$REPO_DIR" && openenv validate ); then
  pass "openenv validate"
else
  fail "openenv validate"; stop "Step 3"
fi

printf "\n${GREEN}${BOLD}All 3/3 checks passed.${NC}\n\n"