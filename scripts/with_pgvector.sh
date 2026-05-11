#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-pgvector/pgvector:pg16}"
CONTAINER_NAME="${CONTAINER_NAME:-langchain-rag-pg}"
VOLUME_NAME="${VOLUME_NAME:-${CONTAINER_NAME}-data}"
DATA_DIR="${DATA_DIR:-}"
HOST_PORT="${HOST_PORT:-5432}"
DB_NAME="${DB_NAME:-langchain_rag}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command...>"
  echo "Example: $0 .venv/bin/python build_index.py"
  exit 1
fi

env_args=()
while [[ $# -gt 0 && "$1" == *=* ]]; do
  env_args+=("$1")
  shift
done

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [VAR=value ...] <command...>"
  echo "Example: $0 RAG_REPO_PATH=... .venv/bin/python build_index.py"
  exit 1
fi

cleanup() {
  if docker ps -aq --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

if docker ps -aq --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

volume_arg="${VOLUME_NAME}:/var/lib/postgresql/data"
if [[ -n "${DATA_DIR}" ]]; then
  DATA_DIR="${DATA_DIR/#\~/${HOME}}"
  mkdir -p "${DATA_DIR}"
  volume_arg="${DATA_DIR}:/var/lib/postgresql/data"
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  -e POSTGRES_USER="${DB_USER}" \
  -e POSTGRES_PASSWORD="${DB_PASSWORD}" \
  -e POSTGRES_DB="${DB_NAME}" \
  -v "${volume_arg}" \
  -p "${HOST_PORT}:5432" \
  "${IMAGE}" >/dev/null

export DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@localhost:${HOST_PORT}/${DB_NAME}"
export RAG_BACKEND="postgres"

printf 'Waiting for Postgres'
for _ in $(seq 1 60); do
  if docker exec "${CONTAINER_NAME}" pg_isready -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; then
    printf '\n'
    break
  fi
  printf '.'
  sleep 1
done

if ! docker exec "${CONTAINER_NAME}" pg_isready -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; then
  echo
  echo "Postgres did not become ready in time"
  exit 1
fi

# Ensure the pgvector extension exists before running the caller's command.
docker exec "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" -c 'CREATE EXTENSION IF NOT EXISTS vector;' >/dev/null

echo "Running: $*"
env "${env_args[@]}" "$@"
