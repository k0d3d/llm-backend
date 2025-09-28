#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MIGRATIONS_DIR="${PROJECT_ROOT}/db"

usage() {
  cat <<'EOF'
Usage: run_migrations.sh [--dir PATH] [--dry-run]

Options:
  --dir PATH   Path to directory containing *.sql migration files (default: ./db)
  --dry-run    Print the files that would be executed without running them

Environment:
  DATABASE_URL  PostgreSQL connection string (postgresql://user:pass@host:port/dbname)

The script executes the SQL files in lexical order using psql. Each file should be idempotent
(e.g. use CREATE TABLE IF NOT EXISTS / ALTER TABLE ... IF NOT EXISTS) so the script can be run
multiple times safely during deploys.
EOF
}

MIGRATIONS_PATH_OVERRIDE=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      shift || { echo "missing argument for --dir" >&2; exit 1; }
      MIGRATIONS_PATH_OVERRIDE="$1"
      ;;
    --dry-run)
      DRY_RUN="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift || true
endwhile

if [[ -n "${MIGRATIONS_PATH_OVERRIDE}" ]]; then
  MIGRATIONS_DIR="${MIGRATIONS_PATH_OVERRIDE}"
fi

if [[ ! -d "${MIGRATIONS_DIR}" ]]; then
  echo "Migration directory not found: ${MIGRATIONS_DIR}" >&2
  exit 1
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL environment variable is required" >&2
  exit 1
fi

PSQL_CMD=(psql "${DATABASE_URL}" --set "ON_ERROR_STOP=1")

mapfile -t MIGRATION_FILES < <(find "${MIGRATIONS_DIR}" -maxdepth 1 -type f -name '*.sql' | sort)

if [[ ${#MIGRATION_FILES[@]} -eq 0 ]]; then
  echo "No SQL files found in ${MIGRATIONS_DIR}" >&2
  exit 0
fi

echo "📦 Running migrations against ${DATABASE_URL}" >&2

for migration_file in "${MIGRATION_FILES[@]}"; do
  echo "➡️  Applying $(basename "${migration_file}")" >&2
  if [[ "${DRY_RUN}" == "true" ]]; then
    continue
  fi
  "${PSQL_CMD[@]}" --file "${migration_file}"
  echo "✅ Done" >&2
done

echo "🎉 Migrations complete" >&2
