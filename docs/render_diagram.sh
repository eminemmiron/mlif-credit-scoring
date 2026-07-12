#!/usr/bin/env bash
# Перерисовать docs/architecture.puml -> SVG + PNG.
# Рендер идёт внутри контейнера spark-master (там есть Java).
# plantuml.jar и graphviz ставятся при первом запуске.
set -euo pipefail

JAR="/opt/app/docs/plantuml.jar"
VER="1.2024.7"

docker exec -u root -e NO_PROXY='*' -e no_proxy='*' spark-master bash -c "
  set -e
  command -v dot >/dev/null 2>&1 || {
    echo '>>> ставлю graphviz...'
    apt-get update -qq && apt-get install -y -qq graphviz
  }
  [ -f '$JAR' ] || {
    echo '>>> качаю plantuml.jar...'
    curl -fsSL -o '$JAR' \
      https://github.com/plantuml/plantuml/releases/download/v${VER}/plantuml-${VER}.jar
  }
  cd /opt/app
  java -jar '$JAR' -tsvg docs/architecture.puml
  java -jar '$JAR' -tpng docs/architecture.puml
  echo '>>> готово: docs/architecture.svg, docs/architecture.png'
"
