#!/bin/zsh
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR" || exit 1
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  print "Не найдено окружение приложения. Инструкция установки находится в README.md."
  print "Нажми любую клавишу, чтобы закрыть окно."
  read -k 1
  exit 1
fi
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/run.py"
RESULT_CODE=$?
if [[ $RESULT_CODE -ne 0 ]]; then
  print "\nПриложение завершилось с ошибкой. Подробности выше. Нажми любую клавишу."
  read -k 1
fi
