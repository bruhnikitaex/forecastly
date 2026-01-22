#!/bin/bash
# Установка всех расширений VS Code для проекта Forecastly
# Для Linux/Mac

echo ""
echo "========================================"
echo "Установка VS Code расширений для Forecastly"
echo "========================================"
echo ""

# Обязательные расширения (основные)
echo "[1/4] Устанавливаю основные расширения Python..."
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy

# Форматирование и линтинг
echo "[2/4] Устанавливаю форматирование и линтинг..."
code --install-extension ms-python.black-formatter
code --install-extension ms-python.flake8
code --install-extension ms-python.mypy-type-checker
code --install-extension charliermarsh.ruff

# Git и контейнеризация
echo "[3/4] Устанавливаю Git, Docker и API расширения..."
code --install-extension eamodio.gitlens
code --install-extension ms-vscode.docker
code --install-extension ms-vscode-remote.remote-containers
code --install-extension ms-vscode.rest-client

# Данные и утилиты
echo "[4/4] Устанавливаю дополнительные расширения..."
code --install-extension redhat.vscode-yaml
code --install-extension tamasfe.even-better-toml
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter-keymap
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension formulahendry.code-runner
code --install-extension gruntfuggly.todo-tree
code --install-extension yzhang.markdown-all-in-one
code --install-extension DavidAnson.vscode-markdownlint
code --install-extension mechatroner.rainbow-csv
code --install-extension naumovs.color-highlight
code --install-extension oderwat.indent-rainbow
code --install-extension streetsidesoftware.code-spell-checker
code --install-extension streetsidesoftware.code-spell-checker-russian
code --install-extension GitHub.copilot

echo ""
echo "========================================"
echo "Установка завершена!"
echo "========================================"
echo ""
echo "Проверьте установленные расширения:"
echo "code --list-extensions"
echo ""
