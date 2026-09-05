# ai-agents-platform

Личные AI-агенты, живущие в кластере: Telegram как интерфейс, Azure AI Foundry как
модель, MCP как способ дать агенту руки.

## Что это

Один репозиторий, один образ, четыре пода.

```
Telegram ──► agent-lead ──┬──► mcp-github       (готовый образ от GitHub)
                          ├──► mcp-cloudflare   (наш код: DNS, WAF, туннели)
                          ├──► mcp-cluster      (наш код: чтение кластера)
                          └──► Azure AI Foundry (/openai/v1/)
```

**Агент — это конфиг, а не код.** Агенты отличаются четырьмя вещами: системный
промпт, список разрешённых инструментов, модель, лимиты. Всё это лежит в
`agents/*.yaml`. Добавить архитектора, кодера или ревьювера = добавить YAML-файл.

**Разделение — на уровне подов, а не репозиториев.** У `mcp-cluster` есть ClusterRole,
но нет токенов. У `mcp-cloudflare` есть Cloudflare-токен, но нет доступа к кластеру.
У агента есть LLM и GitHub PAT, но нет ClusterRole вообще.

## Структура

| Путь | Что |
|---|---|
| `src/agentcore/` | Агент: loop, MCP-клиент, LLM-клиент, policy, память, Telegram |
| `src/mcp_common/` | Общий скелет наших MCP-серверов (транспорт, health, ошибки) |
| `src/mcp_cloudflare/` | MCP-сервер: DNS, WAF, туннели |
| `src/mcp_cluster/` | MCP-сервер: read-only Kubernetes |
| `agents/` | Профили агентов |
| `tests/` | Юнит-тесты policy, каталога инструментов, туннелей, памяти |

Манифесты — в репозитории `personal-k8s`, в `applications/agents/`.

## Как это работает

1. Ты пишешь боту в Telegram.
2. `AgentLoop` собирает каталог инструментов со всех MCP-серверов, фильтрует его по
   профилю и отдаёт модели.
3. Модель предлагает вызовы инструментов, `Policy` их проверяет, `MCPPool` выполняет.
4. Каждый вызов пишется в stdout структурированным JSON → Alloy → Loki → Grafana.

### Команды бота

`/new` — забыть контекст · `/model` — сменить модель · `/models` — список ·
`/tools` — подключённые инструменты · `/status` — режимы записи и лимиты · `/cancel` — прервать

## Права

Реальные границы находятся вне процесса и не поддаются уговорам модели:

- GitHub PAT — fine-grained, ровно на нужные репозитории
- `mcp-github` запускается с `--toolsets`, зафиксированными в git
- `mcp-cluster` имеет только `get/list/watch`
- `mcp-cloudflare` экспортирует ровно те инструменты, что мы написали

`src/agentcore/policy.py` добавляет то, что зависит от аргументов, а не от личности —
прежде всего запрет на прямой push в GitOps-репозиторий. Плохой манифест в
`personal-k8s` доезжает до живого кластера примерно за минуту, поэтому это
правило действует независимо от режима записи.

Переключатели режимов лежат в ConfigMap, закомментированными альтернативами:
`GITHUB_WRITE_MODE`, `CLOUDFLARE_WRITE_MODE`.

## AI stats

После каждого agent turn `AgentLoop` добавляет в финальный ответ компактный footer
`📊 AI stats` с моделью, шагами, stop reason, usage (input/output/cached/reasoning/
billable), общей длительностью и числом/временем реально выполненных tool-вызовов.

`cached` — это prompt cache провайдера (`cached_tokens`), а не KV-cache модели.
Внутренний размер и состояние KV-cache, GPU memory и cache eviction провайдер API
не сообщает, поэтому платформа не выводит и не оценивает эти значения.

## Разработка

Ничего не собирается локально — всё проверяется в CI (`ruff`, импорт всех точек
входа, `pytest`), образ собирается там же под `linux/arm64` и уезжает в OCIR,
дальше Flux.

Если всё-таки нужно локально:

```bash
python -m venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env   # заполнить
.venv/bin/pytest -q
AGENT_PROFILE=lead .venv/bin/python -m agentcore
```

## Заметки, которые сэкономят время

- **MCP SDK 2.x переименовал почти всё.** `streamablehttp_client` → `streamable_http_client`,
  `FastMCP` → `MCPServer`, поля моделей стали snake_case (`input_schema`, `is_error`,
  `structured_content`), HTTP-библиотека — `httpx2`. Практически вся документация в
  сети показывает имена из 1.x.
- **Описание инструмента у Azure ограничено 1024 символами**, имя — 64. У GitHub MCP
  описания длиннее; без обрезки в `toolset.py` запрос падает с 400.
- **GitHub MCP в режиме `http` не читает токен из окружения** — он требует
  `Authorization: Bearer` в каждом запросе. `GITHUB_PERSONAL_ACCESS_TOKEN` работает
  только в режиме `stdio`.
- **Ingress туннеля Cloudflare упорядочен и заканчивается catch-all.** Правило,
  добавленное после него, не сработает никогда.
- **WAF — это Rulesets API**, а не старый `firewall/rules`: правила живут в
  entrypoint-ruleset фазы `http_request_firewall_custom` и меняются целиком через PUT.
