# Changelog

All notable changes to this project will be documented in this file.

# [5.0.0](https://github.com/qtsone/workflows-mcp/compare/v4.4.0...v5.0.0) (2025-11-04)


### Code Refactoring

* unify int/float types to num and fix composition bug ([0537afc](https://github.com/qtsone/workflows-mcp/commit/0537afce73701e0d175dcaf5a0e0eaefa1c92e16))


### Features

* **LLMCall:** BREAKING CHANGE Add support for LLM Executor ([4d831fe](https://github.com/qtsone/workflows-mcp/commit/4d831fe59526d09898bb07abf7ed0a07309d6390))
* **LLMCall:** BREAKING CHANGE Add support for LLM Executor ([#18](https://github.com/qtsone/workflows-mcp/issues/18)) ([5a54cb9](https://github.com/qtsone/workflows-mcp/commit/5a54cb93508448a1fbdecbd793bf1a612c05066d))


### BREAKING CHANGES

* **LLMCall:** **
- Workflow YAML files using `type: int` or `type: float` should migrate
to `type: num`
* **
- Workflow YAML files using `type: int` or `type: float` should migrate to `type: num`

# [4.4.0](https://github.com/qtsone/workflows-mcp/compare/v4.3.0...v4.4.0) (2025-11-02)


### Bug Fixes

* **release:** use deploy key ([18e1fc9](https://github.com/qtsone/workflows-mcp/commit/18e1fc9f7d1ed43685561a002f9233f178cdcbb4))


### Features

* add comprehensive secrets management system ([31d2732](https://github.com/qtsone/workflows-mcp/commit/31d2732d30365283639c320891632c4f7e508865))
* add comprehensive secrets management system ([#17](https://github.com/qtsone/workflows-mcp/issues/17)) ([6927464](https://github.com/qtsone/workflows-mcp/commit/69274647653228b183258f251ba3ff83b200135a))
* **tests:** normalise snapshots ([91f4059](https://github.com/qtsone/workflows-mcp/commit/91f4059cf4b9578b6ad5c7420c6b009c4f175d96))

# [4.3.0](https://github.com/qtsone/workflows-mcp/compare/v4.2.0...v4.3.0) (2025-11-01)


### Features

* add HttpCall block executor for HTTP/REST API integration ([9ca2b70](https://github.com/qtsone/workflows-mcp/commit/9ca2b7002a85d38cdef5e94952509340bf34f940))
* add HttpCall block executor for HTTP/REST API integration ([#16](https://github.com/qtsone/workflows-mcp/issues/16)) ([76d1714](https://github.com/qtsone/workflows-mcp/commit/76d171483af915c4d468d4e14fec2710fc836231))

# [4.2.0](https://github.com/qtsone/workflows-mcp/compare/v4.1.0...v4.2.0) (2025-11-01)


### Bug Fixes

* **checkpointing:** properly handle checkpoint config for nested workflows ([b79740b](https://github.com/qtsone/workflows-mcp/commit/b79740b5b3990137ffbb6fcab75044323dff0e92))


### Features

* **workflows:** add process-todo-create-issues workflow template ([e932de3](https://github.com/qtsone/workflows-mcp/commit/e932de3b420a13a1907d64192095504602deb456))
* **workflows:** add process-todo-create-issues workflow template ([#11](https://github.com/qtsone/workflows-mcp/issues/11)) ([309564d](https://github.com/qtsone/workflows-mcp/commit/309564dd37b0259f2e04ed4d26d81385e981bfd8))

# [4.1.0](https://github.com/qtsone/workflows-mcp/compare/v4.0.3...v4.1.0) (2025-10-30)


### Bug Fixes

* **server:** improve graceful shutdown and fix naming convention ([86b66eb](https://github.com/qtsone/workflows-mcp/commit/86b66eb0bfa91d8bb7e04e70d7c322971e585897))


### Features

* **github:** add interactive issue creation workflow ([52da77f](https://github.com/qtsone/workflows-mcp/commit/52da77f2cd046f146a853d8d309733c07f432527))

## [4.0.3](https://github.com/qtsone/workflows-mcp/compare/v4.0.2...v4.0.3) (2025-10-27)


### Bug Fixes

* **git:** quote condition value to fix YAML syntax ([97ae68f](https://github.com/qtsone/workflows-mcp/commit/97ae68f144a06d6c0b0bc38552291ea0de113617))
* **git:** use env vars for safe commit message handling ([#6](https://github.com/qtsone/workflows-mcp/issues/6)) ([807e61c](https://github.com/qtsone/workflows-mcp/commit/807e61c1d27b3df97abf92d64925d85c099518bb))
* **git:** use env vars for safe message handling (GitHub Actions pattern) ([48297ea](https://github.com/qtsone/workflows-mcp/commit/48297eab62805de89c25159c330cbf86c7b8f2a0))
* **yaml:** quote all unquoted {{...}} variable references ([0e2b476](https://github.com/qtsone/workflows-mcp/commit/0e2b476ff71462a2609f18299bf43eed43c9461c))

## [4.0.2](https://github.com/qtsone/workflows-mcp/compare/v4.0.1...v4.0.2) (2025-10-27)


### Bug Fixes

* **git:** fix boolean expression syntax and shell expansion issues ([daa1d75](https://github.com/qtsone/workflows-mcp/commit/daa1d75427248a2a5550d0d0a63a1826781934ea))
* **git:** fix boolean expression syntax and shell expansion issues ([#4](https://github.com/qtsone/workflows-mcp/issues/4)) ([176c60d](https://github.com/qtsone/workflows-mcp/commit/176c60dae04fee2884ace06c82b7c10385cedcaa))

## [4.0.1](https://github.com/qtsone/workflows-mcp/compare/v4.0.0...v4.0.1) (2025-10-26)


### Bug Fixes

* remove EchoBlock in favor of Shell executor ([#1](https://github.com/qtsone/workflows-mcp/issues/1)) ([678d259](https://github.com/qtsone/workflows-mcp/commit/678d2597f28627eb79b3adec23c50bde0d629814))

# [4.0.0](https://github.com/qtsone/workflows-mcp/compare/v3.3.0...v4.0.0) (2025-10-26)


* chore!: migrate to AGPL-3.0 license ([3431aab](https://github.com/qtsone/workflows-mcp/commit/3431aab67acad56c7ee4f3f37fc7293a3e9749f2))


### BREAKING CHANGES

* License changed from GPL-3.0 to AGPL-3.0-or-later. This license change requires users to provide source code access when the software is used over a network. All existing functionality is preserved.
