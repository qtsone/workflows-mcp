# Changelog

All notable changes to this project will be documented in this file.

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
