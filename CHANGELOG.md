# Changelog

All notable changes to this project will be documented in this file.

# [7.1.0](https://github.com/qtsone/workflows-mcp/compare/v7.0.0...v7.1.0) (2025-11-18)


### Features

* **ReadFiles:** simplify content output for single file reads ([25aa538](https://github.com/qtsone/workflows-mcp/commit/25aa538ffcdc7b22ec422e3f1b7dc887763c932c))
* **ReadFiles:** simplify content output for single file reads ([#33](https://github.com/qtsone/workflows-mcp/issues/33)) ([7430945](https://github.com/qtsone/workflows-mcp/commit/743094517b2e72505a21c445eae2aafe0381a44e))

# [7.0.0](https://github.com/qtsone/workflows-mcp/compare/v6.5.0...v7.0.0) (2025-11-17)


* feat(engine)!: introduce workflow-scoped tmp path and migrate templates ([4422538](https://github.com/qtsone/workflows-mcp/commit/4422538c271b725e2f2d4719b2fefda607cfa565))


### Bug Fixes

* **git-commit:** replace escape sequence handling in commit message template ([9b57b1a](https://github.com/qtsone/workflows-mcp/commit/9b57b1ac08da0e01d5cb0b5112d8a500dbd45fb7))
* **tests:** replace bash-specific substring syntax with POSIX-compatible printf ([edd966c](https://github.com/qtsone/workflows-mcp/commit/edd966c703b642ed767caaeafffccd09ed8514dc))


### Code Refactoring

* remove render template executor and related functionality ([3f7cd09](https://github.com/qtsone/workflows-mcp/commit/3f7cd0952ccda9f4cb34f41edc9f71bfc9166b92))


### Features

* **engine:** add comprehensive interpolation support for EditFile operations ([a29f325](https://github.com/qtsone/workflows-mcp/commit/a29f325a285d8510bb5af11ceee67b41aa9a3f28))
* **engine:** implement unified variable resolver with rule-based transformation pipeline ([befea5f](https://github.com/qtsone/workflows-mcp/commit/befea5f3587048e162cd67c5fd1dc8887e29ccff))
* **file operations:** add readfiles support with glob patterns and base path ([aad24e7](https://github.com/qtsone/workflows-mcp/commit/aad24e7f17204a2ac04678d29ae8f7f667762adb))
* **workflows-mcp:** Add profile fallback for portable LLM workflows ([85df48f](https://github.com/qtsone/workflows-mcp/commit/85df48f94fb2b6ec63712ed84e17566f17b331f7))


### Performance Improvements

* **ci:** improve ci pipeline efficiency ([67bd33c](https://github.com/qtsone/workflows-mcp/commit/67bd33c6212c407d6a53f3984ebc4dedbd2a62da))


### BREAKING CHANGES

* This changes runtime behavior that may break existing workflows:

What breaks and why:
- $SCRATCH usage removed/deprecated: templates and workflows that relied on the literal $SCRATCH path or shell environment substitution will no longer write/read to that path. The code now expects workflow templates to use the jinja variable {{tmp}} which is resolved to the execution scratch dir.
- Absolute path allowance tightened: previously some $SCRATCH-based handling allowed absolute output paths; executors now only permit absolute paths if the path is inside the configured scratch directory or the output schema explicitly sets unsafe=true. Absolute paths outside the scratch directory will be disallowed, which can change behavior for blocks that wrote to arbitrary absolute locations.
- Secrets materialization semantics changed: if secrets were pre-materialized as a plain dict in the jinja context, the resolver now merges existing values with newly fetched secrets instead of treating them the same as a SecretProxy. This can alter how secret values are looked up/overwritten.
- validate.py now recursively scans YAML (*.yaml, *.yml) which may surface additional validation errors for files that were previously ignored.

Migration steps for users:
1. Update all workflows and templates to replace any occurrences of $SCRATCH with {{tmp}} in commands, file paths, and CreateFile/ReadFiles inputs.
2. For scripts that relied on absolute paths previously allowed via $SCRATCH, either:
   - change to use paths under {{tmp}} (recommended), or
   - explicitly mark outputs as unsafe in the output schema (not recommended unless necessary).
3. Ensure any code that injected secrets into the jinja context provides a SecretProxy where possible; if an existing dict is used intentionally, verify merge semantics meet expectations.
4. Re-run validation (validate.py) and tests to catch any additional YAML files that now get scanned.

Alternative approaches if migration is not immediately possible:
- As a stopgap, workflows can set an environment variable SCRATCH inside block inputs to mimic prior behavior, or modify blocks to write to a path referenced by {{tmp}} by injecting the same value into the jinja context. However, migrating to {{tmp}} is the supported path forward.

Add a workflow-scoped jinja variable `{{tmp}}` (mapped to the execution scratch dir) and migrate templates/tests to use it; tighten Shell executor path handling; improve secret materialization; add a local LLM mock for tests and update snapshots. Key changes:
- expose "tmp" in workflow jinja context (workflow_runner)
- replace $SCRATCH usages in templates and tests with {{tmp}} and update schema examples
- ShellExecutor: treat raw paths as already-resolved values, allow absolute paths only when they reside inside the scratch dir or when output schema marks unsafe
- UnifiedVariableResolver: handle already-materialized secrets (dict) by merging with newly fetched secrets instead of overwriting
- add a local LLM mock and tests for LLM executor; add/update snapshots
- broaden validate.py to scan YAML recursively
- remove/replace several README/template docs and update many test workflows

These changes unify temporary-path handling, improve security around absolute paths, fix secret materialization issues, and make LLM-related tests hermetic.
* removes the RenderTemplate executor and all associated code, including schema validation, template rendering logic, and test workflows that depended on it. this change simplifies the file operation executors by removing the template rendering capability and updates existing workflows to use alternative approaches like Shell commands or CreateFile blocks. the RenderTemplate functionality was replaced with direct shell command execution and content rendering in templates, reducing complexity and removing the dependency on jinja2 templating for these use cases.
* **engine:** Replace legacy VariableResolver with new architecture that provides:
- Expression classification for optimal Jinja2 routing
- Rule-based transformation pipeline (security, syntax, namespace)
- Enhanced context with BlockProxy and SecretProxy
- SandboxedEnvironment for template rendering security
- Clean separation of concerns between rules and evaluation

Key components:
- UnifiedVariableResolver: Main resolver orchestrating transformation pipeline
- ExpressionClassifier: Routes expressions to appropriate evaluation methods
- TransformRule system: Modular, priority-based transformation rules
- BlockProxy: ADR-007 shortcuts and nested block access
- SecretProxy: Lazy secret loading with audit logging
- Security rules: Forbidden namespace blocking, secret tracking
- Syntax rules: Bracket notation normalization, special character handling

Updates:
- RenderTemplateExecutor uses SandboxedEnvironment instead of Environment
- BlockOrchestrator switches to UnifiedVariableResolver
- Legacy variables.py preserved as variables.py.bak
- Test workflows and snapshots reorganized for new resolver

# [6.5.0](https://github.com/qtsone/workflows-mcp/compare/v6.4.0...v6.5.0) (2025-11-12)


### Features

* **engine:** add comprehensive interpolation support for EditFile operations ([f3b890e](https://github.com/qtsone/workflows-mcp/commit/f3b890e0763462f635dc345b28e74df313038461))
* **engine:** add EditFile block for deterministic file editing ([029a188](https://github.com/qtsone/workflows-mcp/commit/029a188db281a13919ea83b547148fe9993e2d7c))
* **engine:** add EditFile block for deterministic file editing ([#28](https://github.com/qtsone/workflows-mcp/issues/28)) ([0b64c3a](https://github.com/qtsone/workflows-mcp/commit/0b64c3ad88a3308fdb89201f0f81985e5562fc75))
* **tests:** enhance variable resolution test suite with comprehensive validation ([bb048f2](https://github.com/qtsone/workflows-mcp/commit/bb048f280a3816b06c3f3b0e5c7a158cd3941de2))

# [6.4.0](https://github.com/qtsone/workflows-mcp/compare/v6.3.0...v6.4.0) (2025-11-12)


### Bug Fixes

* make job queue status filter test resilient to race conditions ([de50e10](https://github.com/qtsone/workflows-mcp/commit/de50e10bdef9601be0bf2a82b95cfdd8445c8dc4)), closes [#27](https://github.com/qtsone/workflows-mcp/issues/27)


### Features

* unified job architecture with async execution and queue support ([#27](https://github.com/qtsone/workflows-mcp/issues/27)) ([bca1d1c](https://github.com/qtsone/workflows-mcp/commit/bca1d1c4a81e684254fe3fb4d59fdb38be8bad5b))

# [6.3.0](https://github.com/qtsone/workflows-mcp/compare/v6.2.0...v6.3.0) (2025-11-10)


### Features

* **for_each:** add dynamic key interpolation and fractal execution s… ([#24](https://github.com/qtsone/workflows-mcp/issues/24)) ([3f404e5](https://github.com/qtsone/workflows-mcp/commit/3f404e5ddf7b01ee0fff6238b169ccf402edf5ac))
* **for_each:** add dynamic key interpolation and fractal execution support ([a938de6](https://github.com/qtsone/workflows-mcp/commit/a938de67a6e0e9855fa3c2fde91836f5eecc33fc))

# [6.2.0](https://github.com/qtsone/workflows-mcp/compare/v6.1.1...v6.2.0) (2025-11-08)


### Features

* **llm:** add profile-based configuration system ([d7acdb8](https://github.com/qtsone/workflows-mcp/commit/d7acdb80dc739068b3a0f7ae98b0d06fb4d95866))
* **llm:** add profile-based configuration system ([#23](https://github.com/qtsone/workflows-mcp/issues/23)) ([d0a124c](https://github.com/qtsone/workflows-mcp/commit/d0a124cfc05717b845377f62b768cc53ac125ffb))

## [6.1.1](https://github.com/qtsone/workflows-mcp/compare/v6.1.0...v6.1.1) (2025-11-07)


### Bug Fixes

* **schema:** resolve LLMProvider $ref by moving nested $defs to root level ([4b402f7](https://github.com/qtsone/workflows-mcp/commit/4b402f7c776cff6db3ef838774de58b7763b6919))
* **schema:** resolve LLMProvider $ref by moving nested $defs to root level ([#21](https://github.com/qtsone/workflows-mcp/issues/21)) ([5f13750](https://github.com/qtsone/workflows-mcp/commit/5f13750b12b70e2f40086cc72f9372d9ed16fa74))

# [6.1.0](https://github.com/qtsone/workflows-mcp/compare/v6.0.0...v6.1.0) (2025-11-07)


### Features

* **engine:** add interpolatable field support for dynamic type resolution ([652a1ce](https://github.com/qtsone/workflows-mcp/commit/652a1ce703c97bef4d4b9b344cc9697fe8e14dd0))
* **engine:** add interpolatable field support for dynamic type resolution ([#20](https://github.com/qtsone/workflows-mcp/issues/20)) ([c860c44](https://github.com/qtsone/workflows-mcp/commit/c860c44355282820cc1b88256987abb00b332996))

# [6.0.0](https://github.com/qtsone/workflows-mcp/compare/v5.0.0...v6.0.0) (2025-11-06)


* refactor(engine)!: replace _internal dict with strongly-typed ExecutionInternal model ([15df1b1](https://github.com/qtsone/workflows-mcp/commit/15df1b1b4605d1800d33efe32a55b8e3707b7fb2))


### Features

* **ADR-009:** add outcome field to distinguish operation failure from executor crash ([e3e2d45](https://github.com/qtsone/workflows-mcp/commit/e3e2d45176d2f83d86f5bc8a96b68d27c94cb765))
* **engine:** implement for_each type preservation and empty collection handling ([0e2f8ab](https://github.com/qtsone/workflows-mcp/commit/0e2f8abb9492498b9104c160cca1e8820cac49fe))


### BREAKING CHANGES

* HttpCallInput field names changed. Update workflows to use
'json' instead of 'body_json' and 'content' instead of 'body_text'.
* **ADR-009:** NodeMeta now includes 'outcome' field

ADR-009 Implementation - Phase 2: Outcome Field

Changes:
- Added 'outcome' field to NodeMeta with values: success, failure, crash, n/a
- Updated all NodeMeta factory methods to set appropriate outcome values
- Modified orchestrator.py to pass outcome="crash" for executor exceptions
- For_each parent aggregates outcome from children (crash > failure > success)
- Fixed resume_workflow status check from .is_failed() to .failed property
- Added model_validator to Execution for checkpoint deserialization

Key Outcomes:
- Operation failure: executor runs, operation fails (e.g., exit 1) - outcome="failure"
- Executor crash: executor crashes with exception - outcome="crash"
- Skipped blocks: never executed - outcome="n/a"
- Successful blocks: operation succeeded - outcome="success"

Test Status: 88/91 tests passing (96.7%)
- 2 resume_workflow tests failing (investigation ongoing - MCP server crash issue)
- 1 secrets-http-auth test failing (external httpbin.org 503 error)

Files Modified:
- src/workflows_mcp/engine/node_meta.py: Added outcome field, updated factories
- src/workflows_mcp/engine/orchestrator.py: Pass outcome="crash" for exceptions
- src/workflows_mcp/engine/workflow_runner.py: Fixed .is_failed() → .failed
- src/workflows_mcp/engine/execution.py: Added model_validator for checkpoint deserialization

See: docs/adr/ADR-009-FOR-EACH-ABSTRACTION.md

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
