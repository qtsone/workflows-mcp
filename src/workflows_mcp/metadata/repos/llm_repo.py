from __future__ import annotations

import json
from sqlite3 import Connection
from typing import Any

import yaml

from workflows_mcp.engine.llm_config import LLMConfig, ProfileConfig, ProviderConfig


class SQLiteLLMConfigRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def load_config(self) -> LLMConfig:
        provider_rows = self._conn.execute(
            "SELECT provider_name, config_json FROM llm_providers ORDER BY provider_name ASC"
        ).fetchall()
        profile_rows = self._conn.execute(
            """
            SELECT profile_name, provider_name, model, config_json
            FROM llm_profiles
            ORDER BY profile_name ASC
            """
        ).fetchall()
        default_profile_row = self._conn.execute(
            "SELECT value FROM server_settings WHERE key = ?",
            ("llm.default_profile",),
        ).fetchone()

        providers: dict[str, ProviderConfig] = {}
        for row in provider_rows:
            providers[str(row[0])] = ProviderConfig(**json.loads(str(row[1])))

        profiles: dict[str, ProfileConfig] = {}
        for row in profile_rows:
            profile_payload: dict[str, Any] = json.loads(str(row[3]))
            profile_payload["provider"] = str(row[1])
            profile_payload["model"] = str(row[2])
            profiles[str(row[0])] = ProfileConfig(**profile_payload)

        default_profile = str(default_profile_row[0]) if default_profile_row is not None else None
        config = LLMConfig(
            providers=providers,
            profiles=profiles,
            default_profile=default_profile,
        )
        config.validate_profile_provider_references()
        return config

    def replace_config(self, config: LLMConfig) -> None:
        config.validate_profile_provider_references()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute("DELETE FROM llm_profiles")
            self._conn.execute("DELETE FROM llm_providers")

            for provider_name, provider_config in config.providers.items():
                payload = provider_config.model_dump(mode="json", exclude_none=True)
                self._conn.execute(
                    """
                    INSERT INTO llm_providers (provider_name, config_json)
                    VALUES (?, ?)
                    """,
                    (provider_name, json.dumps(payload, sort_keys=True)),
                )

            for profile_name, profile_config in config.profiles.items():
                payload = profile_config.model_dump(mode="json", exclude_none=True)
                provider_name = str(payload.pop("provider"))
                model_name = str(payload.pop("model"))
                self._conn.execute(
                    """
                    INSERT INTO llm_profiles (profile_name, provider_name, model, config_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (profile_name, provider_name, model_name, json.dumps(payload, sort_keys=True)),
                )

            if config.default_profile is None:
                self._conn.execute(
                    "DELETE FROM server_settings WHERE key = ?",
                    ("llm.default_profile",),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO server_settings (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    ("llm.default_profile", config.default_profile),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def preview_yaml(self, raw_yaml: str) -> LLMConfig:
        parsed = yaml.safe_load(raw_yaml)
        if not isinstance(parsed, dict):
            raise ValueError("LLM config YAML must parse to a mapping")
        config = LLMConfig(**parsed)
        config.validate_profile_provider_references()
        return config

    def import_yaml(self, raw_yaml: str) -> LLMConfig:
        config = self.preview_yaml(raw_yaml)
        self.replace_config(config)
        return config

    def export_yaml(self) -> str:
        config = self.load_config()
        config_dict = config.model_dump(mode="python", exclude_none=True)
        return yaml.safe_dump(config_dict, sort_keys=True)
