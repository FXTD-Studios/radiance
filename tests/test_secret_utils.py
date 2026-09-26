from __future__ import annotations

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from secret_utils import normalize_env_name, resolve_secret


class TestSecretUtils(unittest.TestCase):
    def test_normalize_env_name_accepts_plain_or_prefixed_names(self):
        self.assertEqual(normalize_env_name("RADIANCE_TOKEN"), "RADIANCE_TOKEN")
        self.assertEqual(normalize_env_name("env:RADIANCE_TOKEN"), "RADIANCE_TOKEN")

    def test_normalize_env_name_rejects_non_env_strings(self):
        self.assertEqual(normalize_env_name("sk-secret-value"), "")
        self.assertEqual(normalize_env_name("MY TOKEN"), "")

    def test_resolve_secret_prefers_named_environment_value(self):
        env = {"CUSTOM_TOKEN": "custom", "DEFAULT_TOKEN": "default"}
        result = resolve_secret(
            explicit_value="direct",
            env_var="CUSTOM_TOKEN",
            default_env_var="DEFAULT_TOKEN",
            environ=env,
        )
        self.assertEqual(result, "custom")

    def test_resolve_secret_falls_back_to_default_env_then_direct_value(self):
        self.assertEqual(
            resolve_secret("direct", "", "DEFAULT_TOKEN", {"DEFAULT_TOKEN": "default"}),
            "default",
        )
        self.assertEqual(resolve_secret("direct", "", "", {}), "direct")


if __name__ == "__main__":
    unittest.main()
