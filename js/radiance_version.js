/**
 * The package version, for the frontend (the viewer's header shows it).
 *
 * tests/test_version_sync.py fails if this differs from pyproject.toml, so it
 * cannot go stale the way the viewer's hard-coded "v3.5" did.
 */
export const RADIANCE_VERSION = "4.0.0";
