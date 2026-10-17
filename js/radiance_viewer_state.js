/**
 * The viewer's saved display preferences: one schema, one loader.
 *
 * These were about thirty separate localStorage reads in the constructor, each
 * with its own default and no check on what came back. A stale or hand-edited
 * value (a scope mode that no longer exists, "NaN" for a peak level) went
 * straight into the viewer state. Here every preference has a type, a default
 * and, where the allowed set is known, a list of values; anything else loads as
 * the default.
 *
 * Scope: per-browser display preferences only. Workflow content (the grade,
 * pinned references, media) is not stored here. Storage keys are unchanged, so
 * nothing written by an earlier version needs migrating; the version marker and
 * the migration runner exist so the next change that does need one has a place.
 *
 * Pure: takes a Storage-like object, never touches the DOM or `window`.
 */
import { SCOPE_SCALES, LEVELS } from "./radiance_scope_units.js";

export const STATE_VERSION = 1;
export const VERSION_KEY = 'radiance_viewer_prefs_version';

const SCOPE_MODES = ['parade', 'waveform', 'histogram', 'vectorscope', 'chromaticity', 'falsecolor'];

/**
 * field: the viewer property. key: the localStorage key (unchanged).
 * type 'enum' needs `values`; 'bool' stores '0' / '1' (read: anything but the
 * `off` string is on, as before); 'int' needs `min`; 'string' is any
 * non-empty text up to `max` characters, for lists the viewer owns.
 */
export const PREFS = Object.freeze([
    { field: 'compareFit', key: 'radiance_compare_fit', type: 'enum', values: ['pixel', 'fit'], default: 'pixel' },
    { field: 'scopeMode', key: 'radiance_scope_mode', type: 'enum', values: SCOPE_MODES, default: 'parade' },
    { field: 'scopeScale', key: 'radiance_scope_scale', type: 'enum', values: SCOPE_SCALES.map((s) => s.id), default: 'cv10' },
    { field: 'scopeLevels', key: 'radiance_scope_levels', type: 'enum', values: LEVELS.map((l) => l.id), default: 'data' },
    { field: 'scopeTransformed', key: 'radiance_scope_xform', type: 'bool', off: '0', default: true },
    { field: 'scopeHlgPeak', key: 'radiance_scope_hlg_peak', type: 'int', min: 100, default: 1000 },
    { field: 'safeAreaPreset', key: 'radiance_safe_preset', type: 'string', max: 32, default: 'modern' },
    { field: 'matteMode', key: 'radiance_matte', type: 'string', max: 32, default: 'off' },
    { field: 'pixelFilter', key: 'radiance_pixel_filter', type: 'enum', values: ['nearest', 'linear'], default: 'nearest' },
    { field: 'timeDisplay', key: 'radiance_time_display', type: 'enum', values: ['frames', 'seconds', 'timecode'], default: 'frames' },
    { field: 'displayTarget', key: 'radiance_display_target', type: 'string', max: 32, default: 'sRGB' },
    // 'manual' is a runtime state (a LUT picked by hand), never a saved choice.
    { field: 'viewMode', key: 'radiance_view_mode', type: 'string', max: 32, default: 'auto', reject: ['manual'] },
]);

const BY_FIELD = new Map(PREFS.map((p) => [p.field, p]));

function read(storage, key) {
    try { return storage?.getItem?.(key) ?? null; } catch { return null; }
}

/** Parse one stored string for a preference. Returns the default if it is unusable. */
export function parsePref(pref, raw) {
    if (raw === null || raw === undefined) return pref.default;
    switch (pref.type) {
        case 'enum':
            return pref.values.includes(raw) ? raw : pref.default;
        case 'bool':
            return raw !== pref.off;
        case 'int': {
            const n = parseInt(raw, 10);
            return Number.isFinite(n) && n >= (pref.min ?? -Infinity) ? n : pref.default;
        }
        case 'string':
            if (typeof raw !== 'string' || !raw || raw.length > (pref.max ?? 64)) return pref.default;
            return pref.reject?.includes(raw) ? pref.default : raw;
        default:
            return pref.default;
    }
}

/** What to store for a value, or null when the value is not valid for the preference. */
export function serializePref(pref, value) {
    switch (pref.type) {
        case 'enum': return pref.values.includes(value) ? value : null;
        case 'bool': return value ? '1' : pref.off;
        case 'int': return Number.isFinite(value) && value >= (pref.min ?? -Infinity) ? String(Math.round(value)) : null;
        case 'string':
            return typeof value === 'string' && value && value.length <= (pref.max ?? 64) && !pref.reject?.includes(value)
                ? value : null;
        default: return null;
    }
}

/**
 * Apply migrations from the stored version up to STATE_VERSION, then record the
 * version. `migrations` maps a version N to a function that upgrades N -> N+1.
 * A migration that throws stops the run at that step (the version is not
 * advanced past it) and the error is returned; loading carries on with defaults.
 */
export function migrate(storage, migrations = MIGRATIONS, target = STATE_VERSION) {
    let v = parseInt(read(storage, VERSION_KEY), 10);
    if (!Number.isFinite(v) || v < 1) v = 1;       // anything before the marker is version 1
    let error = null;
    while (v < target) {
        try { migrations[v]?.(storage); } catch (e) { error = e; break; }
        v += 1;
    }
    try { if (!error) storage?.setItem?.(VERSION_KEY, String(Math.max(v, target))); } catch { /* storage is optional */ }
    return { version: v, error };
}

export const MIGRATIONS = Object.freeze({});

/** Every preference, validated, with defaults for anything missing or unusable. */
export function loadPrefs(storage) {
    migrate(storage);
    const out = {};
    for (const p of PREFS) out[p.field] = parsePref(p, read(storage, p.key));
    return out;
}

/** Save one preference. Returns false (and writes nothing) for an invalid value. */
export function savePref(storage, field, value) {
    const pref = BY_FIELD.get(field);
    if (!pref) return false;
    const s = serializePref(pref, value);
    if (s === null) return false;
    try { storage?.setItem?.(pref.key, s); return true; } catch { return false; }
}
