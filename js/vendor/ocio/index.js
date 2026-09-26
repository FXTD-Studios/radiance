// ── Radiance vendoring patch ────────────────────────────────────────────────
// Upstream resolves the Emscripten module through the package's "#ocio-wasm"
// imports map and the .wasm URL through ./wasm-url.js. Neither exists outside
// an npm install, and ComfyUI serves this directory as static files, so both
// are resolved here instead. This is the ONLY change from the published
// @bb-studio/ocio 0.0.7 source -- see README.md in this directory.
let _ocioWasmModule = null;
async function _loadOcioWasmModule() {
  if (_ocioWasmModule) return _ocioWasmModule;
  const isNode = typeof process !== 'undefined' && !!process.versions?.node;
  const mod = isNode
    ? await import('./ocio-wasm.node.js')
    : await import('./ocio-wasm.js');
  _ocioWasmModule = mod.default;
  return _ocioWasmModule;
}
const DEFAULT_WASM_URL = new URL('./ocio-wasm.wasm', import.meta.url).href;
// ── end patch ───────────────────────────────────────────────────────────────

export const ACES_CG_V2_CONFIG = 'ocio://cg-config-v2.2.0_aces-v1.3_ocio-v2.4';
export const ACES_STUDIO_V2_CONFIG = 'ocio://studio-config-v2.2.0_aces-v1.3_ocio-v2.4';
export const ACES_CG_V4_CONFIG = 'ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5';
export const ACES_STUDIO_V4_CONFIG = 'ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5';

export const TransformDirection = Object.freeze({
  FORWARD: 0,
  INVERSE: 1
});

export const OptimizationFlags = Object.freeze({
  DEFAULT: -2147483648,
  NONE: 0x00000000,
  LOSSLESS: 0x089c3fc3,
  VERY_GOOD: 0x0f9c3fc3,
  GOOD: 0x0fdc3fc3,
  DRAFT: -1
});

export const Interpolation = Object.freeze({
  UNKNOWN: 0,
  NEAREST: 1,
  LINEAR: 2,
  TETRAHEDRAL: 3,
  CUBIC: 4,
  DEFAULT: 254,
  BEST: 255
});

export const CDLStyle = Object.freeze({
  ASC: 0,
  NO_CLAMP: 1
});

const DEFAULT_GPU_SHADER_FUNCTION = 'OCIODisplay';
const DEFAULT_GPU_RESOURCE_PREFIX = 'ocio';
const GPU_UNIFORM_TYPES = Object.freeze([
  'double',
  'bool',
  'float3',
  'vector_float',
  'vector_int',
  'unknown'
]);

function assertTypedArray(value, type, name) {
  if (!(value instanceof type)) {
    throw new TypeError(`${name} must be a ${type.name}`);
  }
}

function toDirection(value) {
  if (value === TransformDirection.INVERSE || value === 'inverse') {
    return TransformDirection.INVERSE;
  }
  return TransformDirection.FORWARD;
}

function toOptimizationFlags(value) {
  if (value === undefined || value === null || value === 'default') {
    return OptimizationFlags.DEFAULT;
  }
  if (typeof value === 'number') {
    return value;
  }
  const key = String(value).toUpperCase().replace(/[-\s]/g, '_');
  if (Object.hasOwn(OptimizationFlags, key)) {
    return OptimizationFlags[key];
  }
  throw new Error(`Unknown OCIO optimization mode: ${value}`);
}

function toEnumValue(value, values, aliases, label, defaultValue) {
  if (value === undefined || value === null || value === '') {
    return defaultValue;
  }
  if (typeof value === 'number' && Object.values(values).includes(value)) {
    return value;
  }
  const key = String(value).toLowerCase().replace(/[\s_]/g, '-');
  const resolvedKey = aliases[key] ?? key.toUpperCase().replace(/-/g, '_');
  if (Object.hasOwn(values, resolvedKey)) {
    return values[resolvedKey];
  }
  throw new Error(`Unknown OCIO ${label}: ${value}`);
}

function toInterpolation(value) {
  return toEnumValue(
    value,
    Interpolation,
    { tetra: 'TETRAHEDRAL', trilinear: 'LINEAR' },
    'interpolation',
    Interpolation.DEFAULT
  );
}

function toCDLStyle(value) {
  return toEnumValue(
    value,
    CDLStyle,
    { 'no-clamp': 'NO_CLAMP', noclamp: 'NO_CLAMP' },
    'CDL style',
    CDLStyle.NO_CLAMP
  );
}

function requireNonEmptyString(value, name) {
  const text = value == null ? '' : String(value).trim();
  if (!text) {
    throw new TypeError(`${name} must be a non-empty string`);
  }
  return text;
}

function toOptionalBoolean(value, defaultValue, name) {
  if (value === undefined) {
    return defaultValue;
  }
  if (typeof value !== 'boolean') {
    throw new TypeError(`${name} must be a boolean`);
  }
  return value;
}

function normalizeGpuLanguage(value) {
  if (value === undefined || value === null || value === '' || value === 'glsl') {
    return 'glsl_es_3.0';
  }
  const language = String(value).toLowerCase().replace(/-/g, '_');
  if (language === 'webgl' || language === 'webgl1') {
    return 'glsl_es_1.0';
  }
  if (language === 'webgl2') {
    return 'glsl_es_3.0';
  }
  return language;
}

function toPositiveInteger(value, name) {
  if (value === undefined || value === null) {
    return 0;
  }
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    throw new RangeError(`${name} must be a positive finite number`);
  }
  return Math.floor(number);
}

export async function createOCIO(options = {}) {
  const moduleFactory = options.moduleFactory ?? (await _loadOcioWasmModule());
  const userLocateFile = options.locateFile;
  const wasmUrl = options.wasmUrl == null ? DEFAULT_WASM_URL : String(options.wasmUrl);
  const moduleOptions = {
    ...options.moduleOptions,
    locateFile(path, prefix) {
      if (userLocateFile) {
        return userLocateFile(path, prefix);
      }
      if (path === 'ocio-wasm.wasm') {
        return wasmUrl;
      }
      return prefix + path;
    }
  };

  return new OCIO(await moduleFactory(moduleOptions));
}

export class OCIO {
  constructor(module) {
    this.module = module;
  }

  get version() {
    return this._string('ocio_get_version');
  }

  get versionHex() {
    return this.module._ocio_get_version_hex();
  }

  get lastError() {
    return this._string('ocio_get_last_error');
  }

  clearAllCaches() {
    this.module._ocio_clear_all_caches();
  }

  listBuiltinConfigs() {
    const count = this.module._ocio_builtin_config_get_count();
    const configs = [];
    for (let index = 0; index < count; index += 1) {
      configs.push({
        name: this._string('ocio_builtin_config_get_name', index),
        uiName: this._string('ocio_builtin_config_get_ui_name', index),
        recommended: this.module._ocio_builtin_config_is_recommended(index) === 1
      });
    }
    return configs;
  }

  getBuiltinConfigYaml(name) {
    return this._withCString(name, (namePtr) => this._string('ocio_builtin_config_get_yaml', namePtr));
  }

  listFileTransformFormats() {
    const count = this.module._ocio_file_transform_get_num_formats();
    return Array.from({ length: count }, (_, index) => ({
      name: this._string('ocio_file_transform_get_format_name', index),
      extension: this._string('ocio_file_transform_get_format_extension', index)
    }));
  }

  isFileTransformFormatSupported(extension) {
    const value = requireNonEmptyString(extension, 'OCIO file-transform extension');
    return this._withCString(value, (extensionPtr) => (
      this.module._ocio_file_transform_is_format_extension_supported(extensionPtr) === 1
    ));
  }

  createBuiltinConfig(name = ACES_CG_V4_CONFIG) {
    const handle = this._withCString(name, (namePtr) => this.module._ocio_config_create_builtin(namePtr));
    this._assertHandle(handle, `Could not create built-in OCIO config ${name}`);
    return new Config(this, handle);
  }

  createConfigFromFile(path) {
    const handle = this._withCString(path, (pathPtr) => this.module._ocio_config_create_from_file(pathPtr));
    this._assertHandle(handle, `Could not create OCIO config from ${path}`);
    return new Config(this, handle);
  }

  createConfigFromString(text, options = {}) {
    const workingDir = options.workingDir ?? '';
    const handle = this._withCStrings([text, workingDir], ([textPtr, workingDirPtr]) => (
      this.module._ocio_config_create_from_string(textPtr, workingDirPtr)
    ));
    this._assertHandle(handle, 'Could not create OCIO config from string');
    return new Config(this, handle);
  }

  mkdirp(path) {
    const parts = path.split('/').filter(Boolean);
    let current = path.startsWith('/') ? '' : '.';
    for (const part of parts) {
      current = current === '' ? `/${part}` : `${current}/${part}`;
      try {
        this.module.FS.mkdir(current);
      } catch (error) {
        if (error?.errno !== 20) {
          throw error;
        }
      }
    }
  }

  writeFile(path, data) {
    const directory = path.slice(0, path.lastIndexOf('/'));
    if (directory) {
      this.mkdirp(directory);
    }
    this.module.FS.writeFile(path, data);
  }

  readFile(path, options) {
    return this.module.FS.readFile(path, options);
  }

  _assertHandle(handle, message) {
    if (!handle) {
      throw new Error(`${message}: ${this.lastError}`);
    }
  }

  _assertStatus(status, message) {
    if (!status) {
      throw new Error(`${message}: ${this.lastError}`);
    }
  }

  _string(functionName, ...args) {
    const pointer = this.module[`_${functionName}`](...args);
    return pointer ? this.module.UTF8ToString(pointer) : '';
  }

  _withCString(value, callback) {
    const text = value == null ? '' : String(value);
    const byteLength = this.module.lengthBytesUTF8(text) + 1;
    const pointer = this.module._malloc(byteLength);
    this.module.stringToUTF8(text, pointer, byteLength);
    try {
      return callback(pointer);
    } finally {
      this.module._free(pointer);
    }
  }

  _withCStrings(values, callback) {
    const pointers = [];
    try {
      for (const value of values) {
        const text = value == null ? '' : String(value);
        const byteLength = this.module.lengthBytesUTF8(text) + 1;
        const pointer = this.module._malloc(byteLength);
        this.module.stringToUTF8(text, pointer, byteLength);
        pointers.push(pointer);
      }
      return callback(pointers);
    } finally {
      for (const pointer of pointers) {
        this.module._free(pointer);
      }
    }
  }
}

export class Config {
  constructor(ocio, handle) {
    this.ocio = ocio;
    this.handle = handle;
  }

  get version() {
    return {
      major: this.ocio.module._ocio_config_get_major_version(this.handle),
      minor: this.ocio.module._ocio_config_get_minor_version(this.handle)
    };
  }

  validate() {
    this.ocio._assertStatus(this.ocio.module._ocio_config_validate(this.handle), 'OCIO config validation failed');
    return true;
  }

  dispose() {
    if (this.handle) {
      this.ocio.module._ocio_config_release(this.handle);
      this.handle = 0;
    }
  }

  listRoles() {
    const count = this.ocio.module._ocio_config_get_num_roles(this.handle);
    return Array.from({ length: count }, (_, index) => ({
      name: this.ocio._string('ocio_config_get_role_name', this.handle, index),
      colorSpace: this.ocio._string('ocio_config_get_role_color_space', this.handle, index)
    }));
  }

  listColorSpaces() {
    const count = this.ocio.module._ocio_config_get_num_color_spaces(this.handle);
    return Array.from({ length: count }, (_, index) => {
      const name = this.ocio._string('ocio_config_get_color_space_name', this.handle, index);
      return this.getColorSpace(name);
    });
  }

  getColorSpace(name) {
    return this.ocio._withCString(name, (namePtr) => {
      const aliasCount = this.ocio.module._ocio_config_get_num_color_space_aliases(this.handle, namePtr);
      const categoryCount = this.ocio.module._ocio_config_get_num_color_space_categories(this.handle, namePtr);
      return {
        name,
        canonicalName: this.ocio._string('ocio_config_get_canonical_name', this.handle, namePtr),
        family: this.ocio._string('ocio_config_get_color_space_family', this.handle, namePtr),
        encoding: this.ocio._string('ocio_config_get_color_space_encoding', this.handle, namePtr),
        description: this.ocio._string('ocio_config_get_color_space_description', this.handle, namePtr),
        isData: this.ocio.module._ocio_config_get_color_space_is_data(this.handle, namePtr) === 1,
        referenceSpace: this.ocio.module._ocio_config_get_color_space_reference_space(this.handle, namePtr),
        aliases: Array.from({ length: aliasCount }, (_, index) => (
          this.ocio._string('ocio_config_get_color_space_alias', this.handle, namePtr, index)
        )),
        categories: Array.from({ length: categoryCount }, (_, index) => (
          this.ocio._string('ocio_config_get_color_space_category', this.handle, namePtr, index)
        ))
      };
    });
  }

  getFileRule(index) {
    const ruleIndex = Number(index);
    if (!Number.isInteger(ruleIndex) || ruleIndex < 0) {
      throw new RangeError('OCIO file rule index must be a non-negative integer');
    }

    const customKeyCount = this.ocio.module._ocio_config_get_file_rule_custom_key_count(
      this.handle,
      ruleIndex
    );
    const custom = {};
    for (let keyIndex = 0; keyIndex < customKeyCount; keyIndex += 1) {
      const name = this.ocio._string(
        'ocio_config_get_file_rule_custom_key_name',
        this.handle,
        ruleIndex,
        keyIndex
      );
      custom[name] = this.ocio._string(
        'ocio_config_get_file_rule_custom_key_value',
        this.handle,
        ruleIndex,
        keyIndex
      );
    }

    return {
      index: ruleIndex,
      name: this.ocio._string('ocio_config_get_file_rule_name', this.handle, ruleIndex),
      colorSpace: this.ocio._string('ocio_config_get_file_rule_color_space', this.handle, ruleIndex),
      pattern: this.ocio._string('ocio_config_get_file_rule_pattern', this.handle, ruleIndex),
      extension: this.ocio._string('ocio_config_get_file_rule_extension', this.handle, ruleIndex),
      regex: this.ocio._string('ocio_config_get_file_rule_regex', this.handle, ruleIndex),
      custom
    };
  }

  listFileRules() {
    const count = this.ocio.module._ocio_config_get_num_file_rules(this.handle);
    return Array.from({ length: count }, (_, index) => this.getFileRule(index));
  }

  matchFileRule(filePath) {
    return this.ocio._withCString(filePath, (filePathPtr) => {
      const colorSpace = this.ocio._string(
        'ocio_config_get_color_space_from_filepath',
        this.handle,
        filePathPtr
      );
      if (!colorSpace) {
        return null;
      }

      const ruleIndex = this.ocio.module._ocio_config_get_file_rule_index_from_filepath(
        this.handle,
        filePathPtr
      );
      const rule = this.getFileRule(ruleIndex);
      const isDefaultRule =
        this.ocio.module._ocio_config_filepath_only_matches_default_rule(
          this.handle,
          filePathPtr
        ) === 1;
      return {
        colorSpace,
        ruleIndex,
        ruleName: rule.name,
        isDefaultRule,
        custom: rule.custom
      };
    });
  }

  listDisplays() {
    const count = this.ocio.module._ocio_config_get_num_displays(this.handle);
    return Array.from({ length: count }, (_, index) => this.ocio._string('ocio_config_get_display', this.handle, index));
  }

  getDefaultDisplay() {
    return this.ocio._string('ocio_config_get_default_display', this.handle);
  }

  listViews(display) {
    return this.ocio._withCString(display, (displayPtr) => {
      const count = this.ocio.module._ocio_config_get_num_views(this.handle, displayPtr);
      return Array.from({ length: count }, (_, index) => {
        const name = this.ocio._string('ocio_config_get_view', this.handle, displayPtr, index);
        return this.getView(display, name);
      });
    });
  }

  getView(display, view) {
    return this.ocio._withCStrings([display, view], ([displayPtr, viewPtr]) => ({
      name: view,
      transform: this.ocio._string('ocio_config_get_view_transform_name', this.handle, displayPtr, viewPtr),
      colorSpace: this.ocio._string('ocio_config_get_view_color_space_name', this.handle, displayPtr, viewPtr),
      looks: this.ocio._string('ocio_config_get_view_looks', this.handle, displayPtr, viewPtr),
      description: this.ocio._string('ocio_config_get_view_description', this.handle, displayPtr, viewPtr)
    }));
  }

  getDefaultView(display, colorSpace) {
    if (colorSpace) {
      return this.ocio._withCStrings([display, colorSpace], ([displayPtr, colorSpacePtr]) => (
        this.ocio._string('ocio_config_get_default_view_for_color_space', this.handle, displayPtr, colorSpacePtr)
      ));
    }
    return this.ocio._withCString(display, (displayPtr) => (
      this.ocio._string('ocio_config_get_default_view', this.handle, displayPtr)
    ));
  }

  listLooks() {
    const count = this.ocio.module._ocio_config_get_num_looks(this.handle);
    return Array.from({ length: count }, (_, index) => this.ocio._string('ocio_config_get_look_name', this.handle, index));
  }

  getLook(name) {
    const resolvedName = requireNonEmptyString(name, 'OCIO look name');
    return this.ocio._withCString(resolvedName, (namePtr) => {
      const processSpace = this.ocio._string(
        'ocio_config_get_look_process_space',
        this.handle,
        namePtr
      );
      if (!processSpace) {
        throw new Error(`Could not get OCIO look ${resolvedName}: ${this.ocio.lastError}`);
      }
      return {
        name: resolvedName,
        processSpace,
        description: this.ocio._string(
          'ocio_config_get_look_description',
          this.handle,
          namePtr
        ),
        hasForwardTransform:
          this.ocio.module._ocio_config_look_has_transform(
            this.handle,
            namePtr,
            TransformDirection.FORWARD
          ) === 1,
        hasInverseTransform:
          this.ocio.module._ocio_config_look_has_transform(
            this.handle,
            namePtr,
            TransformDirection.INVERSE
          ) === 1
      };
    });
  }

  getLooksResultColorSpace(looks, options = {}) {
    const resolvedLooks = requireNonEmptyString(looks, 'OCIO looks');
    return this._withContext(options.context, (contextHandle) => (
      this.ocio._withCString(resolvedLooks, (looksPtr) => {
        const colorSpace = this.ocio._string(
          'ocio_config_get_looks_result_color_space',
          this.handle,
          contextHandle,
          looksPtr
        );
        if (!colorSpace) {
          throw new Error(
            `Could not resolve OCIO looks ${resolvedLooks}: ${this.ocio.lastError}`
          );
        }
        return colorSpace;
      })
    ));
  }

  listViewTransforms() {
    const count = this.ocio.module._ocio_config_get_num_view_transforms(this.handle);
    return Array.from({ length: count }, (_, index) => (
      this.ocio._string('ocio_config_get_view_transform_name_by_index', this.handle, index)
    ));
  }

  listNamedTransforms() {
    const count = this.ocio.module._ocio_config_get_num_named_transforms(this.handle);
    return Array.from({ length: count }, (_, index) => (
      this.ocio._string('ocio_config_get_named_transform_name', this.handle, index)
    ));
  }

  getNamedTransform(name) {
    const resolvedName = requireNonEmptyString(name, 'OCIO named-transform name');
    return this.ocio._withCString(resolvedName, (namePtr) => {
      const canonicalName = this.ocio._string(
        'ocio_config_get_named_transform_canonical_name',
        this.handle,
        namePtr
      );
      if (!canonicalName) {
        throw new Error(
          `Could not get OCIO named transform ${resolvedName}: ${this.ocio.lastError}`
        );
      }
      const aliasCount = this.ocio.module._ocio_config_get_num_named_transform_aliases(
        this.handle,
        namePtr
      );
      const categoryCount = this.ocio.module._ocio_config_get_num_named_transform_categories(
        this.handle,
        namePtr
      );
      return {
        name: canonicalName,
        family: this.ocio._string(
          'ocio_config_get_named_transform_family',
          this.handle,
          namePtr
        ),
        description: this.ocio._string(
          'ocio_config_get_named_transform_description',
          this.handle,
          namePtr
        ),
        encoding: this.ocio._string(
          'ocio_config_get_named_transform_encoding',
          this.handle,
          namePtr
        ),
        aliases: Array.from({ length: aliasCount }, (_, index) => this.ocio._string(
          'ocio_config_get_named_transform_alias',
          this.handle,
          namePtr,
          index
        )),
        categories: Array.from({ length: categoryCount }, (_, index) => this.ocio._string(
          'ocio_config_get_named_transform_category',
          this.handle,
          namePtr,
          index
        )),
        hasForwardTransform:
          this.ocio.module._ocio_config_named_transform_has_transform(
            this.handle,
            namePtr,
            TransformDirection.FORWARD
          ) === 1,
        hasInverseTransform:
          this.ocio.module._ocio_config_named_transform_has_transform(
            this.handle,
            namePtr,
            TransformDirection.INVERSE
          ) === 1
      };
    });
  }

  createColorSpaceProcessor(source, destination, options = {}) {
    const optimization = toOptimizationFlags(options.optimization);
    const handle = this._withContext(options.context, (contextHandle) => (
      this.ocio._withCStrings([source, destination], ([sourcePtr, destinationPtr]) => (
        this.ocio.module._ocio_processor_create_color_space(
          this.handle,
          contextHandle,
          sourcePtr,
          destinationPtr,
          optimization
        )
      ))
    ));
    this.ocio._assertHandle(handle, `Could not create OCIO ColorSpace processor ${source} -> ${destination}`);
    return new Processor(this.ocio, handle);
  }

  createDisplayViewProcessor({
    source,
    display,
    view,
    direction = TransformDirection.FORWARD,
    optimization,
    context,
    looksBypass,
    dataBypass
  } = {}) {
    if (looksBypass !== undefined || dataBypass !== undefined) {
      return this.createGroupTransformProcessor([
        {
          type: 'displayView',
          source,
          display,
          view,
          direction,
          looksBypass,
          dataBypass
        }
      ], { optimization, context });
    }
    const optimizationFlags = toOptimizationFlags(optimization);
    const transformDirection = toDirection(direction);
    const handle = this._withContext(context, (contextHandle) => (
      this.ocio._withCStrings([source, display, view], ([sourcePtr, displayPtr, viewPtr]) => (
        this.ocio.module._ocio_processor_create_display_view(
          this.handle,
          contextHandle,
          sourcePtr,
          displayPtr,
          viewPtr,
          transformDirection,
          optimizationFlags
        )
      ))
    ));
    this.ocio._assertHandle(handle, `Could not create OCIO Display/View processor ${source} -> ${display}/${view}`);
    return new Processor(this.ocio, handle);
  }

  createNamedTransformProcessor(name, options = {}) {
    const resolvedName = requireNonEmptyString(name, 'OCIO named-transform name');
    const optimization = toOptimizationFlags(options.optimization);
    const direction = toDirection(options.direction);
    const handle = this._withContext(options.context, (contextHandle) => (
      this.ocio._withCString(resolvedName, (namePtr) => (
        this.ocio.module._ocio_processor_create_named_transform(
          this.handle,
          contextHandle,
          namePtr,
          direction,
          optimization
        )
      ))
    ));
    this.ocio._assertHandle(handle, `Could not create OCIO NamedTransform processor ${resolvedName}`);
    return new Processor(this.ocio, handle);
  }

  createFileTransformProcessor(options = {}) {
    return this.createGroupTransformProcessor(
      [{ type: 'file', ...options }],
      { optimization: options.optimization, context: options.context }
    );
  }

  createLookTransformProcessor(options = {}) {
    return this.createGroupTransformProcessor(
      [{ type: 'look', ...options }],
      { optimization: options.optimization, context: options.context }
    );
  }

  createGroupTransformProcessor(transforms, options = {}) {
    if (!Array.isArray(transforms) || transforms.length === 0) {
      throw new TypeError('OCIO group transforms require a non-empty transform array');
    }

    const groupHandle = this.ocio.module._ocio_group_transform_create();
    this.ocio._assertHandle(groupHandle, 'Could not create OCIO GroupTransform');
    try {
      transforms.forEach((transform, index) => {
        try {
          this._appendGroupTransform(groupHandle, transform);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(`Invalid OCIO group transform at index ${index}: ${message}`, {
            cause: error
          });
        }
      });

      const optimization = toOptimizationFlags(options.optimization);
      const direction = toDirection(options.direction);
      const handle = this._withContext(options.context, (contextHandle) => (
        this.ocio.module._ocio_processor_create_group_transform(
          this.handle,
          contextHandle,
          groupHandle,
          direction,
          optimization
        )
      ));
      this.ocio._assertHandle(handle, 'Could not create OCIO GroupTransform processor');
      return new Processor(this.ocio, handle);
    } finally {
      this.ocio.module._ocio_group_transform_release(groupHandle);
    }
  }

  _appendGroupTransform(groupHandle, transform) {
    if (!transform || typeof transform !== 'object' || Array.isArray(transform)) {
      throw new TypeError('OCIO transform descriptor must be an object');
    }

    const direction = toDirection(transform.direction);
    if (transform.type === 'colorSpace') {
      const source = requireNonEmptyString(transform.source, 'ColorSpaceTransform source');
      const destination = requireNonEmptyString(
        transform.destination,
        'ColorSpaceTransform destination'
      );
      const dataBypass = toOptionalBoolean(
        transform.dataBypass,
        true,
        'ColorSpaceTransform dataBypass'
      );
      this.ocio._withCStrings([source, destination], ([sourcePtr, destinationPtr]) => {
        this.ocio._assertStatus(
          this.ocio.module._ocio_group_transform_append_color_space(
            groupHandle,
            sourcePtr,
            destinationPtr,
            direction,
            dataBypass ? 1 : 0
          ),
          `Could not append OCIO ColorSpaceTransform ${source} -> ${destination}`
        );
      });
      return;
    }

    if (transform.type === 'file') {
      const src = requireNonEmptyString(transform.src, 'FileTransform src');
      const cccId = transform.cccId == null ? '' : String(transform.cccId);
      const interpolation = toInterpolation(transform.interpolation);
      const cdlStyle = toCDLStyle(transform.cdlStyle);
      this.ocio._withCStrings([src, cccId], ([srcPtr, cccIdPtr]) => {
        this.ocio._assertStatus(
          this.ocio.module._ocio_group_transform_append_file(
            groupHandle,
            srcPtr,
            direction,
            interpolation,
            cccIdPtr,
            cdlStyle
          ),
          `Could not append OCIO FileTransform ${src}`
        );
      });
      return;
    }

    if (transform.type === 'look') {
      const source = requireNonEmptyString(transform.source, 'LookTransform source');
      const destination = requireNonEmptyString(
        transform.destination,
        'LookTransform destination'
      );
      const looks = requireNonEmptyString(transform.looks, 'LookTransform looks');
      const skipColorSpaceConversion = toOptionalBoolean(
        transform.skipColorSpaceConversion,
        false,
        'LookTransform skipColorSpaceConversion'
      );
      this.ocio._withCStrings(
        [source, destination, looks],
        ([sourcePtr, destinationPtr, looksPtr]) => {
          this.ocio._assertStatus(
            this.ocio.module._ocio_group_transform_append_look(
              groupHandle,
              sourcePtr,
              destinationPtr,
              looksPtr,
              direction,
              skipColorSpaceConversion ? 1 : 0
            ),
            `Could not append OCIO LookTransform ${looks}`
          );
        }
      );
      return;
    }

    if (transform.type === 'displayView') {
      const source = requireNonEmptyString(transform.source, 'DisplayViewTransform source');
      const display = requireNonEmptyString(transform.display, 'DisplayViewTransform display');
      const view = requireNonEmptyString(transform.view, 'DisplayViewTransform view');
      const looksBypass = toOptionalBoolean(
        transform.looksBypass,
        false,
        'DisplayViewTransform looksBypass'
      );
      const dataBypass = toOptionalBoolean(
        transform.dataBypass,
        true,
        'DisplayViewTransform dataBypass'
      );
      this.ocio._withCStrings(
        [source, display, view],
        ([sourcePtr, displayPtr, viewPtr]) => {
          this.ocio._assertStatus(
            this.ocio.module._ocio_group_transform_append_display_view(
              groupHandle,
              sourcePtr,
              displayPtr,
              viewPtr,
              direction,
              looksBypass ? 1 : 0,
              dataBypass ? 1 : 0
            ),
            `Could not append OCIO DisplayViewTransform ${display}/${view}`
          );
        }
      );
      return;
    }

    if (transform.type === 'named') {
      const name = requireNonEmptyString(transform.name, 'NamedTransform name');
      this.ocio._withCString(name, (namePtr) => {
        this.ocio._assertStatus(
          this.ocio.module._ocio_group_transform_append_named(
            groupHandle,
            this.handle,
            namePtr,
            direction
          ),
          `Could not append OCIO NamedTransform ${name}`
        );
      });
      return;
    }

    throw new Error(`Unknown OCIO group transform type: ${transform.type}`);
  }

  _withContext(context, callback) {
    const entries = Object.entries(context ?? {}).sort(([left], [right]) => left.localeCompare(right));
    if (entries.length === 0) {
      return callback(0);
    }

    const contextHandle = this.ocio.module._ocio_context_create(this.handle);
    this.ocio._assertHandle(contextHandle, 'Could not create OCIO context');
    try {
      for (const [name, value] of entries) {
        if (!name.trim() || typeof value !== 'string') {
          throw new TypeError('OCIO context variables require non-empty names and string values');
        }
        this.ocio._withCStrings([name, value], ([namePtr, valuePtr]) => {
          this.ocio._assertStatus(
            this.ocio.module._ocio_context_set_string_var(contextHandle, namePtr, valuePtr),
            `Could not set OCIO context variable ${name}`
          );
        });
      }
      return callback(contextHandle);
    } finally {
      this.ocio.module._ocio_context_release(contextHandle);
    }
  }
}

export class Processor {
  constructor(ocio, handle) {
    this.ocio = ocio;
    this.handle = handle;
  }

  get cacheId() {
    return this.ocio._string('ocio_processor_get_cache_id', this.handle);
  }

  get isNoOp() {
    return this.ocio.module._ocio_processor_is_noop(this.handle) === 1;
  }

  get isIdentity() {
    return this.ocio.module._ocio_processor_is_identity(this.handle) === 1;
  }

  dispose() {
    if (this.handle) {
      this.ocio.module._ocio_processor_release(this.handle);
      this.handle = 0;
    }
  }

  applyRGBF32(rgb, options = {}) {
    assertTypedArray(rgb, Float32Array, 'rgb');
    if (rgb.length % 3 !== 0) {
      throw new RangeError('rgb length must be divisible by 3');
    }
    const target = options.copy ? new Float32Array(rgb) : rgb;
    return this._applyFloat32(target, 3, 'ocio_processor_apply_rgb_f32');
  }

  applyRGBAF32(rgba, options = {}) {
    assertTypedArray(rgba, Float32Array, 'rgba');
    if (rgba.length % 4 !== 0) {
      throw new RangeError('rgba length must be divisible by 4');
    }
    const target = options.copy ? new Float32Array(rgba) : rgba;
    return this._applyFloat32(target, 4, 'ocio_processor_apply_rgba_f32');
  }

  applyRGBA8(rgba, options = {}) {
    if (!(rgba instanceof Uint8Array) && !(rgba instanceof Uint8ClampedArray)) {
      throw new TypeError('rgba must be a Uint8Array or Uint8ClampedArray');
    }
    if (rgba.length % 4 !== 0) {
      throw new RangeError('rgba length must be divisible by 4');
    }

    const target = options.copy ? new rgba.constructor(rgba) : rgba;
    const byteLength = target.byteLength;
    const pointer = this.ocio.module._malloc(byteLength);
    this.ocio.module.HEAPU8.set(target, pointer);
    try {
      this.ocio._assertStatus(
        this.ocio.module._ocio_processor_apply_rgba_u8(this.handle, pointer, target.length / 4),
        'OCIO Uint8 RGBA processing failed'
      );
      target.set(this.ocio.module.HEAPU8.subarray(pointer, pointer + byteLength));
      return target;
    } finally {
      this.ocio.module._free(pointer);
    }
  }

  getGpuShaderInfo(options = {}) {
    const language = normalizeGpuLanguage(options.language);
    const functionName = options.functionName ?? DEFAULT_GPU_SHADER_FUNCTION;
    const resourcePrefix = options.resourcePrefix ?? DEFAULT_GPU_RESOURCE_PREFIX;
    const textureMaxWidth = toPositiveInteger(options.textureMaxWidth, 'textureMaxWidth');
    const allowTexture1D = options.allowTexture1D === true ? 1 : 0;

    this.ocio._withCStrings([language, functionName, resourcePrefix], ([languagePtr, functionNamePtr, resourcePrefixPtr]) => {
      this.ocio._assertStatus(
        this.ocio.module._ocio_processor_extract_gpu_shader_info(
          this.handle,
          languagePtr,
          functionNamePtr,
          resourcePrefixPtr,
          textureMaxWidth,
          allowTexture1D
        ),
        'OCIO GPU shader extraction failed'
      );
    });

    return {
      shaderText: this.ocio._string('ocio_processor_get_gpu_shader_text', this.handle),
      functionName: this.ocio._string('ocio_processor_get_gpu_shader_function_name', this.handle),
      language: this.ocio._string('ocio_processor_get_gpu_shader_language', this.handle),
      cacheId: this.ocio._string('ocio_processor_get_gpu_shader_cache_id', this.handle),
      uniformBufferSize: this.ocio.module._ocio_processor_get_gpu_shader_uniform_buffer_size(this.handle),
      textures: this._getGpuShaderTextures(),
      uniforms: this._getGpuShaderUniforms()
    };
  }

  getGpuShaderText(options = {}) {
    return this.getGpuShaderInfo(options).shaderText;
  }

  _applyFloat32(target, channels, functionName) {
    const byteLength = target.byteLength;
    const pointer = this.ocio.module._malloc(byteLength);
    this.ocio.module.HEAPF32.set(target, pointer / Float32Array.BYTES_PER_ELEMENT);
    try {
      this.ocio._assertStatus(
        this.ocio.module[`_${functionName}`](this.handle, pointer, target.length / channels),
        'OCIO Float32 processing failed'
      );
      target.set(this.ocio.module.HEAPF32.subarray(
        pointer / Float32Array.BYTES_PER_ELEMENT,
        pointer / Float32Array.BYTES_PER_ELEMENT + target.length
      ));
      return target;
    } finally {
      this.ocio.module._free(pointer);
    }
  }

  _getGpuShaderTextures() {
    const count = this.ocio.module._ocio_processor_get_gpu_shader_texture_count(this.handle);
    return Array.from({ length: count }, (_, index) => {
      const valueCount = this.ocio.module._ocio_processor_get_gpu_shader_texture_value_count(this.handle, index);
      const valuesPointer = this.ocio.module._ocio_processor_get_gpu_shader_texture_values(this.handle, index);
      if (!valuesPointer && valueCount > 0) {
        throw new Error(`Could not read OCIO GPU texture values: ${this.ocio.lastError}`);
      }
      const valuesOffset = valuesPointer / Float32Array.BYTES_PER_ELEMENT;
      return {
        name: this.ocio._string('ocio_processor_get_gpu_shader_texture_name', this.handle, index),
        samplerName: this.ocio._string('ocio_processor_get_gpu_shader_texture_sampler_name', this.handle, index),
        width: this.ocio.module._ocio_processor_get_gpu_shader_texture_width(this.handle, index),
        height: this.ocio.module._ocio_processor_get_gpu_shader_texture_height(this.handle, index),
        depth: this.ocio.module._ocio_processor_get_gpu_shader_texture_depth(this.handle, index),
        dimensions: this.ocio.module._ocio_processor_get_gpu_shader_texture_dimensions(this.handle, index),
        channels: this.ocio.module._ocio_processor_get_gpu_shader_texture_channels(this.handle, index),
        interpolation: this.ocio._string('ocio_processor_get_gpu_shader_texture_interpolation', this.handle, index),
        values: new Float32Array(this.ocio.module.HEAPF32.subarray(valuesOffset, valuesOffset + valueCount))
      };
    });
  }

  _getGpuShaderUniforms() {
    const count = this.ocio.module._ocio_processor_get_gpu_shader_uniform_count(this.handle);
    return Array.from({ length: count }, (_, index) => {
      const typeId = this.ocio.module._ocio_processor_get_gpu_shader_uniform_type(this.handle, index);
      const type = GPU_UNIFORM_TYPES[typeId] ?? 'unknown';
      const valueCount = this.ocio.module._ocio_processor_get_gpu_shader_uniform_value_count(this.handle, index);
      let value;
      if (type === 'bool') {
        value = this.ocio.module._ocio_processor_get_gpu_shader_uniform_value_i32(this.handle, index, 0) !== 0;
      } else if (type === 'vector_int') {
        value = Array.from({ length: valueCount }, (_, valueIndex) => (
          this.ocio.module._ocio_processor_get_gpu_shader_uniform_value_i32(this.handle, index, valueIndex)
        ));
      } else if (valueCount === 1) {
        value = this.ocio.module._ocio_processor_get_gpu_shader_uniform_value_f64(this.handle, index, 0);
      } else {
        value = Array.from({ length: valueCount }, (_, valueIndex) => (
          this.ocio.module._ocio_processor_get_gpu_shader_uniform_value_f64(this.handle, index, valueIndex)
        ));
      }

      return {
        name: this.ocio._string('ocio_processor_get_gpu_shader_uniform_name', this.handle, index),
        type,
        bufferOffset: this.ocio.module._ocio_processor_get_gpu_shader_uniform_buffer_offset(this.handle, index),
        value
      };
    });
  }
}
