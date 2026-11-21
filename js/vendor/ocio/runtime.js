const DEFAULT_MAX_RGB_CACHE_ENTRIES = 2048;

const hashString = (value) => {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
};

const hashBytes = (hash, bytes) => {
  let result = hash;
  for (const byte of bytes) {
    result ^= byte;
    result = Math.imul(result, 16777619);
  }
  return result;
};

const toUint8Array = (value, label) => {
  if (value instanceof Uint8Array) return new Uint8Array(value);
  if (!ArrayBuffer.isView(value)) {
    throw new TypeError(`${label} must be an ArrayBufferView`);
  }
  return new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
};

const normalizeRelativePath = (value, label = 'OCIO package path') => {
  if (typeof value !== 'string') throw new TypeError(`${label} must be a string`);
  const normalized = value.replaceAll('\\', '/').replace(/^\/+/, '');
  const segments = normalized.split('/');
  if (!normalized || segments.some((segment) => !segment || segment === '.' || segment === '..')) {
    throw new Error(`Invalid ${label} "${value}"`);
  }
  return normalized;
};

const normalizeVirtualPath = (value) => {
  if (typeof value !== 'string') throw new TypeError('OCIO virtual path must be a string');
  const normalized = value.replaceAll('\\', '/');
  if (!normalized.startsWith('/')) {
    throw new Error(`OCIO virtual path must be absolute: "${value}"`);
  }
  return `/${normalizeRelativePath(normalized, 'OCIO virtual path')}`;
};

const normalizeContext = (context = {}) => {
  if (!context || typeof context !== 'object' || Array.isArray(context)) {
    throw new TypeError('OCIO context must be an object');
  }
  const entries = Object.entries(context);
  for (const [name, value] of entries) {
    if (!name.trim() || typeof value !== 'string') {
      throw new TypeError('OCIO context variables require non-empty names and string values');
    }
  }
  return Object.fromEntries(entries.sort(([left], [right]) => left.localeCompare(right)));
};

const canonicalize = (value) => {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, canonicalize(entry)])
  );
};

const stableStringify = (value) => JSON.stringify(canonicalize(value));

const freezeDeep = (value) => {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const entry of Object.values(value)) freezeDeep(entry);
  return value;
};

const fingerprintPackage = (source) => {
  let hash = 2166136261;
  for (const file of source.files) {
    hash = hashBytes(hash, new TextEncoder().encode(file.relativePath));
    hash = hashBytes(hash, file.data);
  }
  return (hash >>> 0).toString(36);
};

export const normalizeOcioConfigPackage = (source) => {
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    throw new TypeError('OCIO config package must be an object');
  }
  if (!Array.isArray(source.files) || source.files.length === 0) {
    throw new TypeError('OCIO config package requires a non-empty files array');
  }

  const configRelativePath = normalizeRelativePath(
    source.configRelativePath,
    'OCIO config relative path'
  );
  const seen = new Set();
  const files = source.files.map((file, index) => {
    if (!file || typeof file !== 'object' || Array.isArray(file)) {
      throw new TypeError(`OCIO config package file ${index} must be an object`);
    }
    const relativePath = normalizeRelativePath(
      file.relativePath,
      `OCIO config package file ${index} path`
    );
    if (seen.has(relativePath)) {
      throw new Error(`Duplicate OCIO config package path "${relativePath}"`);
    }
    seen.add(relativePath);
    return {
      relativePath,
      data: toUint8Array(file.data, `OCIO config package file ${relativePath}`)
    };
  });

  if (!seen.has(configRelativePath)) {
    throw new Error(`OCIO config package is missing "${configRelativePath}"`);
  }
  return { configRelativePath, files };
};

const normalizeProcessorRequest = (request) => {
  if (!request || typeof request !== 'object' || Array.isArray(request)) {
    throw new TypeError('OCIO processor request must be an object');
  }
  const transforms = Array.isArray(request.transforms)
    ? request.transforms
    : request.transforms
      ? [request.transforms]
      : [];
  if (transforms.length === 0) {
    throw new TypeError('OCIO processor request requires at least one transform');
  }
  return {
    transforms: transforms.map((transform) => canonicalize(transform)),
    optimization: request.optimization ?? 'default',
    direction: request.direction ?? 'forward',
    context: normalizeContext(request.context)
  };
};

const assertRgb = (color) => {
  if (!Array.isArray(color) && !(color instanceof Float32Array)) {
    throw new TypeError('OCIO RGB transform requires a three-channel array');
  }
  if (color.length !== 3 || !Array.from(color).every(Number.isFinite)) {
    throw new TypeError('OCIO RGB transform requires three finite values');
  }
};

export class OcioRuntime {
  constructor(ocio, options = {}) {
    if (!ocio || typeof ocio.createBuiltinConfig !== 'function') {
      throw new TypeError('OcioRuntime requires an OCIO instance');
    }
    const maxRgbCacheEntries = options.maxRgbCacheEntries ?? DEFAULT_MAX_RGB_CACHE_ENTRIES;
    if (!Number.isInteger(maxRgbCacheEntries) || maxRgbCacheEntries < 1) {
      throw new RangeError('maxRgbCacheEntries must be a positive integer');
    }
    this.ocio = ocio;
    this.maxRgbCacheEntries = maxRgbCacheEntries;
    this.config = null;
    this.configInfo = null;
    this.configGeneration = 0;
    this.processorCache = new Map();
    this.gpuShaderCache = new Map();
    this.webGpuShaderCache = new Map();
    this.rgbTransformCache = new Map();
    this.latestFailure = null;
    this.disposed = false;
  }

  get activeConfigId() {
    return this.configInfo?.id ?? null;
  }

  getConfigInfo() {
    return this.configInfo;
  }

  loadBuiltinConfig(name) {
    this._assertUsable();
    const id = String(name ?? '').trim();
    if (!id) throw new TypeError('Built-in OCIO config name must be non-empty');
    return this._loadCandidate(id, () => this.ocio.createBuiltinConfig(id));
  }

  inspectBuiltinConfig(name) {
    this._assertUsable();
    const id = String(name ?? '').trim();
    if (!id) throw new TypeError('Built-in OCIO config name must be non-empty');
    return this._inspectCandidate(id, () => this.ocio.createBuiltinConfig(id));
  }

  loadConfigPackage(source, options = {}) {
    this._assertUsable();
    const mounted = this._mountConfigPackage(source, options);
    return this._loadCandidate(mounted.id, () => this.ocio.createConfigFromFile(mounted.path));
  }

  inspectConfigPackage(source, options = {}) {
    this._assertUsable();
    const mounted = this._mountConfigPackage(source, options);
    return this._inspectCandidate(mounted.id, () => this.ocio.createConfigFromFile(mounted.path));
  }

  mountFile(path, data) {
    this._assertUsable();
    const resolvedPath = normalizeVirtualPath(path);
    this.ocio.writeFile(resolvedPath, toUint8Array(data, `OCIO virtual file ${resolvedPath}`));
    return resolvedPath;
  }

  matchFileRule(filePath) {
    const config = this._requireConfig();
    const path = String(filePath ?? '').trim();
    if (!path) throw new TypeError('OCIO file-rule matching requires a non-empty path');
    return config.matchFileRule(path);
  }

  getViews(display) {
    const config = this._requireConfig();
    const resolvedDisplay = String(display ?? '').trim();
    if (!resolvedDisplay) throw new TypeError('OCIO display name must be non-empty');
    return config.listViews(resolvedDisplay);
  }

  getDefaultView(display, colorSpace) {
    const config = this._requireConfig();
    const resolvedDisplay = String(display ?? '').trim();
    if (!resolvedDisplay) throw new TypeError('OCIO display name must be non-empty');
    const resolvedColorSpace = colorSpace == null ? '' : String(colorSpace).trim();
    return resolvedColorSpace
      ? config.getDefaultView(resolvedDisplay, resolvedColorSpace)
      : config.getDefaultView(resolvedDisplay);
  }

  getGpuShaderInfo(request, options = {}) {
    const normalized = normalizeProcessorRequest(request);
    const processorEntry = this._getProcessor(normalized);
    if (processorEntry.processor.isNoOp || processorEntry.processor.isIdentity) return null;
    const shaderOptions = this._resolveShaderOptions(processorEntry.key, options);
    const key = `${processorEntry.key}|glsl:${stableStringify(shaderOptions)}`;
    const cached = this.gpuShaderCache.get(key);
    if (cached) return cached.value;
    try {
      const value = processorEntry.processor.getGpuShaderInfo(shaderOptions);
      this.gpuShaderCache.set(key, { contextKey: processorEntry.contextKey, value });
      return value;
    } catch (error) {
      this._recordFailure(error);
      throw error;
    }
  }

  getWebGpuShaderInfo(request, options = {}) {
    const normalized = normalizeProcessorRequest(request);
    const processorEntry = this._getProcessor(normalized);
    if (processorEntry.processor.isNoOp || processorEntry.processor.isIdentity) {
      return Promise.resolve(null);
    }
    const shaderOptions = this._resolveShaderOptions(processorEntry.key, options);
    delete shaderOptions.language;
    delete shaderOptions.allowTexture1D;
    const key = `${processorEntry.key}|wgsl:${stableStringify(shaderOptions)}`;
    const cached = this.webGpuShaderCache.get(key);
    if (cached) return cached.promise;

    const promise = processorEntry.processor.getWebGpuShaderInfo(shaderOptions).catch((error) => {
      this._recordFailure(error);
      if (this.webGpuShaderCache.get(key)?.promise === promise) {
        this.webGpuShaderCache.delete(key);
      }
      throw error;
    });
    this.webGpuShaderCache.set(key, { contextKey: processorEntry.contextKey, promise });
    return promise;
  }

  transformRgb(source, destination, color, options = {}) {
    this._requireConfig();
    assertRgb(color);
    const src = String(source ?? '').trim();
    const dst = String(destination ?? '').trim();
    if (!src || !dst) throw new TypeError('OCIO RGB transform requires source and destination');
    if (src === dst) return Array.from(color);

    const normalized = normalizeProcessorRequest({
      transforms: { type: 'colorSpace', source: src, destination: dst },
      optimization: options.optimization ?? 'lossless',
      context: options.context
    });
    const processorEntry = this._getProcessor(normalized);
    const colorKey = Array.from(color).join(',');
    const key = `${processorEntry.key}|rgb:${colorKey}`;
    const cached = this.rgbTransformCache.get(key);
    if (cached) {
      this.rgbTransformCache.delete(key);
      this.rgbTransformCache.set(key, cached);
      return [...cached.value];
    }

    try {
      const input = new Float32Array(color);
      const output = processorEntry.processor.isNoOp || processorEntry.processor.isIdentity
        ? input
        : processorEntry.processor.applyRGBF32(input);
      const value = [output[0], output[1], output[2]];
      this.rgbTransformCache.set(key, { contextKey: processorEntry.contextKey, value });
      while (this.rgbTransformCache.size > this.maxRgbCacheEntries) {
        const oldestKey = this.rgbTransformCache.keys().next().value;
        this.rgbTransformCache.delete(oldestKey);
      }
      return [...value];
    } catch (error) {
      this._recordFailure(error);
      throw error;
    }
  }

  invalidateContext(context) {
    this._assertUsable();
    const contextKey = stableStringify(normalizeContext(context));
    for (const [key, entry] of this.processorCache) {
      if (entry.contextKey !== contextKey) continue;
      entry.processor.dispose();
      this.processorCache.delete(key);
    }
    for (const cache of [this.gpuShaderCache, this.webGpuShaderCache, this.rgbTransformCache]) {
      for (const [key, entry] of cache) {
        if (entry.contextKey === contextKey) cache.delete(key);
      }
    }
  }

  clearCaches() {
    for (const entry of this.processorCache.values()) entry.processor.dispose();
    this.processorCache.clear();
    this.gpuShaderCache.clear();
    this.webGpuShaderCache.clear();
    this.rgbTransformCache.clear();
  }

  getDiagnostics() {
    return freezeDeep({
      activeConfigId: this.activeConfigId,
      processorCacheEntries: this.processorCache.size,
      gpuShaderCacheEntries: this.gpuShaderCache.size,
      webGpuShaderCacheEntries: this.webGpuShaderCache.size,
      rgbTransformCacheEntries: this.rgbTransformCache.size,
      latestFailure: this.latestFailure
    });
  }

  dispose() {
    if (this.disposed) return;
    this.clearCaches();
    this.config?.dispose();
    this.config = null;
    this.configInfo = null;
    this.disposed = true;
  }

  _loadCandidate(id, createConfig) {
    let candidate = null;
    try {
      candidate = createConfig();
      candidate.validate();
      const info = this._createConfigInfo(id, candidate);
      this.clearCaches();
      this.config?.dispose();
      this.config = candidate;
      this.configInfo = info;
      this.configGeneration += 1;
      this.latestFailure = null;
      candidate = null;
      return info;
    } catch (error) {
      this._recordFailure(error);
      throw error;
    } finally {
      candidate?.dispose();
    }
  }

  _inspectCandidate(id, createConfig) {
    let candidate = null;
    try {
      candidate = createConfig();
      candidate.validate();
      return this._createConfigInfo(id, candidate);
    } catch (error) {
      this._recordFailure(error);
      throw error;
    } finally {
      candidate?.dispose();
    }
  }

  _mountConfigPackage(source, options) {
    const normalized = normalizeOcioConfigPackage(source);
    const fingerprint = fingerprintPackage(normalized);
    const requestedId = options.id == null ? '' : String(options.id).trim();
    const id = requestedId || `package:${fingerprint}`;
    const root = `/ocio-runtime/configs/${hashString(`${id}:${fingerprint}`)}`;
    this.ocio.mkdirp(root);
    for (const file of normalized.files) {
      const segments = file.relativePath.split('/');
      if (segments.length > 1) {
        this.ocio.mkdirp(`${root}/${segments.slice(0, -1).join('/')}`);
      }
      this.ocio.writeFile(`${root}/${file.relativePath}`, file.data);
    }
    return { id, path: `${root}/${normalized.configRelativePath}` };
  }

  _createConfigInfo(id, config) {
    const displays = config.listDisplays();
    const viewsByDisplay = Object.fromEntries(
      displays.map((display) => [display, config.listViews(display)])
    );
    const defaultViewsByDisplay = Object.fromEntries(
      displays.map((display) => [
        display,
        config.getDefaultView(display) || viewsByDisplay[display]?.[0]?.name || ''
      ])
    );
    const roles = config.listRoles();
    const defaultViewsByRole = Object.fromEntries(
      roles.map(({ name, colorSpace }) => [
        name,
        Object.fromEntries(
          displays.map((display) => [
            display,
            config.getDefaultView(display, colorSpace) || defaultViewsByDisplay[display] || ''
          ])
        )
      ])
    );
    return freezeDeep({
      id,
      version: config.version,
      ocioVersion: this.ocio.version,
      ocioVersionHex: this.ocio.versionHex,
      builtinConfigs: this.ocio.listBuiltinConfigs(),
      colorSpaces: config.listColorSpaces(),
      roles,
      displays,
      viewsByDisplay,
      defaultDisplay: config.getDefaultDisplay() || displays[0] || '',
      defaultViewsByDisplay,
      defaultViewsByRole,
      looks: config.listLooks().map((name) => config.getLook(name)),
      namedTransforms: config.listNamedTransforms().map((name) => config.getNamedTransform(name)),
      fileRules: config.listFileRules(),
      fileTransformFormats: this.ocio.listFileTransformFormats()
    });
  }

  _getProcessor(normalizedRequest) {
    const config = this._requireConfig();
    const contextKey = stableStringify(normalizedRequest.context);
    const requestKey = stableStringify({
      transforms: normalizedRequest.transforms,
      optimization: normalizedRequest.optimization,
      direction: normalizedRequest.direction
    });
    const key = `${this.configGeneration}|${contextKey}|${requestKey}`;
    const cached = this.processorCache.get(key);
    if (cached) return cached;

    try {
      const processor = config.createGroupTransformProcessor(normalizedRequest.transforms, {
        optimization: normalizedRequest.optimization,
        direction: normalizedRequest.direction,
        context: normalizedRequest.context
      });
      const entry = { key, contextKey, processor };
      this.processorCache.set(key, entry);
      return entry;
    } catch (error) {
      this._recordFailure(error);
      throw error;
    }
  }

  _resolveShaderOptions(processorKey, options) {
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new TypeError('OCIO shader options must be an object');
    }
    const suffix = hashString(processorKey);
    return {
      ...options,
      functionName: options.functionName ?? `ocio_runtime_${suffix}`,
      resourcePrefix: options.resourcePrefix ?? `ocio_runtime_${suffix}`
    };
  }

  _requireConfig() {
    this._assertUsable();
    if (!this.config) throw new Error('OcioRuntime has no active config');
    return this.config;
  }

  _assertUsable() {
    if (this.disposed) throw new Error('OcioRuntime has been disposed');
  }

  _recordFailure(error) {
    this.latestFailure = error instanceof Error ? (error.stack ?? error.message) : String(error);
  }
}
