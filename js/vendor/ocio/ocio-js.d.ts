export const ACES_CG_V2_CONFIG: 'ocio://cg-config-v2.2.0_aces-v1.3_ocio-v2.4';
export const ACES_STUDIO_V2_CONFIG: 'ocio://studio-config-v2.2.0_aces-v1.3_ocio-v2.4';
export const ACES_CG_V4_CONFIG: 'ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5';
export const ACES_STUDIO_V4_CONFIG: 'ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5';

export const TransformDirection: Readonly<{
  FORWARD: 0;
  INVERSE: 1;
}>;

export const OptimizationFlags: Readonly<{
  DEFAULT: number;
  NONE: number;
  LOSSLESS: number;
  VERY_GOOD: number;
  GOOD: number;
  DRAFT: -1;
}>;

export const Interpolation: Readonly<{
  UNKNOWN: 0;
  NEAREST: 1;
  LINEAR: 2;
  TETRAHEDRAL: 3;
  CUBIC: 4;
  DEFAULT: 254;
  BEST: 255;
}>;

export const CDLStyle: Readonly<{
  ASC: 0;
  NO_CLAMP: 1;
}>;

export type TransformDirectionValue = 0 | 1 | 'forward' | 'inverse';
export type OptimizationValue =
  | number
  | 'default'
  | 'none'
  | 'lossless'
  | 'very-good'
  | 'good'
  | 'draft';
export type InterpolationValue =
  | 0
  | 1
  | 2
  | 3
  | 4
  | 254
  | 255
  | 'unknown'
  | 'nearest'
  | 'linear'
  | 'trilinear'
  | 'tetra'
  | 'tetrahedral'
  | 'cubic'
  | 'default'
  | 'best';
export type CDLStyleValue = 0 | 1 | 'asc' | 'no-clamp' | 'no_clamp' | 'noclamp';
export type OcioContextVariables = Readonly<Record<string, string>>;

export interface ProcessorOptions {
  direction?: TransformDirectionValue;
  optimization?: OptimizationValue;
  context?: OcioContextVariables;
}

export interface CreateOCIOOptions {
  moduleFactory?: (options?: unknown) => Promise<unknown>;
  moduleOptions?: Record<string, unknown>;
  wasmUrl?: string;
  locateFile?: (path: string, prefix: string) => string;
}

export interface BuiltinConfigInfo {
  name: string;
  uiName: string;
  recommended: boolean;
}

export interface ColorSpaceInfo {
  name: string;
  canonicalName: string;
  family: string;
  encoding: string;
  description: string;
  isData: boolean;
  referenceSpace: number;
  aliases: string[];
  categories: string[];
}

export interface RoleInfo {
  name: string;
  colorSpace: string;
}

export interface ViewInfo {
  name: string;
  transform: string;
  colorSpace: string;
  looks: string;
  description: string;
}

export interface FileRuleInfo {
  index: number;
  name: string;
  colorSpace: string;
  pattern: string;
  extension: string;
  regex: string;
  custom: Record<string, string>;
}

export interface FileRuleMatch {
  colorSpace: string;
  ruleIndex: number;
  ruleName: string;
  isDefaultRule: boolean;
  custom: Record<string, string>;
}

export interface FileTransformFormatInfo {
  name: string;
  extension: string;
}

export interface LookInfo {
  name: string;
  processSpace: string;
  description: string;
  hasForwardTransform: boolean;
  hasInverseTransform: boolean;
}

export interface NamedTransformInfo {
  name: string;
  family: string;
  description: string;
  encoding: string;
  aliases: string[];
  categories: string[];
  hasForwardTransform: boolean;
  hasInverseTransform: boolean;
}

export interface DisplayViewTransformOptions extends ProcessorOptions {
  source: string;
  display: string;
  view: string;
  looksBypass?: boolean;
  dataBypass?: boolean;
}

export type DisplayViewProcessorOptions = DisplayViewTransformOptions;

export interface ColorSpaceTransformDescriptor {
  type: 'colorSpace';
  source: string;
  destination: string;
  direction?: TransformDirectionValue;
  dataBypass?: boolean;
}

export interface FileTransformDescriptor {
  type: 'file';
  src: string;
  direction?: TransformDirectionValue;
  interpolation?: InterpolationValue;
  cccId?: string;
  cdlStyle?: CDLStyleValue;
}

export interface LookTransformDescriptor {
  type: 'look';
  source: string;
  destination: string;
  looks: string;
  direction?: TransformDirectionValue;
  skipColorSpaceConversion?: boolean;
}

export interface DisplayViewTransformDescriptor {
  type: 'displayView';
  source: string;
  display: string;
  view: string;
  direction?: TransformDirectionValue;
  looksBypass?: boolean;
  dataBypass?: boolean;
}

export interface NamedTransformDescriptor {
  type: 'named';
  name: string;
  direction?: TransformDirectionValue;
}

export type TransformDescriptor =
  | ColorSpaceTransformDescriptor
  | FileTransformDescriptor
  | LookTransformDescriptor
  | DisplayViewTransformDescriptor
  | NamedTransformDescriptor;

export interface FileTransformProcessorOptions
  extends Omit<FileTransformDescriptor, 'type'>,
    Omit<ProcessorOptions, 'direction'> {}

export interface LookTransformProcessorOptions
  extends Omit<LookTransformDescriptor, 'type'>,
    Omit<ProcessorOptions, 'direction'> {}

export type GpuShaderLanguage =
  | 'glsl'
  | 'glsl_1.2'
  | 'glsl_1.3'
  | 'glsl_4.0'
  | 'glsl_vk_4.6'
  | 'glsl_es_1.0'
  | 'glsl_es_3.0'
  | 'webgl'
  | 'webgl1'
  | 'webgl2';

export interface GpuShaderOptions {
  language?: GpuShaderLanguage;
  functionName?: string;
  resourcePrefix?: string;
  textureMaxWidth?: number;
  allowTexture1D?: boolean;
}

export interface OcioGpuTexture {
  name: string;
  samplerName: string;
  width: number;
  height: number;
  depth: number;
  dimensions: 1 | 2 | 3;
  channels: 1 | 3;
  interpolation: string;
  values: Float32Array;
}

export interface OcioGpuUniform {
  name: string;
  type: 'double' | 'bool' | 'float3' | 'vector_float' | 'vector_int' | 'unknown';
  bufferOffset: number;
  value: number | boolean | number[];
}

export interface GpuShaderInfo {
  shaderText: string;
  functionName: string;
  language: string;
  cacheId: string;
  uniformBufferSize: number;
  textures: OcioGpuTexture[];
  uniforms: OcioGpuUniform[];
}


export interface WebGpuBinding {
  group: number;
  binding: number;
}

/** Options for WebGPU shader extraction. WGSL is generated through OCIO Vulkan GLSL + Naga. */
export interface WebGpuShaderOptions
  extends Omit<GpuShaderOptions, 'language' | 'allowTexture1D'> {}

/** OCIO LUT metadata plus the texture and sampler bindings used by the generated WGSL. */
export interface OcioWebGpuTexture extends OcioGpuTexture {
  texture: WebGpuBinding;
  sampler: WebGpuBinding;
}

export interface WebGpuShaderInfo {
  /** Embeddable WGSL containing `functionName` plus an internal translation entry point. */
  shaderText: string;
  /** Original OCIO Vulkan GLSL before normalization/translation. Useful for diagnostics. */
  sourceShaderText: string;
  /** Callable OCIO transform function in `shaderText`; call it from your own WGSL entry point. */
  functionName: string;
  language: 'wgsl';
  sourceLanguage: 'glsl_vk_4.6';
  /** OCIO GPU shader cache identifier, suitable for application-side pipeline/resource caches. */
  cacheId: string;
  uniformBufferSize: number;
  /** Bind-group location of the OCIO uniform buffer, or null when the shader has none. */
  uniformBinding: WebGpuBinding | null;
  textures: OcioWebGpuTexture[];
  uniforms: OcioGpuUniform[];
}

export interface OcioConfigPackageFile {
  relativePath: string;
  data: ArrayBufferView;
}

export interface OcioConfigPackage {
  configRelativePath: string;
  files: readonly OcioConfigPackageFile[];
}

export interface NormalizedOcioConfigPackage {
  configRelativePath: string;
  files: Array<{ relativePath: string; data: Uint8Array }>;
}

export interface OcioProcessorRequest {
  transforms: TransformDescriptor | readonly TransformDescriptor[];
  optimization?: OptimizationValue;
  direction?: TransformDirectionValue;
  context?: OcioContextVariables;
}

export interface OcioRuntimeConfigInfo {
  id: string;
  version: { major: number; minor: number };
  ocioVersion: string;
  ocioVersionHex: number;
  builtinConfigs: BuiltinConfigInfo[];
  colorSpaces: ColorSpaceInfo[];
  roles: RoleInfo[];
  displays: string[];
  viewsByDisplay: Record<string, ViewInfo[]>;
  defaultDisplay: string;
  defaultViewsByDisplay: Record<string, string>;
  /** Color-space-aware default view for each OCIO role and display. */
  defaultViewsByRole: Record<string, Record<string, string>>;
  looks: LookInfo[];
  namedTransforms: NamedTransformInfo[];
  fileRules: FileRuleInfo[];
  fileTransformFormats: FileTransformFormatInfo[];
}

export interface OcioRuntimeDiagnostics {
  activeConfigId: string | null;
  processorCacheEntries: number;
  gpuShaderCacheEntries: number;
  webGpuShaderCacheEntries: number;
  rgbTransformCacheEntries: number;
  latestFailure: string | null;
}

export interface CreateOcioRuntimeOptions extends CreateOCIOOptions {
  maxRgbCacheEntries?: number;
}

export function normalizeOcioConfigPackage(source: OcioConfigPackage): NormalizedOcioConfigPackage;

export class OcioRuntime {
  readonly ocio: OCIO;
  readonly activeConfigId: string | null;

  constructor(ocio: OCIO, options?: { maxRgbCacheEntries?: number });
  getConfigInfo(): OcioRuntimeConfigInfo | null;
  loadBuiltinConfig(name: string): OcioRuntimeConfigInfo;
  inspectBuiltinConfig(name: string): OcioRuntimeConfigInfo;
  loadConfigPackage(source: OcioConfigPackage, options?: { id?: string }): OcioRuntimeConfigInfo;
  inspectConfigPackage(source: OcioConfigPackage, options?: { id?: string }): OcioRuntimeConfigInfo;
  mountFile(path: string, data: ArrayBufferView): string;
  matchFileRule(filePath: string): FileRuleMatch | null;
  getViews(display: string): ViewInfo[];
  getDefaultView(display: string, colorSpace?: string): string;
  getGpuShaderInfo(request: OcioProcessorRequest, options?: GpuShaderOptions): GpuShaderInfo | null;
  getWebGpuShaderInfo(
    request: OcioProcessorRequest,
    options?: WebGpuShaderOptions,
  ): Promise<WebGpuShaderInfo | null>;
  transformRgb(
    source: string,
    destination: string,
    color: readonly [number, number, number] | Float32Array,
    options?: { optimization?: OptimizationValue; context?: OcioContextVariables },
  ): [number, number, number];
  invalidateContext(context: OcioContextVariables): void;
  clearCaches(): void;
  getDiagnostics(): OcioRuntimeDiagnostics;
  dispose(): void;
}

export function createOcioRuntime(options?: CreateOcioRuntimeOptions): Promise<OcioRuntime>;

export function createOCIO(options?: CreateOCIOOptions): Promise<OCIO>;

export class OCIO {
  readonly module: unknown;
  readonly version: string;
  readonly versionHex: number;
  readonly lastError: string;

  clearAllCaches(): void;
  listBuiltinConfigs(): BuiltinConfigInfo[];
  getBuiltinConfigYaml(name: string): string;
  listFileTransformFormats(): FileTransformFormatInfo[];
  isFileTransformFormatSupported(extension: string): boolean;
  createBuiltinConfig(name?: string): Config;
  createConfigFromFile(path: string): Config;
  createConfigFromString(text: string, options?: { workingDir?: string }): Config;
  mkdirp(path: string): void;
  writeFile(path: string, data: string | ArrayBufferView): void;
  readFile(path: string, options?: unknown): Uint8Array | string;
}

export class Config {
  readonly version: { major: number; minor: number };
  validate(): true;
  dispose(): void;
  listRoles(): RoleInfo[];
  listColorSpaces(): ColorSpaceInfo[];
  getColorSpace(name: string): ColorSpaceInfo;
  getFileRule(index: number): FileRuleInfo;
  listFileRules(): FileRuleInfo[];
  matchFileRule(filePath: string): FileRuleMatch | null;
  getViews(display: string): ViewInfo[];
  getDefaultView(display: string, colorSpace?: string): string;
  listDisplays(): string[];
  getDefaultDisplay(): string;
  listViews(display: string): ViewInfo[];
  getView(display: string, view: string): ViewInfo;
  getDefaultView(display: string, colorSpace?: string): string;
  listLooks(): string[];
  getLook(name: string): LookInfo;
  getLooksResultColorSpace(
    looks: string,
    options?: { context?: OcioContextVariables },
  ): string;
  listViewTransforms(): string[];
  listNamedTransforms(): string[];
  getNamedTransform(name: string): NamedTransformInfo;
  createColorSpaceProcessor(source: string, destination: string, options?: {
    optimization?: OptimizationValue;
    context?: OcioContextVariables;
  }): Processor;
  createDisplayViewProcessor(options: DisplayViewProcessorOptions): Processor;
  createNamedTransformProcessor(name: string, options?: ProcessorOptions): Processor;
  createFileTransformProcessor(options: FileTransformProcessorOptions): Processor;
  createLookTransformProcessor(options: LookTransformProcessorOptions): Processor;
  createGroupTransformProcessor(
    transforms: readonly TransformDescriptor[],
    options?: ProcessorOptions,
  ): Processor;
}

export class Processor {
  readonly cacheId: string;
  readonly isNoOp: boolean;
  readonly isIdentity: boolean;
  dispose(): void;
  applyRGBF32(rgb: Float32Array, options?: { copy?: boolean }): Float32Array;
  applyRGBAF32(rgba: Float32Array, options?: { copy?: boolean }): Float32Array;
  applyRGBA8(rgba: Uint8Array | Uint8ClampedArray, options?: { copy?: boolean }): Uint8Array | Uint8ClampedArray;
  getGpuShaderInfo(options?: GpuShaderOptions): GpuShaderInfo;
  getGpuShaderText(options?: GpuShaderOptions): string;
  /** Generate WGSL plus OCIO LUT/uniform binding metadata for WebGPU integration. */
  getWebGpuShaderInfo(options?: WebGpuShaderOptions): Promise<WebGpuShaderInfo>;
  /** Generate only the translated WGSL text. */
  getWebGpuShaderText(options?: WebGpuShaderOptions): Promise<string>;
}
