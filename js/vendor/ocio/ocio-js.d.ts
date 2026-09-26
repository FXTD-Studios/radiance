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
}
