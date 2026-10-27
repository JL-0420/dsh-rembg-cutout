/**
 * dsh-rembg-cutout —— 本地抠图（u2net / u2netp）+ 自动裁边。
 *
 * 两个面向模型的工具：
 *   - remove_background：把图片主体抠出来，输出透明 PNG（可顺便裁到主体外框）
 *   - image_trim：自动裁掉四周纯色留白（不抠图，只裁边）
 *
 * 运行时完全本地：onnxruntime + 模型都在本机，不联网、不计费（首次使用会装一次
 * onnxruntime、下一次模型）。依赖复用 DSH 自带的 Python（已带 numpy + Pillow），
 * 只为 onnxruntime 单独建一个 target 目录，不污染系统环境。
 */

import { spawn } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, delimiter, dirname, extname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PY_DIR = join(HERE, 'py')

const MODELS = ['u2netp', 'u2net']

/** 单次 python 调用的默认超时（毫秒）。 */
const DEFAULT_TIMEOUT_MS = 180000

export const name = 'rembg-cutout'

/**
 * `attachments` / `workspaceRegistry` 都按“可用则用”处理：
 * 缺了只是不能内联预览 / 相对路径退回进程目录，不影响抠图本身。
 */
export const inject = ['tools', 'workspaceRegistry', 'attachments']

const CONFIG_FIELDS = {
  pythonPath: {
    kind: 'string',
    description: '执行抠图的 Python 解释器；留空则自动找 DSH 自带 Python（带 numpy/Pillow），再退回 PATH 里的 python。',
  },
  dataDir: {
    kind: 'string',
    description: '存放 onnxruntime 与模型的目录；留空用 <DSH_HOME>/rembg-cutout。',
  },
  pipIndexUrl: {
    kind: 'string',
    default: 'https://pypi.tuna.tsinghua.edu.cn/simple',
    description: '首次安装 onnxruntime 时使用的 pip 源。国内网络建议保留默认镜像。',
  },
  defaultModel: {
    kind: 'string',
    default: 'u2netp',
    description: '默认模型。u2netp 约 4.7MB、快；u2net 约 176MB、边缘更细。',
  },
  timeoutMs: {
    kind: 'number',
    default: DEFAULT_TIMEOUT_MS,
    description: '单次抠图/裁边的最长执行时间。',
  },
}

function validateConfig(candidate) {
  const input = candidate === null || candidate === undefined ? {} : candidate
  if (typeof input !== 'object' || Array.isArray(input)) {
    return { issues: [{ message: 'config must be an object' }] }
  }
  const source = /** @type {Record<string, unknown>} */ (input)
  const value = {}
  const issues = []
  for (const [key, spec] of Object.entries(CONFIG_FIELDS)) {
    const raw = source[key]
    if (raw === undefined || raw === null) {
      if ('default' in spec) value[key] = spec.default
      continue
    }
    if (spec.kind === 'string') {
      if (typeof raw !== 'string') {
        issues.push({ message: `\`${key}\` must be a string`, path: [key] })
        continue
      }
      value[key] = raw
      continue
    }
    if (spec.kind === 'number') {
      if (typeof raw !== 'number' || !Number.isFinite(raw)) {
        issues.push({ message: `\`${key}\` must be a finite number`, path: [key] })
        continue
      }
      value[key] = raw
      continue
    }
    value[key] = raw
  }
  for (const key of Object.keys(source)) {
    if (!(key in CONFIG_FIELDS)) {
      issues.push({ message: `unknown config field \`${key}\``, path: [key] })
    }
  }
  if (issues.length > 0) return { issues }
  return { value }
}

export const Config = {
  '~standard': {
    version: 1,
    vendor: 'dsh-rembg-cutout',
    validate: validateConfig,
  },
  get jsonSchema() {
    return {
      type: 'object',
      properties: Object.fromEntries(
        Object.entries(CONFIG_FIELDS).map(([key, spec]) => [
          key,
          {
            type: spec.kind === 'number' ? 'number' : 'string',
            description: spec.description,
            ...('default' in spec ? { default: spec.default } : {}),
          },
        ]),
      ),
      additionalProperties: false,
    }
  },
}

/* ------------------------------------------------------------------ */
/* 进程与路径小工具                                                    */
/* ------------------------------------------------------------------ */

/**
 * 跑一个子进程并收集输出。
 * @param {string} command
 * @param {string[]} args
 * @param {{cwd?: string, env?: Record<string, string | undefined>, timeoutMs?: number, signal?: AbortSignal}} options
 * @returns {Promise<{code: number, stdout: string, stderr: string}>}
 */
function runProcess(command, args, options = {}) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    let settled = false
    const timeout = options.timeoutMs === undefined
      ? null
      : setTimeout(() => finish(124, `timeout after ${options.timeoutMs}ms`), options.timeoutMs)
    if (options.signal) {
      options.signal.addEventListener('abort', () => finish(130, 'aborted'), { once: true })
    }
    function finish(code, reason) {
      if (settled) return
      settled = true
      if (timeout) clearTimeout(timeout)
      try { child.kill() } catch { /* already gone */ }
      if (reason) stderr += `\n[${reason}]`
      resolvePromise({ code, stdout, stderr })
    }
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk) => { stdout += chunk })
    child.stderr.on('data', (chunk) => { stderr += chunk })
    child.on('error', (error) => {
      if (settled) return
      settled = true
      if (timeout) clearTimeout(timeout)
      rejectPromise(error)
    })
    child.on('close', (code) => finish(code === null ? 1 : code))
  })
}

/** DSH_HOME 或 ~/.dsh。 */
function dshHome() {
  const fromEnv = process.env.DSH_HOME
  if (typeof fromEnv === 'string' && fromEnv.trim().length > 0) return fromEnv.trim()
  return join(homedir(), '.dsh')
}

/**
 * 找 DSH 自带的 Python 解释器（dsh-runtimes/<版本>/dependencies/python/...）。
 * @returns {string | null}
 */
function findBundledPython() {
  const root = join(dshHome(), 'dsh-runtimes')
  if (!existsSync(root)) return null
  let entries
  try {
    entries = readdirSync(root, { withFileTypes: true })
  } catch {
    return null
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const candidates = [
      join(root, entry.name, 'dependencies', 'python', 'python.exe'),
      join(root, entry.name, 'dependencies', 'python', 'bin', 'python3'),
    ]
    for (const candidate of candidates) {
      if (existsSync(candidate)) return candidate
    }
  }
  return null
}

/** PATH 里第一个 python。 */
function findPathPython() {
  const paths = (process.env.PATH ?? '').split(delimiter).filter(Boolean)
  const names = process.platform === 'win32' ? ['python.exe', 'python3.exe'] : ['python3', 'python']
  for (const dir of paths) {
    for (const fileName of names) {
      const full = join(dir, fileName)
      if (existsSync(full)) return full
    }
  }
  return null
}

/** 路径里带空格时加引号，方便错误信息里直接复制执行。 */
function quote(value) {
  return /\s/.test(value) ? `"${value}"` : value
}

function expandHome(input) {
  if (input === '~' || input.startsWith('~/') || input.startsWith('~\\')) {
    return join(homedir(), input.slice(1))
  }
  return input
}

function toAbsolute(input, cwd) {
  const expanded = expandHome(String(input))
  return isAbsolute(expanded) ? resolve(expanded) : resolve(cwd, expanded)
}

/** 找调用会话的工作目录；拿不到就退回进程目录。 */
function workspaceRootFor(registry, sessionId) {
  if (registry === undefined || registry === null) return null
  if (typeof sessionId !== 'string' || sessionId.length === 0) return null
  try {
    const workspaces = registry.list()
    if (!Array.isArray(workspaces)) return null
    for (const workspace of workspaces) {
      const ids = workspace?.sessionIds
      if (Array.isArray(ids) && ids.includes(sessionId) && typeof workspace.path === 'string') {
        return workspace.path
      }
    }
  } catch {
    return null
  }
  return null
}

/** 默认输出路径：<输入目录>/<输入名>_<后缀>.png */
function defaultOutput(inputPath, suffix) {
  const ext = extname(inputPath)
  const stem = basename(inputPath, ext)
  return join(dirname(inputPath), `${stem}_${suffix}.png`)
}

/**
 * 读产物并交给 attachment 服务做内联预览；失败不影响主结果。
 * @param {any} ctx
 * @param {string} path
 * @param {string} name
 * @returns {Promise<Record<string, unknown> | null>}
 */
async function previewAttachment(ctx, path, name) {
  if (ctx.attachments === undefined || ctx.attachments === null) return null
  try {
    const bytes = readFileSync(path)
    const ref = await ctx.attachments.saveImage({
      data: new Uint8Array(bytes),
      mediaType: 'image/png',
      name,
    })
    return ref ?? null
  } catch (error) {
    ctx.logger?.warn?.(`rembg-cutout: attachment preview failed: ${String(error)}`)
    return null
  }
}

/** 从脚本输出里捞最后一行 JSON。 */
function parseJsonLine(stdout) {
  const lines = String(stdout).split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i]
    if (!line.startsWith('{')) continue
    try {
      return JSON.parse(line)
    } catch {
      /* 继续往前找 */
    }
  }
  return null
}

/* ------------------------------------------------------------------ */
/* 运行时自举：找到 python、备好 onnxruntime                            */
/* ------------------------------------------------------------------ */

class Runtime {
  constructor(config) {
    this.config = config
    this.dataDir = config.dataDir && config.dataDir.length > 0
      ? config.dataDir
      : join(dshHome(), 'rembg-cutout')
    this.pylibsDir = join(this.dataDir, 'pylibs')
    this.modelsDir = join(this.dataDir, 'models')
    /** @type {Promise<{python: string}> | null} */
    this.ready = null
  }

  async prepare(signal) {
    if (this.ready !== null) return this.ready
    this.ready = this.prepareInner(signal).catch((error) => {
      this.ready = null
      throw error
    })
    return this.ready
  }

  async prepareInner(signal) {
    const python = await this.findPython(signal)
    await this.ensureOnnxRuntime(python, signal)
    return { python }
  }

  async findPython(signal) {
    const configured = String(this.config.pythonPath ?? '').trim()
    if (configured.length > 0) {
      const result = await runProcess(configured, ['-c', 'import PIL, numpy'], { signal, timeoutMs: 30000 })
      if (result.code !== 0) {
        throw new Error(
          `配置的 pythonPath 不可用：${configured}\n` +
          `它需要能 import PIL 和 numpy。${result.stderr.trim().slice(0, 400)}`,
        )
      }
      return configured
    }

    const candidates = [findBundledPython(), findPathPython()].filter(Boolean)
    if (candidates.length === 0) {
      throw new Error(
        '找不到可用的 Python。请安装 Python 3.10+ 并确保 Pillow 可用，或在插件的 `rembg-cutout` 行配置 `pythonPath` 指定解释器。',
      )
    }
    for (const candidate of candidates) {
      const result = await runProcess(candidate, ['-c', 'import PIL, numpy'], { signal, timeoutMs: 30000 })
      if (result.code === 0) return candidate
    }
    throw new Error(
      '找到 Python，但它缺 Pillow 或 numpy：\n' +
      candidates.map((c) => `  - ${c}`).join('\n') +
      '\n请安装（pip install pillow numpy），或在插件行配置 `pythonPath` 指向 DSH 自带的 Python（它已自带这两个包）。',
    )
  }

  async ensureOnnxRuntime(python, signal) {
    if (isOnnxRuntimeInstalled(this.pylibsDir)) return
    const result = await runProcess(python, [
      '-m', 'pip', 'install',
      '--target', this.pylibsDir,
      '--no-deps',
      '--no-warn-script-location',
      '--quiet',
      '-i', this.config.pipIndexUrl,
      'onnxruntime',
    ], { signal, timeoutMs: 600000, env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' } })
    if (result.code !== 0 || !isOnnxRuntimeInstalled(this.pylibsDir)) {
      throw new Error(
        '安装 onnxruntime 失败（首次使用只需装一次）。\n' +
        '可以先手动执行下面这条命令，再重试：\n' +
        `  ${quote(python)} -m pip install --target ${quote(this.pylibsDir)} -i ${this.config.pipIndexUrl} onnxruntime\n` +
        `输出：${(result.stderr || result.stdout).trim().slice(0, 600)}`,
      )
    }
  }

  /** 脚本运行环境：把 pylibs 加到 PYTHONPATH 前面。 */
  envFor() {
    const prev = process.env.PYTHONPATH
    const pythonPath = prev && prev.length > 0
      ? `${this.pylibsDir}${delimiter}${prev}`
      : this.pylibsDir
    return { ...process.env, PYTHONPATH: pythonPath }
  }
}

function isOnnxRuntimeInstalled(pylibsDir) {
  try {
    return existsSync(join(pylibsDir, 'onnxruntime', '__init__.py'))
  } catch {
    return false
  }
}

/* ------------------------------------------------------------------ */
/* 工具注册                                                            */
/* ------------------------------------------------------------------ */

const TOOL_REMOVE = 'remove_background'
const TOOL_TRIM = 'image_trim'

const REMOVE_DESCRIPTION = [
  '把图片主体抠出来，输出透明背景 PNG；可顺便裁到主体外框。',
  '纯本地运行（onnxruntime + u2net/u2netp 模型）：不花钱、不上传、断网可用。',
  '适用：照片、插画、商品图、表情包素材去背景。',
].join(' ')

const TRIM_DESCRIPTION = [
  '自动裁掉图片四周的纯色留白（白边/黑边/任意纯色底），或按透明通道裁到内容边界。',
  '只裁边、不抠图：适合先把生图的大留白去掉再进下一步处理。',
].join(' ')

export function apply(ctx, config) {
  const runtime = new Runtime(config)
  const defaultModel = MODELS.includes(String(config.defaultModel)) ? String(config.defaultModel) : 'u2netp'
  const timeoutMs = Number(config.timeoutMs) > 0 ? Number(config.timeoutMs) : DEFAULT_TIMEOUT_MS

  /**
   * 跑 py/ 下的脚本并解析 stdout 的 JSON。
   * @param {string} script - 文件名，如 cutout.py
   * @param {string[]} args
   * @param {any} exec
   * @param {string} cwd
   */
  async function runScript(script, args, exec, cwd) {
    const { python } = await runtime.prepare(exec?.signal)
    const result = await runProcess(python, [join(PY_DIR, script), ...args], {
      cwd,
      env: runtime.envFor(),
      timeoutMs,
      signal: exec?.signal,
    })
    if (result.code !== 0) {
      const detail = (result.stderr || result.stdout).trim()
      throw new Error(
        `${script} 执行失败（退出码 ${result.code}）。\n` +
        (detail.length > 0 ? detail.slice(-1200) : '（没有输出）'),
      )
    }
    const payload = parseJsonLine(result.stdout)
    if (payload === null) {
      throw new Error(`${script} 没有输出可解析的 JSON。原始输出：${result.stdout.slice(-600)}`)
    }
    return payload
  }

  ctx.effect(() => ctx.tools.register({
    name: TOOL_REMOVE,
    description: REMOVE_DESCRIPTION,
    parameters: {
      type: 'object',
      properties: {
        input: {
          type: 'string',
          description: '输入图片路径。绝对路径，或相对当前会话工作目录的路径（如 attachments/xxx.png）。',
        },
        output: {
          type: 'string',
          description: '输出 PNG 路径。默认与输入同目录、文件名加 `_cutout` 后缀。',
        },
        model: {
          type: 'string',
          enum: MODELS,
          description: 'u2netp（默认，4.7MB、快）或 u2net（176MB、发丝等细节更好）。',
        },
        crop: {
          type: 'boolean',
          description: '是否裁到主体外框。默认 true，抠完只剩主体那一块。',
        },
        padding: {
          type: 'integer',
          minimum: 0,
          description: '裁剪时四周留白像素，默认 0。',
        },
        feather: {
          type: 'number',
          minimum: 0,
          description: '边缘羽化半径（像素），默认 0。主体边缘太硬时给 0.5~2。',
        },
        threshold: {
          type: 'integer',
          minimum: 1,
          maximum: 254,
          description: 'alpha 阈值 1~254，默认 128。抠多了调低，抠漏了调高。',
        },
        background: {
          type: 'string',
          description: '背景：`transparent`（默认，透明 PNG）或 `#rrggbb` 纯色底。',
        },
      },
      required: ['input'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          inputSize: { type: 'object' },
          outputSize: { type: 'object' },
          model: { type: 'string' },
          coverage: { type: 'number' },
          bbox: { type: 'object' },
          attachment: { type: 'object' },
        },
        required: ['path'],
        additionalProperties: true,
      },
      render(_args, value) {
        const lines = [
          `抠图完成：${value.path}`,
          value.outputSize ? `尺寸：${value.outputSize.width}x${value.outputSize.height}` : '',
          value.model ? `模型：${value.model}` : '',
        ].filter(Boolean)
        const blocks = [{ type: 'text', text: lines.join('\n') }]
        if (value.attachment) blocks.push({ type: 'image', attachment: value.attachment })
        return blocks
      },
    },
    async execute(args, exec) {
      const cwd = workspaceRootFor(ctx.workspaceRegistry, exec?.agent?.id) ?? process.cwd()
      const inputPath = toAbsolute(args.input, cwd)
      if (!existsSync(inputPath) || !statSync(inputPath).isFile()) {
        throw new Error(`找不到输入图片：${inputPath}`)
      }
      const outputPath = typeof args.output === 'string' && args.output.length > 0
        ? toAbsolute(args.output, cwd)
        : defaultOutput(inputPath, 'cutout')

      const model = typeof args.model === 'string' && MODELS.includes(args.model) ? args.model : defaultModel
      const cliArgs = [
        '--input', inputPath,
        '--output', outputPath,
        '--model', model,
        '--model-dir', runtime.modelsDir,
        '--threshold', String(args.threshold === undefined ? 128 : args.threshold),
        '--padding', String(args.padding === undefined ? 0 : args.padding),
        '--feather', String(args.feather === undefined ? 0 : args.feather),
        '--bg', typeof args.background === 'string' && args.background.length > 0 ? args.background : 'transparent',
        '--json',
      ]
      if (args.crop !== false) cliArgs.push('--crop')

      const result = await runScript('cutout.py', cliArgs, exec, cwd)
      const attachment = await previewAttachment(ctx, result.output, basename(result.output))
      return { ...result, path: result.output, attachment: attachment ?? undefined }
    },
  }))

  ctx.effect(() => ctx.tools.register({
    name: TOOL_TRIM,
    description: TRIM_DESCRIPTION,
    parameters: {
      type: 'object',
      properties: {
        input: { type: 'string', description: '输入图片路径（绝对或相对会话工作目录）。' },
        output: { type: 'string', description: '输出路径，默认同目录加 `_trim` 后缀。' },
        tolerance: {
          type: 'integer',
          minimum: 0,
          maximum: 255,
          description: '颜色容差，默认 8。背景有渐变/噪点就调大（如 16~32）。',
        },
        mode: {
          type: 'string',
          enum: ['auto', 'alpha', 'color'],
          description: 'auto（默认）：有透明通道按透明裁，否则按纯色底裁；alpha：只按透明通道；color：只按纯色底。',
        },
        padding: {
          type: 'integer',
          minimum: 0,
          description: '裁完后四周保留的像素，默认 0。',
        },
      },
      required: ['input'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          inputSize: { type: 'object' },
          outputSize: { type: 'object' },
          bbox: { type: 'object' },
          attachment: { type: 'object' },
        },
        required: ['path'],
        additionalProperties: true,
      },
      render(_args, value) {
        const lines = [
          `裁边完成：${value.path}`,
          value.outputSize ? `尺寸：${value.outputSize.width}x${value.outputSize.height}` : '',
        ].filter(Boolean)
        const blocks = [{ type: 'text', text: lines.join('\n') }]
        if (value.attachment) blocks.push({ type: 'image', attachment: value.attachment })
        return blocks
      },
    },
    async execute(args, exec) {
      const cwd = workspaceRootFor(ctx.workspaceRegistry, exec?.agent?.id) ?? process.cwd()
      const inputPath = toAbsolute(args.input, cwd)
      if (!existsSync(inputPath) || !statSync(inputPath).isFile()) {
        throw new Error(`找不到输入图片：${inputPath}`)
      }
      const outputPath = typeof args.output === 'string' && args.output.length > 0
        ? toAbsolute(args.output, cwd)
        : defaultOutput(inputPath, 'trim')

      const cliArgs = [
        '--input', inputPath,
        '--output', outputPath,
        '--tolerance', String(args.tolerance === undefined ? 8 : args.tolerance),
        '--mode', typeof args.mode === 'string' ? args.mode : 'auto',
        '--padding', String(args.padding === undefined ? 0 : args.padding),
      ]
      const result = await runScript('trim.py', cliArgs, exec, cwd)
      const attachment = await previewAttachment(ctx, result.output, basename(result.output))
      return { ...result, path: result.output, attachment: attachment ?? undefined }
    },
  }))
}
