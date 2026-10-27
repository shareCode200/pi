#!/usr/bin/env node
/**
 * Pi 调试启动器 —— 供 PyCharm / IDE 的 Node.js 运行配置使用。
 *
 * 设计要点（为什么不是 spawn 子进程）：
 * IDE 的调试器 attach 到的是「本进程」。如果这里 spawn 一个子进程去跑 Pi，
 * 调试器只会停在启动器上，真正的 agent 循环在子进程里，IDE 看不到。
 * 所以必须「进程内导入」——让 Pi 和启动器处于同一个进程、同一个 inspector 会话。
 *
 * 存在的第二个理由：
 * Pi 是 TypeScript monorepo，内部包通过 tsconfig.json 的 paths 别名互相引用
 * （如 `@earendil-works/pi-agent-core` -> packages/agent/src/index.ts）。
 * 直接跑 cli.ts 会走 node_modules 软链，一旦某个包没构建或软链失效，
 * 就会静默加载到 stale 的 dist 文件，断点打在 src 上永远不命中。
 * 这里先 import source-resolver.ts（它调用 registerHooks 按 tsconfig paths 解析），
 * 再 import cli.ts，保证所有内部引用都指向源码。
 *
 * 调试友好性：
 * Node 原生剥离 TS 类型（type stripping）不生成 source map，
 * 但行号与列号保持不变 —— 所以 IDE 断点可以直接命中 .ts 源码行。
 *
 * 用法：
 *   node scripts/debug-launch.mjs                          # 交互模式（TUI）
 *   node scripts/debug-launch.mjs --mode json -p "hi"      # 非交互（调试更友好）
 */
import { pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// 关闭 fd/rg 自动下载：本地已有则无副作用，离线机器上避免启动卡在网络请求上
process.env.PI_OFFLINE ??= "1";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");

// 顺序很关键：先加载 resolver 注册模块解析钩子，再加载 CLI
await import(pathToFileURL(resolve(repoRoot, "packages/coding-agent/src/experimental/source-resolver.ts")).href);
await import(pathToFileURL(resolve(repoRoot, "packages/coding-agent/src/cli.ts")).href);
