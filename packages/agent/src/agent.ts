import {
	type ClientIdentity,
	createInitialSystemMessage,
	getCurrentSystemMessage,
	getCurrentSystemPrompt,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type TextContent,
	type ThinkingBudgets,
	type Transport,
	toToolDeclaration,
} from "@earendil-works/pi-ai";
import { runAgentLoop, runAgentLoopContinue } from "./agent-loop.ts";
import { getDefaultStreamFn } from "./stream-fn.ts";
import type {
	AfterToolCallContext,
	AfterToolCallResult,
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentLoopTurnUpdate,
	AgentMessage,
	AgentState,
	AgentTool,
	BeforeToolCallContext,
	BeforeToolCallResult,
	FinishTurn,
	PrepareNextTurnContext,
	PrepareRequest,
	QueueMode,
	StreamFn,
	ToolExecutionMode,
} from "./types.ts";

export type { QueueMode } from "./types.ts";

// 把 Agent 内部消息转换成要发给 LLM 的消息时的默认实现。
// AgentMessage 是 Agent 这一层使用的完整消息类型（可能带时间戳、usage 等额外字段），
// 而 LLM 只认识 system / user / assistant / toolResult 这四类角色，
// 所以这里只保留这四种角色、把其余类型的消息过滤掉。
function defaultConvertToLlm(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(message) =>
			message.role === "system" ||
			message.role === "user" ||
			message.role === "assistant" ||
			message.role === "toolResult",
	);
}

// 一份全零的 token 用量统计占位对象。
// 当一次运行失败或中断、根本没有拿到真实用量数据时，用它填充消息的 usage 字段，
// 保证消息结构完整，避免上层读取 usage 时遇到 undefined。
const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

// 模型信息尚未确定时使用的占位模型。
// 构造 Agent 时如果调用方没提供 model，就先塞这个 unknown 占位值，
// 等真正发起请求前再被具体的模型信息覆盖。
const DEFAULT_MODEL = {
	id: "unknown",
	name: "unknown",
	api: "unknown",
	provider: "unknown",
	baseUrl: "",
	reasoning: false,
	input: [],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 0,
	maxTokens: 0,
} satisfies Model<any>;

// Agent 内部可变状态的类型。
// 与对外暴露的 AgentState 相比，这里的 isStreaming / pendingToolCalls 等字段是可变的
// （isStreaming 不是只读 getter、pendingToolCalls 用 Set 而不是数组）。
// 内部用这个类型做状态管理，对外再收敛成只读视角。
type MutableAgentState = Omit<AgentState, "isStreaming" | "streamingMessage" | "pendingToolCalls" | "errorMessage"> & {
	isStreaming: boolean;
	streamingMessage?: AgentMessage;
	pendingToolCalls: Set<string>;
	errorMessage?: string;
};

// 构造 Agent 时可传入的初始状态。
// 除了 pendingToolCalls / isStreaming / streamingMessage / errorMessage 这几个运行期字段之外，
// 其余状态字段（如 systemPrompt、tools、messages、model 等）都可以通过它做初始化。
/** Initial state for {@link Agent}. `systemPrompt` and `tools` become the leading system message unless `messages` already starts with one. */
export type AgentInitialState = Partial<
	Omit<AgentState, "pendingToolCalls" | "isStreaming" | "streamingMessage" | "errorMessage">
>;

// 根据初始状态构造 Agent 内部的可变状态对象。
// 要点：
// 1. tools / messages 都做了 slice() 拷贝，避免直接引用外部数组被意外修改。
// 2. 如果传入的 messages 开头不是 system 消息，就把 systemPrompt + 工具声明合成一条
//    初始 system 消息插到最前面，保证对话始终以 system 角色开头。
function createMutableAgentState(initialState?: AgentInitialState): MutableAgentState {
	let tools = initialState?.tools?.slice() ?? [];
	let messages = initialState?.messages?.slice() ?? [];
	const initialMessage = createInitialSystemMessage(initialState?.systemPrompt, tools.map(toToolDeclaration));
	if (messages[0]?.role !== "system" && initialMessage) messages.unshift(initialMessage);

	return {
		get systemPrompt() {
			return getCurrentSystemPrompt(messages);
		},
		model: initialState?.model ?? DEFAULT_MODEL,
		thinkingLevel: initialState?.thinkingLevel ?? "off",
		get tools() {
			return tools;
		},
		set tools(nextTools: AgentTool<any>[]) {
			tools = nextTools.slice();
		},
		get messages() {
			return messages;
		},
		set messages(nextMessages: AgentMessage[]) {
			messages = nextMessages.slice();
		},
		isStreaming: false,
		streamingMessage: undefined,
		pendingToolCalls: new Set<string>(),
		errorMessage: undefined,
	};
}

// 构造 Agent 时可选的各种钩子和配置项。
// 这些几乎都是「可插拔扩展点」：调用方通过传入不同的回调，定制 Agent 的行为，
// 而不需要改动 Agent 本身的循环逻辑。这是 Pi 可扩展设计的核心载体之一。
//
// 【设计思想·「配置即扩展」——用注入替代继承】
// Agent 的行为定制不是靠「继承 Agent 类重写方法」，而是靠构造时注入一堆回调
// （convertToLlm / transformContext / beforeToolCall / afterToolCall / finishTurn…）。
// 这符合「组合优于继承」的原则：每个能力都是可独立替换的单元，测试时可以随意
// mock 其中任意一个；也解释了为什么 Pi 内核能保持极简——复杂能力都以钩子的形式
// 被「塞进来」，而不是写死在循环里。
/** Options for constructing an {@link Agent}. */
export interface AgentOptions {
	initialState?: AgentInitialState;
	convertToLlm?: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	streamFn: StreamFn;
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	onPayload?: SimpleStreamOptions["onPayload"];
	onResponse?: SimpleStreamOptions["onResponse"];
	onProviderStreamEvent?: SimpleStreamOptions["onProviderStreamEvent"];
	beforeToolCall?: (context: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined>;
	afterToolCall?: (context: AfterToolCallContext, signal?: AbortSignal) => Promise<AfterToolCallResult | undefined>;
	finishTurn?: FinishTurn;
	prepareRequest?: PrepareRequest;
	prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	steeringMode?: QueueMode;
	followUpMode?: QueueMode;
	sessionId?: string;
	thinkingBudgets?: ThinkingBudgets;
	transport?: Transport;
	maxRetryDelayMs?: number;
	toolExecution?: ToolExecutionMode;
	/**
	 * How requests identify the app to providers, merged field by field over `DEFAULT_AGENT_IDENTITY`
	 * (`{ attribution: PI_ATTRIBUTION }`). Pass `{ attribution: null }` to send no attribution headers.
	 */
	identity?: ClientIdentity;
}

// 待处理消息队列。
// 用于暂存「steering（转向）消息」和「follow-up（后续）消息」——
// 这两类消息不是立刻发给模型，而是要等当前 assistant 回合结束后再注入。
// mode 控制每次取消息时的粒度：all = 一次取全部，one-at-a-time = 一次只取一条。
//
// 【设计思想·steering 与 follow-up 的语义区分——「插话」vs「续跑」】
// Pi 把「运行途中新增的消息」细分成两个队列，语义完全不同：
//   - steering（转向）：等当前回合结束就立刻注入，用来「改变方向」（比如用户打断
//     说"别做这个了，先看那个"）；
//   - follow-up（后续）：只有 agent 本应停止时才会执行，用来「追加任务」（比如"做完
//     再顺便跑一下测试"）。
// 这种区分让「人类与正在运行的 agent 协作」变得可控：既能打断、又能追加，
// 而不用粗暴地 abort 再重启。这是交互式 agent 体验设计上很关键的一层考量。
class PendingMessageQueue {
	private messages: AgentMessage[] = [];
	public mode: QueueMode;

	constructor(mode: QueueMode) {
		this.mode = mode;
	}

	// 入队：往队列末尾追加一条消息。
	enqueue(message: AgentMessage): void {
		this.messages.push(message);
	}

	// 判断队列里是否还有待处理消息。
	hasItems(): boolean {
		return this.messages.length > 0;
	}

	// 只预览、不消费：返回下一轮要处理的消息。
	// all 模式返回全部副本；one-at-a-time 模式只返回队首那一条。
	peek(): AgentMessage[] {
		if (this.mode === "all") return this.messages.slice();
		const first = this.messages[0];
		return first ? [first] : [];
	}

	// 取出并移除本轮要处理的消息（等价于 peek 之后把这些消息从队列删掉）。
	drain(): AgentMessage[] {
		const drained = this.peek();
		this.messages = this.messages.slice(drained.length);
		return drained;
	}

	// 清空队列。
	clear(): void {
		this.messages = [];
	}
}

// 一次「运行」的句柄。
// promise + resolve 配合，让外部可以通过 waitForIdle() 等待本次运行结束；
// abortController 用于随时中止本次运行。
type ActiveRun = {
	promise: Promise<void>;
	resolve: () => void;
	abortController: AbortController;
};

/**
 * Stateful wrapper around the low-level agent loop.
 *
 * `Agent` owns the current transcript, emits lifecycle events, executes tools,
 * and exposes queueing APIs for steering and follow-up messages.
 */
// Agent 类是底层 agent 循环（见 agent-loop.ts）之上的「有状态封装」。
// 它负责：持有当前对话记录、发出生命周期事件、执行工具，
// 以及提供 steer() / followUp() 之类的消息排队接口。
// 上层（如 coding-agent 包）直接跟这个类打交道，而不是跟裸的循环函数打交道。
//
// 【设计思想·「纯函数内核 + 有状态外壳」的分层】
// Pi 把 agent 拆成两层：agent-loop.ts 是纯函数式的循环（无状态、可复用、可测试），
// 而本文件的 Agent 类是有状态的外壳——持有 transcript、管理队列、暴露 abort/waitForIdle
// 等生命周期 API。这样上层拿到的是一个「对象」而非「函数」，天然适合长生命周期的
// 交互式应用；而底层的纯函数循环又可以被任何框架（甚至你自己写的）复用。
// 一句话：核心逻辑越纯、越容易被复用；状态封装越薄、越容易替换。
export class Agent {
	private _state: MutableAgentState;
	private readonly listeners = new Set<(event: AgentEvent, signal: AbortSignal) => Promise<void> | void>();
	private readonly steeringQueue: PendingMessageQueue;
	private readonly followUpQueue: PendingMessageQueue;

	public convertToLlm: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	public transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	public streamFunction: StreamFn;
	public getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	public onPayload?: SimpleStreamOptions["onPayload"];
	public onResponse?: SimpleStreamOptions["onResponse"];
	public onProviderStreamEvent?: SimpleStreamOptions["onProviderStreamEvent"];
	public beforeToolCall?: (
		context: BeforeToolCallContext,
		signal?: AbortSignal,
	) => Promise<BeforeToolCallResult | undefined>;
	public afterToolCall?: (
		context: AfterToolCallContext,
		signal?: AbortSignal,
	) => Promise<AfterToolCallResult | undefined>;
	public finishTurn?: FinishTurn;
	public prepareRequest?: PrepareRequest;
	public prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	public prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	private activeRun?: ActiveRun;
	/** Session identifier forwarded to providers for cache-aware backends. */
	public sessionId?: string;
	/** Optional per-level thinking token budgets forwarded to the stream function. */
	public thinkingBudgets?: ThinkingBudgets;
	/** Preferred transport forwarded to the stream function. */
	public transport: Transport;
	/** Optional cap for provider-requested retry delays. */
	public maxRetryDelayMs?: number;
	/** Tool execution strategy for assistant messages that contain multiple tool calls. */
	public toolExecution: ToolExecutionMode;
	/** Request identity forwarded to the stream function, over `DEFAULT_AGENT_IDENTITY`. */
	public identity?: ClientIdentity;

	// 构造函数：把 AgentOptions 里的各项配置和钩子摊平到实例字段上。
	// 值得注意的两点：
	// 1. runtimeOptions 兜底处理 —— 兼容某些旧编译产物可能不传 options 或 streamFn 的情况。
	// 2. 大量字段用了 ?? 默认值：convertToLlm 默认用 defaultConvertToLlm、
	//    streamFunction 默认用 getDefaultStreamFn()、steering/followUp 队列默认 one-at-a-time、
	//    transport 默认 auto、toolExecution 默认 parallel。
	constructor(options: AgentOptions) {
		// Older compiled consumers may omit options or streamFn even though the current API requires them.
		const runtimeOptions: Partial<AgentOptions> = options ?? {};
		this._state = createMutableAgentState(runtimeOptions.initialState);
		this.convertToLlm = runtimeOptions.convertToLlm ?? defaultConvertToLlm;
		this.transformContext = runtimeOptions.transformContext;
		this.streamFunction = runtimeOptions.streamFn ?? getDefaultStreamFn();
		this.getApiKey = runtimeOptions.getApiKey;
		this.onPayload = runtimeOptions.onPayload;
		this.onResponse = runtimeOptions.onResponse;
		this.onProviderStreamEvent = runtimeOptions.onProviderStreamEvent;
		this.beforeToolCall = runtimeOptions.beforeToolCall;
		this.afterToolCall = runtimeOptions.afterToolCall;
		this.finishTurn = runtimeOptions.finishTurn;
		this.prepareRequest = runtimeOptions.prepareRequest;
		this.prepareNextTurn = runtimeOptions.prepareNextTurn;
		this.prepareNextTurnWithContext = runtimeOptions.prepareNextTurnWithContext;
		this.steeringQueue = new PendingMessageQueue(runtimeOptions.steeringMode ?? "one-at-a-time");
		this.followUpQueue = new PendingMessageQueue(runtimeOptions.followUpMode ?? "one-at-a-time");
		this.sessionId = runtimeOptions.sessionId;
		this.thinkingBudgets = runtimeOptions.thinkingBudgets;
		this.transport = runtimeOptions.transport ?? "auto";
		this.maxRetryDelayMs = runtimeOptions.maxRetryDelayMs;
		this.toolExecution = runtimeOptions.toolExecution ?? "parallel";
		this.identity = runtimeOptions.identity;
	}

	/**
	 * Subscribe to agent lifecycle events.
	 *
	 * Listener promises are awaited in subscription order and are included in
	 * the current run's settlement. Listeners also receive the active abort
	 * signal for the current run.
	 *
	 * `agent_end` is the final emitted event for a run, but the agent does not
	 * become idle until all awaited listeners for that event have settled.
	 */
	// 订阅 Agent 的生命周期事件。
	// 返回一个取消订阅函数，调用它即可移除监听。
	// 监听器的 Promise 会按订阅顺序被 await，并纳入本次运行的整体结束判定；
	// 监听器还会收到当前运行的 abort signal，用于感知中断。
	subscribe(listener: (event: AgentEvent, signal: AbortSignal) => Promise<void> | void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	// 对外暴露当前 Agent 状态（只读视角）。
	// 注意：直接赋值 state.tools 或 state.messages 会触发 setter，内部做副本拷贝，
	// 而不是原地替换引用。
	get state(): AgentState {
		return this._state;
	}

	// steering 队列的消费模式（all / one-at-a-time）的读写访问器。
	/** Controls how queued steering messages are drained. */
	set steeringMode(mode: QueueMode) {
		this.steeringQueue.mode = mode;
	}

	get steeringMode(): QueueMode {
		return this.steeringQueue.mode;
	}

	// follow-up 队列的消费模式（all / one-at-a-time）的读写访问器。
	/** Controls how queued follow-up messages are drained. */
	set followUpMode(mode: QueueMode) {
		this.followUpQueue.mode = mode;
	}

	get followUpMode(): QueueMode {
		return this.followUpQueue.mode;
	}

	// 排队一条「转向」消息：等当前 assistant 回合结束后注入。
	/** Queue a message to be injected after the current assistant turn finishes. */
	steer(message: AgentMessage): void {
		this.steeringQueue.enqueue(message);
	}

	// 排队一条「后续」消息：只有当 Agent 本应停下来时才执行。
	/** Queue a message to run only after the agent would otherwise stop. */
	followUp(message: AgentMessage): void {
		this.followUpQueue.enqueue(message);
	}

	// 清空转向队列。
	/** Remove all queued steering messages. */
	clearSteeringQueue(): void {
		this.steeringQueue.clear();
	}

	// 清空后续队列。
	/** Remove all queued follow-up messages. */
	clearFollowUpQueue(): void {
		this.followUpQueue.clear();
	}

	// 同时清空转向队列和后续队列。
	/** Remove all queued steering and follow-up messages. */
	clearAllQueues(): void {
		this.clearSteeringQueue();
		this.clearFollowUpQueue();
	}

	// 判断是否还有排队中的消息（两类队列任一个非空即为 true）。
	/** Returns true when either queue still contains pending messages. */
	hasQueuedMessages(): boolean {
		return this.steeringQueue.hasItems() || this.followUpQueue.hasItems();
	}

	// 预览下一轮将处理的消息，但不消费。
	// 优先级：转向队列优先；转向队列为空时才看后续队列。
	/** Preview the messages selected for the next turn without consuming them. */
	peekQueuedMessages(): AgentMessage[] {
		const steering = this.steeringQueue.peek();
		return steering.length > 0 ? steering : this.followUpQueue.peek();
	}

	// 当前运行的中止信号（没有活跃运行时为 undefined）。
	/** Active abort signal for the current run, if any. */
	get signal(): AbortSignal | undefined {
		return this.activeRun?.abortController.signal;
	}

	// 中止当前运行（若存在）。
	/** Abort the current run, if one is active. */
	abort(): void {
		this.activeRun?.abortController.abort();
	}

	/**
	 * Resolve when the current run and all awaited event listeners have finished.
	 *
	 * This resolves after `agent_end` listeners settle.
	 */
	// 等待当前运行彻底结束（含所有已 await 的事件监听器）后 resolve。
	// 没有活跃运行时立即 resolve。
	waitForIdle(): Promise<void> {
		return this.activeRun?.promise ?? Promise.resolve();
	}

	// 重置对话：清空对话历史和两个队列，但保留系统基线（system 消息）。
	// 相当于「回到刚启动、只有 system prompt 和工具声明」的状态。
	/** Clear conversation state and queues while retaining the replayed prompt/tool baseline. */
	reset(): void {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before resetting.");
		}

		const baseline = getCurrentSystemMessage(this._state.messages);
		this._state.messages = baseline ? [baseline] : [];
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this._state.errorMessage = undefined;
		this.clearFollowUpQueue();
		this.clearSteeringQueue();
	}

	// 发起一个新的 prompt，支持三种输入形态：
	// 1. 单个消息对象；
	// 2. 消息数组（批量）；
	// 3. 字符串（可附带图片）。
	// 归一化后交给 runPromptMessages 执行一次完整的 agent 循环。
	/** Start a new prompt from text, a single message, or a batch of messages. */
	async prompt(message: AgentMessage | AgentMessage[]): Promise<void>;
	async prompt(input: string, images?: ImageContent[]): Promise<void>;
	async prompt(input: string | AgentMessage | AgentMessage[], images?: ImageContent[]): Promise<void> {
		// 已有运行在进行时拒绝新 prompt，要求先等待完成或改用 steer/followUp 排队。
		if (this.activeRun) {
			throw new Error(
				"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.",
			);
		}
		const messages = this.normalizePromptInput(input, images);
		await this.runPromptMessages(messages);
	}

	// 从当前对话记录继续往下跑（不新增消息）。
	// 用于「重试」场景：上下文里已经有 user 或 toolResult 消息，需要再请求一次模型。
	/** Continue from the current transcript. The last message must be a user or tool-result message. */
	async continue(): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before continuing.");
		}

		// 没有任何可继续的消息（空记录或全 system）时报错。
		const lastMessage = this._state.messages[this._state.messages.length - 1];
		if (!lastMessage || this._state.messages.every((message) => message.role === "system")) {
			throw new Error("No messages to continue from");
		}

		// 最后一条是 assistant 消息时，不能直接请求模型（模型不接受 assistant 结尾的续写），
		// 而是优先消费队列里排队的消息：
		// 先看转向队列，再看后续队列，都没有才报错。
		if (lastMessage.role === "assistant") {
			const queuedSteering = this.steeringQueue.drain();
			if (queuedSteering.length > 0) {
				await this.runPromptMessages(queuedSteering, { skipInitialSteeringPoll: true });
				return;
			}

			const queuedFollowUps = this.followUpQueue.drain();
			if (queuedFollowUps.length > 0) {
				await this.runPromptMessages(queuedFollowUps);
				return;
			}

			throw new Error("Cannot continue from message role: assistant");
		}

		await this.runContinuation();
	}

	// 把 prompt 的三种输入形态统一归一化成 AgentMessage 数组：
	// 数组直接返回；单个消息对象包成单元素数组；字符串则包成一条 user 消息（可带图片）。
	private normalizePromptInput(
		input: string | AgentMessage | AgentMessage[],
		images?: ImageContent[],
	): AgentMessage[] {
		if (Array.isArray(input)) {
			return input;
		}

		if (typeof input !== "string") {
			return [input];
		}

		const content: Array<TextContent | ImageContent> = [{ type: "text", text: input }];
		if (images && images.length > 0) {
			content.push(...images);
		}
		return [{ role: "user", content, timestamp: Date.now() }];
	}

	// 执行一次完整 agent 循环：把传入的消息作为本轮输入，
	// 包在 runWithLifecycle 里跑（统一处理活动运行标记、异常兜底、收尾）。
	private async runPromptMessages(
		messages: AgentMessage[],
		options: { skipInitialSteeringPoll?: boolean } = {},
	): Promise<void> {
		await this.runWithLifecycle(async (signal) => {
			await runAgentLoop(
				messages,
				this.createContextSnapshot(),
				this.createLoopConfig(options),
				(event) => this.processEvents(event),
				signal,
				this.streamFunction,
			);
		});
	}

	// 执行「续写」循环：不加新消息，直接基于当前上下文继续请求模型。
	private async runContinuation(): Promise<void> {
		await this.runWithLifecycle(async (signal) => {
			await runAgentLoopContinue(
				this.createContextSnapshot(),
				this.createLoopConfig(),
				(event) => this.processEvents(event),
				signal,
				this.streamFunction,
			);
		});
	}

// 生成当前上下文的快照（拷贝 messages 和 tools），
// 传给底层循环，避免循环执行过程中外部对状态的修改影响本次运行。
//
// 【设计思想·「上下文快照」隔离运行期与外部修改】
// 一次运行开始前，把 messages 和 tools 各做一次 slice() 拷贝再交给循环。
// 这样运行过程中，外部无论怎么改 Agent 的状态（比如用户又 steer 了新消息、或换了工具集），
// 都不会污染正在进行的这一轮。这让「并发的输入」与「正在执行的运行」互不干扰，
// 是保证循环可预测、可复现的重要一环——尤其对 durable 的崩溃恢复语义很关键。
private createContextSnapshot(): AgentContext {
		return {
			messages: this._state.messages.slice(),
			tools: this._state.tools.slice(),
		};
	}

	// 把 Agent 实例上分散的配置和钩子，汇总成一份给底层循环用的 AgentLoopConfig。
	// 其中 getSteeringMessages / getFollowUpMessages 分别对接两个内部队列；
	// skipInitialSteeringPoll 用于「继续」场景首次跳过转向轮询，避免重复消费。
	private createLoopConfig(options: { skipInitialSteeringPoll?: boolean } = {}): AgentLoopConfig {
		let skipInitialSteeringPoll = options.skipInitialSteeringPoll === true;
		return {
			model: this._state.model,
			reasoning: this._state.thinkingLevel === "off" ? undefined : this._state.thinkingLevel,
			sessionId: this.sessionId,
			onPayload: this.onPayload,
			onResponse: this.onResponse,
			onProviderStreamEvent: this.onProviderStreamEvent,
			transport: this.transport,
			thinkingBudgets: this.thinkingBudgets,
			maxRetryDelayMs: this.maxRetryDelayMs,
			identity: this.identity,
			toolExecution: this.toolExecution,
			beforeToolCall: this.beforeToolCall,
			afterToolCall: this.afterToolCall,
			finishTurn: this.finishTurn,
			prepareRequest: this.prepareRequest,
			prepareNextTurn:
				this.prepareNextTurnWithContext || this.prepareNextTurn
					? async (context) => {
							if (this.prepareNextTurnWithContext) {
								return await this.prepareNextTurnWithContext(context, this.signal);
							}
							return await this.prepareNextTurn?.(this.signal);
						}
					: undefined,
			convertToLlm: this.convertToLlm,
			transformContext: this.transformContext,
			getApiKey: this.getApiKey,
			getSteeringMessages: async () => {
				if (skipInitialSteeringPoll) {
					skipInitialSteeringPoll = false;
					return [];
				}
				return this.steeringQueue.drain();
			},
			getFollowUpMessages: async () => this.followUpQueue.drain(),
		};
	}

// 统一的生命周期包装器：任何一次运行（prompt / continue）都从这里进出。
// 职责：
// 1. 防止重入（已有活跃运行时直接报错）。
// 2. 创建本次运行的 AbortController 和 Promise，登记到 activeRun。
// 3. 把状态置为 isStreaming，执行 executor。
// 4. 异常时调用 handleRunFailure 兜底；无论如何最后调用 finishRun 收尾。
//
// 【设计思想·「AbortController 贯穿全链路」——把中断建模成一等公民】
// 每次运行都有一个专属的 AbortController，其 signal 会被一路传给底层循环、streamFunction、
// 工具执行，乃至事件监听器。用户想中断（Ctrl+C / abort()）时，只需触发这一个 signal，
// 所有正在 await 的环节都能感知到并优雅退出，而不是粗暴杀进程。这种「取消信号显式下传」
// 的模式，让 agent 在面对长任务时也能被干净地打断——是健壮性设计的核心。
private async runWithLifecycle(executor: (signal: AbortSignal) => Promise<void>): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing.");
		}

		const abortController = new AbortController();
		let resolvePromise = () => {};
		const promise = new Promise<void>((resolve) => {
			resolvePromise = resolve;
		});
		this.activeRun = { promise, resolve: resolvePromise, abortController };

		this._state.isStreaming = true;
		this._state.streamingMessage = undefined;
		this._state.errorMessage = undefined;

		try {
			await executor(abortController.signal);
		} catch (error) {
			await this.handleRunFailure(error, abortController.signal.aborted);
		} finally {
			this.finishRun();
		}
	}

	// 运行失败 / 被中止时的兜底处理：
	// 构造一条空的 assistant 失败消息（stopReason 为 aborted 或 error），
	// 然后照样按正常顺序补发 message_start → message_end → turn_end → agent_end 事件，
	// 让上层监听器能收到一个「这次运行结束了」的完整信号，而不是静默丢失。
	private async handleRunFailure(error: unknown, aborted: boolean): Promise<void> {
		const failureMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: this._state.model.api,
			provider: this._state.model.provider,
			model: this._state.model.id,
			usage: EMPTY_USAGE,
			stopReason: aborted ? "aborted" : "error",
			errorMessage: error instanceof Error ? error.message : String(error),
			timestamp: Date.now(),
		} satisfies AgentMessage;
		await this.processEvents({ type: "message_start", message: failureMessage });
		await this.processEvents({ type: "message_end", message: failureMessage });
		await this.processEvents({ type: "turn_end", message: failureMessage, toolResults: [] });
		await this.processEvents({ type: "agent_end", messages: [failureMessage] });
	}

	// 收尾：清掉流式标记和待执行工具集合，resolve 本次运行的 promise，注销 activeRun。
	private finishRun(): void {
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this.activeRun?.resolve();
		this.activeRun = undefined;
	}

	/**
	 * Reduce internal state for a loop event, then await listeners.
	 *
	 * `agent_end` only means no further loop events will be emitted. The run is
	 * considered idle later, after all awaited listeners for `agent_end` finish
	 * and `finishRun()` clears runtime-owned state.
	 */
// 处理一个循环事件，分两步：
// 第一步，根据事件类型更新 Agent 内部状态——
//   message_start/message_update 更新当前流式消息；
//   message_end 把消息真正追加进对话历史；
//   tool_execution_start/end 维护「正在执行中的工具调用」集合；
//   turn_end 若 assistant 消息带 errorMessage 则记录到 errorMessage；
//   agent_end 清空流式消息。
// 第二步，把事件按订阅顺序逐个 await 转发给所有监听器。
//
// 【设计思想·「归约 + 广播」的事件处理模式】
// 一个事件到了这里做两件事：先「归约」（reducer）——根据事件类型把 Agent 内部状态更新到
// 最新；再「广播」——按订阅顺序 await 转发给所有监听器。这其实是函数式编程里
// reducer 思想的落地：Agent 的状态 = 一系列事件折叠（fold）的结果，任何时刻的状态都
// 可从事件流重建。这为 durable 包的「持久化 + 崩溃恢复」提供了天然基础——因为状态
// 本质上是事件流的函数，重放事件就能恢复状态。
private async processEvents(event: AgentEvent): Promise<void> {
		switch (event.type) {
			case "message_start":
				this._state.streamingMessage = event.message;
				break;

			case "message_update":
				this._state.streamingMessage = event.message;
				break;

			case "message_end":
				this._state.streamingMessage = undefined;
				this._state.messages.push(event.message);
				break;

			case "tool_execution_start": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.add(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "tool_execution_end": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.delete(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "turn_end":
				if (event.message.role === "assistant" && event.message.errorMessage) {
					this._state.errorMessage = event.message.errorMessage;
				}
				break;

			case "agent_end":
				this._state.streamingMessage = undefined;
				break;
		}

		const signal = this.activeRun?.abortController.signal;
		if (!signal) {
			throw new Error("Agent listener invoked outside active run");
		}
		for (const listener of this.listeners) {
			await listener(event, signal);
		}
	}
}
