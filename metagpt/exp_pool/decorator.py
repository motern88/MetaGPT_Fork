"""Experience Decorator."""

import asyncio
import functools
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from metagpt.config2 import config
from metagpt.exp_pool.context_builders import BaseContextBuilder, SimpleContextBuilder
from metagpt.exp_pool.manager import ExperienceManager, get_exp_manager
from metagpt.exp_pool.perfect_judges import BasePerfectJudge, SimplePerfectJudge
from metagpt.exp_pool.schema import (
    LOG_NEW_EXPERIENCE_PREFIX,
    Experience,
    Metric,
    QueryType,
    Score,
)
from metagpt.exp_pool.scorers import BaseScorer, SimpleScorer
from metagpt.exp_pool.serializers import BaseSerializer, SimpleSerializer
from metagpt.logs import logger
from metagpt.utils.async_helper import NestAsyncio
from metagpt.utils.exceptions import handle_exception

# 返回类型的泛型类型变量
ReturnType = TypeVar("ReturnType")

# exp_cache 装饰器
def exp_cache(
    _func: Optional[Callable[..., ReturnType]] = None,
    query_type: QueryType = QueryType.SEMANTIC,
    manager: Optional[ExperienceManager] = None,
    scorer: Optional[BaseScorer] = None,
    perfect_judge: Optional[BasePerfectJudge] = None,
    context_builder: Optional[BaseContextBuilder] = None,
    serializer: Optional[BaseSerializer] = None,
    tag: Optional[str] = None,
):
    """装饰器，用于获取完美经验，否则执行函数并创建新经验。

    注意：
        1. 该装饰器可用于同步和异步函数。
        2. 函数必须有一个 `req` 参数，并且必须作为关键字参数传入。
        3. 如果 `config.exp_pool.enabled` 为 False，装饰器将直接执行函数。
        4. 如果 `config.exp_pool.enable_write` 为 False，装饰器将跳过评估和保存经验。
        5. 如果 `config.exp_pool.enable_read` 为 False，装饰器将跳过从经验池中读取数据。

    参数：
        _func：为了使装饰器更灵活，允许直接使用 `@exp_cache` 装饰器，而无需 `@exp_cache()`。
        query_type：查询时使用的查询类型，默认是语义查询（`QueryType.SEMANTIC`）。
        manager：获取、评估和保存经验的方式，默认为 `exp_manager`。
        scorer：评估经验的方式，默认使用 `SimpleScorer()`。
        perfect_judge：判断经验是否完美的方式，默认使用 `SimplePerfectJudge()`。
        context_builder：构建上下文的方式，默认使用 `SimpleContextBuilder()`。
        serializer：序列化请求和函数的返回值，反序列化存储的响应，默认使用 `SimpleSerializer()`。
        tag：经验的标签，默认使用 `ClassName.method_name` 或 `function_name`。

    返回：
        装饰器函数，用于包装目标函数。
    """

    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        async def get_or_create(args: Any, kwargs: Any) -> ReturnType:
            if not config.exp_pool.enabled:
                rsp = func(*args, **kwargs)
                return await rsp if asyncio.iscoroutine(rsp) else rsp

            # 创建经验缓存处理器
            handler = ExpCacheHandler(
                func=func,
                args=args,
                kwargs=kwargs,
                query_type=query_type,
                exp_manager=manager,
                exp_scorer=scorer,
                exp_perfect_judge=perfect_judge,
                context_builder=context_builder,
                serializer=serializer,
                tag=tag,
            )

            # 获取经验
            await handler.fetch_experiences()

            # 如果找到完美经验，直接返回
            if exp := await handler.get_one_perfect_exp():
                return exp

            # 执行函数
            await handler.execute_function()

            # 如果允许写入经验池，则处理经验并保存
            if config.exp_pool.enable_write:
                await handler.process_experience()

            return handler._raw_resp

        return ExpCacheHandler.choose_wrapper(func, get_or_create)

    return decorator(_func) if _func else decorator


# 经验缓存处理类
class ExpCacheHandler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 函数和相关参数
    func: Callable
    args: Any
    kwargs: Any
    query_type: QueryType = QueryType.SEMANTIC
    exp_manager: Optional[ExperienceManager] = None
    exp_scorer: Optional[BaseScorer] = None
    exp_perfect_judge: Optional[BasePerfectJudge] = None
    context_builder: Optional[BaseContextBuilder] = None
    serializer: Optional[BaseSerializer] = None
    tag: Optional[str] = None

    # 存储经验、请求和响应等数据
    _exps: list[Experience] = None
    _req: str = ""
    _resp: str = ""
    _raw_resp: Any = None
    _score: Score = None

    @model_validator(mode="after")
    def initialize(self):
        """初始化默认值，如果可选参数为 None，则设置默认值。

        由于装饰器可能传递 None，因此需要手动初始化可选参数的默认值。
        """

        self._validate_params()

        self.exp_manager = self.exp_manager or get_exp_manager()
        self.exp_scorer = self.exp_scorer or SimpleScorer()
        self.exp_perfect_judge = self.exp_perfect_judge or SimplePerfectJudge()
        self.context_builder = self.context_builder or SimpleContextBuilder()
        self.serializer = self.serializer or SimpleSerializer()
        self.tag = self.tag or self._generate_tag()

        self._req = self.serializer.serialize_req(**self.kwargs)

        return self

    # 获取经验
    async def fetch_experiences(self):
        """通过查询类型获取经验。"""

        self._exps = await self.exp_manager.query_exps(self._req, query_type=self.query_type, tag=self.tag)
        logger.info(f"Found {len(self._exps)} experiences for tag '{self.tag}'")

    # 获取完美经验并解析响应
    async def get_one_perfect_exp(self) -> Optional[Any]:
        """获取一个完美的经验并解析其响应。"""

        for exp in self._exps:
            if await self.exp_perfect_judge.is_perfect_exp(exp, self._req, *self.args, **self.kwargs):
                logger.info(f"Got one perfect experience for req '{exp.req[:20]}...'")
                return self.serializer.deserialize_resp(exp.resp)

        return None

    # 执行目标函数并获取响应
    async def execute_function(self):
        """执行目标函数并获取响应。"""

        self._raw_resp = await self._execute_function()
        self._resp = self.serializer.serialize_resp(self._raw_resp)

    @handle_exception
    async def process_experience(self):
        """处理经验。

        评估并保存经验。
        使用 `handle_exception` 确保程序的健壮性，不会中止后续操作。
        """

        await self.evaluate_experience()
        self.save_experience()

    # 评估经验并保存评分
    async def evaluate_experience(self):
        """评估经验并保存评分。"""

        self._score = await self.exp_scorer.evaluate(self._req, self._resp)

    # 保存经验
    def save_experience(self):
        """保存新的经验。"""

        exp = Experience(req=self._req, resp=self._resp, tag=self.tag, metric=Metric(score=self._score))
        self.exp_manager.create_exp(exp)
        self._log_exp(exp)

    @staticmethod
    def choose_wrapper(func, wrapped_func):
        """根据目标函数是否为异步函数选择包装器。

        如果目标函数是异步函数，则使用异步包装器；否则使用同步包装器。
        """

        async def async_wrapper(*args, **kwargs):
            return await wrapped_func(args, kwargs)

        def sync_wrapper(*args, **kwargs):
            NestAsyncio.apply_once()
            return asyncio.get_event_loop().run_until_complete(wrapped_func(args, kwargs))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    # 验证参数是否有效
    def _validate_params(self):
        if "req" not in self.kwargs:
            raise ValueError("`req` 必须作为关键字参数传入。")

    # 生成经验的标签
    def _generate_tag(self) -> str:
        """根据函数生成标签。

        如果第一个参数是类实例，则使用 "ClassName.method_name"；否则使用 "function_name"。
        """

        if self.args and hasattr(self.args[0], "__class__"):
            cls_name = type(self.args[0]).__name__
            return f"{cls_name}.{self.func.__name__}"

        return self.func.__name__

    # 构建上下文
    async def _build_context(self) -> str:
        self.context_builder.exps = self._exps

        return await self.context_builder.build(self.kwargs["req"])

    # 执行目标函数
    async def _execute_function(self):
        self.kwargs["req"] = await self._build_context()

        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*self.args, **self.kwargs)

        return self.func(*self.args, **self.kwargs)

    # 记录经验
    def _log_exp(self, exp: Experience):
        log_entry = exp.model_dump_json(include={"uuid", "req", "resp", "tag"})

        logger.debug(f"{LOG_NEW_EXPERIENCE_PREFIX}{log_entry}")
