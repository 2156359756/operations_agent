"""
品牌官号运营Agent - FastAPI后端
知原药业品牌官号内容助理
支持多模型动态配置
"""

import os
import re
import json
import uuid
import base64
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from enum import Enum

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# 加载环境变量
load_dotenv()

# 获取配置
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# 配置路径
BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
TASK_LOGS_DIR = BASE_DIR / "task_logs"
USER_CONFIG_DIR = BASE_DIR / "user_config"
GENERATED_IMAGES_DIR = BASE_DIR / "generated_images"

# 确保目录存在
TASK_LOGS_DIR.mkdir(exist_ok=True)
USER_CONFIG_DIR.mkdir(exist_ok=True)
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)

# 模型分类
MODEL_TYPE_TEXT = "text"
MODEL_TYPE_IMAGE = "image"
MODEL_TYPE_VIDEO = "video"
VALID_MODEL_TYPES = (MODEL_TYPE_TEXT, MODEL_TYPE_IMAGE, MODEL_TYPE_VIDEO)

# 接口风格：文本模型用 openai / anthropic，图片模型用 openai / dashscope
API_STYLE_OPENAI = "openai"
API_STYLE_ANTHROPIC = "anthropic"
API_STYLE_DASHSCOPE = "dashscope"
VALID_API_STYLES = (API_STYLE_OPENAI, API_STYLE_ANTHROPIC, API_STYLE_DASHSCOPE)

# 单次生成的最大输出长度，避免长脚本被供应商默认值截断
MAX_OUTPUT_TOKENS = 4096

# 全局配置缓存
config_cache: Dict[str, Any] = {}

# 用户模型配置（支持动态添加）
USER_MODELS_FILE = USER_CONFIG_DIR / "models.json"
user_models: List[Dict[str, Any]] = []


DEFAULT_TEXT_MODELS = [
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model_type": MODEL_TYPE_TEXT,
        "api_style": API_STYLE_OPENAI,
        "enabled": True,
        "is_default": True
    },
    {
        "id": "deepseek-chat",
        "name": "DeepSeek Chat",
        "provider": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model_type": MODEL_TYPE_TEXT,
        "api_style": API_STYLE_OPENAI,
        "enabled": True,
        "is_default": False
    },
    {
        "id": "gpt-image-1",
        "name": "GPT Image 1",
        "provider": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model_type": MODEL_TYPE_IMAGE,
        "api_style": API_STYLE_OPENAI,
        "supports_edit": True,
        "enabled": True,
        "is_default": True
    },
    {
        "id": "wanx2.1-t2i-turbo",
        "name": "通义万相 2.1 Turbo",
        "provider": "阿里云",
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "api_key": "",
        "model_type": MODEL_TYPE_IMAGE,
        "api_style": API_STYLE_DASHSCOPE,
        "supports_edit": False,
        "enabled": True,
        "is_default": False
    }
]


def normalize_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """补齐模型配置字段，兼容早期没有分类信息的配置文件"""
    model.setdefault("api_key", "")
    model.setdefault("enabled", True)
    model.setdefault("is_default", False)

    # 早期配置没有 model_type，一律按文本模型处理
    if model.get("model_type") not in VALID_MODEL_TYPES:
        model["model_type"] = MODEL_TYPE_TEXT

    if model.get("api_style") not in VALID_API_STYLES:
        base_url = (model.get("base_url") or "").lower()
        if "dashscope" in base_url:
            model["api_style"] = API_STYLE_DASHSCOPE
        elif "anthropic" in base_url:
            model["api_style"] = API_STYLE_ANTHROPIC
        else:
            model["api_style"] = API_STYLE_OPENAI

    if model["model_type"] == MODEL_TYPE_IMAGE:
        # OpenAI 风格才有图片编辑端点，DashScope 文生图没有
        model.setdefault("supports_edit", model["api_style"] == API_STYLE_OPENAI)
        # quality 只在填了值时才发给接口。有的中转站必须带（如 gpt-image 系列的 low），
        # 有的中转站不认这个参数会直接报 400，所以默认留空＝不发送
        model.setdefault("quality", "")
    else:
        model.pop("supports_edit", None)
        model.pop("quality", None)

    return model


def load_user_models():
    """加载用户配置的模型"""
    global user_models
    if USER_MODELS_FILE.exists():
        with open(USER_MODELS_FILE, "r", encoding="utf-8") as f:
            user_models = json.load(f)
        user_models = [normalize_model(m) for m in user_models]
        ensure_single_default()
    else:
        user_models = [normalize_model(dict(m)) for m in DEFAULT_TEXT_MODELS]
    save_user_models()


def ensure_single_default():
    """每个分类各自保留一个默认模型"""
    for model_type in VALID_MODEL_TYPES:
        same_type = [m for m in user_models if m.get("model_type") == model_type]
        if not same_type:
            continue
        defaults = [m for m in same_type if m.get("is_default")]
        if len(defaults) > 1:
            for m in defaults[1:]:
                m["is_default"] = False
        elif not defaults:
            # 优先选一个已填密钥且启用的
            pick = next((m for m in same_type if m.get("enabled") and m.get("api_key")), None)
            pick = pick or next((m for m in same_type if m.get("enabled")), None)
            if pick:
                pick["is_default"] = True


def get_default_model(model_type: str) -> Optional[Dict[str, Any]]:
    """取某个分类下正在使用的模型"""
    return next(
        (m for m in user_models
         if m.get("model_type") == model_type and m.get("enabled") and m.get("is_default")),
        None
    )


def resolve_model(model_id: Optional[str], model_type: str) -> Optional[Dict[str, Any]]:
    """按 model_id 取指定分类的模型，未指定时回退到该分类的默认模型"""
    if model_id:
        model = next(
            (m for m in user_models
             if m["id"] == model_id and m.get("enabled") and m.get("model_type") == model_type),
            None
        )
        if not model:
            raise HTTPException(
                status_code=400,
                detail=f"指定的模型不可用，请确认它已启用且属于{'图片' if model_type == MODEL_TYPE_IMAGE else '文本'}模型分类"
            )
        return model
    return get_default_model(model_type)


def require_model(model_id: Optional[str], model_type: str) -> Dict[str, Any]:
    """取模型，没有可用的就报一个能指导用户下一步操作的错误"""
    model = resolve_model(model_id, model_type)
    if not model:
        label = {MODEL_TYPE_TEXT: "文本", MODEL_TYPE_IMAGE: "图片", MODEL_TYPE_VIDEO: "视频"}.get(
            model_type, model_type
        )
        raise HTTPException(
            status_code=400,
            detail=f"还没有可用的{label}模型，请在「管理API与模型」中添加{label}模型并填写密钥"
        )
    return model


def save_user_models():
    """保存用户配置的模型"""
    with open(USER_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(user_models, f, ensure_ascii=False, indent=2)


def load_all_configs():
    """加载所有配置文件到缓存"""
    global config_cache
    
    config_files = [
        "brand_facts.json",
        "content_rules.json", 
        "forbidden_words.json",
        "image_prompt_template.json",
        "history_samples.json"
    ]
    
    for filename in config_files:
        filepath = CONFIG_DIR / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                key = filename.replace(".json", "")
                config_cache[key] = json.load(f)


def save_task_log(task_id: str, log_data: Dict[str, Any]):
    """保存任务日志"""
    log_file = TASK_LOGS_DIR / f"{task_id}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def get_task_log(task_id: str) -> Optional[Dict[str, Any]]:
    """获取单个任务日志"""
    log_file = TASK_LOGS_DIR / f"{task_id}.json"
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# 五个步骤的固定顺序，用于前端跳转校验和状态展示
STEP_KEYS = [
    "generate_topics",
    "generate_outline",
    "generate_script",
    "generate_image_prompts",
    "export_package"
]


def new_task_log(task_id: str) -> Dict[str, Any]:
    """初始化一个聚合任务文档：一个任务一个文件，五个步骤都写在 steps 里"""
    return {
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "status": "created",
        "input": {},
        "steps": {},
        "images": []
    }


def load_or_create_task(task_id: Optional[str]) -> Dict[str, Any]:
    """按 task_id 取出聚合任务文档，没有就新建一个。
    这样五个步骤共用同一个 task_id，导出时才能把选题/大纲/脚本/图片串起来。"""
    if task_id:
        task = get_task_log(task_id)
        if task:
            # 兼容早期「一步一个文件」的旧日志
            if "steps" not in task:
                task = migrate_legacy_task(task)
            return task
        return new_task_log(task_id)
    return new_task_log(str(uuid.uuid4()))


def migrate_legacy_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """把旧格式（顶层 step/result）的日志收敛到 steps 结构里"""
    legacy_step = task.get("step")
    task.setdefault("images", [])
    task["steps"] = {}
    if legacy_step:
        task["steps"][legacy_step] = {
            "created_at": task.get("created_at"),
            "model_used": task.get("model_used"),
            "input": task.get("input", {}),
            "llm_input": task.get("llm_input"),
            "llm_output": task.get("llm_output"),
            "result": task.get("result", {}),
            "compliance_scan": task.get("compliance_scan")
        }
    return task


def record_step(
    task: Dict[str, Any],
    step: str,
    status: str,
    step_data: Dict[str, Any],
    task_input: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """写入某一步的结果。重新生成同一步会覆盖旧结果，并清掉它的下游步骤，
    避免导出时混进已经作废的内容。"""
    task.setdefault("steps", {})
    task.setdefault("images", [])

    step_data = dict(step_data)
    step_data["created_at"] = datetime.now().isoformat()
    task["steps"][step] = step_data

    if task_input:
        merged = dict(task.get("input") or {})
        merged.update({k: v for k, v in task_input.items() if v is not None})
        task["input"] = merged

    # 重新生成后，后面的步骤全部作废
    if step in STEP_KEYS:
        stale = STEP_KEYS[STEP_KEYS.index(step) + 1:]
        for key in stale:
            task["steps"].pop(key, None)
        # 图片依赖图片提示词，提示词重生成时一并清掉
        if step in ("generate_topics", "generate_outline", "generate_script", "generate_image_prompts"):
            task["images"] = []

    task["status"] = status
    task["updated_at"] = datetime.now().isoformat()
    save_task_log(task["task_id"], task)
    return task


def step_result(task: Dict[str, Any], step: str) -> Dict[str, Any]:
    """取某一步的 result，没有就返回空字典"""
    return (task.get("steps", {}).get(step) or {}).get("result", {}) or {}


def get_task_list() -> List[Dict[str, Any]]:
    """获取所有任务日志列表"""
    tasks = []
    for log_file in TASK_LOGS_DIR.glob("*.json"):
        # 排除导出日志
        if "_export" in log_file.stem:
            continue
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        steps = data.get("steps") or {}
        task_input = data.get("input") or {}
        tasks.append({
            "task_id": data.get("task_id"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at", data.get("created_at")),
            "platform": task_input.get("platform"),
            "topic": task_input.get("topic"),
            "status": data.get("status", "unknown"),
            "completed_steps": [k for k in STEP_KEYS if k in steps],
            "image_count": len(data.get("images") or [])
        })

    tasks.sort(key=lambda t: t.get("updated_at") or "", reverse=True)
    return tasks[:50]  # 限制返回数量


# 系统提示词
SYSTEM_PROMPT = """你是知原药业品牌官号内容助理，严格基于给到的brand_facts品牌事实库创作，不能编造产品信息。

业务范围：皮肤健康、功效护肤、皮肤科科普。

核心原则：
1. 禁止输出疗效承诺、治病效果、虚假对比
2. 禁止编造产品功效、医疗描述
3. 输出的内容必须遵守content_rules平台规则
4. 严格规避forbidden_words禁用词库中的所有词汇
5. 知识库缺少信息时，必须主动告警说明

输出规范：
- 每一份输出末尾必须强制加上：「AI初稿，需业务+医学法规人工审核后方可对外使用」
- 涉及成分、功效描述必须基于产品事实库
- 安全提示和注意事项必须完整

图片提示词输出规范：
- 使用英文撰写正向提示词，便于在AI绘图工具中使用
- 负面提示词要包含：禁止医疗效果对比、禁止药品外观特写、禁止治疗前后对比图
- 每套提示词必须附带合规提示
"""


# ============ Pydantic模型 ============

class GenerateTopicsRequest(BaseModel):
    # task_id 由前端在整条流程里复用，五个步骤写进同一个任务文档
    task_id: Optional[str] = Field(None, description="已有任务ID，缺省时新建任务")
    platform: str = Field(..., description="目标平台")
    column: str = Field(..., description="栏目")
    target_audience: str = Field(..., description="目标人群")
    topic: str = Field(..., description="创作主题")
    brand: Optional[str] = Field(None, description="指定品牌")


class GenerateOutlineRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="已有任务ID")
    selected_topic: Dict[str, Any] = Field(..., description="选中的选题")
    platform: str
    column: str
    target_audience: str
    outline_feedback: Optional[str] = Field(None, description="大纲修改反馈")


class GenerateScriptRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="已有任务ID")
    outline: str = Field(..., description="大纲")
    selected_topic: Dict[str, Any]
    platform: str
    column: str
    target_audience: str
    script_feedback: Optional[str] = Field(None, description="修改反馈")


class GenerateImagePromptsRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="已有任务ID")
    script: str = Field(..., description="脚本内容")
    selected_topic: Dict[str, Any]
    platform: str
    prompt_count: int = Field(default=4, ge=1, le=8, description="生成几套图片提示词")


class ExportPackageRequest(BaseModel):
    task_id: str = Field(..., description="任务ID")


# 模型配置相关模型
class ModelConfig(BaseModel):
    # model_ 是 pydantic 的保留前缀，显式放开才能使用 model_type 字段
    model_config = {"protected_namespaces": ()}

    id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="供应商")
    base_url: str = Field(..., description="API地址")
    api_key: str = Field(default="", description="API密钥")
    model_type: str = Field(default=MODEL_TYPE_TEXT, description="模型分类")
    api_style: str = Field(default=API_STYLE_OPENAI, description="接口风格")
    enabled: bool = Field(default=True, description="是否启用")
    is_default: bool = Field(default=False, description="是否为该分类的默认模型")


class AddModelRequest(BaseModel):
    # model_ 是 pydantic 的保留前缀，这里显式放开以便使用 model_id 字段
    model_config = {"protected_namespaces": ()}

    name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="供应商")
    base_url: str = Field(..., description="API地址")
    api_key: str = Field(..., description="API密钥")
    model_id: str = Field(..., description="模型ID")
    model_type: str = Field(default=MODEL_TYPE_TEXT, description="text / image / video")
    api_style: str = Field(default=API_STYLE_OPENAI, description="openai / dashscope")
    supports_edit: Optional[bool] = Field(None, description="图片模型是否支持基于原图编辑")
    quality: Optional[str] = Field(None, description="图片画质，留空则不发送该参数")


class UpdateModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    name: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_type: Optional[str] = None
    api_style: Optional[str] = None
    supports_edit: Optional[bool] = None
    quality: Optional[str] = None
    enabled: Optional[bool] = None
    is_default: Optional[bool] = None


class GenerateImageRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    task_id: str = Field(..., description="所属任务ID")
    prompt_index: int = Field(..., description="选中的图片提示词序号")
    positive_prompt: str = Field(..., description="正向提示词")
    negative_prompt: Optional[str] = Field(None, description="负面提示词")
    size: str = Field(default="1024x1024", description="图片尺寸")
    model_id: Optional[str] = Field(None, description="指定图片模型，缺省用默认图片模型")


class OptimizeImageRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    task_id: str = Field(..., description="所属任务ID")
    image_id: str = Field(..., description="要优化的图片ID")
    feedback: str = Field(..., description="用户的修改意见")
    mode: str = Field(default="rewrite", description="rewrite=改写提示词重绘，edit=基于原图编辑")
    model_id: Optional[str] = Field(None, description="指定图片模型")
    text_model_id: Optional[str] = Field(None, description="改写提示词所用的文本模型")


class UseModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str = Field(..., description="要使用的模型ID")


# ============ LLM调用 ============

def extract_api_error(response: httpx.Response) -> str:
    """从各家错误响应里取出可读的错误信息"""
    try:
        data = response.json()
    except Exception:
        return response.text[:500]

    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str) and err:
            return err
        # DashScope 的错误结构
        if data.get("message"):
            return str(data["message"])
        if data.get("code"):
            return str(data["code"])
    return response.text[:500]


def diagnose_api_failure(status_code: int, detail: str) -> str:
    """把供应商的报错翻译成「该找谁解决」。
    最常见的困惑是中转站上游没权限，但用户会以为是自己配置错了。"""
    text = (detail or "").lower()

    # 中转站自己的上游被拒：不是本地配置问题，改参数也没用
    if "upstream" in text or status_code == 502:
        if "forbidden" in text or "access" in text:
            return (
                "这是中转站「自己的上游」被拒绝了，不是你的配置问题。"
                "密钥能通过鉴权（模型列表可以正常读取），但中转站向它的上游请求该模型时被拒。"
                "请联系中转站管理员确认账号是否有该模型的权限或余额，改本地参数无法解决。"
            )
        return "中转站上游服务暂时不可用，属于服务方问题，稍后重试或联系中转站管理员。"

    if status_code == 401 or "invalid api key" in text or "unauthorized" in text:
        return "密钥无效或已过期，请重新复制粘贴API密钥。"
    if status_code == 404:
        return "接口地址不对。请确认API地址是否需要带 /v1 结尾，以及模型ID是否与供应商文档一致。"
    if status_code == 429 or "rate limit" in text or "quota" in text:
        return "触发限流或额度不足，请稍后重试或检查账户余额。"
    if "not supported" in text or "requires an image model" in text:
        return "该模型不支持这个接口。请确认模型分类（文本/图片）和接口风格是否选对。"
    if status_code == 400:
        return "请求参数被拒绝。图片模型可尝试调整「画质」设置（留空＝不发送该参数）。"
    return ""


def require_api_key(model: Dict[str, Any], label: str = "该模型") -> str:
    """取出可用的密钥，占位符一并视为未配置"""
    api_key = model.get("api_key") or API_KEY
    if not api_key or api_key == "your_api_key_here":
        raise HTTPException(
            status_code=400,
            detail=f"{label}尚未配置API密钥，请在「管理API与模型」中填写"
        )
    return api_key


async def call_llm(
    messages: List[Dict],
    model_config: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    expect_json: bool = False
) -> str:
    """调用文本大模型"""

    model = model_config or get_default_model(MODEL_TYPE_TEXT)
    if not model:
        raise HTTPException(
            status_code=400,
            detail="还没有可用的文本模型，请在「管理API与模型」中添加并填写密钥"
        )

    api_key = require_api_key(model, f"文本模型「{model.get('name', model['id'])}」")
    base_url = (model.get("base_url") or BASE_URL).rstrip("/")
    model_name = model.get("id", LLM_MODEL)
    api_style = model.get("api_style", API_STYLE_OPENAI)

    if api_style == API_STYLE_ANTHROPIC:
        # Anthropic 的 system 提示是顶层参数，不能作为 messages 里的一条
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat_messages = [m for m in messages if m.get("role") != "system"]
        url = f"{base_url}/messages"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": model_name,
            "messages": chat_messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": temperature
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
    else:
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": MAX_OUTPUT_TOKENS
        }
        if expect_json:
            # 让供应商直接约束成JSON，减少解析失败
            payload["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(url, headers=headers, json=payload)

            # 个别网关不认 response_format，去掉重试一次
            if response.status_code == 400 and payload.pop("response_format", None):
                response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=502 if response.status_code >= 500 else response.status_code,
                    detail=f"文本模型调用失败: {extract_api_error(response)}"
                )

            result = response.json()

            # 兼容 OpenAI 与 Anthropic 两种返回结构
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            if "content" in result and isinstance(result["content"], list):
                return "".join(
                    part.get("text", "") for part in result["content"]
                    if part.get("type") == "text"
                )
            return str(result)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="文本模型请求超时，请稍后重试")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接到文本模型服务，请检查API地址配置")


def parse_llm_json(llm_output: str, expect: str = "array") -> Any:
    """解析模型返回的JSON，容忍 markdown 代码块包裹"""
    text = llm_output.strip()

    # 去掉 ```json ... ``` 包裹
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 退一步，从正文里截取最外层的 JSON 片段（非贪婪不适用，这里取首尾配对）
    if expect == "array":
        start, end = text.find("["), text.rfind("]")
    else:
        start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise HTTPException(
        status_code=502,
        detail="模型返回的内容不是合法JSON，无法解析。可以重试一次，或换一个文本模型。"
    )


# ============ 图片模型调用 ============

def save_image_bytes(task_id: str, image_bytes: bytes, ext: str = "png") -> Dict[str, str]:
    """把图片落到本地 generated_images，返回可直接在前端展示的URL"""
    image_id = f"{task_id[:8]}_{uuid.uuid4().hex[:8]}"
    filename = f"{image_id}.{ext}"
    filepath = GENERATED_IMAGES_DIR / filename
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return {
        "image_id": image_id,
        "filename": filename,
        "url": f"/generated_images/{filename}"
    }


async def download_image(client: httpx.AsyncClient, url: str) -> bytes:
    """把供应商返回的临时图片URL下载到本地，URL通常只有几小时有效期"""
    response = await client.get(url, timeout=120.0)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="图片下载失败，供应商返回的链接不可用")
    return response.content


async def call_image_model(
    model: Dict[str, Any],
    positive_prompt: str,
    negative_prompt: Optional[str] = None,
    size: str = "1024x1024",
    base_image_path: Optional[Path] = None
) -> bytes:
    """调用图片模型出图，返回图片二进制。
    base_image_path 有值时走图片编辑端点（需要模型 supports_edit）。"""

    api_key = require_api_key(model, f"图片模型「{model.get('name', model['id'])}」")
    base_url = (model.get("base_url") or "").rstrip("/")
    api_style = model.get("api_style", API_STYLE_OPENAI)
    model_name = model["id"]

    if base_image_path and not model.get("supports_edit"):
        raise HTTPException(
            status_code=400,
            detail=f"模型「{model.get('name')}」不支持基于原图编辑，请改用「重新生成」方式优化"
        )

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            if api_style == API_STYLE_DASHSCOPE:
                # DashScope 是异步任务制：先提交拿 task_id，再轮询结果
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "enable"
                }
                payload = {
                    "model": model_name,
                    "input": {"prompt": positive_prompt},
                    "parameters": {"n": 1, "size": size.replace("x", "*")}
                }
                if negative_prompt:
                    payload["input"]["negative_prompt"] = negative_prompt

                submit = await client.post(
                    f"{base_url}/services/aigc/text2image/image-synthesis",
                    headers=headers,
                    json=payload
                )
                if submit.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"图片生成任务提交失败: {extract_api_error(submit)}"
                    )

                dashscope_task_id = (submit.json().get("output") or {}).get("task_id")
                if not dashscope_task_id:
                    raise HTTPException(status_code=502, detail="图片模型未返回任务ID，无法获取结果")

                poll_headers = {"Authorization": f"Bearer {api_key}"}
                # 最长等约 5 分钟
                for _ in range(100):
                    await asyncio.sleep(3)
                    poll = await client.get(
                        f"{base_url}/tasks/{dashscope_task_id}",
                        headers=poll_headers
                    )
                    if poll.status_code != 200:
                        continue
                    output = poll.json().get("output") or {}
                    status = output.get("task_status")
                    if status == "SUCCEEDED":
                        results = output.get("results") or []
                        image_url = next((r.get("url") for r in results if r.get("url")), None)
                        if not image_url:
                            raise HTTPException(status_code=502, detail="图片生成成功但未返回图片地址")
                        return await download_image(client, image_url)
                    if status in ("FAILED", "CANCELED", "UNKNOWN"):
                        reason = output.get("message") or output.get("code") or status
                        raise HTTPException(status_code=502, detail=f"图片生成失败: {reason}")

                raise HTTPException(status_code=504, detail="图片生成超时，请稍后重试")

            # OpenAI 风格
            if base_image_path:
                # 图片编辑端点要用 multipart
                headers = {"Authorization": f"Bearer {api_key}"}
                files = {
                    "image": (base_image_path.name, base_image_path.read_bytes(), "image/png")
                }
                data = {
                    "model": model_name,
                    "prompt": positive_prompt,
                    "n": "1",
                    "size": size
                }
                if model.get("quality"):
                    data["quality"] = model["quality"]
                response = await client.post(
                    f"{base_url}/images/edits",
                    headers=headers,
                    files=files,
                    data=data
                )
            else:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "prompt": positive_prompt,
                    "n": 1,
                    "size": size
                }
                if model.get("quality"):
                    payload["quality"] = model["quality"]
                response = await client.post(
                    f"{base_url}/images/generations",
                    headers=headers,
                    json=payload
                )

            if response.status_code != 200:
                detail = extract_api_error(response)
                hint = diagnose_api_failure(response.status_code, detail)
                raise HTTPException(
                    status_code=502 if response.status_code >= 500 else response.status_code,
                    detail=f"图片生成失败: {detail}" + (f"\n\n{hint}" if hint else "")
                )

            items = response.json().get("data") or []
            if not items:
                raise HTTPException(status_code=502, detail="图片模型没有返回任何图片")

            first = items[0]
            if first.get("b64_json"):
                return base64.b64decode(first["b64_json"])
            if first.get("url"):
                return await download_image(client, first["url"])
            raise HTTPException(status_code=502, detail="图片模型返回结构无法识别")

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="图片模型请求超时，请稍后重试")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接到图片模型服务，请检查API地址配置")


def scan_forbidden_words(text: str) -> List[Dict[str, Any]]:
    """扫描文本中的禁用词和医学风险词"""
    forbidden_config = config_cache.get("forbidden_words", {})
    forbidden_words = forbidden_config.get("forbidden_words", [])
    medical_risk_words = forbidden_config.get("medical_risk_words", [])
    
    violations = []
    
    for word in forbidden_words:
        if word in text:
            violations.append({
                "word": word,
                "category": "forbidden",
                "severity": "critical",
                "description": "禁用表达",
                "action": "必须删除或修改"
            })
    
    for word in medical_risk_words:
        if word in text:
            if not any(v["word"] == word for v in violations):
                violations.append({
                    "word": word,
                    "category": "medical_risk",
                    "severity": "medium",
                    "description": "医学风险词",
                    "action": "需人工复核确认表述准确性"
                })
    
    return violations


# ============ FastAPI应用 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载配置
    load_all_configs()
    load_user_models()
    yield

app = FastAPI(
    title="品牌官号运营Agent",
    description="知原药业品牌官号内容创作助手 - 支持多模型配置",
    version="1.1.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
# 生成的图片直接以静态资源暴露，前端和导出的Markdown都引用这个路径
app.mount(
    "/generated_images",
    StaticFiles(directory=str(GENERATED_IMAGES_DIR)),
    name="generated_images"
)


@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


# ============ 模型配置API ============

@app.get("/api/models")
async def get_models():
    """获取所有已配置的模型"""
    # 隐藏API密钥
    safe_models = []
    for m in user_models:
        safe_model = m.copy()
        if safe_model.get("api_key"):
            safe_model["api_key"] = "***" + safe_model["api_key"][-4:] if len(safe_model["api_key"]) > 4 else "***"
        safe_models.append(safe_model)
    return {"models": safe_models}


@app.post("/api/models")
async def add_model(request: AddModelRequest):
    """添加新模型"""
    # 检查是否已存在
    if any(m["id"] == request.model_id for m in user_models):
        raise HTTPException(status_code=400, detail="该模型ID已存在")

    if request.model_type not in VALID_MODEL_TYPES:
        raise HTTPException(status_code=400, detail="模型分类只能是 text / image / video")
    if request.api_style not in VALID_API_STYLES:
        raise HTTPException(status_code=400, detail="接口风格只能是 openai / anthropic / dashscope")

    new_model = {
        "id": request.model_id,
        "name": request.name,
        "provider": request.provider,
        "base_url": request.base_url.rstrip("/"),
        "api_key": request.api_key,
        "model_type": request.model_type,
        "api_style": request.api_style,
        "enabled": True,
        "is_default": False
    }
    if request.supports_edit is not None:
        new_model["supports_edit"] = request.supports_edit
    if request.quality is not None:
        new_model["quality"] = request.quality.strip()

    normalize_model(new_model)
    user_models.append(new_model)
    # 该分类下还没有默认模型时，新加的直接顶上
    ensure_single_default()
    save_user_models()

    return {"success": True, "message": "模型添加成功", "model": new_model}


@app.put("/api/models/{model_id}")
async def update_model(model_id: str, request: UpdateModelRequest):
    """更新模型配置"""
    model = next((m for m in user_models if m["id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    # 更新字段
    if request.name is not None:
        model["name"] = request.name
    if request.provider is not None:
        model["provider"] = request.provider
    if request.base_url is not None:
        model["base_url"] = request.base_url.rstrip("/")
    # 前端回显的密钥是掩码，含 * 的一律视为“没有改动”
    if request.api_key is not None and "*" not in request.api_key:
        model["api_key"] = request.api_key
    if request.model_type is not None:
        if request.model_type not in VALID_MODEL_TYPES:
            raise HTTPException(status_code=400, detail="模型分类只能是 text / image / video")
        model["model_type"] = request.model_type
    if request.api_style is not None:
        if request.api_style not in VALID_API_STYLES:
            raise HTTPException(status_code=400, detail="接口风格只能是 openai / anthropic / dashscope")
        model["api_style"] = request.api_style
    if request.supports_edit is not None:
        model["supports_edit"] = request.supports_edit
    if request.quality is not None:
        model["quality"] = request.quality.strip()
    if request.enabled is not None:
        model["enabled"] = request.enabled
    if request.is_default is not None:
        if request.is_default:
            if not model.get("enabled"):
                raise HTTPException(status_code=400, detail="该模型已停用，无法设为使用中")
            # 只在同一分类内取消默认，别把其他分类的默认模型清掉
            for m in user_models:
                if m.get("model_type") == model.get("model_type"):
                    m["is_default"] = False
        model["is_default"] = request.is_default

    normalize_model(model)
    ensure_single_default()
    save_user_models()

    return {"success": True, "message": "模型更新成功"}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """删除模型"""
    global user_models
    original_count = len(user_models)
    user_models = [m for m in user_models if m["id"] != model_id]
    
    if len(user_models) == original_count:
        raise HTTPException(status_code=404, detail="模型不存在")

    # 删掉的可能正是「使用中」的那个，补一个默认出来，否则该分类会没有可用模型
    ensure_single_default()
    save_user_models()
    return {"success": True, "message": "模型删除成功"}


@app.post("/api/models/use")
async def use_model(request: UseModelRequest):
    """选择使用的模型（设为默认）"""
    model = next((m for m in user_models if m["id"] == request.model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    if not model.get("enabled"):
        raise HTTPException(status_code=400, detail="该模型未启用")
    
    # 只在同一分类内切换：文本模型和图片模型各有一个“使用中”
    model_type = model.get("model_type", MODEL_TYPE_TEXT)
    for m in user_models:
        if m.get("model_type") == model_type:
            m["is_default"] = (m["id"] == request.model_id)

    save_user_models()

    type_label = {"text": "文本", "image": "图片", "video": "视频"}.get(model_type, model_type)
    return {
        "success": True,
        "message": f"已将{type_label}模型切换为：{model['name']}",
        "model_type": model_type
    }


@app.post("/api/models/test")
async def test_model(model_id: str):
    """测试模型连接：文本模型发一句话，图片模型只校验端点可达"""
    model = next((m for m in user_models if m["id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    api_key = model.get("api_key") or API_KEY
    if not api_key or api_key == "your_api_key_here":
        raise HTTPException(status_code=400, detail="API密钥未配置，请先为该模型填写API密钥")

    model_type = model.get("model_type", MODEL_TYPE_TEXT)
    api_style = model.get("api_style", API_STYLE_OPENAI)
    base_url = (model.get("base_url") or "").rstrip("/")

    try:
        if model_type == MODEL_TYPE_TEXT:
            if api_style == API_STYLE_ANTHROPIC:
                url = f"{base_url}/messages"
                headers = {
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
            else:
                url = f"{base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            payload = {
                "model": model["id"],
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 16
            }
        else:
            # 图片/视频模型：真跑一次出图太慢也费钱，这里用一个极小的请求探连通性
            if api_style == API_STYLE_DASHSCOPE:
                url = f"{base_url}/services/aigc/text2image/image-synthesis"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "enable"
                }
                payload = {
                    "model": model["id"],
                    "input": {"prompt": "a blue circle on white background"},
                    "parameters": {"n": 1, "size": "1024*1024"}
                }
            else:
                url = f"{base_url}/images/generations"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model["id"],
                    "prompt": "a blue circle on white background",
                    "n": 1,
                    "size": "1024x1024"
                }
                if model.get("quality"):
                    payload["quality"] = model["quality"]

        timeout = 60.0 if model_type == MODEL_TYPE_TEXT else 180.0
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                return {"success": True, "message": "连接测试成功"}

            detail = extract_api_error(response)
            return {
                "success": False,
                "message": f"连接失败: {detail}",
                "diagnosis": diagnose_api_failure(response.status_code, detail)
            }

    except httpx.TimeoutException:
        return {"success": False, "message": "连接超时"}
    except httpx.ConnectError:
        return {"success": False, "message": "无法连接到该API地址，请检查填写是否正确"}
    except Exception as e:
        return {"success": False, "message": f"连接错误: {str(e)}"}


# ============ 配置API ============

@app.get("/api/config")
async def get_config():
    """获取所有配置概要"""
    return {
        "brands": [b["name"] for b in config_cache.get("brand_facts", {}).get("brands", [])],
        "platforms": [p["name"] for p in config_cache.get("content_rules", {}).get("platforms", [])],
        "columns": [c["name"] for c in config_cache.get("content_rules", {}).get("columns", [])],
        "audiences": [a["name"] for a in config_cache.get("content_rules", {}).get("target_audiences", [])]
    }


@app.get("/api/config/{config_name}")
async def get_config_detail(config_name: str):
    """获取指定配置的详细内容"""
    if config_name in config_cache:
        return config_cache[config_name]
    raise HTTPException(status_code=404, detail="配置不存在")


@app.get("/api/config/{config_name}/edit")
async def edit_config(config_name: str, content: str):
    """更新配置文件"""
    filepath = CONFIG_DIR / f"{config_name}.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="配置文件不存在")
    
    try:
        # 验证JSON格式
        json_data = json.loads(content)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # 重新加载缓存
        config_cache[config_name] = json_data
        
        return {"success": True, "message": "配置已更新"}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON格式错误")


# ============ 任务API ============

@app.get("/api/tasks")
async def list_tasks():
    """获取任务列表"""
    return get_task_list()


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """获取单个任务的完整记录"""
    task = get_task_log(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.post("/api/topics")
async def generate_topics(request: GenerateTopicsRequest, model_id: Optional[str] = None):
    """接口1：生成选题建议"""
    task = load_or_create_task(request.task_id)
    task_id = task["task_id"]

    # 获取要使用的文本模型（未指定则用文本分类的默认模型）
    model_config = resolve_model(model_id, MODEL_TYPE_TEXT)

    # 构建提示词
    brand_info = ""
    if request.brand:
        for b in config_cache.get("brand_facts", {}).get("brands", []):
            if b["name"] == request.brand:
                brand_info = json.dumps(b, ensure_ascii=False)
                break
    
    prompt = f"""请基于以下信息生成5个选题建议：

目标平台：{request.platform}
栏目：{request.column}
目标人群：{request.target_audience}
创作主题：{request.topic}

品牌信息（优先使用）：
{brand_info if brand_info else "不指定品牌，基于整体产品线创作"}

品牌事实库：
{json.dumps(config_cache.get("brand_facts", {}), ensure_ascii=False, indent=2)}

平台规则：
{json.dumps(config_cache.get("content_rules", {}), ensure_ascii=False, indent=2)}

历史优质样例：
{json.dumps(config_cache.get("history_samples", {}).get("samples", [])[:2], ensure_ascii=False, indent=2)}

要求：
1. 每条选题包含：选题标题、适配平台、目标受众、核心立意、潜在合规风险提示
2. 选题要有创意，符合平台调性
3. 标注每条选题可能存在的合规风险点
4. 输出JSON数组格式
"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        llm_output = await call_llm(messages, model_config, expect_json=True)
        topics = parse_llm_json(llm_output, expect="array")

        # 个别模型会把数组包在对象里返回
        if isinstance(topics, dict):
            topics = next(
                (v for v in topics.values() if isinstance(v, list)),
                [topics]
            )

        record_step(
            task,
            "generate_topics",
            "topics_generated",
            {
                "model_used": (model_config or {}).get("id", "default"),
                "input": request.model_dump(),
                "llm_input": prompt,
                "llm_output": llm_output,
                "result": {"topics": topics}
            },
            task_input={
                "platform": request.platform,
                "column": request.column,
                "target_audience": request.target_audience,
                "topic": request.topic,
                "brand": request.brand
            }
        )

        return {
            "task_id": task_id,
            "topics": topics
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/outline")
async def generate_outline(request: GenerateOutlineRequest, model_id: Optional[str] = None):
    """接口2：生成内容大纲"""
    task = load_or_create_task(request.task_id)
    task_id = task["task_id"]

    model_config = resolve_model(model_id, MODEL_TYPE_TEXT)

    feedback_section = ""
    if request.outline_feedback:
        feedback_section = f"\n\n用户对大纲的修改反馈（请据此调整）：\n{request.outline_feedback}"

    prompt = f"""请根据以下选题生成详细的内容大纲：

选题信息：
{json.dumps(request.selected_topic, ensure_ascii=False, indent=2)}

目标平台：{request.platform}
栏目：{request.column}
目标人群：{request.target_audience}
{feedback_section}

品牌事实库：
{json.dumps(config_cache.get("brand_facts", {}), ensure_ascii=False, indent=2)}

平台规则：
{json.dumps(config_cache.get("content_rules", {}), ensure_ascii=False, indent=2)}

历史优质样例大纲：
{json.dumps([s.get("outline", []) for s in config_cache.get("history_samples", {}).get("samples", [])[:2]], ensure_ascii=False, indent=2)}

要求：
1. 大纲结构清晰，包含开场引入、核心内容、结尾总结
2. 每个章节有简要说明
3. 符合平台内容规范
4. 标注各部分的内容重点
5. 输出格式：markdown大纲
"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        llm_output = await call_llm(messages, model_config)
        
        record_step(
            task,
            "generate_outline",
            "outline_generated",
            {
                "model_used": (model_config or {}).get("id", "default"),
                "input": request.model_dump(),
                "llm_input": prompt,
                "llm_output": llm_output,
                "result": {
                    "outline": llm_output,
                    "selected_topic": request.selected_topic
                }
            },
            task_input={
                "platform": request.platform,
                "column": request.column,
                "target_audience": request.target_audience
            }
        )

        return {
            "task_id": task_id,
            "outline": llm_output
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/script")
async def generate_script(request: GenerateScriptRequest, model_id: Optional[str] = None):
    """接口3：生成完整脚本初稿并自动合规扫描"""
    task = load_or_create_task(request.task_id)
    task_id = task["task_id"]

    model_config = resolve_model(model_id, MODEL_TYPE_TEXT)

    feedback_section = ""
    if request.script_feedback:
        feedback_section = f"\n\n用户修改反馈（请根据反馈调整）：\n{request.script_feedback}"
    
    prompt = f"""请根据以下大纲生成完整的脚本初稿：

选题信息：
{json.dumps(request.selected_topic, ensure_ascii=False, indent=2)}

确认的大纲：
{request.outline}

目标平台：{request.platform}
栏目：{request.column}
目标人群：{request.target_audience}
{feedback_section}

品牌事实库：
{json.dumps(config_cache.get("brand_facts", {}), ensure_ascii=False, indent=2)}

平台规则：
{json.dumps(config_cache.get("content_rules", {}), ensure_ascii=False, indent=2)}

历史优质样例脚本：
{json.dumps([s.get("script_sample", "") for s in config_cache.get("history_samples", {}).get("samples", [])[:2]], ensure_ascii=False, indent=2)}

禁用词库（必须严格规避）：
{json.dumps(config_cache.get("forbidden_words", {}).get("forbidden_words", [])[:50], ensure_ascii=False, indent=2)}

要求：
1. 严格基于大纲展开，内容完整
2. 语言风格符合目标平台调性
3. 禁止使用任何禁用词
4. 禁止编造产品功效、医疗描述
5. 禁止输出疗效对比、治疗承诺类内容
6. 每份输出末尾必须强制加上：「AI初稿，需业务+医学法规人工审核后方可对外使用」
7. 涉及产品功效时，仅描述注册的功效范围，不做疗效承诺
"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        llm_output = await call_llm(messages, model_config)
        
        # 扫描禁用词
        violations = scan_forbidden_words(llm_output)
        
        record_step(
            task,
            "generate_script",
            "script_generated",
            {
                "model_used": (model_config or {}).get("id", "default"),
                "input": request.model_dump(),
                "llm_input": prompt,
                "llm_output": llm_output,
                "compliance_scan": {
                    "violations": violations,
                    "scan_time": datetime.now().isoformat()
                },
                "result": {
                    "script": llm_output,
                    "violations": violations,
                    "selected_topic": request.selected_topic
                }
            },
            task_input={
                "platform": request.platform,
                "column": request.column,
                "target_audience": request.target_audience
            }
        )

        return {
            "task_id": task_id,
            "script": llm_output,
            "violations": violations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/image-prompts")
async def generate_image_prompts(request: GenerateImagePromptsRequest, model_id: Optional[str] = None):
    """接口4：根据脚本生成多套图片提示词"""
    task = load_or_create_task(request.task_id)
    task_id = task["task_id"]

    model_config = resolve_model(model_id, MODEL_TYPE_TEXT)

    prompt = f"""请根据以下脚本内容，生成{request.prompt_count}套图片生成提示词：

脚本内容：
{request.script}

选题信息：
{json.dumps(request.selected_topic, ensure_ascii=False, indent=2)}

目标平台：{request.platform}

图片提示词模板库：
{json.dumps(config_cache.get("image_prompt_template", {}), ensure_ascii=False, indent=2)}

要求：
1. 生成{request.prompt_count}套风格/构图各不相同的提示词，覆盖：小红书封面、小红书插图、短视频画面、社交海报等用途
2. 正向提示词用英文撰写，要包含画面主体、构图、光线、色调、质感等要素，便于直接投喂给AI绘图模型
3. 每套提示词包含：场景描述、正向提示词（英文）、负面提示词（英文+中文风险提示）
4. 提示词要与脚本内容高度相关
5. 严禁生成任何医疗效果对比、治疗前后对比相关提示词
6. 画面中不要出现任何文字、logo、药品包装特写
7. 输出JSON数组格式，每项包含：image_type, name, positive_prompt_en, negative_prompt, scene_description, recommended_size（从 1024x1024 / 1024x1792 / 1792x1024 中选一个最适合该用途的）
"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        llm_output = await call_llm(messages, model_config, expect_json=True)
        prompts = parse_llm_json(llm_output, expect="array")

        if isinstance(prompts, dict):
            prompts = next(
                (v for v in prompts.values() if isinstance(v, list)),
                [prompts]
            )

        compliance_reminder = config_cache.get("image_prompt_template", {}).get("compliance_reminder",
            "【重要提示】AI生成图片仅作为设计参考，对外发布前必须经过医学、法务合规审核；严禁制作皮肤前后疗效对比图，禁止暗示治疗效果。")

        record_step(
            task,
            "generate_image_prompts",
            "image_prompts_generated",
            {
                "model_used": (model_config or {}).get("id", "default"),
                "input": request.model_dump(),
                "llm_input": prompt,
                "llm_output": llm_output,
                "result": {
                    "image_prompts": prompts,
                    "compliance_reminder": compliance_reminder
                }
            },
            task_input={"platform": request.platform}
        )

        return {
            "task_id": task_id,
            "image_prompts": prompts,
            "compliance_reminder": compliance_reminder
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 图片生成 / 优化 ============

def append_image(task: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """把一张生成好的图片挂到任务上。
    不走 record_step，否则会把图片提示词那一步的下游数据清掉。"""
    task.setdefault("images", [])
    task["images"].append(record)
    task["updated_at"] = datetime.now().isoformat()
    save_task_log(task["task_id"], task)
    return record


def get_prompt_meta(task: Dict[str, Any], prompt_index: int) -> Dict[str, Any]:
    """取出第 prompt_index 套图片提示词，用于给图片打标签"""
    prompts = step_result(task, "generate_image_prompts").get("image_prompts") or []
    if 0 <= prompt_index < len(prompts):
        item = prompts[prompt_index]
        if isinstance(item, dict):
            return item
    return {}


async def rewrite_image_prompt(
    original_prompt: str,
    feedback: str,
    text_model: Dict[str, Any]
) -> str:
    """让文本模型根据用户意见改写英文正向提示词"""
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite prompts for AI image generation. "
                "Return ONLY the rewritten English prompt, no explanation, no quotes, no markdown."
            )
        },
        {
            "role": "user",
            "content": (
                f"Original image prompt:\n{original_prompt}\n\n"
                f"The user wants these changes (may be written in Chinese):\n{feedback}\n\n"
                "Rewrite the prompt so it keeps everything the user did not ask to change, "
                "and applies the requested changes. Keep it a single descriptive English prompt. "
                "Never include text, logos, medicine packaging close-ups, or any before/after "
                "treatment comparison in the scene."
            )
        }
    ]
    rewritten = await call_llm(messages, text_model, temperature=0.5)
    cleaned = rewritten.strip().strip("`").strip()
    # 模型偶尔会回一整段解释，取最长的一行兜底
    if "\n" in cleaned:
        lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
        cleaned = max(lines, key=len) if lines else cleaned
    return cleaned or original_prompt


@app.post("/api/images/generate")
async def generate_image(request: GenerateImageRequest):
    """接口4.5：用图片模型把选中的那套提示词画出来"""
    task = get_task_log(request.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在，请先完成前面的步骤")
    if "steps" not in task:
        task = migrate_legacy_task(task)

    model = require_model(request.model_id, MODEL_TYPE_IMAGE)

    image_bytes = await call_image_model(
        model,
        positive_prompt=request.positive_prompt,
        negative_prompt=request.negative_prompt,
        size=request.size
    )

    saved = save_image_bytes(request.task_id, image_bytes)
    meta = get_prompt_meta(task, request.prompt_index)

    record = {
        **saved,
        "prompt_index": request.prompt_index,
        "prompt_name": meta.get("name") or meta.get("image_type") or f"图片方案 {request.prompt_index + 1}",
        "positive_prompt": request.positive_prompt,
        "negative_prompt": request.negative_prompt,
        "size": request.size,
        "model_used": model["id"],
        "model_name": model.get("name", model["id"]),
        "source": "generate",
        "parent_image_id": None,
        "feedback": None,
        "created_at": datetime.now().isoformat()
    }
    append_image(task, record)

    return {"task_id": request.task_id, "image": record}


@app.post("/api/images/optimize")
async def optimize_image(request: OptimizeImageRequest):
    """接口4.6：根据用户意见优化已生成的图片。
    rewrite=先用文本模型改写提示词再重绘；edit=把原图喂给图片模型做局部修改。"""
    task = get_task_log(request.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if "steps" not in task:
        task = migrate_legacy_task(task)

    source_image = next(
        (img for img in (task.get("images") or []) if img.get("image_id") == request.image_id),
        None
    )
    if not source_image:
        raise HTTPException(status_code=404, detail="找不到要优化的图片")

    if request.mode not in ("rewrite", "edit"):
        raise HTTPException(status_code=400, detail="优化方式只能是 rewrite 或 edit")

    image_model = require_model(request.model_id, MODEL_TYPE_IMAGE)
    original_prompt = source_image.get("positive_prompt") or ""

    if request.mode == "edit":
        base_path = GENERATED_IMAGES_DIR / source_image["filename"]
        if not base_path.exists():
            raise HTTPException(status_code=404, detail="原图文件已丢失，请改用「改写提示词重绘」")
        final_prompt = f"{original_prompt}\n\nApply these changes: {request.feedback}".strip()
        image_bytes = await call_image_model(
            image_model,
            positive_prompt=final_prompt,
            negative_prompt=source_image.get("negative_prompt"),
            size=source_image.get("size", "1024x1024"),
            base_image_path=base_path
        )
    else:
        text_model = require_model(request.text_model_id, MODEL_TYPE_TEXT)
        final_prompt = await rewrite_image_prompt(original_prompt, request.feedback, text_model)
        image_bytes = await call_image_model(
            image_model,
            positive_prompt=final_prompt,
            negative_prompt=source_image.get("negative_prompt"),
            size=source_image.get("size", "1024x1024")
        )

    saved = save_image_bytes(request.task_id, image_bytes)
    record = {
        **saved,
        "prompt_index": source_image.get("prompt_index"),
        "prompt_name": source_image.get("prompt_name"),
        "positive_prompt": final_prompt,
        "negative_prompt": source_image.get("negative_prompt"),
        "size": source_image.get("size", "1024x1024"),
        "model_used": image_model["id"],
        "model_name": image_model.get("name", image_model["id"]),
        "source": f"optimize_{request.mode}",
        "parent_image_id": request.image_id,
        "feedback": request.feedback,
        "created_at": datetime.now().isoformat()
    }
    append_image(task, record)

    return {"task_id": request.task_id, "image": record, "rewritten_prompt": final_prompt}


@app.delete("/api/images/{task_id}/{image_id}")
async def delete_image(task_id: str, image_id: str):
    """删掉一张不要的图（同时清理本地文件）"""
    task = get_task_log(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    images = task.get("images") or []
    target = next((img for img in images if img.get("image_id") == image_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="图片不存在")

    task["images"] = [img for img in images if img.get("image_id") != image_id]
    task["updated_at"] = datetime.now().isoformat()
    save_task_log(task_id, task)

    filepath = GENERATED_IMAGES_DIR / target.get("filename", "")
    if filepath.exists() and filepath.parent == GENERATED_IMAGES_DIR:
        try:
            filepath.unlink()
        except OSError:
            pass

    return {"success": True, "message": "图片已删除"}


@app.post("/api/export")
async def export_package(request: ExportPackageRequest):
    """接口5：导出Markdown发布物料包"""
    task_id = request.task_id

    task = get_task_log(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="未找到该任务记录")
    if "steps" not in task:
        task = migrate_legacy_task(task)

    topic_step = step_result(task, "generate_topics")
    outline_step = step_result(task, "generate_outline")
    script_step = step_result(task, "generate_script")
    image_step = step_result(task, "generate_image_prompts")

    if not script_step.get("script"):
        raise HTTPException(
            status_code=400,
            detail="该任务还没有生成脚本，请先完成第3步再导出"
        )

    task_input = task.get("input") or {}
    images = task.get("images") or []

    # 按提示词序号把图片归到对应的方案下
    images_by_prompt: Dict[Any, List[Dict[str, Any]]] = {}
    for img in images:
        images_by_prompt.setdefault(img.get("prompt_index"), []).append(img)

    # 构建Markdown
    md_content = []
    md_content.append("# 品牌官号内容发布物料包")
    md_content.append("")
    md_content.append(f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append("")

    # 选题信息
    md_content.append("## 选题信息")
    md_content.append("")
    md_content.append(f"- **平台**：{task_input.get('platform', 'N/A')}")
    md_content.append(f"- **栏目**：{task_input.get('column', 'N/A')}")
    md_content.append(f"- **目标人群**：{task_input.get('target_audience', 'N/A')}")
    md_content.append(f"- **创作主题**：{task_input.get('topic', 'N/A')}")
    md_content.append("")

    selected = (
        script_step.get("selected_topic")
        or outline_step.get("selected_topic")
        or {}
    )
    if selected:
        md_content.append("### 选中的选题")
        md_content.append("")
        md_content.append(f"**标题**：{selected.get('title', selected.get('选题标题', 'N/A'))}")
        md_content.append("")
        md_content.append(f"**核心立意**：{selected.get('core_idea', selected.get('核心立意', 'N/A'))}")
        md_content.append("")

    # 大纲
    if outline_step.get("outline"):
        md_content.append("## 内容大纲")
        md_content.append("")
        md_content.append(outline_step["outline"])
        md_content.append("")

    # 脚本
    md_content.append("## 脚本初稿")
    md_content.append("")
    md_content.append("> 合规提醒：AI生成内容，请务必经过业务+医学法规人工审核后方可使用")
    md_content.append("")
    md_content.append(script_step.get("script", ""))
    md_content.append("")

    violations = script_step.get("violations") or []
    if violations:
        md_content.append("### 合规风险提示")
        md_content.append("")
        for v in violations:
            severity = "[严重]" if v.get("severity") == "critical" else "[中等]"
            md_content.append(f"- {severity} **{v.get('word')}**：{v.get('description')} → {v.get('action')}")
        md_content.append("")

    # 图片提示词 + 已生成的图片
    image_prompts = image_step.get("image_prompts") or []
    if image_prompts or images:
        compliance_reminder = image_step.get(
            "compliance_reminder",
            "【重要提示】AI生成图片仅作为设计参考，对外发布前必须经过医学、法务合规审核"
        )

        md_content.append("## 图片素材")
        md_content.append("")
        md_content.append(f"> {compliance_reminder}")
        md_content.append("")

        for i, prompt in enumerate(image_prompts, 1):
            md_content.append(f"### {i}. {prompt.get('name', prompt.get('image_type', f'图片{i}'))}")
            md_content.append("")
            md_content.append(f"**场景**：{prompt.get('scene_description', 'N/A')}")
            md_content.append("")
            md_content.append("**正向提示词（英文）**：")
            md_content.append("```")
            md_content.append(prompt.get('positive_prompt_en', prompt.get('positive_prompt', 'N/A')))
            md_content.append("```")
            md_content.append("")
            md_content.append("**负面提示词**：")
            md_content.append("```")
            md_content.append(prompt.get('negative_prompt', 'N/A'))
            md_content.append("```")
            md_content.append("")

            # 这套提示词已经出过图，就把图片插在提示词下面
            for img in images_by_prompt.get(i - 1, []):
                label = "优化后" if img.get("parent_image_id") else "初稿"
                md_content.append(f"![{prompt.get('name', f'图片{i}')} - {label}]({img['url']})")
                md_content.append("")
                detail = [f"模型：{img.get('model_name', img.get('model_used', 'N/A'))}", f"尺寸：{img.get('size', 'N/A')}"]
                if img.get("feedback"):
                    detail.append(f"优化意见：{img['feedback']}")
                md_content.append(f"> {label} · " + " · ".join(detail))
                md_content.append("")

        # 没有归到任何提示词下的图片（提示词被重新生成过等情况）
        orphans = [
            img for img in images
            if not isinstance(img.get("prompt_index"), int)
            or img["prompt_index"] >= len(image_prompts)
        ]
        if orphans:
            md_content.append("### 其他已生成图片")
            md_content.append("")
            for img in orphans:
                md_content.append(f"![{img.get('prompt_name', '生成图片')}]({img['url']})")
                md_content.append("")

        md_content.append(f"**本物料包共包含 {len(images)} 张已生成图片**")
        md_content.append("")

    # 页脚
    md_content.append("---")
    md_content.append("")
    md_content.append("## 使用说明")
    md_content.append("")
    md_content.append("1. 本物料包由AI辅助生成，仅供参考")
    md_content.append("2. **发布前必须完成**：业务审核 + 医学法规审核")
    md_content.append("3. 图片为AI生成初稿，需设计人员精修并经合规审核后使用")
    md_content.append("4. 严禁制作皮肤前后疗效对比图")
    md_content.append("5. 如有问题，请联系品牌运营团队")
    md_content.append("")
    md_content.append("---")
    md_content.append(f"*本物料包由品牌官号运营Agent生成 | 知原药业*")
    
    markdown_text = "\n".join(md_content)
    
    # 保存导出日志
    export_log = {
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "status": "exported",
        "step": "export_package",
        "export_content": markdown_text,
        "image_count": len(images),
        "export_time": datetime.now().isoformat()
    }
    save_task_log(f"{task_id}_export", export_log)

    # 把导出也记成一步，这样进度条上第5步会显示成已完成
    record_step(
        task,
        "export_package",
        "exported",
        {
            "result": {
                "exported_at": datetime.now().isoformat(),
                "image_count": len(images)
            }
        }
    )

    return {
        "task_id": task_id,
        "markdown": markdown_text,
        "image_count": len(images),
        "filename": f"品牌官号物料包_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    }


# ============ 健康检查 ============

@app.get("/api/health")
async def health_check():
    """健康检查"""
    default_model = next((m for m in user_models if m.get("is_default") and m.get("enabled")), None)
    return {
        "status": "ok",
        "version": "1.1.0",
        "default_model": default_model["name"] if default_model else None,
        "models_count": len([m for m in user_models if m.get("enabled")])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
