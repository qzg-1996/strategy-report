# -*- coding: utf-8 -*-
"""AI 分析模块：调用 Kimi API（Moonshot），实现消息自动解读与辅助筛选

设计要点：
  - API Key 存本地 ai_config.json（已加入 .gitignore，不会上传GitHub）
  - 云端部署时在 Streamlit Cloud 后台 Settings -> Secrets 配置 KIMI_API_KEY
  - 调用失败自动降级，不影响系统其他功能
"""
import json
import os

import requests

from config import BASE_DIR

CONFIG_PATH = os.path.join(BASE_DIR, "ai_config.json")
DEFAULTS = {"api_key": "", "base_url": "https://api.moonshot.cn/v1",
            "model": "kimi-k3.1-pro"}

# 模型偏好顺序：官方持续下线旧模型（moonshot-v1系已于2026-08-31下线），
# 404时自动查询账号可用模型，按此顺序挑选，以后模型再换代也不用改代码
PREFERRED_MODELS = ["kimi-k3.1-pro", "kimi-k3", "kimi-k2.7-code",
                    "kimi-k2.6", "kimi-k2.5"]

SYSTEM_PROMPT = (
    "你是黑色系大宗商品产业分析师，服务于一家钢贸企业经营管理部。"
    "基于提供的当日行情数据与财经快讯，输出简洁专业的每日解读。要求：\n"
    "1. 开头一句话定调：今日消息面对钢材价格整体利多/利空/中性（含强度）；\n"
    "2. 挑出最重要的3-5条消息逐条点评：影响路径、影响程度（强/中/弱）、"
    "作用品种（成材/原料/宏观）；\n"
    "3. 结尾结合基差分位、月差结构给出1-2条操作提示（套保/基差交易视角）；\n"
    "4. 全文不超过400字，直接给结论，不要客套话，不要重复原文标题。")


def load_config():
    """读取AI配置：优先本地文件，其次Streamlit secrets（云端）"""
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            cfg.update(json.load(open(CONFIG_PATH, encoding="utf-8")))
        except Exception:
            pass
    if not cfg["api_key"]:
        try:
            import streamlit as st
            cfg["api_key"] = st.secrets.get("KIMI_API_KEY", "")
        except Exception:
            pass
    return cfg


def save_config(api_key, model=None):
    cfg = load_config()
    cfg["api_key"] = api_key.strip()
    if model:
        cfg["model"] = model
    json.dump(cfg, open(CONFIG_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def is_ready():
    return bool(load_config()["api_key"])


def _pick_available_model(cfg):
    """用Key查询账号当前可用模型列表，按偏好顺序挑一个（应对官方模型换代下线）"""
    r = requests.get(f"{cfg['base_url'].rstrip('/')}/models",
                     headers={"Authorization": f"Bearer {cfg['api_key']}"},
                     timeout=30)
    r.raise_for_status()
    ids = [m.get("id", "") for m in r.json().get("data", [])]
    for want in PREFERRED_MODELS:
        if want in ids:
            return want
    for mid in ids:  # 兜底：任意一个kimi系对话模型
        if mid.startswith("kimi"):
            return mid
    raise RuntimeError(f"账号无可用模型，models接口返回：{ids[:5]}")


def _post_chat(cfg, model, messages, max_tokens):
    # K3系列模型：temperature/top_p等采样参数被服务端锁死（固定1.0/0.95），
    # 传了直接400报错，所以一律不传；max_tokens已弃用，改用max_completion_tokens
    payload = {"model": model, "messages": messages,
               "max_completion_tokens": max_tokens}
    if model.startswith("kimi-k3"):
        # K3支持reasoning_effort（low/high/max，默认max）。每日解读/筛选是小任务，
        # 用low档：出结果更快、reasoning token消耗更少（思考token也计费）
        payload["reasoning_effort"] = "low"
    r = requests.post(
        f"{cfg['base_url'].rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=90)
    if r.status_code == 404:
        # 模型不存在/已下线：抛HTTPError让上层走自动换模型逻辑
        raise requests.HTTPError("404 Not Found", response=r)
    if r.status_code != 200:
        # 把服务端返回的错误原文抛出来，便于排查（截图即可定位）
        raise RuntimeError(f"AI调用失败 {r.status_code}：{r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _chat(messages, max_tokens=800, temperature=0.3):
    """调用Kimi API（OpenAI兼容协议）；404多为模型已下线，自动换可用模型重试一次
    temperature参数保留兼容但不再发送（新模型锁死采样参数）"""
    cfg = load_config()
    if not cfg["api_key"]:
        raise RuntimeError("未配置API Key")
    try:
        return _post_chat(cfg, cfg["model"], messages, max_tokens)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            model = _pick_available_model(cfg)
            cfg["model"] = model
            try:  # 本地环境顺手写回配置，云端只读文件系统则跳过
                json.dump(cfg, open(CONFIG_PATH, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
            except Exception:
                pass
            return _post_chat(cfg, model, messages, max_tokens)
        raise


def daily_interpret(market_snapshot, news_list):
    """每日AI解读：行情快照 + 消息列表 -> 分析文字"""
    news_text = "\n".join(
        f"- [{n['category']}] {n['pub_time'][:16]} {n['title']}"
        + (f"（{n['summary'][:80]}）" if n["summary"] != n["title"] else "")
        for n in news_list)
    user = (f"【当日行情快照】\n{market_snapshot}\n\n"
            f"【当日相关快讯（已经初步筛选）】\n{news_text or '（今日暂无黑色系相关快讯）'}")
    return _chat([{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}])


def ai_screen_news(raw_news):
    """AI辅助筛选：从原始快讯中挑出与钢贸/黑色系相关的条目
    raw_news: [{"idx","title"}...]  返回AI认为相关的idx列表及一句话理由"""
    items = "\n".join(f"{n['idx']}. {n['title']}" for n in raw_news)
    prompt = (
        "以下是今日财经快讯列表。请挑出与钢材、黑色金属、钢铁产业链、"
        "房地产基建、宏观货币政策直接或间接相关的条目。\n"
        "输出格式：每行一条「序号|相关理由（10字内）」，不要输出其他内容；"
        "若全部无关，只输出「无」。\n\n" + items)
    resp = _chat([{"role": "user", "content": prompt}], max_tokens=500)
    out = []
    for line in resp.strip().splitlines():
        line = line.strip()
        if not line or line == "无":
            continue
        if "|" in line:
            num, reason = line.split("|", 1)
            num = "".join(ch for ch in num if ch.isdigit())
            if num:
                out.append({"idx": int(num), "reason": reason.strip()})
    return out
