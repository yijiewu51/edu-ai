import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Literal

import aiosqlite
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

DB_PATH = "career_reports.db"
RATE_LIMIT_SECONDS = 60
_rate_limit: dict[str, float] = {}  # session_id -> last_request_time

SYSTEM_PROMPT = """你是一位专注于帮助中国30-45岁职场人完成职业转型的资深职业规划师。
你提供务实、具体、有数据支撑的建议，深刻理解中国当前就业市场趋势和AI对各行业的影响。
始终用简体中文回答。风格：直接、专业、鼓励，避免空话套话，给出可立刻行动的建议。"""


# ── Database ───────────────────────────────────────────────────────────────

async def init_db(db: aiosqlite.Connection):
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            session_id      TEXT    NOT NULL,
            job_title       TEXT    NOT NULL,
            years_exp       TEXT    NOT NULL,
            age_range       TEXT    NOT NULL,
            skills          TEXT    NOT NULL,
            career_goals    TEXT    NOT NULL,
            report_html     TEXT,
            report_json     TEXT,
            ip_address      TEXT,
            user_agent      TEXT,
            generation_ms   INTEGER
        )
    """)
    await db.commit()


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await aiosqlite.connect(DB_PATH)
    await init_db(app.state.db)
    yield
    await app.state.db.close()


app = FastAPI(lifespan=lifespan, title="AI职业诊断报告")


# ── Models ─────────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_title: str = Field(min_length=1, max_length=100)
    years_exp: Literal["1-3年", "3-5年", "5-10年", "10年以上"]
    age_range: Literal["30-35岁", "35-40岁", "40-45岁", "45岁以上"]
    skills: list[str] = Field(min_length=1, max_length=30)
    career_goals: str = Field(min_length=5, max_length=1000)


# ── AI Integration ─────────────────────────────────────────────────────────

def parse_claude_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


async def generate_report_data(req: ReportRequest) -> dict:
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    skills_str = "、".join(req.skills)
    prompt = f"""用户职业信息：
- 当前职位：{req.job_title}
- 工作年限：{req.years_exp}
- 年龄段：{req.age_range}
- 现有技能：{skills_str}
- 职业目标：{req.career_goals}

请以严格JSON格式返回分析结果，结构如下（不要有任何额外文字）：
{{
  "summary": "一句话概括该用户的职业现状和转型方向（20字以内）",
  "skill_gap_analysis": "针对该用户的技能缺口深度分析，结合当前AI替代就业趋势，2-3段，每段100字左右",
  "top_5_skills": [
    {{"rank": 1, "skill": "技能名称", "reason": "为何该技能对此用户最重要", "learning_weeks": 4, "difficulty": "入门/中级/进阶"}}
  ],
  "learning_path": [
    {{"phase": "第1-2个月", "focus": "学习重点", "goal": "达成目标", "resources": ["推荐资源1", "推荐资源2"]}}
  ],
  "job_market_outlook": "该转型方向的就业市场前景，包含薪资区间和需求趋势，150字左右",
  "risk_assessment": "不转型继续现有路线的风险分析，100字左右",
  "encouragement": "针对该用户具体情况的个性化鼓励，引用其工作年限和目标，50字左右"
}}"""

    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=0.7,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    try:
        return parse_claude_json(raw)
    except json.JSONDecodeError:
        # Log for debugging and raise
        print(f"[ERROR] Failed to parse Claude JSON. Raw response:\n{raw[:500]}")
        raise


# ── Report HTML Renderer ───────────────────────────────────────────────────

SKILL_COLORS = ["bg-red-500", "bg-orange-500", "bg-yellow-500", "bg-blue-500", "bg-purple-500"]
DIFFICULTY_COLORS = {
    "入门": "bg-green-100 text-green-800",
    "中级": "bg-yellow-100 text-yellow-800",
    "进阶": "bg-red-100 text-red-800",
}


def render_report_html(data: dict, req: ReportRequest) -> str:
    skills_html = ""
    for i, skill in enumerate(data.get("top_5_skills", [])[:5]):
        color = SKILL_COLORS[i]
        diff = skill.get("difficulty", "中级")
        diff_cls = DIFFICULTY_COLORS.get(diff, "bg-gray-100 text-gray-800")
        skills_html += f"""
        <div class="flex items-start gap-3 p-4 bg-white rounded-xl border border-gray-100 shadow-sm">
          <div class="flex-shrink-0 w-8 h-8 {color} text-white rounded-full flex items-center justify-center font-bold text-sm">
            {skill.get('rank', i+1)}
          </div>
          <div class="flex-1">
            <div class="flex items-center gap-2 mb-1">
              <span class="font-semibold text-gray-900">{skill.get('skill', '')}</span>
              <span class="text-xs px-2 py-0.5 rounded-full {diff_cls}">{diff}</span>
              <span class="text-xs text-gray-400">约{skill.get('learning_weeks', 4)}周</span>
            </div>
            <p class="text-sm text-gray-600">{skill.get('reason', '')}</p>
          </div>
        </div>"""

    path_html = ""
    phases = data.get("learning_path", [])
    for i, phase in enumerate(phases):
        is_last = i == len(phases) - 1
        resources = "、".join(phase.get("resources", []))
        connector = "" if is_last else '<div class="w-0.5 h-6 bg-blue-200 ml-4 my-1"></div>'
        path_html += f"""
        <div class="flex items-start gap-3">
          <div class="flex-shrink-0 flex flex-col items-center">
            <div class="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-bold">{i+1}</div>
            {connector}
          </div>
          <div class="flex-1 pb-2">
            <div class="font-semibold text-blue-900 text-sm">{phase.get('phase', '')}</div>
            <div class="font-medium text-gray-800 mt-0.5">{phase.get('focus', '')}</div>
            <div class="text-sm text-gray-500 mt-0.5">目标：{phase.get('goal', '')}</div>
            <div class="text-xs text-blue-600 mt-1">推荐：{resources}</div>
          </div>
        </div>"""

    skills_tags = "".join(
        f'<span class="inline-block bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded-full">{s}</span>'
        for s in req.skills
    )

    report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M')}"

    return f"""
<div class="space-y-6" id="report-output">
  <!-- Header -->
  <div class="bg-gradient-to-br from-blue-600 to-purple-700 rounded-2xl p-6 text-white shadow-lg">
    <div class="flex items-start justify-between">
      <div>
        <div class="text-blue-200 text-xs mb-1">AI职业诊断报告 · {report_id}</div>
        <h2 class="text-2xl font-bold">{data.get('summary', 'AI职业规划报告')}</h2>
        <p class="text-blue-200 text-sm mt-2">{req.job_title} · {req.years_exp} · {req.age_range}</p>
      </div>
      <div class="text-5xl opacity-20">🧭</div>
    </div>
    <div class="mt-4 flex flex-wrap gap-2">
      {skills_tags}
    </div>
  </div>

  <!-- Skill Gap Analysis -->
  <div class="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
    <h3 class="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
      <span class="w-2 h-6 bg-blue-600 rounded-full inline-block"></span>
      技能缺口分析
    </h3>
    <div class="text-gray-700 leading-relaxed text-sm whitespace-pre-line">{data.get('skill_gap_analysis', '')}</div>
  </div>

  <!-- Top 5 Skills -->
  <div class="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
    <h3 class="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
      <span class="w-2 h-6 bg-orange-500 rounded-full inline-block"></span>
      最优先学习的5项技能
    </h3>
    <div class="space-y-3">
      {skills_html}
    </div>
  </div>

  <!-- Learning Path -->
  <div class="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
    <h3 class="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
      <span class="w-2 h-6 bg-green-500 rounded-full inline-block"></span>
      个性化学习路径
    </h3>
    <div class="space-y-1">
      {path_html}
    </div>
  </div>

  <!-- Market Outlook + Risk -->
  <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
    <div class="bg-blue-50 rounded-2xl p-5 border border-blue-100">
      <h3 class="font-bold text-blue-900 mb-2 text-sm">📈 市场前景</h3>
      <p class="text-blue-800 text-sm leading-relaxed">{data.get('job_market_outlook', '')}</p>
    </div>
    <div class="bg-amber-50 rounded-2xl p-5 border border-amber-100">
      <h3 class="font-bold text-amber-900 mb-2 text-sm">⚠️ 不转型的风险</h3>
      <p class="text-amber-800 text-sm leading-relaxed">{data.get('risk_assessment', '')}</p>
    </div>
  </div>

  <!-- Encouragement -->
  <div class="bg-gradient-to-r from-purple-50 to-blue-50 rounded-2xl p-6 border-l-4 border-purple-600">
    <p class="text-purple-900 font-medium text-base leading-relaxed">💬 {data.get('encouragement', '')}</p>
  </div>

  <!-- Footer -->
  <div class="text-center text-xs text-gray-400 pb-4">
    由 AI职业诊断平台 生成 · {datetime.now().strftime('%Y年%m月%d日')} · 仅供参考
  </div>
</div>"""


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("index.html")


@app.post("/api/report")
async def create_report(req: ReportRequest, request: Request):
    db: aiosqlite.Connection = request.app.state.db
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")

    # Rate limiting
    now = time.time()
    last = _rate_limit.get(req.session_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        wait = int(RATE_LIMIT_SECONDS - (now - last))
        raise HTTPException(429, f"请求过于频繁，请{wait}秒后重试")
    _rate_limit[req.session_id] = now

    start_ms = int(time.time() * 1000)

    try:
        report_data = await generate_report_data(req)
    except json.JSONDecodeError:
        raise HTTPException(500, "AI生成报告格式异常，请稍后重试")
    except Exception as e:
        print(f"[ERROR] Claude API error: {e}")
        raise HTTPException(500, "AI服务暂时不可用，请稍后重试")

    report_html = render_report_html(report_data, req)
    gen_ms = int(time.time() * 1000) - start_ms

    await db.execute(
        """INSERT INTO submissions
           (session_id, job_title, years_exp, age_range, skills, career_goals,
            report_html, report_json, ip_address, user_agent, generation_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            req.session_id,
            req.job_title,
            req.years_exp,
            req.age_range,
            json.dumps(req.skills, ensure_ascii=False),
            req.career_goals,
            report_html,
            json.dumps(report_data, ensure_ascii=False),
            ip,
            ua,
            gen_ms,
        ),
    )
    await db.commit()

    async with db.execute("SELECT last_insert_rowid()") as cursor:
        row = await cursor.fetchone()
        submission_id = row[0]

    return {"submission_id": submission_id, "report_html": report_html}


@app.get("/api/health")
async def health(request: Request):
    db: aiosqlite.Connection = request.app.state.db
    async with db.execute("SELECT COUNT(*) FROM submissions") as cursor:
        row = await cursor.fetchone()
        total = row[0]
    return {"status": "ok", "total_submissions": total}


@app.get("/api/stats")
async def stats(request: Request):
    db: aiosqlite.Connection = request.app.state.db

    async with db.execute("SELECT COUNT(*) FROM submissions") as c:
        total = (await c.fetchone())[0]

    async with db.execute(
        "SELECT job_title, COUNT(*) as n FROM submissions GROUP BY job_title ORDER BY n DESC LIMIT 10"
    ) as c:
        top_jobs = [{"job": r[0], "count": r[1]} for r in await c.fetchall()]

    async with db.execute(
        "SELECT AVG(generation_ms) FROM submissions WHERE generation_ms IS NOT NULL"
    ) as c:
        avg_ms = (await c.fetchone())[0]

    return {
        "total_submissions": total,
        "top_jobs": top_jobs,
        "avg_generation_ms": round(avg_ms or 0),
    }
