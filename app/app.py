from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, re, json
import asyncio

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Need OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2267246 ABAP Snippet Assessment")

OBSOLETE_FIELDS = [
    "MARC-MEGRU", "MARC-USEQU", "MARC-ALTSL", "MARC-MDACH",
    "MARC-DPLFS", "MARC-DPLPU", "MARC-DPLHO",
    "MARD-DISKZ", "MARD-LSOBS", "MARD-LMINB", "MARD-LBSTF",
    "MARC-FHORI"
]
OBSOLETE_DATAELEMS = sorted({f.split("-")[1] for f in OBSOLETE_FIELDS})

SQL_FIELD_RE = re.compile(
    r"\b(" + "|".join([f.replace("-", r"[\-~>]") for f in OBSOLETE_FIELDS]) + r")\b",
    re.IGNORECASE
)
DECL_FIELD_RE = re.compile(
    r"\b(DATA|TYPES|FIELD\-SYMBOLS|CONSTANTS|PARAMETERS)\b[^.\n]*?\b(TYPE|LIKE)\b\s+("
    + "|".join(OBSOLETE_DATAELEMS) + r")\b",
    re.IGNORECASE
)
STRUCT_FIELD_RE = re.compile(
    r"\b(\w+)\s+TYPE\s+(" + "|".join(OBSOLETE_DATAELEMS) + r")\b",
    re.IGNORECASE
)

class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    code: Optional[str] = ""
    findings: Optional[List[Finding]] = Field(default_factory=list)

def remediation_comment(field: str) -> str:
    return f"Remove or redesign. {field.upper()} is obsolete in S/4HANA (SAP Note 2267246)."

def find_obsolete_usages(code: str, unit: Unit) -> List[Finding]:
    findings = []
    for m in SQL_FIELD_RE.finditer(code or ""):
        field = m.group(1)
        findings.append(Finding(
            pgm_name=unit.pgm_name,
            inc_name=unit.inc_name,
            type=unit.type,
            name=unit.name,
            issue_type="OBSOLETE_FIELD",
            severity="WARNING",
            message=f"Obsolete field usage: {field}",
            suggestion=remediation_comment(field),
            snippet=field,
        ))
    for m in DECL_FIELD_RE.finditer(code or ""):
        data_elem = m.group(3)
        findings.append(Finding(
            pgm_name=unit.pgm_name,
            inc_name=unit.inc_name,
            type=unit.type,
            name=unit.name,
            issue_type="OBSOLETE_FIELD_DECL",
            severity="WARNING",
            message=f"Declaration of obsolete field: {data_elem}",
            suggestion=remediation_comment(data_elem),
            snippet=data_elem,
        ))
    for m in STRUCT_FIELD_RE.finditer(code or ""):
        compname, data_elem = m.group(1), m.group(2)
        findings.append(Finding(
            pgm_name=unit.pgm_name,
            inc_name=unit.inc_name,
            type=unit.type,
            name=unit.name,
            issue_type="OBSOLETE_STRUCT",
            severity="WARNING",
            message=f"Struct field uses obsolete datatype: {data_elem} in {compname}",
            suggestion=remediation_comment(data_elem),
            snippet=data_elem,
        ))
    return findings

SYSTEM_MSG = """
You are a senior ABAP expert. Output ONLY JSON as response.
For every provided payload .findings[].snippet,
write a bullet point that:
- Displays the exact offending code (use .snippet)
- Explains the necessary action to fix it (from .suggestion, if present).
- Bullet points should contain both offending code snippet and the fix, shown inline.
- Do NOT omit any snippet; all must be covered, no matter how many there are.
- Only show actual ABAP code for each snippet with its specific action.
""".strip()

USER_TEMPLATE = """
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Start line: {start_line}
End line: {end_line}

ABAP code context (optional):
{code}

findings (JSON list of findings, each with .snippet and .suggestion if present):
{findings_json}

Instructions:
1. Write a 1-paragraph assessment summarizing obsolete field risks in human language.
2. Write a llm_prompt field: for every finding, add a bullet point with
   - The exact code (snippet field)
   - The action required (from suggestion field, if any).
   - Do not compress, omit, or refer to them by index; always display code inline.

Return valid JSON with:
{{
  "assessment": "<paragraph>",
  "llm_prompt": "<action bullets>"
}}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt_sync(unit: Unit) -> Dict[str, str]:
    findings_json = json.dumps([f.model_dump() for f in (unit.findings or [])], ensure_ascii=False, indent=2)
    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name or "",
            "start_line": unit.start_line or 0,
            "end_line": unit.end_line or 0,
            "code": unit.code or "",
            "findings_json": findings_json,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.post("/assess-2267246-strict")
async def assess_obsolete_snippet(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    for u in units:
        if not u.findings or len(u.findings) == 0:
            if u.code:
                u.findings = find_obsolete_usages(u.code, u)
        if not u.findings or len(u.findings) == 0:
            continue  # skip if negative
            
        # LLM call in background thread
        llm_out = await loop.run_in_executor(
            None, llm_assess_and_prompt_sync, u
        )
        obj = u.model_dump()
        obj["assessment"] = llm_out.get("assessment", "")
        prompt_out = llm_out.get("llm_prompt", "")
        if isinstance(prompt_out, list):
            prompt_out = "\n".join(str(x) for x in prompt_out if x is not None)
        obj["llm_prompt"] = prompt_out
        obj.pop("findings", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}