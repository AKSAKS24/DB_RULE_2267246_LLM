# app_2267246_strict.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import os, re, json

# ---- LLM Setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2267246 Obsolete Field Assessment - Strict Schema")

# ===== Obsolete fields =====
OBSOLETE_FIELDS = [
    "MARC-MEGRU", "MARC-USEQU", "MARC-ALTSL", "MARC-MDACH",
    "MARC-DPLFS", "MARC-DPLPU", "MARC-DPLHO",
    "MARD-DISKZ", "MARD-LSOBS", "MARD-LMINB", "MARD-LBSTF",
    "MARC-FHORI"
]
OBSOLETE_DATAELEMS = sorted({f.split("-")[1] for f in OBSOLETE_FIELDS})

# ===== Regex patterns =====
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

# ===== Strict Models =====
class SelectItem(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def no_none_in_list(cls, v: List[str]) -> List[str]:
        return [x for x in v if x is not None]

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str                # now required
    code: Optional[str] = "" # we still allow ABAP code to be passed in so we can generate selects
    selects: List[SelectItem] = Field(default_factory=list)

# ===== Detection logic =====
def remediation_comment(field: str) -> str:
    return f"* TODO: {field.upper()} is obsolete in S/4HANA (SAP Note 2267246). Functionality omitted; remove or redesign."

def parse_and_build_selectitems(code: str) -> List[SelectItem]:
    findings: List[SelectItem] = []

    # SQL or qualified usage (marc-megru etc.)
    for m in SQL_FIELD_RE.finditer(code):
        full = m.group(1)
        table = full.split("-")[0].split("~")[0].split(">")[0].upper()
        fld = full.upper()
        findings.append(
            SelectItem(
                table=table,
                target_type="TABLE",
                target_name="",
                used_fields=[fld],
                suggested_fields=[fld],
                suggested_statement=f"{full}  {remediation_comment(full)}"
            )
        )

    # Declaration usage
    for m in DECL_FIELD_RE.finditer(code):
        data_elem = m.group(3)
        findings.append(
            SelectItem(
                table="",
                target_type="DATA",
                target_name="",
                used_fields=[data_elem.upper()],
                suggested_fields=[data_elem.upper()],
                suggested_statement=f"{data_elem}  {remediation_comment(data_elem)}"
            )
        )

    # Structure component usage
    for m in STRUCT_FIELD_RE.finditer(code):
        compname, data_elem = m.group(1), m.group(2)
        findings.append(
            SelectItem(
                table="",
                target_type="STRUCT_COMP",
                target_name=compname,
                used_fields=[data_elem.upper()],
                suggested_fields=[data_elem.upper()],
                suggested_statement=f"{data_elem}  {remediation_comment(data_elem)}"
            )
        )

    return findings

# ===== Summariser =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    field_count: Dict[str, int] = {}
    flagged = []
    for s in unit.selects:
        for f in s.used_fields:
            field_count[f.upper()] = field_count.get(f.upper(), 0) + 1
            flagged.append({"field": f, "reason": remediation_comment(f)})
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_occurrences": len(unit.selects),
            "fields_frequency": field_count,
            "note_2267246_flags": flagged
        }
    }

# ===== LLM prompt =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2267246. Output strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP code usage in light of SAP Note 2267246 (Obsolete MARC/MARD fields in S/4HANA).

We provide program/include/unit metadata and analysis findings (under "selects" with table, used_fields, suggested_fields, and suggested_statement).

Your tasks:
1) Produce a concise assessment of impact.
2) Produce an actionable LLM remediation prompt to append TODO comments to obsolete field usages.

Return ONLY strict JSON:
{{
  "assessment": "<concise note 2267246 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Summary:
{plan_json}

selects (JSON):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE)
    ]
)
llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan = summarize_selects(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], ensure_ascii=False, indent=2)
    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ===== API Endpoint =====
@app.post("/assess-2267246-strict")
async def assess_obsolete_fields(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        # Build selects from code if provided
        if u.code:
            u.selects = parse_and_build_selectitems(u.code)

        if not u.selects:
            obj = u.model_dump()
            obj.pop("selects", None)
            obj["assessment"] = "No usage of obsolete MARC/MARD fields found (SAP Note 2267246)."
            obj["llm_prompt"] = ""
            out.append(obj)
            continue

        llm_out = llm_assess_and_prompt(u)
        obj = u.model_dump()
        obj.pop("selects", None)
        obj.pop("code", None)  # don't return raw code
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        out.append(obj)

    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}