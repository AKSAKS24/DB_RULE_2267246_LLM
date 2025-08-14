from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, re, json

# ---- Env setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# LangChain deps
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2267246 Obsolete Field Detection & Remediation")

# ===== Obsolete fields from MARC/MARD =====
OBSOLETE_FIELDS = [
    "MARC-MEGRU", "MARC-USEQU", "MARC-ALTSL", "MARC-MDACH",
    "MARC-DPLFS", "MARC-DPLPU", "MARC-DPLHO",
    "MARD-DISKZ", "MARD-LSOBS", "MARD-LMINB", "MARD-LBSTF",
    "MARC-FHORI"
]
# Data element names (part after the dash)
OBSOLETE_DATAELEMS = sorted({f.split("-")[1] for f in OBSOLETE_FIELDS})

# ===== Regex patterns =====
# 1. SQL usage of obsolete fields (MARC-XXXX / MARD-XXXX with -, ~, ->)
SQL_FIELD_RE = re.compile(
    r"\b(" + "|".join([f.replace("-", r"[\-~>]") for f in OBSOLETE_FIELDS]) + r")\b",
    re.IGNORECASE
)

# 2. Declaration usage: TYPE/LIKE <obsolete_dataelem>
DECL_FIELD_RE = re.compile(
    r"\b(DATA|TYPES|FIELD\-SYMBOLS|CONSTANTS|PARAMETERS)\b[^.\n]*?\b(TYPE|LIKE)\b\s+(" 
    + "|".join(OBSOLETE_DATAELEMS) + r")\b",
    re.IGNORECASE
)

# Also catch inside structures: comp TYPE MEGRU
STRUCT_FIELD_RE = re.compile(
    r"\b(\w+)\s+TYPE\s+(" + "|".join(OBSOLETE_DATAELEMS) + r")\b",
    re.IGNORECASE
)

# ===== Models =====
class SelectItem(BaseModel):
    table: Optional[str] = None
    field: str
    location: str
    suggested_statement: str

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = None
    code: Optional[str] = ""
    selects: List[SelectItem] = []

# ===== Helper functions =====
def remediation_comment(field: str) -> str:
    return f"* TODO: {field.upper()} is obsolete in S/4HANA (SAP Note 2267246). Functionality is omitted; remove or redesign logic."

def remediate_usage(field_text: str) -> str:
    return f"{field_text}  {remediation_comment(field_text)}"

def parse_and_fill_selects(unit: Unit) -> List[SelectItem]:
    """Scan for obsolete field usages in SQL, declarations, and structures."""
    code = unit.code or ""
    findings: List[SelectItem] = []

    # 1. SQL field usage
    for m in SQL_FIELD_RE.finditer(code):
        full = m.group(1)
        table = full.split("-")[0].split("~")[0].split(">")[0].upper()
        findings.append(SelectItem(
            table=table,
            field=full,
            location="SQL",
            suggested_statement=remediate_usage(full)
        ))

    # 2. Declaration usage
    for m in DECL_FIELD_RE.finditer(code):
        full = m.group(0)
        data_elem = m.group(3)
        findings.append(SelectItem(
            table=None,
            field=data_elem,
            location="Declaration",
            suggested_statement=remediate_usage(data_elem)
        ))

    # 3. Structure component usage
    for m in STRUCT_FIELD_RE.finditer(code):
        compname, data_elem = m.group(1), m.group(2)
        findings.append(SelectItem(
            table=None,
            field=data_elem,
            location=f"Structure component '{compname}'",
            suggested_statement=remediate_usage(data_elem)
        ))

    return findings

# ===== Summariser =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    field_count: Dict[str, int] = {}
    flagged = []
    for s in unit.selects:
        field_up = s.field.upper()
        field_count[field_up] = field_count.get(field_up, 0) + 1
        flagged.append({"field": s.field, "reason": remediation_comment(s.field)})
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

# ===== LLM Prompt Setup =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2267246. Output strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP code usage in light of SAP Note 2267246 (Obsolete MARC/MARD fields in S/4HANA).

From S/4HANA onwards, the following fields are obsolete and their functionality is omitted:
{field_list}

We provide program/include/unit metadata, and analysis findings.

Your tasks:
1) Produce a concise **assessment** highlighting:
   - Which statements or declarations reference obsolete fields.
   - Why these fields are obsolete (functionality omitted).
   - Potential functional and data impact.
2) Produce an **LLM remediation prompt** to:
   - Scan ABAP code for these field usages in SQL, declarations, and structure components.
   - Add a TODO comment after each usage indicating obsolescence and SAP Note number.
   - Output strictly in JSON with: original_code, remediated_code, changes[].

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

Analysis:
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

# ===== LLM Call =====
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
            "field_list": ", ".join(OBSOLETE_FIELDS),
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ===== API POST =====
@app.post("/assess-2267246")
def assess_obsolete_fields(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
        u.selects = parse_and_fill_selects(u)
        if not u.selects:
            # No obsolete fields found
            obj = u.model_dump()
            obj.pop("selects", None)
            obj["assessment"] = "No usage of obsolete MARC/MARD fields found (SAP Note 2267246)."
            obj["llm_prompt"] = ""
            out.append(obj)
            continue

        # Found at least 1 occurrence → call LLM
        llm_out = llm_assess_and_prompt(u)
        obj = u.model_dump()
        obj.pop("selects", None)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        out.append(obj)

    return out