# AI Resume Screening System

> **LangChain + Groq (LLaMA 3.1) + LangSmith | PDF Input → Score → Strong / Average / Weak**

---

## 📋 Problem Statement

Build an AI tool for a recruiter:
- **Input** → PDF Resume + Job Description
- **Process** → Skill Extraction + Matching + Scoring
- **Output** → Fit Score (0–100) + Label + Explanation

---

## 🏗️ Pipeline Architecture

```
PDF File
    │
    ▼  PyMuPDF (fitz)
Plain Text
    │
    ▼  extraction_chain.invoke()
Extracted Skills (JSON)
    │
    ▼  matching_chain.invoke()
Match Analysis (JSON)
    │
    ▼  scoring_chain.invoke()
Score 0–100 (JSON)
    │
    ▼  assign_label()  ← rule-based, no LLM
🟢 STRONG / 🟡 AVERAGE / 🔴 WEAK
    │
    ▼  explanation_chain.invoke()
Recruiter Explanation (Plain Text)
    │
    ▼  @traceable
LangSmith Trace
```

Each chain follows LangChain LCEL syntax:
```python
chain = PromptTemplate | ChatGroq | StrOutputParser
result = chain.invoke({...})
```

---

## 🔧 Tech Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | Groq `llama-3.1-8b-instant` | Free API, no credit card |
| Framework | LangChain LCEL | Modular `\|` pipe chains |
| Tracing | LangSmith | All 4 steps visible per run |
| PDF Reader | PyMuPDF (fitz) | Reliable multi-column extraction |
| Env Mgmt | python-dotenv | Secure `.env` key loading |
| Language | Python 3.13 (Anaconda) | Jupyter Notebook |

---

## 📁 Project Structure

```
AI-Resume-Screening/
├── AI_Resume_Screening_System_with_Tracing.ipynb   ← Main notebook
├── strong_resume.pdf                                ← Strong candidate PDF
├── average_resume.pdf                               ← Average candidate PDF
├── weak_resume.pdf                                  ← Weak candidate PDF
├── resume_screening_results.json                    ← Exported results
├── .env                                             ← API keys (create venv and add API_Keys to it)
└── README.md
```

---

## ⚙️ Setup & Installation

### Step 1 — Clone / Download
```bash
git clone https://github.com/piyushsb007/Gen-ai.git
cd Gen-ai.git
```

### Step 2 — Install Dependencies
```bash
pip install langchain langchain-groq langsmith pymupdf python-dotenv
```

Or inside the notebook:
```python
import sys
!{sys.executable} -m pip install langchain langchain-groq langsmith pymupdf python-dotenv
```

### Step 3 — Get Free API Keys

**Groq API Key (Free — no credit card):**
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up → API Keys → Create New Key
3. Copy key starting with `gsk_...`

**LangSmith API Key (Free):**
1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up → Settings → API Keys → Create
3. Copy key starting with `ls__...`

### Step 4 — Create `.env` File
Create a file named `.env` in the project root:
```
GROQ_API_KEY=gsk_YOUR_GROQ_KEY_HERE
LANGCHAIN_API_KEY=ls__YOUR_LANGSMITH_KEY_HERE
```

### Step 5 — Add Your PDF Resumes
Place your PDF files in the same directory and update the paths in the notebook:
```python
PDF_PATHS = {
    "Strong Candidate"  : "strong_resume.pdf",
    "Average Candidate" : "average_resume.pdf",
    "Weak Candidate"    : "weak_resume.pdf",
}
```

### Step 6 — Run the Notebook
Open `AI_Resume_Screening_System_with_Tracing.ipynb` in Jupyter and run all cells top to bottom.

---

## Notebook Cell Guide

| Cell | Section | What It Does |
|---|---|---|
| 1 | Install | `pip install` all required packages |
| 2 | Env Setup | Load API keys from `.env`, configure LangSmith tracing |
| 3 | PDF Extractor | `extract_text_from_pdf()` using PyMuPDF |
| 4 | Load PDFs | Read all 3 resume PDFs into `RESUMES` dict |
| 5 | Job Description | Define `JOB_DESCRIPTION` (Senior Data Scientist) |
| 6 | Prompt Templates | 4 `PromptTemplate` objects with anti-hallucination rules |
| 7 | LCEL Chains | Build chains: `prompt \| llm \| str_parser` |
| 8 | Pipeline Function | `screen_candidate()` with `@traceable` |
| 9 | Run All | Loop through all 3 candidates |
| 10 | Results | Summary table + score bars |
| 11 | Debug Run | Vague resume → anti-hallucination test |
| 12 | Export | Save results to `resume_screening_results.json` |

---

## 📊 Scoring Rubric

| Category | Max Points | What It Measures |
|---|---|---|
| Required Skills | 50 | Proportion of JD required skills matched |
| Experience | 25 | Years of experience + production ML work |
| Education | 10 | Relevance to CS / Stats / Math |
| Nice-to-Have | 10 | NLP, Spark, Docker, Kaggle etc. |
| Achievements | 5 | Publications, rankings, certifications |
| **Total** | **100** | |

### Label Thresholds
```
score >= 75  →  🟢 STRONG
score >= 45  →  🟡 AVERAGE
score <  45  →  🔴 WEAK
```

---

## ✅ Execution Results (April 17, 2026)

| Candidate | Score | Label | Correct? |
|---|---|---|---|
| strong_resume | 82/100 | 🟢 STRONG | ✅ |
| average_resume | 45/100 | 🟡 AVERAGE | ✅ |
| weak_resume | 30/100 | 🔴 WEAK | ✅ |
| debug_vague_resume | 10/100 | 🔴 WEAK | ✅ Anti-hallucination PASSED |

**Pipeline accuracy: 3/3 correct labels (100%)**

---

## 🔍 LangSmith Tracing

All runs are automatically traced under project `AI-Resume-Screener-Groq`.

### View Traces
1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Click Projects → `AI-Resume-Screener-Groq`
3. You will see 4 runs: 3 main + 1 debug

### What Each Trace Shows
- Nested chain calls (4 steps per run)
- Input/output for each step
- Latency and token usage
- Tags for filtering (`resume-screening`, `groq`, `debug`, etc.)

### Debug Run
The `debug_vague_resume` run is tagged `debug` + `anti-hallucination-test`. It tests whether the pipeline correctly scores a vague resume low, even if it claims "10 years of experience". Result: **10/100** — proves anti-hallucination rules work.

---

##  Prompt Engineering Decisions

### Anti-Hallucination Rules (in all prompts)
```
STRICT RULES:
- Only extract information EXPLICITLY written in the resume.
- Do NOT infer or assume any skills not present in the resume text.
- If a field has no data in the resume, write "Not mentioned".
```

### Few-Shot Prompting (Explanation Prompt)
Three complete example outputs are embedded in the explanation prompt. This teaches the LLM the expected tone, sentence count, and RECOMMENDATION format without additional fine-tuning.

### JSON Output Constraints (Steps 1–3)
All structured steps end with:
```
Respond ONLY with valid JSON. No explanation, no markdown fences.
```
A `safe_parse_json()` helper strips any markdown code fences (` ```json ... ``` `) that the LLM might add anyway.

---

## 📦 Dependencies

```
langchain>=1.2.15
langchain-groq>=1.1.2
langsmith>=0.7.30
pymupdf>=1.27.2
python-dotenv>=1.1.0
```

---

*Built using LangChain, Groq, and LangSmith*
