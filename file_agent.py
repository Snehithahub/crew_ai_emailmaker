"""Email Crafting Crew.

A 4-agent CrewAI pipeline configured from agents.yaml / tasks.yaml:

    message_analyst  ->  formal_writer   (parallel)  ->  chief_editor
                     ->  friendly_writer (parallel)

Run:
    python file_agent.py
    python file_agent.py -m "pls review OBM PR asap, thx" -r "Priya" -s "Snehitha" -t formal
"""


import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from crewai import LLM, Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, after_kickoff, agent, before_kickoff, crew, llm, task, tool
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL = "groq/openai/gpt-oss-120b"

DEFAULT_INPUTS = {
    "message": "we have made changes for OBM,u can test now LOL",
    "recipient_name": "Team",
    "sender_name": "Snehitha",
    "preferred_tone": "professional and friendly",
    "extra_context": "None",
}


# ---------------------------------------------------------------------------
# Abbreviations (extend by dropping an abbreviations.json next to this file:
# {"ETA": "estimated time of arrival", ...})
# ---------------------------------------------------------------------------

ABBREVIATIONS: dict[str, str] = {
    "AFAIK": "as far as I know",
    "ASAP": "as soon as possible",
    "BRB": "be right back",
    "BTW": "by the way",
    "EOD": "end of day",
    "EOW": "end of week",
    "ETA": "estimated time of arrival",
    "FYI": "for your information",
    "IMO": "in my opinion",
    "IMHO": "in my humble opinion",
    "LOL": "laugh out loud",
    "NP": "no problem",
    "OBM": "obligation management use case",
    "OOO": "out of office",
    "PFA": "please find attached",
    "PR": "pull request",
    "PLS": "please",
    "PLZ": "please",
    "THX": "thanks",
    "TY": "thank you",
    "TBD": "to be decided",
    "WFH": "work from home",
    "U": "you",
    "UR": "your",
    "R": "are",
}

_custom_abbrev_file = BASE_DIR / "abbreviations.json"
if _custom_abbrev_file.exists():
    ABBREVIATIONS.update({k.upper(): v for k, v in json.loads(_custom_abbrev_file.read_text()).items()})

# Whole-word, case-insensitive match (so "PR" doesn't match inside "PRODUCT").
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, ABBREVIATIONS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# Chat-only fillers that should never reach a professional email.
_DROP_WORDS = {"LOL", "BRB"}


def find_abbreviations(text: str) -> list[str]:
    return sorted({m.group(0).upper() for m in _ABBREV_PATTERN.finditer(text)})


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str = Field(..., description="The full text (message or email) to process.")


class AbbreviationExpanderTool(BaseTool):
    name: str = "abbreviation_expander"
    description: str = (
        "Finds chat short forms and abbreviations (e.g. AFAIK, LOL, OBM, ASAP, u) "
        "as whole words in a text and returns the text with them expanded, plus the "
        "list of replacements. Chat fillers like LOL are flagged for removal."
    )
    args_schema: type[BaseModel] = TextInput

    def _run(self, text: str) -> str:
        replacements: list[str] = []

        def _replace(match: re.Match[str]) -> str:
            key = match.group(0).upper()
            if key in _DROP_WORDS:
                replacements.append(f"{match.group(0)} -> (remove: chat filler, '{ABBREVIATIONS[key]}')")
                return ""
            replacements.append(f"{match.group(0)} -> {ABBREVIATIONS[key]}")
            return ABBREVIATIONS[key]

        expanded = _ABBREV_PATTERN.sub(_replace, text)
        expanded = re.sub(r"\s+([.,!?])", r"\1", re.sub(r"\s{2,}", " ", expanded)).strip()
        if not replacements:
            return f"No abbreviations found. Text can be used as-is:\n{text}"
        return "Replacements:\n- " + "\n- ".join(replacements) + f"\n\nExpanded text:\n{expanded}"


class EmailQualityCheckerTool(BaseTool):
    name: str = "email_quality_checker"
    description: str = (
        "Checks a full email draft for: subject line, greeting, sign-off, body length, "
        "leftover abbreviations/slang, emojis and excessive exclamation marks. "
        "Returns PASS or a list of issues to fix."
    )
    args_schema: type[BaseModel] = TextInput

    def _run(self, text: str) -> str:
        issues: list[str] = []
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        lowered = text.lower()

        if not any(line.lower().startswith("subject:") for line in lines):
            issues.append("Missing a 'Subject:' line.")
        if not re.search(r"^(hi|hello|dear|hey|good (morning|afternoon|evening))\b", lowered, re.MULTILINE):
            issues.append("Missing a greeting (Hi / Hello / Dear ...).")
        if not re.search(r"(regards|best|thanks|thank you|sincerely|cheers|warmly)\s*,?", lowered):
            issues.append("Missing a sign-off (Best regards / Thanks / ...).")

        word_count = len(re.findall(r"\w+", text))
        if word_count > 220:
            issues.append(f"Too long ({word_count} words); aim for under ~180.")
        if word_count < 25:
            issues.append(f"Too short ({word_count} words); the message may be missing context.")

        leftovers = find_abbreviations(text)
        if leftovers:
            issues.append(f"Unexpanded abbreviations/slang: {', '.join(leftovers)}.")
        if re.search(r"[\U0001F300-\U0001FAFF]", text):
            issues.append("Contains emojis.")
        if text.count("!") > 2:
            issues.append("Too many exclamation marks.")

        if issues:
            return "FAIL\n- " + "\n- ".join(issues)
        return f"PASS ({word_count} words)"


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

class MessageAnalysis(BaseModel):
    expanded_message: str = Field(description="The raw message with all abbreviations expanded.")
    abbreviations_found: list[str] = Field(default_factory=list)
    intent: str = Field(description="What the sender wants, e.g. 'status update + request to test'.")
    audience: str
    key_points: list[str] = Field(description="Facts / action items that must appear in the email.")
    urgency: Literal["low", "medium", "high"]
    recommended_tone: str
    ambiguities: list[str] = Field(default_factory=list)


class FinalEmail(BaseModel):
    subject: str
    greeting: str
    body: str
    sign_off: str = Field(description="Closing line plus sender name, e.g. 'Best regards,\\nSnehitha'.")
    based_on: Literal["formal", "friendly", "merged"]
    editor_note: str = Field(description="One line explaining why this version was chosen.")

    def render(self) -> str:
        return f"Subject: {self.subject}\n\n{self.greeting}\n\n{self.body}\n\n{self.sign_off}"


# ---------------------------------------------------------------------------
# Guardrail: validated in code, failing output is sent back to the agent.
# ---------------------------------------------------------------------------

def validate_final_email(output: TaskOutput) -> tuple[bool, Any]:
    email = output.pydantic
    if not isinstance(email, FinalEmail):
        try:
            email = FinalEmail.model_validate_json(output.raw)
        except Exception as exc:  # noqa: BLE001
            return False, f"Output must match the FinalEmail schema: {exc}"

    if not email.subject.strip():
        return False, "Subject line is empty."
    if len(email.body.split()) < 15:
        return False, "Body is too short; include all key points from the analysis."
    leftovers = find_abbreviations(email.render())
    if leftovers:
        return False, f"Expand or remove these abbreviations: {', '.join(leftovers)}."
    return True, email


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------

@CrewBase
class EmailCraftingCrew:
    """Analyse a rough message, draft two versions in parallel, edit into one final email."""

    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"

    # --- LLMs referenced by name in agents.yaml ---------------------------
    @llm
    def precise_llm(self) -> LLM:
        return LLM(model=MODEL, temperature=0)

    @llm
    def creative_llm(self) -> LLM:
        return LLM(model=MODEL, temperature=0.6)

    # --- Tools referenced by name in agents.yaml --------------------------
    @tool
    def abbreviation_expander(self) -> BaseTool:
        return AbbreviationExpanderTool()

    @tool
    def email_quality_checker(self) -> BaseTool:
        return EmailQualityCheckerTool()

    # --- Agents ------------------------------------------------------------
    @agent
    def message_analyst(self) -> Agent:
        return Agent(config=self.agents_config["message_analyst"])

    @agent
    def formal_writer(self) -> Agent:
        return Agent(config=self.agents_config["formal_writer"])

    @agent
    def friendly_writer(self) -> Agent:
        return Agent(config=self.agents_config["friendly_writer"])

    @agent
    def chief_editor(self) -> Agent:
        return Agent(config=self.agents_config["chief_editor"])

    # --- Tasks -------------------------------------------------------------
    @task
    def analyze_message_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_message_task"], output_pydantic=MessageAnalysis)

    @task
    def formal_draft_task(self) -> Task:
        return Task(config=self.tasks_config["formal_draft_task"])

    @task
    def friendly_draft_task(self) -> Task:
        return Task(config=self.tasks_config["friendly_draft_task"])

    @task
    def final_email_task(self) -> Task:
        return Task(
            config=self.tasks_config["final_email_task"],
            output_pydantic=FinalEmail,
            guardrail=validate_final_email,
            guardrail_max_retries=3,
        )

    # --- Lifecycle hooks ---------------------------------------------------
    @before_kickoff
    def prepare_inputs(self, inputs: dict | None) -> dict:
        """Fill defaults so every {placeholder} in the YAML is always defined."""
        merged = {**DEFAULT_INPUTS, **{k: v for k, v in (inputs or {}).items() if v}}
        if not merged["message"].strip():
            raise ValueError("'message' input must not be empty.")
        OUTPUT_DIR.mkdir(exist_ok=True)
        return merged

    @after_kickoff
    def save_results(self, result):
        """Save the final email as Markdown and the full run as JSON."""
        email = result.pydantic
        if isinstance(email, FinalEmail):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            (OUTPUT_DIR / "final_email.md").write_text(email.render() + "\n")
            report = {
                "final_email": email.model_dump(),
                "tasks": [{"task": t.name, "agent": t.agent, "output": t.raw} for t in result.tasks_output],
                "token_usage": result.token_usage.model_dump() if result.token_usage else None,
            }
            (OUTPUT_DIR / f"run_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
        return result

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            max_rpm=30,
            verbose=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Turn a rough message into a polished email.")
    parser.add_argument("-m", "--message", help="Raw message to turn into an email.")
    parser.add_argument("-r", "--recipient", dest="recipient_name", help="Recipient name.")
    parser.add_argument("-s", "--sender", dest="sender_name", help="Sender name.")
    parser.add_argument("-t", "--tone", dest="preferred_tone", help="e.g. formal, friendly, apologetic.")
    parser.add_argument("-c", "--context", dest="extra_context", help="Extra facts the email may use.")
    return {k: v for k, v in vars(parser.parse_args()).items() if v}


def main() -> None:
    result = EmailCraftingCrew().crew().kickoff(inputs=parse_args())
    email = result.pydantic

    print("\n===== FINAL EMAIL =====\n")
    if isinstance(email, FinalEmail):
        print(email.render())
        print(f"\n[based on: {email.based_on}] {email.editor_note}")
    else:
        print(result.raw)
    if result.token_usage:
        print(f"\nTokens used: {result.token_usage.total_tokens}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
