from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _escape(text: Any) -> str:
    return html.escape(str(text))


class ReportCompiler:
    def build_markdown(
        self,
        *,
        case: dict[str, Any],
        plan: dict[str, Any],
        result: dict[str, Any],
        claim_rows: list[dict[str, Any]],
        reexamination: list[dict[str, Any]],
    ) -> str:
        findings = result.get("findings", [])
        survived = [f for f in findings if f.get("final_status") in {"supported", "challenged", "provisional"} and not any(a.get("killed") for a in f.get("falsifications", []))]
        failed = [f for f in findings if f not in survived]
        selected = plan.get("selected_dataset", {})
        lines: list[str] = [
            f"# Orbita Research Dossier: {case['name']}",
            "",
            f"**Case ID:** `{case['id']}`  ",
            f"**Mode:** {case.get('mode', 'open_discovery')}  ",
            f"**Goal supplied:** {case.get('goal') or 'None — open discovery mode'}  ",
            f"**Dataset:** {selected.get('name', '—')}  ",
            f"**Rows × columns:** {selected.get('rows', '—')} × {selected.get('columns', '—')}  ",
            "",
            "## Executive summary",
            "",
        ]
        if survived:
            lines.append(
                f"Orbita froze and tested {len(findings)} candidate relationships. "
                f"{len(survived)} survived the configured held-out and cross-seed attacks; "
                f"{len(failed)} were refuted, unstable, or unresolved."
            )
        else:
            lines.append(
                f"Orbita tested {len(findings)} candidate relationships, but none survived every configured attack. "
                "This is a valid null result and should not be rewritten as a discovery."
            )
        lines += [
            "",
            "> Status labels describe performance inside this finite analysis. They are not proof, a p-value, causal evidence, or external replication.",
            "",
            "## What Orbita received and inferred",
            "",
            f"- Files received: {len(case.get('files', []))}",
            f"- Analysis route: {', '.join(plan.get('routes', [])) or '—'}",
            f"- Candidates generated from locked scout data: {plan.get('candidate_generation', {}).get('generated_candidates', 0)}",
            f"- Scout rows: {plan.get('candidate_generation', {}).get('scout_rows', '—')}",
            f"- Locked confirmation rows: {plan.get('candidate_generation', {}).get('confirmation_rows', '—')}",
            "",
            "### Approved assumptions",
            "",
        ]
        for assumption in plan.get("assumptions", []):
            lines.append(f"- **{assumption.get('id', 'assumption')}** — {assumption.get('statement', '')}")
        if not plan.get("assumptions"):
            lines.append("- No explicit assumptions were recorded.")

        lines += ["", "## Data quality, errors, and artifact guards", ""]
        for item in plan.get("quality_findings", []):
            lines.append(f"- **{item.get('title')}** ({item.get('severity', 'unspecified')}): {item.get('detail')}")
        if not plan.get("quality_findings"):
            lines.append("No automatically detected high-priority data-quality issue was recorded.")

        lines += ["", "## Findings that survived", ""]
        if not survived:
            lines.append("No candidate survived every configured attack.")
        for idx, finding in enumerate(survived, start=1):
            candidate = finding.get("candidate", {})
            verdict = finding.get("verdict", {})
            lines += [
                f"### {idx}. {candidate.get('statement', candidate.get('id'))}",
                "",
                f"- **Final status:** {finding.get('final_status')}",
                f"- **Held-out score:** {_fmt(verdict.get('score'))}",
                f"- **Baseline:** {_fmt(verdict.get('detail', {}).get('baseline'))}",
                f"- **Checks survived:** {', '.join(finding.get('survived', [])) or 'none'}",
                f"- **Candidate type:** {candidate.get('payload', {}).get('kind', 'unspecified')}",
            ]
            for attack in finding.get("falsifications", []):
                lines.append(
                    f"- **{attack.get('name')} check:** {'failed' if attack.get('killed') else 'passed'}; "
                    f"metric={_fmt(attack.get('metric'))}"
                )
            lines += [
                "- **Interpretation:** This relation remained predictive or structured in the locked confirmation checks used here. It does not establish causation.",
                "- **Recommended next test:** Repeat the frozen candidate on an independent dataset; inspect subgroup consistency, outliers, measurement construction, and plausible confounders.",
                "",
            ]

        lines += ["## Candidates that failed or remained unresolved", ""]
        if not failed:
            lines.append("None.")
        for finding in failed:
            killed_by = [a.get("name") for a in finding.get("falsifications", []) if a.get("killed")]
            lines.append(
                f"- **{finding.get('candidate', {}).get('statement', finding.get('candidate', {}).get('id'))}** — "
                f"status `{finding.get('final_status')}`, score {_fmt(finding.get('verdict', {}).get('score'))}; "
                f"failed: {', '.join(killed_by) or 'did not reach the governed threshold'}."
            )

        lines += [
            "",
            "## Assumptions, limitations, and what is not established",
            "",
            "- Candidate generation was exploratory, so the number of candidates tested matters.",
            "- The confirmation partition is internal held-out evidence, not an external replication cohort.",
            "- Linear and group-level screens can miss nonlinear, temporal, causal, or domain-specific structure.",
            "- Rows were treated as independent unless the analysis plan states otherwise.",
            "- Source documents are preserved as provenance and context in v0.1; they do not silently override the dataset.",
            "- A surviving association should be interpreted in the researcher’s domain before any mechanistic or practical claim is made.",
            "",
            "## Highest-value next tests",
            "",
            "1. Re-run the frozen surviving candidates on independently collected data.",
            "2. Add known confounders and repeated-measure structure through a domain-specific analysis plug-in.",
            "3. Use negative controls and measurement-construction checks to identify definitional or derived-variable artifacts.",
            "4. Convert the strongest surviving relation into a preregistered intervention or discriminating experiment where scientifically appropriate.",
            "",
            "## Belief graph and provenance receipt",
            "",
            f"- Persistent claims linked to this case: {len(claim_rows)}",
            f"- Open re-examination tasks: {len(reexamination)}",
            f"- Engine run ID: `{result.get('run_id', '—')}`",
            f"- Hash-ledger path: `{result.get('ledger_path', '—')}`",
            f"- Graph snapshot ID: `{result.get('graph_snapshot_id', '—')}`",
            "",
        ]
        if claim_rows:
            lines.append("| Claim ID | Status | Finding type | Statement |")
            lines.append("|---|---|---|---|")
            for row in claim_rows:
                lines.append(
                    f"| `{row['claim_id']}` | {row['status']} | {row['finding_type']} | {row['canonical_text'].replace('|', '\\|')} |"
                )
        if reexamination:
            lines += ["", "### Re-examination queue", ""]
            for item in reexamination:
                lines.append(
                    f"- `{item['id']}` — **{item['impact']}**: {item['canonical_text']} ({item['reason']})"
                )
        lines += [
            "",
            "## Machine-readable appendix",
            "",
            "The case directory also contains the approved plan, raw engine result, hash-chained ledger, graph snapshot, and JSON report data.",
        ]
        return "\n".join(lines) + "\n"

    def markdown_to_html(self, markdown_text: str, *, title: str) -> str:
        # Dependency-free, deliberately conservative rendering for the local MVP.
        blocks: list[str] = []
        in_table = False
        table_rows: list[list[str]] = []
        for raw in markdown_text.splitlines():
            line = raw.rstrip()
            if line.startswith("|" ) and line.endswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if all(set(cell) <= {"-", ":"} for cell in cells):
                    continue
                table_rows.append(cells)
                in_table = True
                continue
            if in_table:
                blocks.append(self._render_table(table_rows))
                table_rows = []
                in_table = False
            if line.startswith("### "):
                blocks.append(f"<h3>{_escape(line[4:])}</h3>")
            elif line.startswith("## "):
                blocks.append(f"<h2>{_escape(line[3:])}</h2>")
            elif line.startswith("# "):
                blocks.append(f"<h1>{_escape(line[2:])}</h1>")
            elif line.startswith("> "):
                blocks.append(f"<blockquote>{_escape(line[2:])}</blockquote>")
            elif line.startswith("- "):
                blocks.append(f"<p class='bullet'>• {self._inline(line[2:])}</p>")
            elif line and len(line) > 3 and line[0].isdigit() and line[1:3] == ". ":
                blocks.append(f"<p class='bullet'>{self._inline(line)}</p>")
            elif line:
                blocks.append(f"<p>{self._inline(line)}</p>")
        if table_rows:
            blocks.append(self._render_table(table_rows))
        return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{_escape(title)}</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:#0b1020;color:#e9eef8;margin:0;line-height:1.6}}
main{{max-width:980px;margin:0 auto;padding:48px 28px 90px}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:42px;border-top:1px solid #26314a;padding-top:24px}}h3{{margin-top:28px}}
p{{color:#c9d3e4}}strong{{color:#fff}}code{{background:#151e32;padding:2px 5px;border-radius:5px;color:#9ee7ff}}
blockquote{{border-left:4px solid #65d6ff;padding:12px 18px;background:#121a2c;color:#dce7f5}}
.bullet{{margin:5px 0}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:18px 0}}th,td{{border:1px solid #2a3753;padding:9px;text-align:left;vertical-align:top}}th{{background:#151e32}}
</style></head><body><main>{''.join(blocks)}</main></body></html>"""

    def write_bundle(
        self,
        output_dir: str | Path,
        *,
        case: dict[str, Any],
        plan: dict[str, Any],
        result: dict[str, Any],
        claim_rows: list[dict[str, Any]],
        reexamination: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        markdown = self.build_markdown(case=case, plan=plan, result=result, claim_rows=claim_rows, reexamination=reexamination)
        html_text = self.markdown_to_html(markdown, title=f"Orbita Research Dossier — {case['name']}")
        machine = {
            "case": case,
            "plan": plan,
            "result": result,
            "claims": claim_rows,
            "reexamination": reexamination,
        }
        paths = {
            "markdown": output / "research_dossier.md",
            "html": output / "research_dossier.html",
            "json": output / "research_dossier.json",
            "plan": output / "approved_plan.json",
            "result": output / "engine_result.json",
        }
        paths["markdown"].write_text(markdown, encoding="utf-8")
        paths["html"].write_text(html_text, encoding="utf-8")
        paths["json"].write_text(json.dumps(machine, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        paths["plan"].write_text(json.dumps(plan, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        paths["result"].write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        bundle: dict[str, dict[str, Any]] = {}
        for role, path in paths.items():
            raw = path.read_bytes()
            bundle[role] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        return bundle

    def _inline(self, text: str) -> str:
        escaped = _escape(text)
        # Minimal inline formatting after escaping.
        while "**" in escaped:
            start = escaped.find("**")
            end = escaped.find("**", start + 2)
            if end < 0:
                break
            escaped = escaped[:start] + "<strong>" + escaped[start + 2 : end] + "</strong>" + escaped[end + 2 :]
        while "`" in escaped:
            start = escaped.find("`")
            end = escaped.find("`", start + 1)
            if end < 0:
                break
            escaped = escaped[:start] + "<code>" + escaped[start + 1 : end] + "</code>" + escaped[end + 1 :]
        return escaped

    def _render_table(self, rows: list[list[str]]) -> str:
        if not rows:
            return ""
        head = "".join(f"<th>{self._inline(cell)}</th>" for cell in rows[0])
        body = "".join(
            "<tr>" + "".join(f"<td>{self._inline(cell)}</td>" for cell in row) + "</tr>"
            for row in rows[1:]
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
