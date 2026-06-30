"""
Monthly ESG Report Generator
==============================
Builds an executive ESG report from the computed metrics (KPIs, scores,
anomalies, hotspots, recommendations) — all numbers come from the
deterministic/ML modules. The Groq LLM (backend/llm.py) is used ONLY to
write the prose executive summary, grounded in those same numbers, never to
invent figures of its own.

Outputs both a Markdown file and a PDF (via reportlab) to reports/output/.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.queries import get_daily_metrics, compute_kpis
from ml.scoring import compute_scores
from ml.anomaly import detect_anomalies
from ml.hotspot import top_emitters

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _trend_lines(daily_metrics, window_days=30) -> dict:
    """Compare the trailing window vs. the prior window of equal length to surface
    positive/negative trends per facility_type."""
    df = daily_metrics.copy()
    df["date"] = __import__("pandas").to_datetime(df["date"])
    end = df["date"].max()
    mid = end - __import__("pandas").Timedelta(days=window_days)
    start = mid - __import__("pandas").Timedelta(days=window_days)

    recent = df[(df["date"] > mid) & (df["date"] <= end)]
    prior = df[(df["date"] > start) & (df["date"] <= mid)]

    recent_em = recent.groupby("facility_id")["carbon_emissions_kg"].sum()
    prior_em = prior.groupby("facility_id")["carbon_emissions_kg"].sum()

    common = recent_em.index.intersection(prior_em.index)
    pct_change = ((recent_em[common] - prior_em[common]) / prior_em[common].replace(0, 1)) * 100

    improved = pct_change.sort_values().head(5)
    worsened = pct_change.sort_values(ascending=False).head(5)

    return {
        "positive": [(fid, round(float(v), 1)) for fid, v in improved.items() if v < 0],
        "negative": [(fid, round(float(v), 1)) for fid, v in worsened.items() if v > 0],
    }


def build_report_data(window_days: int = 30, use_llm: bool = True) -> dict:
    df = get_daily_metrics()
    if df.empty:
        raise RuntimeError("No data loaded. Run the seed script first.")

    kpis = compute_kpis(df, window_days=window_days)
    scores = compute_scores(df)
    anomalies = detect_anomalies(df[df["date"] >= (df["date"].max() - __import__("pandas").Timedelta(days=window_days))])
    hotspots = top_emitters(df, n=10, window_days=window_days)
    trends = _trend_lines(df, window_days=window_days)

    needs_attention = scores[scores["classification"] == "Needs Attention"].sort_values("overall_score").head(10)

    data = dict(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        period_start=kpis.get("period_start"),
        period_end=kpis.get("period_end"),
        kpis=kpis,
        top_emitters=hotspots.to_dict("records"),
        anomaly_count=len(anomalies),
        high_severity_anomalies=len(anomalies[anomalies["severity"] == "High"]) if not anomalies.empty else 0,
        facilities_needing_attention=needs_attention.to_dict("records"),
        positive_trends=trends["positive"],
        negative_trends=trends["negative"],
    )

    if use_llm:
        try:
            from backend.llm import generate_executive_summary
            data["executive_summary"] = generate_executive_summary(data)
        except Exception as e:
            data["executive_summary"] = (
                f"[Executive summary unavailable — {e}. Showing computed metrics only below.]"
            )
    else:
        data["executive_summary"] = None

    return data


def render_markdown(data: dict) -> str:
    lines = []
    lines.append(f"# Monthly ESG Report")
    lines.append(f"**Period:** {data['period_start']} to {data['period_end']}  ")
    lines.append(f"**Generated:** {data['generated_at']}\n")

    lines.append("## Executive Summary")
    lines.append(data["executive_summary"] or "_Executive summary not generated (LLM unavailable)._")
    lines.append("")

    k = data["kpis"]
    lines.append("## Key Performance Indicators")
    lines.append(f"- Total Carbon Emissions: **{k.get('total_carbon_emissions_kg', 0):,.1f} kg CO2**")
    lines.append(f"- Average Carbon per Shipment: **{k.get('avg_carbon_per_shipment_kg', 0):,.2f} kg CO2**")
    lines.append(f"- Total Energy Consumption: **{k.get('total_energy_kwh', 0):,.1f} kWh**")
    lines.append(f"- Total Water Consumption: **{k.get('total_water_litres', 0):,.1f} L**")
    lines.append(f"- Total Waste Generated: **{k.get('total_waste_kg', 0):,.1f} kg**")
    lines.append(f"- Recycling Rate: **{k.get('recycling_rate_pct', 0):.1f}%**")
    lines.append(f"- Renewable Energy Usage: **{k.get('renewable_energy_pct', 0):.1f}%**")
    lines.append(f"- Carbon Intensity: **{k.get('carbon_intensity_kg_per_unit', 0):.3f} kg CO2/unit**\n")

    lines.append("## Positive Trends")
    if data["positive_trends"]:
        for fid, pct in data["positive_trends"]:
            lines.append(f"- {fid}: emissions down {abs(pct):.1f}%")
    else:
        lines.append("_No significant improving trends this period._")
    lines.append("")

    lines.append("## Negative Trends")
    if data["negative_trends"]:
        for fid, pct in data["negative_trends"]:
            lines.append(f"- {fid}: emissions up {pct:.1f}%")
    else:
        lines.append("_No significant worsening trends this period._")
    lines.append("")

    lines.append("## Carbon Analysis — Top 10 Emitters")
    lines.append("| Facility | Type | Total Emissions (kg) | Per Shipment (kg) |")
    lines.append("|---|---|---|---|")
    for row in data["top_emitters"]:
        lines.append(f"| {row['facility_id']} | {row['facility_type']} | "
                      f"{row['total_emissions_kg']:,.1f} | {row.get('carbon_per_shipment') or 0:.2f} |")
    lines.append("")

    lines.append("## Anomalies")
    lines.append(f"- Total anomalies detected this period: **{data['anomaly_count']}**")
    lines.append(f"- High severity: **{data['high_severity_anomalies']}**\n")

    lines.append("## Facilities Needing Attention")
    if data["facilities_needing_attention"]:
        lines.append("| Facility | Type | Score | Classification |")
        lines.append("|---|---|---|---|")
        for row in data["facilities_needing_attention"]:
            lines.append(f"| {row['facility_id']} | {row['facility_type']} | "
                          f"{row['overall_score']:.1f} | {row['classification']} |")
    else:
        lines.append("_No facilities currently classified as Needs Attention._")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("See per-facility recommendations via `/api/recommendations/{facility_id}` "
                  "or the Facility View page in the dashboard for detailed, prioritized actions.")

    return "\n".join(lines)


def render_pdf(data: dict, markdown_text: str, output_path: Path):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    h2 = ParagraphStyle("h2custom", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = styles["Normal"]
    story = []

    story.append(Paragraph("Monthly ESG Report", styles["Title"]))
    story.append(Paragraph(f"Period: {data['period_start']} to {data['period_end']}", body))
    story.append(Paragraph(f"Generated: {data['generated_at']}", body))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Executive Summary", h2))
    story.append(Paragraph((data["executive_summary"] or "Not available.").replace("\n", "<br/>"), body))

    k = data["kpis"]
    story.append(Paragraph("Key Performance Indicators", h2))
    kpi_table_data = [
        ["Metric", "Value"],
        ["Total Carbon Emissions", f"{k.get('total_carbon_emissions_kg', 0):,.1f} kg CO2"],
        ["Avg Carbon per Shipment", f"{k.get('avg_carbon_per_shipment_kg', 0):,.2f} kg CO2"],
        ["Total Energy Consumption", f"{k.get('total_energy_kwh', 0):,.1f} kWh"],
        ["Total Water Consumption", f"{k.get('total_water_litres', 0):,.1f} L"],
        ["Total Waste Generated", f"{k.get('total_waste_kg', 0):,.1f} kg"],
        ["Recycling Rate", f"{k.get('recycling_rate_pct', 0):.1f}%"],
        ["Renewable Energy Usage", f"{k.get('renewable_energy_pct', 0):.1f}%"],
        ["Carbon Intensity", f"{k.get('carbon_intensity_kg_per_unit', 0):.3f} kg CO2/unit"],
    ]
    t = Table(kpi_table_data, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5339")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F3")]),
    ]))
    story.append(t)

    story.append(Paragraph("Trends", h2))
    pos = "; ".join(f"{fid} (-{abs(p):.1f}%)" for fid, p in data["positive_trends"]) or "None significant"
    neg = "; ".join(f"{fid} (+{p:.1f}%)" for fid, p in data["negative_trends"]) or "None significant"
    story.append(Paragraph(f"<b>Positive:</b> {pos}", body))
    story.append(Paragraph(f"<b>Negative:</b> {neg}", body))

    story.append(Paragraph("Top 10 Carbon Emitters", h2))
    hot_data = [["Facility", "Type", "Total Emissions (kg)"]]
    for row in data["top_emitters"]:
        hot_data.append([row["facility_id"], row["facility_type"], f"{row['total_emissions_kg']:,.1f}"])
    t2 = Table(hot_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5339")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F3")]),
    ]))
    story.append(t2)

    story.append(Paragraph("Anomalies", h2))
    story.append(Paragraph(
        f"Total anomalies detected: {data['anomaly_count']} "
        f"(High severity: {data['high_severity_anomalies']})", body
    ))

    story.append(Paragraph("Facilities Needing Attention", h2))
    if data["facilities_needing_attention"]:
        att_data = [["Facility", "Type", "Score", "Classification"]]
        for row in data["facilities_needing_attention"]:
            att_data.append([row["facility_id"], row["facility_type"],
                              f"{row['overall_score']:.1f}", row["classification"]])
        t3 = Table(att_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8C2F2F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t3)
    else:
        story.append(Paragraph("No facilities currently classified as Needs Attention.", body))

    doc.build(story)


def generate_report(window_days: int = 30, use_llm: bool = True) -> dict:
    """Build the report and write both .md and .pdf to reports/output/."""
    data = build_report_data(window_days=window_days, use_llm=use_llm)
    md_text = render_markdown(data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = OUTPUT_DIR / f"esg_report_{timestamp}.md"
    pdf_path = OUTPUT_DIR / f"esg_report_{timestamp}.pdf"

    md_path.write_text(md_text)
    render_pdf(data, md_text, pdf_path)

    return {"markdown_path": str(md_path), "pdf_path": str(pdf_path), "data": data}


if __name__ == "__main__":
    result = generate_report(window_days=30, use_llm=False)  # use_llm=False: no network access for Groq in this sandbox
    print(f"Markdown report: {result['markdown_path']}")
    print(f"PDF report: {result['pdf_path']}")
