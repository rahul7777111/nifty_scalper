"""Institutional Reporting Engine.

Provides automated PDF report generation using ReportLab, complete with key stress test metrics,
Strategy Health and Survivability scores, and embedded Matplotlib charts.
Also supports fully styled Excel, CSV, and JSON exports.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ReportLab imports for generating highly styled PDFs
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class ReportingEngine:
    """Automated report generation suite for quantitative risk intelligence."""

    def __init__(self, metrics: Dict[str, Any], trades: Union[List[float], np.ndarray]):
        self.metrics = metrics
        self.trades = np.asarray(trades, dtype=np.float64)

    def export_json(self, file_path: str) -> None:
        """Dumps the metrics dictionary to a JSON file."""
        # Remove any non-serializable objects (like numpy arrays)
        clean = {}
        for k, v in self.metrics.items():
            if isinstance(v, (np.ndarray, list)):
                continue  # skip raw array paths
            if isinstance(v, dict):
                clean[k] = {sk: sv for sk, sv in v.items() if not isinstance(sv, np.ndarray)}
            else:
                clean[k] = v
                
        clean["timestamp"] = datetime.now().isoformat()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=4)

    def export_csv(self, file_path: str) -> None:
        """Exports the trade list and rolling P&L statistics to a CSV."""
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Trade Index", "PnL", "Cumulative PnL"])
            cum_pnl = 0.0
            for idx, val in enumerate(self.trades):
                cum_pnl += val
                writer.writerow([idx + 1, val, cum_pnl])

    def export_excel(self, file_path: str) -> None:
        """Generates a professional, fully styled Excel sheet using openpyxl."""
        # Create DataFrames
        trades_df = pd.DataFrame({
            "Trade Index": range(1, len(self.trades) + 1),
            "Trade PnL (₹)": self.trades,
            "Cumulative PnL (₹)": np.cumsum(self.trades)
        })

        clean_metrics = []
        for k, v in self.metrics.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if not isinstance(sv, (np.ndarray, list)):
                        clean_metrics.append({"Metric": f"{k}_{sk}", "Value": sv})
            elif not isinstance(v, (np.ndarray, list)):
                clean_metrics.append({"Metric": k, "Value": v})
                
        metrics_df = pd.DataFrame(clean_metrics)

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            trades_df.to_excel(writer, sheet_name="Trade Log", index=False)
            metrics_df.to_excel(writer, sheet_name="Risk Metrics", index=False)

    def calculate_health_and_survivability_scores(self) -> Tuple[float, float, str]:
        """Calculates a custom Strategy Health Score and Account Survivability Score.

        Health Score: Max 100, based on Profit Factor, Win Rate, and Calmar Ratio.
        Survivability Score: Max 100, based on Risk of Ruin and CVaR / Capital.
        """
        # Profit Factor Component
        pf = self.metrics.get("profit_factor", 1.0)
        if pf == float('inf'):
            pf_score = 30.0
        else:
            pf_score = min(30.0, (pf / 2.0) * 30.0)

        # Win Rate Component
        wr = self.metrics.get("win_rate", 0.50)
        wr_score = (wr / 0.70) * 30.0 if wr < 0.70 else 30.0

        # Calmar Ratio Component
        calmar = self.metrics.get("calmar_ratio", 0.0)
        calmar_score = min(40.0, (calmar / 3.0) * 40.0) if calmar > 0 else 0.0

        health_score = float(max(10.0, min(100.0, pf_score + wr_score + calmar_score)))

        # Survivability Score: Capital adequacy, drawdown depth, ruin risk
        ruin_info = self.metrics.get("ruin", {})
        ruin_prob = ruin_info.get("ruin_probability", 0.0)
        ruin_penalty = ruin_prob * 60.0

        max_dd = self.metrics.get("max_drawdown_percent", 0.0)
        dd_penalty = (max_dd / 0.50) * 40.0 if max_dd < 0.50 else 40.0

        survivability_score = float(max(0.0, min(100.0, 100.0 - ruin_penalty - dd_penalty)))

        if health_score >= 80.0 and survivability_score >= 80.0:
            rating = "Hedge-Fund Quality (Institutional Grade)"
        elif health_score >= 50.0 and survivability_score >= 60.0:
            rating = "Robust Retail Strategy (Moderate Risk)"
        else:
            rating = "Substandard / Fragile (Dangerous Risk of Capital Loss)"

        return health_score, survivability_score, rating

    def generate_pdf_report(self, file_path: str) -> None:
        """Generates a professional hedge-fund-grade PDF risk report using ReportLab.

        Features:
            1. Executive summary header.
            2. Styled tables of core financial statistics.
            3. Detailed stress and black swan analysis sections.
            4. Strategy Health & Survivability scores.
            5. Embedded matplotlib scenario/drawdown charts.
        """
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        
        # Define Custom Color Palette (Navy Deep, Muted Gold, Dark Charcoal)
        primary_color = colors.HexColor("#1A365D")    # Dark Navy
        secondary_color = colors.HexColor("#2B6CB0")  # Slate Blue
        neutral_dark = colors.HexColor("#2D3748")     # Slate Charcoal
        neutral_light = colors.HexColor("#EDF2F7")    # Gray Light
        accent_red = colors.HexColor("#C53030")       # Dark Crimson

        # Custom Paragraph Styles
        styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=primary_color,
            spaceAfter=12
        ))
        
        styles.add(ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=secondary_color,
            spaceAfter=20
        ))
        
        styles.add(ParagraphStyle(
            name="SectionHeadingStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=primary_color,
            spaceBefore=15,
            spaceAfter=8
        ))

        styles.add(ParagraphStyle(
            name="ScoreStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=accent_red,
            spaceAfter=5
        ))

        story = []

        # 1. Header Section
        title = "STRATEGY RISK & STRESS TEST REPORT"
        story.append(Paragraph(title, styles["ReportTitle"]))
        
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subtitle = f"Generated by Antigravity Risk Analysis Engine  |  Date: {timestamp_str}  |  Institutional Grade"
        story.append(Paragraph(subtitle, styles["ReportSubtitle"]))
        story.append(Spacer(1, 10))

        # 2. Strategy Health & Survivability Metrics Block
        health, survivability, rating = self.calculate_health_and_survivability_scores()
        
        score_html = (
            f"<b>STRATEGY HEALTH SCORE:</b> {health:.1f}/100 &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>ACCOUNT SURVIVABILITY SCORE:</b> {survivability:.1f}/100<br/>"
            f"<b>PORTFOLIO RATING:</b> <font color='#C53030'>{rating}</font>"
        )
        story.append(Paragraph(score_html, styles["ScoreStyle"]))
        story.append(Spacer(1, 10))

        # 3. Core Metrics Table
        story.append(Paragraph("I. Key Performance Indicators (KPIs)", styles["SectionHeadingStyle"]))
        
        # Prepare table data
        m = self.metrics
        kpi_data = [
            [
                Paragraph("<b>Metric</b>", styles["Normal"]),
                Paragraph("<b>Value</b>", styles["Normal"]),
                Paragraph("<b>Metric</b>", styles["Normal"]),
                Paragraph("<b>Value</b>", styles["Normal"])
            ],
            [
                "Total Trades", str(m.get("total_trades", 0)),
                "Win Rate", f"{m.get('win_rate', 0.0) * 100:.2f}%",
            ],
            [
                "Total Profit (₹)", f"₹{m.get('total_profit', 0.0):,.2f}",
                "Max Drawdown %", f"{m.get('max_drawdown_percent', 0.0) * 100:.2f}%",
            ],
            [
                "Sharpe Ratio", f"{m.get('sharpe_ratio', 0.0):.2f}",
                "Sortino Ratio", f"{m.get('sortino_ratio', 0.0):.2f}",
            ],
            [
                "Calmar Ratio", f"{m.get('calmar_ratio', 0.0):.2f}",
                "Profit Factor", f"{m.get('profit_factor', 0.0):.2f}",
            ],
            [
                "Ulcer Index", f"{m.get('ulcer_index', 0.0):.2f}",
                "Conditional VaR (₹)", f"₹{m.get('cvar_95', 0.0):,.2f}",
            ]
        ]

        t_kpis = Table(kpi_data, colWidths=[140, 120, 140, 120])
        t_kpis.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), neutral_light),
            ('TEXTCOLOR', (0, 0), (-1, 0), primary_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, neutral_light]),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(t_kpis)
        story.append(Spacer(1, 15))

        # 4. Stress Test and Ruin Risk
        story.append(Paragraph("II. Advanced Stress & Black Swan Analysis", styles["SectionHeadingStyle"]))
        
        ruin = m.get("ruin", {})
        fat_tail = m.get("fat_tail", {})
        
        stress_data = [
            [
                Paragraph("<b>Risk Vector</b>", styles["Normal"]),
                Paragraph("<b>Stress Insight / Indicator</b>", styles["Normal"]),
                Paragraph("<b>Risk Metric</b>", styles["Normal"])
            ],
            [
                "Ruin Probability",
                "Probability of capital breaching a 50% drawdown threshold",
                f"{ruin.get('ruin_probability', 0.0) * 100:.2f}%"
            ],
            [
                "Expected Recovery",
                "Estimated number of trades required to recover from maximum peak-to-trough drawdown",
                f"{ruin.get('expected_recovery_trades', 0.0):.1f} trades"
            ],
            [
                "Capital Adequacy Ratio",
                "Ratio of total starting capital to extreme Value at Risk",
                f"{ruin.get('capital_adequacy_ratio', 0.0):.2f}"
            ],
            [
                "Fat-Tail Detection",
                f"Excess Kurtosis: {fat_tail.get('excess_kurtosis', 0.0):.2f} (Kurtosis > 0.0 indicates heavy tails)",
                "Leptokurtic (Fat-Tailed)" if fat_tail.get("is_fat_tailed", False) else "Normal Distribution"
            ]
        ]

        t_stress = Table(stress_data, colWidths=[120, 300, 100])
        t_stress.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), neutral_light),
            ('TEXTCOLOR', (0, 0), (-1, 0), primary_color),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, neutral_light]),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        story.append(t_stress)
        story.append(Spacer(1, 15))

        # 5. Charts Section (Render and Save a high-resolution Matplotlib plot to embed)
        story.append(Paragraph("III. Visual Performance Distribution", styles["SectionHeadingStyle"]))
        
        chart_path = os.path.join(os.path.dirname(file_path), "report_chart.png")
        
        # Render a dual-panel report chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.5), dpi=300)
        
        # Matplotlib theme adjustments
        fig.patch.set_facecolor("#ffffff")
        ax1.set_facecolor("#fcfcfc")
        ax2.set_facecolor("#fcfcfc")
        
        # Plot 1: Cumulative Equity
        cum_pnl = np.insert(np.cumsum(self.trades), 0, 0.0) + 100000.0
        ax1.plot(cum_pnl, color="#2B6CB0", linewidth=1.5, label="Equity Path")
        ax1.set_title("Strategy Performance (₹)", fontsize=8, fontweight="bold", color="#1A365D")
        ax1.set_xlabel("Trades", fontsize=7)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.tick_params(labelsize=6)
        
        # Plot 2: Drawdown underwater
        peaks = np.maximum.accumulate(cum_pnl)
        dds = (peaks - cum_pnl) / peaks
        ax2.fill_between(range(len(dds)), -dds * 100, 0, color="#C53030", alpha=0.4, label="Drawdown")
        ax2.set_title("Underwater Drawdown (%)", fontsize=8, fontweight="bold", color="#1A365D")
        ax2.set_xlabel("Trades", fontsize=7)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.tick_params(labelsize=6)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

        # Embed Image
        story.append(Image(chart_path, width=520, height=173))
        
        # Build Document
        doc.build(story)
        
        # Clean up temporary chart file safely
        try:
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception:
            pass
