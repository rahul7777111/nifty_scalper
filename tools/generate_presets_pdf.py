import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page counts and premium footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        
        # Draw top banner line
        self.setStrokeColor(colors.HexColor("#2c3e50"))
        self.setLineWidth(1)
        self.line(36, 800, 559, 800)
        
        # Header text
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#34495e"))
        self.drawString(36, 808, "NIFTY-OPTIONS-SCALPER  |  QUANTITATIVE CONFIGURATION REPORT")
        
        # Draw bottom footer line
        self.line(36, 45, 559, 45)
        
        # Footer text
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7f8c8d"))
        self.drawString(36, 30, "CONFIDENTIAL  |  SYSTEM PRESET MECHANICS & IMPORTANCE")
        
        # Page numbering
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(559, 30, page_str)
        self.restoreState()


def build_presets_pdf(output_path):
    # Standard A4 size: 595.27 x 841.89 points
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom high-end styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#2c3e50"),
        alignment=0,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#7f8c8d"),
        spaceAfter=25
    )

    h1_style = ParagraphStyle(
        "H1Style",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2c3e50"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.white
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#2c3e50")
    )

    cell_bold_style = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#2c3e50")
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("System Strategy Presets: Aggressive vs. Conservative", title_style))
    story.append(Paragraph("A Quantitative Reference Guide for GPT Engine Verification & Operational Alignment", subtitle_style))
    story.append(Spacer(1, 10))

    # Executive Overview
    story.append(Paragraph("Executive Overview", h1_style))
    story.append(Paragraph(
        "The <b>NIFTY-OPTIONS-SCALPER</b> utilizes system presets to enforce predefined risk parameters, trading frequencies, "
        "and sizing rules. Understanding the specific mathematical differences between <b>Aggressive</b> and <b>Conservative</b> "
        "modes is crucial for evaluating execution survivability, capital drawdown tolerances, and transaction-cost efficiencies. "
        "This report provides a side-by-side comparison of these options strategy parameters and notes their structural importance "
        "to assist the GPT advisor in rendering high-context recommendations.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Parameter Table
    story.append(Paragraph("Side-by-Side Parameter Comparison", h1_style))
    
    headers = ["Configuration Key / Env Variable", "Aggressive Preset", "Conservative Preset", "Risk & Execution Impact"]
    
    comparisons = [
        ("Timeframe (MSTOCK_TIMEFRAME)", "1m", "3m", "Aggressive uses 1m candles for hyper-fast entry triggers. Conservative filters noise with 3m candles."),
        ("Order Cooldown (MSTOCK_COOLDOWN_SEC)", "20.0s", "45.0s", "Limits spam entries. Aggressive allows rapid-fire execution. Conservative forces a cooling gap."),
        ("Post-Stopout Cooldown (COOLDOWN_AFTER_STOPOUT)", "75.0s", "180.0s", "Halts bot after losing trades to prevent revenge trading. Conservative triples the safety window."),
        ("Max Open Positions (MSTOCK_MAX_OPEN_POSITIONS)", "6", "6", "Hard limit on total inflight trades to cap portfolio exposure."),
        ("Max Order Quantity (ENTRY_MAX_SAME_DIRECTION_QTY)", "600 lots", "150 lots", "Caps maximum transaction lot sizes. Conservative limits execution capacity to 25% of Aggressive."),
        ("Pyramiding Levels (MSTOCK_MAX_PYRAMID_LEVELS)", "3 Levels", "1 Level (Disabled)", "Allows scale-in on profitable runs. Conservative disables multiple scale-in legs to save margin."),
        ("Premium Trail % (MSTOCK_DIR_PREMIUM_TRAIL_PCT)", "7.0%", "10.0%", "Aggressive uses a tighter trail (7%) to lock profits. Conservative allows a wider 10% breathing room."),
        ("Partial Target Multi (DIR_PARTIAL_TARGET_MULT)", "1.2 * ATR", "1.0 * ATR", "ATR multiple for partial profit taking. Conservative locks profits earlier at 1.0x ATR."),
        ("Partial Qty Pct (MSTOCK_DIR_PARTIAL_QTY_PCT)", "35%", "25%", "Portion of position liquidated at the partial target to protect core positions."),
        ("Risk Scale Stopout (RISK_SCALE_STOPOUT_FACTOR)", "0.90", "0.75", "Scales down sizing after losses. Conservative penalizes size heavier (reduces by 25%) to preserve capital."),
        ("Risk Sizer Recovery Wins (RISK_SCALE_RECOVERY_WINS)", "1 Win", "2 Wins", "Consecutive profitable trades required to restore full lot sizes. Conservative forces double verification."),
        ("ATR High Vol Threshold (RISK_SCALE_ATR_HIGH)", "85.0", "70.0", "ATR value above which high-vol sizing penalty triggers. Conservative penalizes at a lower threshold."),
        ("High Vol Size Penalty (RISK_SCALE_ATR_HIGH_FACTOR)", "0.90", "0.80", "Lot size penalty multiplier during high vol. Conservative cuts size by 20% vs. 10% on Aggressive."),
        ("Dynamic Pyramiding (MSTOCK_ENABLE_DYNAMIC_PYRAMIDING)", "True", "False", "Allows algorithmic adjustment of add-on quantities. Disabled on Conservative."),
        ("MTF Hard Filter (MSTOCK_MTF_HARD_FILTER)", "True", "False", "Aggressive enforces strict multi-timeframe candle alignment before entry. Conservative allows single-frame breakout entries."),
        ("Theta Stop Widen Max (THETA_STOP_WIDEN_MAX_MULT)", "2.5x", "1.8x", "Expands stop-losses to accommodate time-decay breathing room. Conservative strictly caps widening to 1.8x."),
        ("Theta Stop Widen Threshold (THETA_STOP_WIDEN_THRESHOLD)", "-8.0", "-4.0", "Point of options time decay acceleration where stop widening triggers. Conservative triggers earlier."),
        ("Exit Ladder Steps (MSTOCK_EXIT_LADDER_STEPS)", "0.25:0.8, 0.25:1.2, 0.30:1.8", "0.20:1.0, 0.20:1.5, 0.20:2.0", "Incremental exit target rules. Conservative scales out smaller portions at higher targets."),
        ("Delta Hedge Tol Factor (DELTA_HEDGE_VOL_LOW/HIGH)", "0.6 to 1.6", "0.8 to 1.3", "Dynamic Greeks neutral corridor. Conservative maintains a tighter delta-neutral band to minimize exposure.")
    ]

    table_data = [[Paragraph(text, th_style) for text in headers]]
    for key, agg, cons, desc in comparisons:
        table_data.append([
            Paragraph(key, cell_bold_style),
            Paragraph(agg, cell_style),
            Paragraph(cons, cell_style),
            Paragraph(desc, cell_style)
        ])

    comp_table = Table(table_data, colWidths=[120, 70, 70, 260])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 15))

    # Analytical Section on Importance
    story.append(Paragraph("Strategic Importance of Presets (For GPT Advisor Queries)", h1_style))
    story.append(Paragraph(
        "When prompting or consulting the GPT advisor regarding system behaviors, these presets serve as the foundational constraints "
        "defining the bot's risk appetite. Here is why this parameter split is extremely critical:",
        body_style
    ))
    
    bullets = [
        "<b>Market Regime Adaptability:</b> In highly trending, directional regimes, the <i>Aggressive</i> preset maximizes efficiency by scaling in (pyramiding up to 3 levels) and keeping order cooldowns low (20s). However, in volatile, range-bound regimes, these same settings can cause rapid-fire stopouts and severe commission slippage.",
        "<b>Transaction and Slippage Friction:</b> With a max quantity of 600 lots and a tight 7% premium trail, the <i>Aggressive</i> preset is vulnerable to transaction costs and market liquidity spikes. The <i>Conservative</i> preset's lower quantity cap (150 lots) and wider trailing stop (10%) ensure trades survive high-spread intervals without getting whipped out.",
        "<b>Time-Decay Failsafes (Theta):</b> The <i>Conservative</i> preset triggers stop widening at a decay score of -4.0 (earlier than Aggressive's -8.0) but strictly restricts widening to 1.8x. This restricts overnight time decay risk on premium holdings, while the Aggressive preset allows a wider 2.5x buffer, expecting high-gamma directional moves to compensate.",
        "<b>Mathematical Drawdown Controls:</b> Sizing recovery in <i>Conservative</i> requires 2 consecutive wins (vs. 1 win in Aggressive) and imposes a 25% size reduction factor (vs. 10%). This is highly optimized for 8GB RAM local laptops, keeping CPU-heavy optimization loops idle and prioritizing operational survival during losing streaks."
    ]

    for b in bullets:
        story.append(Paragraph(f"• {b}", bullet_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)


if __name__ == "__main__":
    out_path = "Strategy_Preset_Parameters_Comparison.pdf"
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    
    print(f"Generating presets comparison PDF at: {out_path}")
    build_presets_pdf(out_path)
    print("PDF generation complete.")
