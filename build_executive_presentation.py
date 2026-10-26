"""
build_executive_presentation.py
Rebuilds SupplyShield_Presentation.pptx with executive-grade typography, layout,
visual hierarchy, and human design aesthetics while strictly preserving 100% of the
original content, names, tasks, examples, and slide sequence.
"""

import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Color Palette: Executive Tech & Consulting ──────────────
DARK_BG       = RGBColor(15, 23, 42)      # Slate 900
LIGHT_BG      = RGBColor(248, 250, 252)   # Slate 50
CARD_WHITE    = RGBColor(255, 255, 255)   # Pure White
CARD_BORDER   = RGBColor(226, 232, 240)   # Slate 200
DARK_CARD_BG  = RGBColor(30, 41, 59)      # Slate 800
DARK_BORDER   = RGBColor(51, 65, 85)      # Slate 700

BRAND_BLUE    = RGBColor(37, 99, 235)     # Royal Blue 600
BRAND_INDIGO  = RGBColor(79, 70, 229)     # Indigo 600
BRAND_TEAL    = RGBColor(13, 148, 136)    # Teal 600
ALERT_AMBER   = RGBColor(217, 119, 6)     # Amber 600
ALERT_RED     = RGBColor(225, 29, 72)     # Rose 600

TEXT_DARK     = RGBColor(15, 23, 42)      # Slate 900
TEXT_MUTED    = RGBColor(71, 85, 105)     # Slate 600
TEXT_LIGHT    = RGBColor(248, 250, 252)   # Slate 50
TEXT_LIGHT_MUT= RGBColor(148, 163, 184)   # Slate 400

FONT_MAIN     = "Segoe UI"
FONT_HEADING  = "Segoe UI"

def set_slide_bg(slide, prs, color):
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = color
    bg_shape.line.fill.background()
    return bg_shape

def add_card(slide, left, top, width, height, fill_color, border_color=None, radius=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_header(slide, kicker_text, title_text, subtitle_text=None, is_dark=False):
    # Kicker
    kicker_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.7), Inches(0.3))
    tf_k = kicker_box.text_frame
    tf_k.word_wrap = True
    tf_k.margin_left = tf_k.margin_top = tf_k.margin_right = tf_k.margin_bottom = 0
    p_k = tf_k.paragraphs[0]
    p_k.text = kicker_text.upper()
    p_k.font.name = FONT_MAIN
    p_k.font.size = Pt(9.5)
    p_k.font.bold = True
    p_k.font.color.rgb = BRAND_BLUE if not is_dark else RGBColor(96, 165, 250)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.65))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_DARK if not is_dark else TEXT_LIGHT

    # Subtitle
    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.7), Inches(0.35))
        tf_s = sub_box.text_frame
        tf_s.word_wrap = True
        tf_s.margin_left = tf_s.margin_top = tf_s.margin_right = tf_s.margin_bottom = 0
        p_s = tf_s.paragraphs[0]
        p_s.text = subtitle_text
        p_s.font.name = FONT_MAIN
        p_s.font.size = Pt(12)
        p_s.font.color.rgb = TEXT_MUTED if not is_dark else TEXT_LIGHT_MUT

def add_footer(slide, current_num, total_num=12, is_dark=False):
    # Rule
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.015)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = DARK_BORDER if is_dark else CARD_BORDER
    rule.line.fill.background()

    # Footer text left
    foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.08), Inches(9.0), Inches(0.3))
    tf = foot_box.text_frame
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "SupplyShield — Multimodal Early-Warning System | Academic Capstone Research"
    p.font.name = FONT_MAIN
    p.font.size = Pt(9)
    p.font.color.rgb = TEXT_LIGHT_MUT if is_dark else RGBColor(148, 163, 184)

    # Footer text right (slide number)
    num_box = slide.shapes.add_textbox(Inches(10.5), Inches(7.08), Inches(2.033), Inches(0.3))
    tf_n = num_box.text_frame
    tf_n.margin_left = tf_n.margin_top = tf_n.margin_right = tf_n.margin_bottom = 0
    p_n = tf_n.paragraphs[0]
    p_n.alignment = PP_ALIGN.RIGHT
    p_n.text = f"{current_num:02d} / {total_num:02d}"
    p_n.font.name = FONT_MAIN
    p_n.font.size = Pt(9)
    p_n.font.bold = True
    p_n.font.color.rgb = BRAND_BLUE if not is_dark else RGBColor(96, 165, 250)

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ============================================================
    # SLIDE 1: Title & Presentation Team
    # ============================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, prs, DARK_BG)

    # Left Hero Column
    hero_card = add_card(s1, Inches(0.8), Inches(0.9), Inches(5.8), Inches(5.7), DARK_CARD_BG, DARK_BORDER)
    # Accent top strip
    accent_strip = add_card(s1, Inches(0.8), Inches(0.9), Inches(5.8), Inches(0.08), BRAND_BLUE)

    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.3), Inches(5.0), Inches(5.0))
    tf = t_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.text = "AMITY UNIVERSITY • CAPSTONE PROJECT"
    p.font.name = FONT_MAIN
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(96, 165, 250)

    p2 = tf.add_paragraph()
    p2.text = "SupplyShield"
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_LIGHT
    p2.space_before = Pt(14)

    p3 = tf.add_paragraph()
    p3.text = "An Explainable Multimodal Early-Warning System for Supply Chain Disruptions"
    p3.font.name = FONT_MAIN
    p3.font.size = Pt(15)
    p3.font.color.rgb = TEXT_LIGHT_MUT
    p3.space_before = Pt(14)

    p4 = tf.add_paragraph()
    p4.text = "Fusing Time-Series Intelligence, Real-Time News NLP, and Geospatial External Events into Calibrated Disruption Risk with Grounded Multi-Tier Evidence."
    p4.font.name = FONT_MAIN
    p4.font.size = Pt(11)
    p4.font.color.rgb = RGBColor(100, 116, 139)
    p4.space_before = Pt(18)

    # Right Team Column
    team_card = add_card(s1, Inches(6.9), Inches(0.9), Inches(5.633), Inches(5.7), DARK_CARD_BG, DARK_BORDER)
    # Accent top strip
    add_card(s1, Inches(6.9), Inches(0.9), Inches(5.633), Inches(0.08), BRAND_INDIGO)

    team_title_box = s1.shapes.add_textbox(Inches(7.3), Inches(1.3), Inches(4.8), Inches(0.4))
    tf_tt = team_title_box.text_frame
    tf_tt.margin_left = tf_tt.margin_top = tf_tt.margin_right = tf_tt.margin_bottom = 0
    p_tt = tf_tt.paragraphs[0]
    p_tt.text = "PRESENTED BY:"
    p_tt.font.name = FONT_MAIN
    p_tt.font.size = Pt(11)
    p_tt.font.bold = True
    p_tt.font.color.rgb = RGBColor(148, 163, 184)

    team_members = [
        ("1", "Anurag Dubey", "A010175625008", "System Lead & Integration"),
        ("2", "Shubham Kumar Bind", "A010175625009", "Time-Series Intelligence"),
        ("3", "Abhigyan Balodi", "A010175625011", "News Intelligence NLP"),
        ("4", "Krishna Sagar", "A010175625012", "Event & Spatial Risk"),
        ("5", "Harshita Rohilla", "A010175625013", "Multimodal Early Warning"),
    ]

    card_y = 1.85
    for num, name, roll, role in team_members:
        # Mini profile card
        add_card(s1, Inches(7.3), Inches(card_y), Inches(4.833), Inches(0.8), RGBColor(15, 23, 42), DARK_BORDER)
        # Badge circle shape
        badge = add_card(s1, Inches(7.45), Inches(card_y + 0.18), Inches(0.44), Inches(0.44), BRAND_BLUE)
        badge_tf = badge.text_frame
        badge_tf.margin_left = badge_tf.margin_top = badge_tf.margin_right = badge_tf.margin_bottom = 0
        bp = badge_tf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        bp.text = num
        bp.font.name = FONT_MAIN
        bp.font.size = Pt(11)
        bp.font.bold = True
        bp.font.color.rgb = TEXT_LIGHT

        # Name and Roll
        mem_box = s1.shapes.add_textbox(Inches(8.05), Inches(card_y + 0.14), Inches(3.9), Inches(0.55))
        mtf = mem_box.text_frame
        mtf.margin_left = mtf.margin_top = mtf.margin_right = mtf.margin_bottom = 0
        mp1 = mtf.paragraphs[0]
        mp1.text = f"{name} — {roll}"
        mp1.font.name = FONT_MAIN
        mp1.font.size = Pt(12)
        mp1.font.bold = True
        mp1.font.color.rgb = TEXT_LIGHT

        mp2 = mtf.add_paragraph()
        mp2.text = role
        mp2.font.name = FONT_MAIN
        mp2.font.size = Pt(10)
        mp2.font.color.rgb = RGBColor(148, 163, 184)

        card_y += 0.92

    add_footer(s1, 1, 12, is_dark=True)

    # ============================================================
    # SLIDE 2: Introduction
    # ============================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2, prs, LIGHT_BG)
    add_header(s2, "01 / FOUNDATION & CONTEXT", "Introduction",
               "Supply chains are global, complex, and vulnerable to cascading disruptions.")

    # Left Column: Problem & Context
    add_card(s2, Inches(0.8), Inches(1.85), Inches(5.6), Inches(4.9), CARD_WHITE, CARD_BORDER)
    # Accent strip
    add_card(s2, Inches(0.8), Inches(1.85), Inches(5.6), Inches(0.06), BRAND_BLUE)

    t_box = s2.shapes.add_textbox(Inches(1.15), Inches(2.15), Inches(4.9), Inches(4.3))
    tf = t_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.text = "THE OPERATIONAL REALITY"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = BRAND_BLUE

    p2 = tf.add_paragraph()
    p2.text = "Supply chains help move products from suppliers to customers. However, problems like floods, strikes, supplier failures, and delivery delays can interrupt this process."
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(13)
    p2.font.color.rgb = TEXT_DARK
    p2.space_before = Pt(14)

    # Warning Box
    add_card(s2, Inches(1.15), Inches(4.6), Inches(4.9), Inches(1.8), RGBColor(254, 242, 242), RGBColor(254, 202, 202))
    w_box = s2.shapes.add_textbox(Inches(1.35), Inches(4.8), Inches(4.5), Inches(1.4))
    wtf = w_box.text_frame
    wtf.word_wrap = True
    wtf.margin_left = wtf.margin_top = wtf.margin_right = wtf.margin_bottom = 0
    wp1 = wtf.paragraphs[0]
    wp1.text = "CRITICAL GAP IN TRADITIONAL SYSTEMS"
    wp1.font.name = FONT_MAIN
    wp1.font.size = Pt(9.5)
    wp1.font.bold = True
    wp1.font.color.rgb = ALERT_RED

    wp2 = wtf.add_paragraph()
    wp2.text = "Traditional systems often identify problems after they have already happened, leaving organizations reactive rather than proactive."
    wp2.font.name = FONT_MAIN
    wp2.font.size = Pt(12)
    wp2.font.color.rgb = RGBColor(159, 18, 57)
    wp2.space_before = Pt(8)

    # Right Column: SupplyShield Aims To
    add_card(s2, Inches(6.75), Inches(1.85), Inches(5.783), Inches(4.9), CARD_WHITE, CARD_BORDER)
    add_card(s2, Inches(6.75), Inches(1.85), Inches(5.783), Inches(0.06), BRAND_TEAL)

    r_box = s2.shapes.add_textbox(Inches(7.1), Inches(2.15), Inches(5.1), Inches(0.4))
    rtf = r_box.text_frame
    rtf.margin_left = rtf.margin_top = rtf.margin_right = rtf.margin_bottom = 0
    rp = rtf.paragraphs[0]
    rp.text = "SUPPLYSHIELD AIMS TO:"
    rp.font.name = FONT_MAIN
    rp.font.size = Pt(11)
    rp.font.bold = True
    rp.font.color.rgb = BRAND_TEAL

    aims = [
        "Identify possible disruptions early.",
        "Analyse information from different sources.",
        "Predict supply-chain risk.",
        "Explain why a warning was generated.",
        "Display results on an interactive dashboard."
    ]

    aim_y = 2.65
    for idx, aim in enumerate(aims, 1):
        add_card(s2, Inches(7.1), Inches(aim_y), Inches(5.1), Inches(0.72), RGBColor(241, 245, 249), CARD_BORDER)
        badge = add_card(s2, Inches(7.25), Inches(aim_y + 0.16), Inches(0.4), Inches(0.4), BRAND_BLUE)
        btf = badge.text_frame
        btf.margin_left = btf.margin_top = btf.margin_right = btf.margin_bottom = 0
        bp = btf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        bp.text = str(idx)
        bp.font.name = FONT_MAIN
        bp.font.size = Pt(11)
        bp.font.bold = True
        bp.font.color.rgb = TEXT_LIGHT

        abox = s2.shapes.add_textbox(Inches(7.8), Inches(aim_y + 0.16), Inches(4.25), Inches(0.45))
        atf = abox.text_frame
        atf.word_wrap = True
        atf.margin_left = atf.margin_top = atf.margin_right = atf.margin_bottom = 0
        ap = atf.paragraphs[0]
        ap.text = aim
        ap.font.name = FONT_MAIN
        ap.font.size = Pt(12)
        ap.font.bold = True
        ap.font.color.rgb = TEXT_DARK

        aim_y += 0.82

    add_footer(s2, 2, 12)

    # ============================================================
    # SLIDE 3: Problem Statement and Objectives
    # ============================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3, prs, LIGHT_BG)
    add_header(s3, "02 / MOTIVATION & SCOPE", "Problem Statement and Objectives",
               "Bridging the latency and black-box gap in modern supply chain management.")

    # Top Problem Banner
    add_card(s3, Inches(0.8), Inches(1.8), Inches(11.733), Inches(1.15), RGBColor(254, 242, 242), RGBColor(254, 202, 202))
    add_card(s3, Inches(0.8), Inches(1.8), Inches(0.1), Inches(1.15), ALERT_RED)

    prob_box = s3.shapes.add_textbox(Inches(1.1), Inches(1.95), Inches(11.2), Inches(0.9))
    ptf = prob_box.text_frame
    ptf.word_wrap = True
    ptf.margin_left = ptf.margin_top = ptf.margin_right = ptf.margin_bottom = 0
    p1 = ptf.paragraphs[0]
    p1.text = "CORE PROBLEM STATEMENT"
    p1.font.name = FONT_MAIN
    p1.font.size = Pt(10)
    p1.font.bold = True
    p1.font.color.rgb = ALERT_RED

    p2 = ptf.add_paragraph()
    p2.text = "Problem: Companies often receive warnings too late, and some AI systems do not clearly explain their predictions."
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(159, 18, 57)
    p2.space_before = Pt(4)

    # 6 Objectives Grid (2 rows x 3 columns)
    objs = [
        ("1", "Analyse sales, inventory, and delivery information.", "Time-series behavioral analysis & lag modeling"),
        ("2", "Understand supply-chain-related news.", "NLP extraction of real-time supplier & port events"),
        ("3", "Identify risks from weather and external events.", "Spatial mapping of floods, storms, and geopolitical delays"),
        ("4", "Combine all the information to predict disruptions.", "Multimodal cross-attention fusion architecture"),
        ("5", "Explain predictions with supporting evidence.", "SHAP attribution & grounded multi-tier evidence"),
        ("6", "Display risk alerts through a dashboard.", "Executive control center with scenario playground"),
    ]

    start_x = 0.8
    start_y = 3.15
    col_w = 3.75
    gap_x = 0.24
    row_h = 1.7
    gap_y = 0.2

    for idx, (num, title_txt, sub_txt) in enumerate(objs):
        row = idx // 3
        col = idx % 3
        cx = start_x + col * (col_w + gap_x)
        cy = start_y + row * (row_h + gap_y)

        # Card
        add_card(s3, Inches(cx), Inches(cy), Inches(col_w), Inches(row_h), CARD_WHITE, CARD_BORDER)
        # Top colored accent
        add_card(s3, Inches(cx), Inches(cy), Inches(col_w), Inches(0.04), BRAND_BLUE)

        # Number badge
        badge = add_card(s3, Inches(cx + 0.2), Inches(cy + 0.2), Inches(0.36), Inches(0.36), BRAND_BLUE)
        btf = badge.text_frame
        btf.margin_left = btf.margin_top = btf.margin_right = btf.margin_bottom = 0
        bp = btf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        bp.text = num
        bp.font.name = FONT_MAIN
        bp.font.size = Pt(11)
        bp.font.bold = True
        bp.font.color.rgb = TEXT_LIGHT

        # Text box
        obox = s3.shapes.add_textbox(Inches(cx + 0.68), Inches(cy + 0.18), Inches(col_w - 0.85), Inches(row_h - 0.3))
        otf = obox.text_frame
        otf.word_wrap = True
        otf.margin_left = otf.margin_top = otf.margin_right = otf.margin_bottom = 0
        op1 = otf.paragraphs[0]
        op1.text = title_txt
        op1.font.name = FONT_MAIN
        op1.font.size = Pt(11.5)
        op1.font.bold = True
        op1.font.color.rgb = TEXT_DARK

        op2 = otf.add_paragraph()
        op2.text = sub_txt
        op2.font.name = FONT_MAIN
        op2.font.size = Pt(9.5)
        op2.font.color.rgb = TEXT_MUTED
        op2.space_before = Pt(4)

    add_footer(s3, 3, 12)

    # ============================================================
    # SLIDE 4: System Architecture
    # ============================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4, prs, LIGHT_BG)
    add_header(s4, "03 / END-TO-END PIPELINE", "System Architecture",
               "Modular multi-tiered architecture connecting ingestion, specialized intelligence, fusion, and explainability.")

    # 5 Horizontal Pipeline Stages
    stages = [
        ("STAGE 1: INGESTION", "Input Data",
         "Sales, inventory, news, weather and location data",
         BRAND_BLUE, ["Daily ERP transactions", "Global RSS news streams", "Weather & geo-event logs"]),

        ("STAGE 2: EXTRACTION", "Intelligence Engines",
         "Specialized Feature Processors",
         BRAND_INDIGO, ["• Time-Series (Inventory & Delay)", "• News NLP (Event Sentiment)", "• Event Risk (Spatial & Weather)"]),

        ("STAGE 3: FUSION", "Multimodal Fusion and Prediction",
         "Cross-Modal Representation & Alignment",
         BRAND_TEAL, ["Feature Alignment & Normalization", "Cross-Attention Modality Bundling", "Calibrated Disruption Classifier"]),

        ("STAGE 4: VERIFICATION", "Explainability and Integration",
         "Transparent Attribution & Grounding",
         ALERT_AMBER, ["SHAP Permutation Explainer", "Multi-Tier Consistency Verifier", "Ensemble Uncertainty Estimation"]),

        ("STAGE 5: PRESENTATION", "Early-Warning Dashboard",
         "Risk, evidence and explanation",
         ALERT_RED, ["Executive Risk Monitoring", "Interactive Evidence Inspector", "Manual What-If Scenario Sandbox"]),
    ]

    col_w = 2.18
    gap_x = 0.2
    start_x = 0.8
    card_y = 1.9
    card_h = 4.85

    for idx, (kicker, title, desc, col_theme, points) in enumerate(stages):
        cx = start_x + idx * (col_w + gap_x)
        # Card container
        add_card(s4, Inches(cx), Inches(card_y), Inches(col_w), Inches(card_h), CARD_WHITE, CARD_BORDER)
        # Header banner
        add_card(s4, Inches(cx), Inches(card_y), Inches(col_w), Inches(0.06), col_theme)

        box = s4.shapes.add_textbox(Inches(cx + 0.15), Inches(card_y + 0.2), Inches(col_w - 0.3), Inches(card_h - 0.4))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = kicker
        p1.font.name = FONT_MAIN
        p1.font.size = Pt(8.5)
        p1.font.bold = True
        p1.font.color.rgb = col_theme

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(13)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_DARK
        p2.space_before = Pt(6)

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.name = FONT_MAIN
        p3.font.size = Pt(10)
        p3.font.color.rgb = TEXT_MUTED
        p3.space_before = Pt(6)

        # Divider
        p4 = tf.add_paragraph()
        p4.text = "─────────────────"
        p4.font.name = FONT_MAIN
        p4.font.size = Pt(7)
        p4.font.color.rgb = CARD_BORDER
        p4.space_before = Pt(8)

        for pt in points:
            p_bullet = tf.add_paragraph()
            p_bullet.text = pt
            p_bullet.font.name = FONT_MAIN
            p_bullet.font.size = Pt(9.5)
            p_bullet.font.color.rgb = TEXT_DARK
            p_bullet.space_before = Pt(6)

    add_footer(s4, 4, 12)

    # ============================================================
    # SLIDE 5: Technologies Used
    # ============================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5, prs, LIGHT_BG)
    add_header(s5, "04 / TECH STACK & INFRASTRUCTURE", "Technologies Used",
               "The project uses these main technologies across backend modeling, data science, and web delivery:")

    techs = [
        ("Python:", "Main programming language.", BRAND_BLUE),
        ("Pandas and NumPy:", "Data cleaning and numerical calculations.", BRAND_BLUE),
        ("Scikit-learn:", "Machine-learning models.", BRAND_BLUE),
        ("PyTorch:", "Neural-network development.", BRAND_INDIGO),
        ("Sentence Transformers:", "Understanding news text.", BRAND_INDIGO),
        ("SHAP:", "Explaining model predictions.", BRAND_INDIGO),
        ("HTML, CSS and JavaScript:", "Creating the web dashboard.", BRAND_TEAL),
        ("Chart.js:", "Displaying charts and risk information.", BRAND_TEAL),
        ("GitHub and Vercel:", "Code management and dashboard deployment.", BRAND_TEAL),
    ]

    start_x = 0.8
    start_y = 1.9
    col_w = 3.75
    gap_x = 0.24
    row_h = 1.5
    gap_y = 0.2

    for idx, (name, role, accent_col) in enumerate(techs):
        row = idx // 3
        col = idx % 3
        cx = start_x + col * (col_w + gap_x)
        cy = start_y + row * (row_h + gap_y)

        add_card(s5, Inches(cx), Inches(cy), Inches(col_w), Inches(row_h), CARD_WHITE, CARD_BORDER)
        add_card(s5, Inches(cx), Inches(cy), Inches(0.06), Inches(row_h), accent_col)

        box = s5.shapes.add_textbox(Inches(cx + 0.25), Inches(cy + 0.25), Inches(col_w - 0.4), Inches(row_h - 0.4))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = name
        p1.font.name = FONT_MAIN
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_DARK

        p2 = tf.add_paragraph()
        p2.text = role
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(6)

    add_footer(s5, 5, 12)

    # ============================================================
    # HELPER FOR SLIDES 6 to 10 (Person 1 to 5)
    # ============================================================
    def build_person_slide(slide_num, kicker, person_title, lib_name, role_summary, tasks, tech_str, example_str, accent_color):
        s = prs.slides.add_slide(blank_layout)
        set_slide_bg(s, prs, LIGHT_BG)
        add_header(s, kicker, person_title, role_summary)

        # Library Badge Top Bar
        badge_card = add_card(s, Inches(0.8), Inches(1.8), Inches(11.733), Inches(0.55), RGBColor(238, 242, 255), RGBColor(199, 210, 254))
        bb_box = s.shapes.add_textbox(Inches(1.0), Inches(1.88), Inches(11.3), Inches(0.4))
        bbtf = bb_box.text_frame
        bbtf.margin_left = bbtf.margin_top = bbtf.margin_right = bbtf.margin_bottom = 0
        bbp = bbtf.paragraphs[0]
        bbp.text = f"CORE MODULE LIBRARY:  {lib_name}"
        bbp.font.name = FONT_MAIN
        bbp.font.size = Pt(11)
        bbp.font.bold = True
        bbp.font.color.rgb = BRAND_INDIGO

        # Left Column: Tasks Card
        add_card(s, Inches(0.8), Inches(2.5), Inches(6.8), Inches(4.25), CARD_WHITE, CARD_BORDER)
        add_card(s, Inches(0.8), Inches(2.5), Inches(6.8), Inches(0.05), accent_color)

        t_box = s.shapes.add_textbox(Inches(1.15), Inches(2.75), Inches(6.1), Inches(3.7))
        tf = t_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_t = tf.paragraphs[0]
        p_t.text = "MAIN TASKS & CAPABILITIES"
        p_t.font.name = FONT_MAIN
        p_t.font.size = Pt(10.5)
        p_t.font.bold = True
        p_t.font.color.rgb = accent_color

        for task in tasks:
            pt = tf.add_paragraph()
            pt.text = f"•  {task}"
            pt.font.name = FONT_MAIN
            pt.font.size = Pt(11.5)
            pt.font.color.rgb = TEXT_DARK
            pt.space_before = Pt(8)

        # Right Column: Technologies & Example Cards
        # Tech Box
        add_card(s, Inches(7.85), Inches(2.5), Inches(4.683), Inches(1.6), CARD_WHITE, CARD_BORDER)
        add_card(s, Inches(7.85), Inches(2.5), Inches(4.683), Inches(0.05), BRAND_BLUE)
        tech_box = s.shapes.add_textbox(Inches(8.15), Inches(2.75), Inches(4.1), Inches(1.2))
        ttf = tech_box.text_frame
        ttf.word_wrap = True
        ttf.margin_left = ttf.margin_top = ttf.margin_right = ttf.margin_bottom = 0
        tp1 = ttf.paragraphs[0]
        tp1.text = "TECHNOLOGY STACK"
        tp1.font.name = FONT_MAIN
        tp1.font.size = Pt(10)
        tp1.font.bold = True
        tp1.font.color.rgb = BRAND_BLUE

        tp2 = ttf.add_paragraph()
        tp2.text = tech_str
        tp2.font.name = FONT_MAIN
        tp2.font.size = Pt(11)
        tp2.font.color.rgb = TEXT_DARK
        tp2.space_before = Pt(6)

        # Example Box
        add_card(s, Inches(7.85), Inches(4.3), Inches(4.683), Inches(2.45), RGBColor(240, 253, 244), RGBColor(187, 247, 208))
        add_card(s, Inches(7.85), Inches(4.3), Inches(0.08), Inches(2.45), BRAND_TEAL)
        ex_box = s.shapes.add_textbox(Inches(8.15), Inches(4.55), Inches(4.1), Inches(2.0))
        etf = ex_box.text_frame
        etf.word_wrap = True
        etf.margin_left = etf.margin_top = etf.margin_right = etf.margin_bottom = 0
        ep1 = etf.paragraphs[0]
        ep1.text = "REAL-WORLD OPERATIONAL EXAMPLE"
        ep1.font.name = FONT_MAIN
        ep1.font.size = Pt(10)
        ep1.font.bold = True
        ep1.font.color.rgb = BRAND_TEAL

        ep2 = etf.add_paragraph()
        ep2.text = example_str
        ep2.font.name = FONT_MAIN
        ep2.font.size = Pt(12)
        ep2.font.bold = True
        ep2.font.color.rgb = RGBColor(20, 83, 45)
        ep2.space_before = Pt(8)

        add_footer(s, slide_num, 12)

    # ============================================================
    # SLIDE 6: Person 1
    # ============================================================
    build_person_slide(
        6, "05 / INDIVIDUAL CONTRIBUTIONS",
        "Person 1: Supply-Chain Time-Series Intelligence",
        "SupplyChainTimeSeries",
        "My role is to analyse historical supply-chain data.",
        [
            "Clean sales, inventory and delivery data.",
            "Analyse demand and supplier lead times.",
            "Create lag features and moving averages.",
            "Detect unusual changes.",
            "Calculate inventory and time-based risk.",
            "Forecast future demand."
        ],
        "Technologies: Python, Pandas, NumPy and Scikit-learn.",
        "Example: If demand increases while inventory decreases, the system can identify a possible shortage risk.",
        BRAND_BLUE
    )

    # ============================================================
    # SLIDE 7: Person 2
    # ============================================================
    build_person_slide(
        7, "05 / INDIVIDUAL CONTRIBUTIONS",
        "Person 2: News Intelligence",
        "NewsIntelligence",
        "My role is to analyse news articles and identify supply-chain-related events.",
        [
            "Process news headlines.",
            "Identify companies, suppliers and locations.",
            "Detect events such as strikes and port closures.",
            "Analyse sentiment and event relevance.",
            "Convert text into useful numerical features."
        ],
        "Technologies: Python, NLP, Sentence Transformers and sentiment analysis.",
        "Example: A news article about a port strike can provide an early signal of possible delivery delays.",
        BRAND_INDIGO
    )

    # ============================================================
    # SLIDE 8: Person 3
    # ============================================================
    build_person_slide(
        8, "05 / INDIVIDUAL CONTRIBUTIONS",
        "Person 3: External Event and Spatial Intelligence",
        "SupplyChainEventRisk",
        "My role is to analyse external events and their possible effects on supply-chain locations.",
        [
            "Identify weather events and natural disasters.",
            "Analyse geopolitical and infrastructure problems.",
            "Match events with locations and dates.",
            "Calculate event severity.",
            "Map risks to suppliers, ports and transport routes."
        ],
        "Technologies: Python, geographical data processing and spatial analysis.",
        "Example: A flood near a supplier's transport route may increase the risk of delivery disruption.",
        BRAND_TEAL
    )

    # ============================================================
    # SLIDE 9: Person 4
    # ============================================================
    build_person_slide(
        9, "05 / INDIVIDUAL CONTRIBUTIONS",
        "Person 4: Multimodal Fusion and Early Warning",
        "MultimodalWarning",
        "My role is to combine information from the first three modules and predict disruption risk.",
        [
            "Combine time-series, news and event features.",
            "Align data from different sources.",
            "Predict disruption probability.",
            "Classify risk and possible disruption type.",
            "Estimate when a disruption may occur."
        ],
        "Technologies: Python, PyTorch, machine learning and multimodal fusion.",
        "Example: Increasing lead time, a port-strike report and a flood warning together may indicate a higher disruption risk.",
        ALERT_AMBER
    )

    # ============================================================
    # SLIDE 10: Person 5
    # ============================================================
    build_person_slide(
        10, "05 / INDIVIDUAL CONTRIBUTIONS",
        "Person 5: Explainability and Integration",
        "ExplainableWarning",
        "My role is to explain the prediction and connect the different modules into one system.",
        [
            "Explain which factors influenced the prediction.",
            "Use SHAP to measure feature contributions.",
            "Check warnings against supporting evidence.",
            "Communicate uncertainty.",
            "Generate simple explanations for users.",
            "Connect the modules to the dashboard."
        ],
        "Technologies: Python, SHAP, SciPy, HTML, CSS, JavaScript and Chart.js.",
        "Example: The system reports high risk and explains that decreasing inventory, increasing lead time and a port-strike report contributed to the warning.",
        ALERT_RED
    )

    # ============================================================
    # SLIDE 11: Dashboard and Expected Results
    # ============================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s11, prs, LIGHT_BG)
    add_header(s11, "06 / INTERFACE & DELIVERABLES", "Dashboard and Expected Results",
               "SupplyShield is designed to provide an interactive dashboard with three main areas:")

    dash_areas = [
        ("1", "Executive Overview", "Shows risk levels, alerts and disruption timelines.", BRAND_BLUE),
        ("2", "Alert Details", "Shows the predicted risk, contributing factors and supporting news evidence.", BRAND_INDIGO),
        ("3", "Manual Prediction Playground", "Lets users change input values and explore different scenarios.", BRAND_TEAL),
    ]

    start_x = 0.8
    card_w = 3.75
    gap_x = 0.24
    card_y = 1.85
    card_h = 2.15

    for idx, (num, title, desc, accent) in enumerate(dash_areas):
        cx = start_x + idx * (card_w + gap_x)
        add_card(s11, Inches(cx), Inches(card_y), Inches(card_w), Inches(card_h), CARD_WHITE, CARD_BORDER)
        add_card(s11, Inches(cx), Inches(card_y), Inches(card_w), Inches(0.06), accent)

        badge = add_card(s11, Inches(cx + 0.25), Inches(card_y + 0.25), Inches(0.4), Inches(0.4), accent)
        btf = badge.text_frame
        btf.margin_left = btf.margin_top = btf.margin_right = btf.margin_bottom = 0
        bp = btf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        bp.text = num
        bp.font.name = FONT_MAIN
        bp.font.size = Pt(11)
        bp.font.bold = True
        bp.font.color.rgb = TEXT_LIGHT

        dbox = s11.shapes.add_textbox(Inches(cx + 0.78), Inches(card_y + 0.22), Inches(card_w - 1.0), Inches(card_h - 0.4))
        dtf = dbox.text_frame
        dtf.word_wrap = True
        dtf.margin_left = dtf.margin_top = dtf.margin_right = dtf.margin_bottom = 0
        dp1 = dtf.paragraphs[0]
        dp1.text = title
        dp1.font.name = FONT_MAIN
        dp1.font.size = Pt(13)
        dp1.font.bold = True
        dp1.font.color.rgb = TEXT_DARK

        dp2 = dtf.add_paragraph()
        dp2.text = desc
        dp2.font.name = FONT_MAIN
        dp2.font.size = Pt(11)
        dp2.font.color.rgb = TEXT_MUTED
        dp2.space_before = Pt(8)

    # Bottom Decision Support Container
    add_card(s11, Inches(0.8), Inches(4.3), Inches(11.733), Inches(2.45), CARD_WHITE, CARD_BORDER)
    add_card(s11, Inches(0.8), Inches(4.3), Inches(11.733), Inches(0.05), BRAND_INDIGO)

    b_box = s11.shapes.add_textbox(Inches(1.1), Inches(4.5), Inches(11.1), Inches(0.4))
    btf = b_box.text_frame
    btf.margin_left = btf.margin_top = btf.margin_right = btf.margin_bottom = 0
    bp = btf.paragraphs[0]
    bp.text = "THE SYSTEM IS INTENDED TO HELP USERS UNDERSTAND:"
    bp.font.name = FONT_MAIN
    bp.font.size = Pt(10.5)
    bp.font.bold = True
    bp.font.color.rgb = BRAND_INDIGO

    questions = [
        ("SEVERITY", "How serious a possible disruption is."),
        ("ROOT CAUSE", "What may be causing it."),
        ("HORIZON", "When it might happen."),
        ("ATTRIBUTION", "Which factors contributed to the warning.")
    ]

    q_w = 2.65
    q_gap = 0.2
    q_start = 1.1
    for idx, (label, q_text) in enumerate(questions):
        qx = q_start + idx * (q_w + q_gap)
        add_card(s11, Inches(qx), Inches(5.0), Inches(q_w), Inches(1.5), RGBColor(248, 250, 252), CARD_BORDER)
        qbox = s11.shapes.add_textbox(Inches(qx + 0.15), Inches(5.15), Inches(q_w - 0.3), Inches(1.2))
        qtf = qbox.text_frame
        qtf.word_wrap = True
        qtf.margin_left = qtf.margin_top = qtf.margin_right = qtf.margin_bottom = 0

        qp1 = qtf.paragraphs[0]
        qp1.text = label
        qp1.font.name = FONT_MAIN
        qp1.font.size = Pt(9)
        qp1.font.bold = True
        qp1.font.color.rgb = BRAND_BLUE

        qp2 = qtf.add_paragraph()
        qp2.text = q_text
        qp2.font.name = FONT_MAIN
        qp2.font.size = Pt(12)
        qp2.font.bold = True
        qp2.font.color.rgb = TEXT_DARK
        qp2.space_before = Pt(6)

    add_footer(s11, 11, 12)

    # ============================================================
    # SLIDE 12: Conclusion and Future Scope
    # ============================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s12, prs, LIGHT_BG)
    add_header(s12, "07 / IMPACT & ROADMAP", "Conclusion and Future Scope",
               "Summary of accomplishments and strategic pathways for next-generation intelligence.")

    # Left: Conclusion Card
    add_card(s12, Inches(0.8), Inches(1.85), Inches(5.6), Inches(4.0), CARD_WHITE, CARD_BORDER)
    add_card(s12, Inches(0.8), Inches(1.85), Inches(5.6), Inches(0.06), BRAND_BLUE)

    c_box = s12.shapes.add_textbox(Inches(1.15), Inches(2.15), Inches(4.9), Inches(3.4))
    ctf = c_box.text_frame
    ctf.word_wrap = True
    ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0

    cp1 = ctf.paragraphs[0]
    cp1.text = "CONCLUSION"
    cp1.font.name = FONT_MAIN
    cp1.font.size = Pt(11)
    cp1.font.bold = True
    cp1.font.color.rgb = BRAND_BLUE

    cp2 = ctf.add_paragraph()
    cp2.text = "SupplyShield combines supply-chain data, news and external events to identify possible disruptions early. It also aims to explain its predictions so businesses can understand and respond to risks."
    cp2.font.name = FONT_MAIN
    cp2.font.size = Pt(13)
    cp2.font.color.rgb = TEXT_DARK
    cp2.space_before = Pt(12)

    # Right: Future Improvements Card
    add_card(s12, Inches(6.75), Inches(1.85), Inches(5.783), Inches(4.0), CARD_WHITE, CARD_BORDER)
    add_card(s12, Inches(6.75), Inches(1.85), Inches(5.783), Inches(0.06), BRAND_TEAL)

    f_box = s12.shapes.add_textbox(Inches(7.1), Inches(2.15), Inches(5.1), Inches(3.4))
    ftf = f_box.text_frame
    ftf.word_wrap = True
    ftf.margin_left = ftf.margin_top = ftf.margin_right = ftf.margin_bottom = 0

    fp1 = ftf.paragraphs[0]
    fp1.text = "FUTURE IMPROVEMENTS"
    fp1.font.name = FONT_MAIN
    fp1.font.size = Pt(11)
    fp1.font.bold = True
    fp1.font.color.rgb = BRAND_TEAL

    future_pts = [
        "Automatic supplier rerouting.",
        "Satellite imagery for port monitoring.",
        "Analysis of deeper supplier networks.",
        "Mobile notifications for urgent alerts."
    ]

    for pt in future_pts:
        fp_item = ftf.add_paragraph()
        fp_item.text = f"•  {pt}"
        fp_item.font.name = FONT_MAIN
        fp_item.font.size = Pt(12)
        fp_item.font.bold = True
        fp_item.font.color.rgb = TEXT_DARK
        fp_item.space_before = Pt(10)

    # Thank you banner at bottom
    add_card(s12, Inches(0.8), Inches(6.0), Inches(11.733), Inches(0.85), DARK_BG, DARK_BORDER)
    ty_box = s12.shapes.add_textbox(Inches(1.0), Inches(6.12), Inches(11.3), Inches(0.6))
    ty_tf = ty_box.text_frame
    ty_tf.margin_left = ty_tf.margin_top = ty_tf.margin_right = ty_tf.margin_bottom = 0
    typ = ty_tf.paragraphs[0]
    typ.alignment = PP_ALIGN.CENTER
    typ.text = "Thank you!  —  Questions & Discussion"
    typ.font.name = FONT_MAIN
    typ.font.size = Pt(14)
    typ.font.bold = True
    typ.font.color.rgb = TEXT_LIGHT

    add_footer(s12, 12, 12)

    # Save presentation
    prs.save("SupplyShield_Presentation.pptx")
    print("Successfully built SupplyShield_Presentation.pptx with executive UI/UX!")

if __name__ == "__main__":
    build_presentation()
