"""
Presentation Data Visualization Script
Generates presentation-ready charts for graduation thesis:
  1. Sector distribution pie chart (donut)
  2. Distress vs Healthy company breakdown
  3. Dataset structure overview
  4. Observation count heatmap (company x year)
  5. Feature category breakdown
  6. Data quality summary
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- GLOBAL STYLING TO MATCH HTML PRESENTATION ---
BG_DARK = '#0b1120'
BG_SURFACE = '#1e293b'
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#cbd5e1'
ACCENT_BLUE = '#3b82f6'
ACCENT_GREEN = '#10b981'
ACCENT_RED = '#ef4444'
ACCENT_PURPLE = '#8b5cf6'
ACCENT_YELLOW = '#f59e0b'
GRID_COLOR = '#ffffff' # Will use alpha down the line

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': 'none', # transparent to blend with html
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'axes.edgecolor': TEXT_MUTED,
    'axes.labelcolor': TEXT_MAIN,
    'xtick.color': TEXT_MUTED,
    'ytick.color': TEXT_MUTED,
    'text.color': TEXT_MAIN,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Inter', 'sans-serif'],
    'axes.grid': False,
    'grid.alpha': 0.1,
    'grid.color': '#ffffff'
})
# --- END GLOBAL STYLING ---

# ── BIST Sector Mapping (65 companies) ───────────────────────────────────────
# Based on BIST GICS / KAP main activity classifications
SECTOR_MAP = {
    # Manufacturing - Automotive & Parts
    "FROTO":  "Otomotiv",        # Ford Otosan
    "TOASO":  "Otomotiv",        # Tofas
    "ASUZU":  "Otomotiv",        # Anadolu Isuzu
    "VESTL":  "Teknoloji/Elektronik",  # Vestel

    # Manufacturing - Consumer Goods
    "ULKER":  "Gida & Icecek",   # Ulker
    "AEFES":  "Gida & Icecek",   # Anadolu Efes
    "SOKM":   "Gida & Icecek",   # Sok Market
    "MGROS":  "Perakende",       # Migros
    "ADEL":   "Gida & Icecek",   # Adel Kalemcilik
    "TBORG":  "Gida & Icecek",   # Turk Bira (Heineken)
    "YATAS":  "Perakende",       # Yatas

    # Energy & Utilities
    "TUPRS":  "Enerji & Petrokimya",  # Tupras
    "PETKM":  "Enerji & Petrokimya",  # Petkim
    "SASA":   "Enerji & Petrokimya",  # Sasa Polyester
    "AKSA":   "Enerji & Petrokimya",  # Aksa Enerji
    "ZOREN":  "Enerji & Petrokimya",  # Zorlu Enerji
    "ENERY":  "Enerji & Petrokimya",  # Enerya
    "ENJSA":  "Enerji & Petrokimya",  # Enerjisa
    "CWENE":  "Enerji & Petrokimya",  # CW Enerji
    "NTGAZ":  "Enerji & Petrokimya",  # Naturelgaz
    "AHGAZ":  "Enerji & Petrokimya",  # Ahlatci Dogalgaz

    # Industrial & Construction
    "ENKAI":  "Insaat & Muhendislik",  # Enka Insaat
    "CIMSA":  "Insaat Malzemeleri",    # Cimsa Cimento
    "NUHCM":  "Insaat Malzemeleri",    # Nuh Cimento
    "SISE":   "Insaat Malzemeleri",    # Sise Cam
    "KRDMD":  "Metal & Celik",         # Kardemir
    "BRKO":   "Metal & Celik",         # Burcelik (distressed)
    "EMKEL":  "Metal & Celik",         # Emek Elektrik
    "ASTOR":  "Metal & Celik",         # Astor Enerji
    "BAGFS":  "Tarim & Kimya",         # Bagfas Gubre
    "KOZAA":  "Madencilik",            # Koza Altin
    "KRVT":   "Madencilik",            # Koza Anadolu

    # Technology & Software
    "LOGO":   "Teknoloji/Elektronik",  # Logo Yazilim
    "DGATE":  "Teknoloji/Elektronik",  # Datagate
    "VESBE":  "Teknoloji/Elektronik",  # Vestel Beyaz Esya
    "ASELS":  "Savunma & Teknoloji",   # Aselsan

    # Transportation & Aviation
    "THYAO":  "Ulasim & Lojistik",     # THY
    "PGSUS":  "Ulasim & Lojistik",     # Pegasus
    "TAVHL":  "Ulasim & Lojistik",     # TAV Havalimanlari

    # Holding & Diversified
    "KCHOL":  "Holding",               # Koc Holding
    "DOHOL":  "Holding",               # Dogan Holding
    "EUHOL":  "Holding",               # Euro Holding (distressed)

    # Real Estate & REITs
    "ISGYO":  "Gayrimenkul (GYO)",     # Is GYO
    "YGYO":   "Gayrimenkul (GYO)",     # Yapi Kredi GYO (distressed)
    "SMRTG":  "Gayrimenkul (GYO)",     # Smart GYO

    # Textile & Apparel
    "ATEKS":  "Tekstil",               # Altinyildiz (distressed)
    "EMNIS":  "Tekstil",               # Emintas (distressed)
    "ORMA":   "Tekstil",               # Orma

    # Agriculture & Others
    "AGROT":  "Tarim & Kimya",         # Agrotex (distressed)
    "YAYLA":  "Gida & Icecek",         # Yayla Agro
    "EKIZ":   "Tarim & Kimya",         # Ekiz Kimya (distressed)
    "BRMEN":  "Kimya",                 # Bursa Mermerciler (distressed)
    "DARDL":  "Tekstil",               # Dardanel

    # Troubled / Distressed sector companies
    "KUVVA":  "Enerji & Petrokimya",   # Kuvva (distressed)
    "MEGAP":  "Perakende",             # Mega Polietilen (distressed)
    "KERVT":  "Tekstil",               # Kervansaray (distressed)
    "KERVN":  "Tekstil",               # Kervansaray (distressed)
    "ROYAL":  "Perakende",             # Royal Hali (distressed)
    "SNPAM":  "Kimya",                 # Sanifoam (distressed)
    "DIRIT":  "Tekstil",               # Diriteks (distressed)
    "VANGD":  "Tekstil",               # Vangolu (distressed)
    "UMPAS":  "Holding",               # Umpas (distressed)
    "EBEBK":  "Gida & Icecek",         # Ebe Bebek
    "CASA":   "Perakende",             # Casa (distressed)
    "UFUK":   "Insaat Malzemeleri",    # Ufuk (distressed)
}

# Labels from bankruptcy CSV
DISTRESSED_FIRMS = {
    "KUVVA", "MEGAP", "SNPAM", "VANGD", "YGYO", "ATEKS", "BRKO", "BRMEN",
    "CASA", "EMKEL", "EMNIS", "KERVN", "KERVT", "EKIZ", "DIRIT", "DARDL",
    "AGROT", "ROYAL", "EUHOL", "UFUK", "UMPAS"
}


def load_data():
    df = pd.read_csv("outputs/dataset.csv")
    companies = sorted(df["company"].unique())
    company_df = df.groupby("company").agg(
        years=("year", "count"),
        label=("bankruptcy_label", "max")
    ).reset_index()
    company_df["sector"] = company_df["company"].map(SECTOR_MAP).fillna("Diger")
    company_df["status"] = company_df["label"].apply(
        lambda x: "Finansal Sikinti" if x == 1 else "Saglikli"
    )
    return df, company_df


def plot_sector_donut(company_df, filename="pres_01_sector_donut.png"):
    """Premium donut chart of sector distribution."""
    sector_counts = company_df.groupby("sector").size().sort_values(ascending=False)

    colors = [
        ACCENT_GREEN, ACCENT_BLUE, ACCENT_PURPLE, ACCENT_RED, ACCENT_YELLOW,
        "#1ABC9C", "#E67E22", "#16A085", "#8E44AD", "#2980B9",
        "#C0392B", "#27AE60", "#D35400", "#7F8C8D"
    ]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    wedges, texts, autotexts = ax.pie(
        sector_counts.values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        startangle=90,
        colors=colors[:len(sector_counts)],
        wedgeprops=dict(width=0.55, edgecolor=BG_SURFACE, linewidth=2.5),
        pctdistance=0.78,
    )
    for t in autotexts:
        t.set_color(TEXT_MAIN)
        t.set_fontsize(10)
        t.set_fontweight("bold")

    # Center text
    ax.text(0, 0.12, "65", ha="center", va="center",
            fontsize=44, fontweight="bold", color=TEXT_MAIN)
    ax.text(0, -0.14, "Şirket", ha="center", va="center",
            fontsize=16, color=TEXT_MUTED)

    # Legend
    legend_labels = [f"{s} ({n})" for s, n in zip(sector_counts.index, sector_counts.values)]
    legend = ax.legend(
        wedges, legend_labels,
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=11, frameon=False,
        labelcolor=TEXT_MAIN,
    )

    ax.set_title("Veri Seti — Sektör Dağılımı\n(BIST 2018-2024)",
                 fontsize=17, fontweight="bold", color=TEXT_MAIN, pad=20)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight",
                facecolor='none')
    plt.close()
    print(f"  [OK] {filename}")


def plot_distress_breakdown(company_df, filename="pres_02_distress_breakdown.png"):
    """Bar chart: healthy vs distressed per sector."""
    pivot = company_df.pivot_table(
        index="sector", columns="status", values="company", aggfunc="count", fill_value=0
    ).sort_values("Saglikli", ascending=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    y = np.arange(len(pivot))
    h = 0.38
    healthy_col = "Saglikli" if "Saglikli" in pivot.columns else pivot.columns[0]
    stress_col = "Finansal Sikinti" if "Finansal Sikinti" in pivot.columns else None

    bars1 = ax.barh(y + h/2, pivot[healthy_col], h, color=ACCENT_GREEN,
                    label="Sağlıklı", edgecolor=BG_SURFACE)
    if stress_col:
        bars2 = ax.barh(y - h/2, pivot[stress_col], h, color=ACCENT_RED,
                        label="Finansal Sıkıntı", edgecolor=BG_SURFACE)
        for bar in bars2:
            w = bar.get_width()
            if w > 0:
                ax.text(w + 0.05, bar.get_y() + bar.get_height()/2,
                        str(int(w)), va="center", color=ACCENT_RED,
                        fontsize=10, fontweight="bold")
    for bar in bars1:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.05, bar.get_y() + bar.get_height()/2,
                    str(int(w)), va="center", color=ACCENT_GREEN,
                    fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, color=TEXT_MAIN, fontsize=11)
    ax.set_xlabel("Şirket Sayısı", color=TEXT_MUTED, fontsize=12)
    ax.set_title("Sektör Bazında Sağlıklı vs Finansal Sıkıntı Şirketleri",
                 fontsize=15, fontweight="bold", color=TEXT_MAIN, pad=15)
    ax.legend(fontsize=12, frameon=False, labelcolor=TEXT_MAIN)
    ax.tick_params(colors=TEXT_MUTED)
    ax.spines[:].set_visible(False)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.tick_params(axis="x", colors=TEXT_MUTED)
    ax.grid(axis="x", alpha=0.1, color=GRID_COLOR)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight",
                facecolor='none')
    plt.close()
    print(f"  [OK] {filename}")


def plot_dataset_structure(df, filename="pres_03_dataset_structure.png"):
    """Infographic-style dataset structure overview."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('none')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor('none')

    def card(x, y, w, h, title, value, subtitle, color, icon=""):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="none", alpha=0.18, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.72, icon + value, ha="center", va="center",
                fontsize=28, fontweight="bold", color=color, zorder=3)
        ax.text(x + w/2, y + h*0.38, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=TEXT_MAIN, zorder=3)
        ax.text(x + w/2, y + h*0.15, subtitle, ha="center", va="center",
                fontsize=9, color=TEXT_MUTED, zorder=3)

    # Row 1: top-level stats
    card(0.4, 5.8, 3.2, 2.8, "Toplam Gözlem", "439",    "Satır",              ACCENT_BLUE)
    card(4.0, 5.8, 3.2, 2.8, "Şirket",         "65",     "BIST Firması",       ACCENT_GREEN)
    card(7.6, 5.8, 3.2, 2.8, "Dönem",          "7 Yıl",  "2018 – 2024",        ACCENT_PURPLE)
    card(11.2,5.8, 3.2, 2.8, "Sektör",         "14",     "BIST Sektörü",       ACCENT_YELLOW)

    # Row 2: label stats
    card(0.4, 2.6, 3.2, 2.8, "Sağlıklı",       "314",    "71.5%",              ACCENT_GREEN)
    card(4.0, 2.6, 3.2, 2.8, "Fin. Sıkıntı",   "125",    "28.5%",              ACCENT_RED)
    card(7.6, 2.6, 3.2, 2.8, "Feature",        "40",     "Değişken",           '#1ABC9C')
    card(11.2,2.6, 3.2, 2.8, "Etiketli Firm.", "21",     "Distressed",         '#E67E22')

    # Feature breakdown at bottom
    # 40 feature = 13 ratios + 9 macro + 18 interactions
    feat_data = [
        ("13", "Finansal\nOranlar",   ACCENT_BLUE, 1.0),
        ("9",  "Makro\n(Güncel)",     ACCENT_GREEN, 3.8),
        ("18", "Etkileşim\nTerimleri",ACCENT_RED, 6.6),
        ("0",  "Gecikmeli\nMakro",    TEXT_MUTED, 9.4),
    ]
    for val, lbl, clr, xpos in feat_data:
        card(xpos, 0.4, 2.8, 2.0, lbl, val, "feature", clr)

    ax.text(8, 9.1 - 0.55, "Veri Seti Yapısı — BIST Finansal Sıkıntı Tahmini",
            ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT_MAIN)
    ax.text(8, 8.6 - 0.55, "Borsa İstanbul 2018-2024 | Finansal Tablolar + Makroekonomik Göstergeler",
            ha="center", va="center", fontsize=12, color=TEXT_MUTED)

    # Feature label separator
    ax.hlines(2.45, 0.4, 14.4, colors=BG_SURFACE, linewidth=1, zorder=1)
    ax.text(1.5, 0.15, "40 Feature = ", color=TEXT_MUTED, fontsize=10)
    ax.text(4.5, 0.15, "13 oran + 9 makro + 18 etkileşim (lag kaldırıldı)", color=TEXT_MAIN, fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight",
                facecolor='none')
    plt.close()
    print(f"  [OK] {filename}")


def plot_observation_heatmap(df, filename="pres_04_observation_heatmap.png"):
    """Year × Company observations heatmap."""
    pivot = df.pivot_table(index="company", columns="year",
                           values="bankruptcy_label", aggfunc="max")
    pivot_obs = df.pivot_table(index="company", columns="year",
                               values="debt_ratio", aggfunc="count")

    # Sort: distressed on top
    distress_flag = pivot.max(axis=1).sort_values(ascending=False)
    pivot_obs = pivot_obs.loc[distress_flag.index]
    label_mask = pivot.loc[distress_flag.index]

    fig, ax = plt.subplots(figsize=(14, 16))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    years = sorted(df["year"].unique())
    companies = list(distress_flag.index)
    n_c = len(companies)
    n_y = len(years)

    cell_h = 0.55
    cell_w = 1.2

    for ci, comp in enumerate(companies):
        for yi, yr in enumerate(years):
            obs = pivot_obs.loc[comp, yr] if yr in pivot_obs.columns else 0
            lbl = label_mask.loc[comp, yr] if yr in label_mask.columns else 0
            if pd.isna(obs) or obs == 0:
                color = BG_SURFACE
            elif lbl == 1:
                color = ACCENT_RED
            else:
                color = ACCENT_GREEN

            rect = plt.Rectangle(
                (yi * cell_w, (n_c - ci - 1) * cell_h),
                cell_w * 0.92, cell_h * 0.88,
                color=color, zorder=2
            )
            ax.add_patch(rect)

    ax.set_xlim(0, n_y * cell_w)
    ax.set_ylim(0, n_c * cell_h)
    ax.set_xticks([(yi + 0.46) * cell_w for yi in range(n_y)])
    ax.set_xticklabels([str(y) for y in years], color=TEXT_MAIN, fontsize=11)
    ax.set_yticks([(n_c - ci - 0.56) * cell_h for ci in range(n_c)])
    ax.set_yticklabels(companies, color=TEXT_MAIN, fontsize=7.5)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    ax.set_title("Gözlem Paneli — Şirket × Yıl\n",
                 fontsize=15, fontweight="bold", color=TEXT_MAIN, pad=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT_GREEN, label="Sağlıklı Gözlem"),
        Patch(facecolor=ACCENT_RED, label="Finansal Sıkıntı"),
        Patch(facecolor=BG_SURFACE, label="Veri Yok"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              frameon=False, labelcolor=TEXT_MAIN, fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight",
                facecolor='none')
    plt.close()
    print(f"  [OK] {filename}")


def plot_feature_breakdown(filename="pres_05_feature_breakdown.png"):
    """Visual breakdown of the 40 features used."""
    categories = {
        "Finansal\nOranlar": {
            "count": 13,
            "color": ACCENT_BLUE,
            "examples": ["debt_ratio", "quick_ratio", "ROA",
                         "asset_turnover", "net_profit_margin"]
        },
        "Makro\nGöstergeler": {
            "count": 9,
            "color": ACCENT_GREEN,
            "examples": ["gdp_growth", "inflation_rate",
                         "interest_rate", "usdtry_change", "unemployment"]
        },
        "Gecikmeli\nMakro (Kaldırıldı)": {
            "count": 0,
            "color": TEXT_MUTED,
            "examples": ["SHAP değeri ~0 olduğu", "tespit edildiği için",
                         "modele daha iyi temsil", "eden etkileşimler eklendi"]
        },
        "Etkileşim\nTerimleri (Mikro×Makro)": {
            "count": 18,
            "color": ACCENT_RED,
            "examples": ["debt_x_interest", "quick_x_unemployment",
                         "margin_x_inflation", "sales_x_gdp", "..."]
        },
    }

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('none')

    n = len(categories)
    for i, (cat, info) in enumerate(categories.items()):
        ax = fig.add_subplot(1, n, i + 1)
        ax.set_facecolor('none')
        ax.axis("off")

        color = info["color"]
        count = info["count"]

        # Big circle
        circle = plt.Circle((0.5, 0.68), 0.28, color=color, alpha=0.15, zorder=1)
        circle_border = plt.Circle((0.5, 0.68), 0.28, color=color, alpha=0.8,
                                   fill=False, linewidth=3, zorder=2)
        ax.add_patch(circle)
        ax.add_patch(circle_border)
        ax.text(0.5, 0.68, str(count), ha="center", va="center",
                fontsize=38, fontweight="bold", color=color, zorder=3,
                transform=ax.transAxes)
        ax.text(0.5, 0.42, cat, ha="center", va="center",
                fontsize=12, fontweight="bold", color=TEXT_MAIN,
                transform=ax.transAxes)

        # Examples
        for j, ex in enumerate(info["examples"]):
            ax.text(0.5, 0.32 - j * 0.06, f"• {ex}", ha="center",
                    fontsize=8.5, color=TEXT_MUTED, transform=ax.transAxes)

    fig.suptitle("40 Feature — Kategori Dağılımı",
                 fontsize=18, fontweight="bold", color="white", y=1.02)
    # Total
    fig.text(0.5, -0.04, "Toplam 40 Feature = 13 + 9 + 18 + 0",
             ha="center", fontsize=13, color="#7F8C8D")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight",
                facecolor="#0D1117")
    plt.close()
    print(f"  [OK] {filename}")


def main():
    print("=" * 60)
    print("  Generating Presentation Charts")
    print("=" * 60)

    df, company_df = load_data()

    print(f"\n  Companies: {len(company_df)}")
    print(f"  Sectors:   {company_df['sector'].nunique()}")
    print(f"  Distressed:{(company_df['status'] == 'Finansal Sikinti').sum()}")
    print()

    plot_sector_donut(company_df)
    plot_distress_breakdown(company_df)
    plot_dataset_structure(df)
    plot_observation_heatmap(df)
    plot_feature_breakdown()

    print(f"\n  All charts saved to: {PLOTS_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
