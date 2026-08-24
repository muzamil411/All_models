"""Beat sheet, camera path and data for the Japan explainer."""
DUR, FPS, W, H = 72.5, 30, 1080, 1920
MARGIN = 62

# ---------------------------------------------------------------- palette
SEA      = (11, 22, 38)
LAND     = (58, 71, 92)
COAST    = (78, 94, 120)
JP       = (212, 24, 61)
JP_EDGE  = (255, 92, 122)
GRID     = (36, 57, 90)
GLOW     = (255, 80, 110)
INK      = (8, 15, 27)
WASHI    = (245, 242, 234)
MUTED    = (147, 163, 184)
FAINT    = (104, 120, 140)
GOLD     = (232, 180, 80)
RED      = (232, 62, 84)

MAP_PALETTE = dict(sea=SEA, land=LAND, coast=COAST, jp=JP, jp_edge=JP_EDGE,
                   grid=GRID, glow=GLOW)

# ---------------------------------------------------------------- camera
# (time, centre lon, centre lat, slippy zoom)
CAM_KEYS = [
    (0.0,  141.0, 25.0, 1.90),
    (4.6,  134.0, 31.0, 2.70),
    (8.6,  137.6, 37.2, 4.85),
    (13.0, 137.8, 37.0, 5.02),
    (17.4, 138.6, 38.4, 5.14),
    (24.0, 136.2, 35.6, 5.20),
    (31.0, 137.2, 36.9, 4.66),
    (41.6, 137.6, 37.0, 5.08),
    (48.2, 138.1, 36.6, 5.24),
    (55.2, 139.2, 36.2, 5.52),
    (62.0, 133.5, 34.2, 3.70),
    (DUR,  131.0, 33.0, 3.38),
]

# beats used for camera punches and sound accents
BEATS = [0.4, 5.6, 11.2, 17.4, 24.0, 31.0, 35.5, 41.6, 48.2, 55.2, 62.0, 68.5]

# ---------------------------------------------------------------- chapters
# (start, end, number, label)
CHAPTERS = [
    (17.4, 23.8, "01", "THE LAND"),
    (24.0, 30.8, "02", "THE PEOPLE"),
    (31.0, 41.4, "03", "THE PROBLEM"),
    (41.6, 48.0, "04", "THE ECONOMY"),
    (48.2, 55.0, "05", "THE MONEY"),
    (55.2, 61.8, "06", "THE ENGINEERING"),
    (62.0, 68.4, "07", "THE OPENING"),
]

# ---------------------------------------------------------------- content
HEADLINES = [
    dict(t0=0.55, t1=5.10, kicker="AUGUST 1945",
         lines=["TWO ATOMIC BOMBS", "FELL ON THIS", "COUNTRY"], y=430, accent=RED),
    dict(t0=11.35, t1=16.85, kicker="BY 1968",
         lines=["IT WAS THE WORLD'S", "#2 ECONOMY"], y=430, accent=GOLD),
    dict(t0=31.35, t1=35.10, kicker="AND YET",
         lines=["IT IS SLOWLY", "DISAPPEARING"], y=350, accent=RED),
    dict(t0=62.35, t1=68.10, kicker="WHICH IS WHY",
         lines=["JAPAN NEEDS", "820,000 FOREIGN", "WORKERS BY 2029"], y=360, accent=GOLD),
]

REVEAL = dict(t0=5.85, t1=10.60, jp="日本", en="JAPAN")

CARDS = [
    dict(t0=17.85, t1=23.55, label="TOTAL AREA",
         value=lambda u: "{:,}".format(int(377975 * u)), unit="km²", accent=GOLD,
         sub="14,125 ISLANDS   ·   2.3× SMALLER THAN PAKISTAN"),
    dict(t0=24.45, t1=30.55, label="POPULATION",
         value=lambda u: "{:,}".format(int(123 * u)), unit="MILLION", accent=GOLD,
         sub="12TH LARGEST IN THE WORLD"),
    dict(t0=42.05, t1=47.75, label="GDP  ·  NOMINAL",
         value=lambda u: "$%.1f" % (4.2 * u), unit="TRILLION", accent=GOLD,
         sub="4TH LARGEST ECONOMY ON EARTH"),
    dict(t0=48.65, t1=54.75, label="AVERAGE SALARY",
         value=lambda u: "$" + "{:,}".format(int(30000 * u)), unit="/ YEAR", accent=GOLD,
         sub="≈  84 LAKH PKR PER YEAR"),
    dict(t0=55.65, t1=61.55, label="SHINKANSEN TOP SPEED",
         value=lambda u: "{:,}".format(int(320 * u)), unit="km/h", accent=RED,
         sub="ZERO PASSENGER DEATHS SINCE 1964"),
]

# ---------------------------------------------------------------- chart
CHART = dict(t0=35.40, t1=41.30)
# (year, millions, is_projection)
POP = [(1950, 83.2, 0), (1960, 93.3, 0), (1970, 103.7, 0), (1980, 116.8, 0),
       (1990, 123.6, 0), (2000, 126.9, 0), (2008, 128.1, 0), (2015, 127.1, 0),
       (2024, 123.8, 0), (2030, 120.1, 1), (2040, 112.4, 1), (2050, 104.7, 1),
       (2060, 96.1, 1), (2070, 87.0, 1)]

# ---------------------------------------------------------------- map fx
# atomic shockwave rings: (fire time, lon, lat, label)
BLASTS = [(1.75, 132.45, 34.39, "HIROSHIMA"), (2.65, 129.87, 32.75, "NAGASAKI")]
# inbound migration arrows: (from lon, lat) -> Japan, fired during THE OPENING
ARROWS = [(121.0, 14.6), (106.8, 10.8), (100.5, 13.7), (95.0, 21.9), (90.4, 23.8), (74.3, 31.5)]
JP_CENTRE = (138.0, 36.5)
TOKYO = (139.69, 35.69)

CTA = dict(t0=68.55, t1=71.95, main="FOLLOW FOR PART 2", sub="WHICH COUNTRY NEXT?")
