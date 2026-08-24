# Shared edit timeline for the Italy re-edit
DUR   = 74.7333
FPS   = 30
W, H  = 1080, 1920

# (start, end, chapter_no, chapter_label, split_screen?)
SECTIONS = [
    (0.00,  4.80, None, "HOOK",          False),
    (4.80,  9.50, "01", "LOCATION",      False),
    (9.50, 12.60, "02", "LAND AREA",     False),
    (12.60,16.70, "03", "POPULATION",    False),
    (16.70,19.90, "04", "CAPITAL",       True ),
    (19.90,23.50, "05", "ROMAN EMPIRE",  False),
    (23.50,27.50, "06", "WORLD WAR II",  False),
    (27.50,33.90, "06", "WORLD WAR II",  False),
    (33.90,41.30, "07", "ECONOMY",       False),
    (41.30,44.90, "07", "ECONOMY",       False),
    (44.90,49.70, "08", "AVERAGE SALARY",False),
    (49.70,56.10, "09", "CITIZENSHIP",   False),
    (56.10,59.20, "09", "CITIZENSHIP",   False),
    (59.20,66.30, "10", "MOVING THERE",  True ),
    (66.30,70.50, "10", "MOVING THERE",  False),
    (70.50,DUR,   None, "OUTRO",         False),
]

# punch-in beats derived from speech phrase gaps + hard cuts
BEATS = [2.0,4.8,6.9,9.5,11.2,12.6,14.9,16.7,19.9,22.6,25.7,27.5,29.8,
         33.9,36.5,39.2,41.3,44.9,46.5,48.5,49.7,51.5,53.2,56.1,58.2,
         61.0,63.4,66.3,68.7,71.4,72.7]

# hard cut points in the source (from scene detection)
HARD_CUTS = [19.97, 41.33, 44.90, 66.27]

# palette
INK    = (11, 16, 26)
GOLD   = (255, 199, 44)
RED    = (233, 58, 62)
GREEN  = (34, 197, 122)
WHITE  = (255, 255, 255)
