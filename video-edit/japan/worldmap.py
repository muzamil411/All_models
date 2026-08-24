"""Vector world map: slippy-map projection, animated camera, PIL rendering."""
import math, warnings, numpy as np
warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- geometry
def load_rings(simplify=0.02):
    import geopandas as gpd
    g = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    g = g[g.name != "Antarctica"].copy()
    g["geometry"] = g.geometry.simplify(simplify)
    out = []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in polys:
            c = np.asarray(p.exterior.coords, dtype=np.float64)
            if len(c) < 3:
                continue
            out.append({"name": row["name"], "iso": row["iso_a3"],
                        "lon": c[:, 0], "lat": c[:, 1],
                        "bbox": (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()),
                        "area": abs(p.area)})
    return out

# ---------------------------------------------------------------- projection
def merc_y(lat):
    lat = np.clip(lat, -85.05, 85.05)
    s = np.sin(np.radians(lat))
    return 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)

class Camera:
    """center lon/lat + zoom (slippy-map z, world = 256*2**z px)."""
    def __init__(self, lon, lat, zoom, w, h, rot=0.0):
        self.lon, self.lat, self.zoom, self.w, self.h, self.rot = lon, lat, zoom, w, h, rot
    @property
    def world(self): return 256.0 * (2.0 ** self.zoom)
    def project(self, lon, lat):
        W = self.world
        x = (np.asarray(lon) + 180.0) / 360.0 * W
        y = merc_y(np.asarray(lat)) * W
        cx = (self.lon + 180.0) / 360.0 * W
        cy = merc_y(self.lat) * W
        dx, dy = x - cx, y - cy
        if self.rot:
            a = math.radians(self.rot); ca, sa = math.cos(a), math.sin(a)
            dx, dy = dx * ca - dy * sa, dx * sa + dy * ca
        return dx + self.w / 2.0, dy + self.h / 2.0
    def visible_bbox(self, pad=1.35):
        """lon/lat bounds of the viewport, generously padded."""
        W = self.world
        dlon = (self.w / W) * 360.0 * pad
        cy = merc_y(self.lat) * W
        def inv_y(py):
            t = py / W
            return math.degrees(2 * math.atan(math.exp((0.5 - t) * 2 * math.pi)) - math.pi / 2)
        lat_hi = inv_y(cy - self.h / 2.0 * pad)
        lat_lo = inv_y(cy + self.h / 2.0 * pad)
        return (self.lon - dlon / 2, lat_lo, self.lon + dlon / 2, lat_hi)

def lerp(a, b, u): return a + (b - a) * u
def smooth(u):     u = max(0.0, min(1.0, u)); return u * u * (3 - 2 * u)
def smoother(u):   u = max(0.0, min(1.0, u)); return u * u * u * (u * (u * 6 - 15) + 10)

# ---------------------------------------------------------------- drawing
class MapRenderer:
    def __init__(self, w, h, ss=2, palette=None):
        self.w, self.h, self.ss = w, h, ss
        self.rings = load_rings()
        self.p = palette or {}
    def render(self, cam, highlight=(), glow=0.0):
        ss = self.ss
        big = Image.new("RGB", (self.w * ss, self.h * ss), self.p["sea"])
        d = ImageDraw.Draw(big)
        c = Camera(cam.lon, cam.lat, cam.zoom + math.log2(ss), self.w * ss, self.h * ss, cam.rot)
        self._graticule(d, c)
        lo_lon, lo_lat, hi_lon, hi_lat = c.visible_bbox()
        hi_set = set(highlight)
        hot = []
        for r in self.rings:
            b = r["bbox"]
            if b[2] < lo_lon or b[0] > hi_lon or b[3] < lo_lat or b[1] > hi_lat:
                continue
            x, y = c.project(r["lon"], r["lat"])
            pts = list(zip(x.tolist(), y.tolist()))
            if r["name"] in hi_set:
                hot.append(pts)
            else:
                d.polygon(pts, fill=self.p["land"], outline=self.p["coast"])
        for pts in hot:
            d.polygon(pts, fill=self.p["jp"], outline=self.p["jp_edge"])
        img = big.resize((self.w, self.h), Image.LANCZOS)
        if glow > 0.01 and hot:
            img = self._glow(img, c, hot, glow)
        return img
    def _glow(self, img, c, hot, amount):
        from PIL import ImageFilter
        ss = self.ss
        m = Image.new("L", (self.w * ss, self.h * ss), 0)
        dm = ImageDraw.Draw(m)
        for pts in hot:
            dm.polygon(pts, fill=255)
        m = m.resize((self.w, self.h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(26))
        layer = Image.new("RGB", img.size, self.p["glow"])
        return Image.composite(Image.blend(img, layer, min(0.85, amount)), img, m)
    def _graticule(self, d, c):
        col, step = self.p["grid"], 10
        for lon in range(-180, 181, step):
            x, y = c.project([lon, lon], [-85, 85])
            d.line([(x[0], y[0]), (x[1], y[1])], fill=col, width=self.ss)
        lats = list(range(-80, 81, step))
        for lat in lats:
            xs = np.linspace(-180, 180, 64)
            x, y = c.project(xs, np.full_like(xs, lat, dtype=float))
            d.line(list(zip(x.tolist(), y.tolist())), fill=col, width=self.ss)
