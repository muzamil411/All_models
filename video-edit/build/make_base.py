import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timeline import *

CW, CH = 1728, 3072            # 3x upscaled working canvas
ZLO, ZHI, ZSPLIT = 1.015, 1.078, 1.040
PUNCH_A, ATK, DEC = 0.055, 0.045, 0.280

def keyframes():
    """One z value per section boundary, alternating lo/hi so the camera never stops
    moving and stays continuous across cuts. Split-screen neighbours get clamped."""
    bounds = [s[0] for s in SECTIONS] + [SECTIONS[-1][1]]
    zk = [ZLO if k % 2 == 0 else ZHI for k in range(len(bounds))]
    for i, (_t0, _t1, _n, _l, split) in enumerate(SECTIONS):
        if split:
            zk[i]     = min(zk[i],     ZSPLIT)
            zk[i + 1] = min(zk[i + 1], ZSPLIT)
    return bounds, zk

def zoom_expr():
    t = "(on/%d)" % FPS
    bounds, zk = keyframes()
    chain = "%.4f" % zk[-1]
    for i in range(len(SECTIONS) - 1, -1, -1):
        t0, t1 = bounds[i], bounds[i + 1]
        z0, z1 = zk[i], zk[i + 1]
        u = "((%s-%.3f)/%.3f)" % (t, t0, max(t1 - t0, 0.001))
        smooth = "(3*pow(%s,2)-2*pow(%s,3))" % (u, u)
        chain = "if(between(%s,%.3f,%.3f),(%.4f+%.4f*%s),%s)" % (t, t0, t1, z0, z1 - z0, smooth, chain)
    punches = []
    for b in BEATS:
        amp = PUNCH_A * (0.6 if any(abs(b - c) < 0.2 for c in HARD_CUTS) else 1.0)
        d = "(%s-%.3f)" % (t, b)
        punches.append("if(gte(%s,%.3f),%.4f*(1-exp(-%s/%.3f))*exp(-%s/%.3f),0)"
                       % (t, b, amp, d, ATK, d, DEC))
    return "min(1.16,(%s)+%s)" % (chain, "+".join(punches))

def py_zoom(t):
    bounds, zk = keyframes()
    z = zk[-1]
    for i in range(len(SECTIONS)):
        t0, t1 = bounds[i], bounds[i + 1]
        if t0 <= t <= t1:
            u = (t - t0) / max(t1 - t0, .001)
            z = zk[i] + (zk[i + 1] - zk[i]) * (3 * u * u - 2 * u ** 3)
            break
    for b in BEATS:
        if t >= b:
            amp = PUNCH_A * (0.6 if any(abs(b - c) < 0.2 for c in HARD_CUTS) else 1.0)
            d = t - b
            z += amp * (1 - math.exp(-d / ATK)) * math.exp(-d / DEC)
    return min(1.16, z)

GRADE = (
    "eq=contrast=1.16:saturation=1.20:gamma=0.972:brightness=0.012,"
    "curves=r='0/0.020 0.20/0.190 0.50/0.520 0.80/0.845 1/1':"
           "g='0/0.015 0.50/0.500 1/1':"
           "b='0/0.058 0.20/0.222 0.50/0.500 0.80/0.788 1/0.955',"
    "vibrance=intensity=0.20"
)

def build_vf():
    return ",".join([
        "scale=%d:%d:flags=lanczos" % (CW, CH),
        "zoompan=z='%s':x='iw/2-(iw/zoom/2)+(26*sin((on/%d)*0.33))':"
        "y='ih/2-(ih/zoom/2)+(18*sin((on/%d)*0.26+1.1))':d=1:s=%dx%d:fps=%d"
            % (zoom_expr(), FPS, FPS, W, H, FPS),
        GRADE,
        "unsharp=5:5:0.85:5:5:0.0",
        "vignette=angle=PI/4.6:mode=forward",
        "noise=alls=6:allf=t+u",
        "format=yuv420p",
    ])

if __name__ == "__main__":
    if "--check" in sys.argv:
        vals = [py_zoom(i / FPS) for i in range(int(DUR * FPS))]
        d = [abs(vals[i+1]-vals[i]) for i in range(len(vals)-1)]
        print("z range %.4f..%.4f  max frame delta %.5f (%.1f px)"
              % (min(vals), max(vals), max(d), max(d)*CW/2))
        for t in [0,2.0,2.15,4.75,4.8,4.95,9.45,9.5,9.65,17,19.9,20.1,59.5,74]:
            print("  t=%5.2f z=%.4f" % (t, py_zoom(t)))
    else:
        print(build_vf())
