"""Quick metric summary across eval logs.
Usage: python -m scripts._eval_metric_summary log1 log2 ...
Reports per-log: N, SR, close_5mm (final_lat<5), close_3mm, min_dist<5mm, min_dist<8mm,
                 final_lat median, min_dist median, final_angle<10deg.
"""
import re, sys
PAT = re.compile(
    r"Episode\s+(\d+)/(\d+).*?\|\s+(\S+)\s+\|.*?dist=([-\d.]+)mm\s+lateral=([-\d.]+)mm\s+"
    r"angle=([-\d.]+)deg\s+min_dist=([-\d.]+)mm\s+sensor=([-\d.]+)"
)
def parse(p):
    eps = []
    with open(p) as f:
        for ln in f:
            m = PAT.search(ln)
            if not m: continue
            _, _, status, dist, lat, ang, mind, sen = m.groups()
            eps.append({"status": status, "dist": float(dist), "lat": float(lat),
                        "ang": float(ang), "min_d": float(mind), "sen": float(sen)})
    return eps
def med(xs):
    xs = sorted(xs); n=len(xs)
    if n==0: return float("nan")
    return xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2])
def rate(xs, thr, op="<"):
    if not xs: return 0.0
    if op=="<": k = sum(1 for x in xs if x < thr)
    else: k = sum(1 for x in xs if x > thr)
    return 100.0*k/len(xs)
def main():
    print(f"{'log':50s} {'N':>3} {'SR':>5} {'lat<5':>6} {'lat<3':>6} {'md<5':>6} {'md<8':>6} {'ang<10':>7} {'lat_med':>8} {'md_med':>8}")
    for p in sys.argv[1:]:
        eps = parse(p)
        if not eps:
            print(f"{p:50s} -- empty --"); continue
        sr = 100.0*sum(1 for e in eps if e["status"].startswith("SUCCESS"))/len(eps)
        lat5 = rate([e["lat"] for e in eps], 5)
        lat3 = rate([e["lat"] for e in eps], 3)
        md5 = rate([e["min_d"] for e in eps], 5)
        md8 = rate([e["min_d"] for e in eps], 8)
        ang10 = rate([e["ang"] for e in eps], 10)
        lm = med([e["lat"] for e in eps])
        mm = med([e["min_d"] for e in eps])
        name = p.split("/")[-1][:50]
        print(f"{name:50s} {len(eps):3d} {sr:5.1f} {lat5:6.1f} {lat3:6.1f} {md5:6.1f} {md8:6.1f} {ang10:7.1f} {lm:8.2f} {mm:8.2f}")
if __name__ == "__main__":
    main()
