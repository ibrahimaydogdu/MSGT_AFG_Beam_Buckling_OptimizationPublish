"""Builds 04_Results/RESULTS_SUMMARY.md: every number is read from stored files (results/**, results/post/**,
VALIDATION_REPORT.md); nothing is typed in. Intended as the data pack for Section 4 (Results and Discussion).
Usage: python -I -X utf8 results_summary.py [--awsm V2]"""
import argparse
import collections
import csv
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
POST = os.path.join(RES, "post")
OUT = os.path.normpath(os.path.join(HERE, "..", "04_Results", "RESULTS_SUMMARY.md"))
ALGS = ("awsm", "nsga2", "moead")
NAMES = {"awsm": "AWSM-ABC-Levy", "nsga2": "NSGA-II", "moead": "MOEA/D"}


def rcsv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def fl(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def med(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def table(header, rows, fmt="{:.4g}"):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(fmt.format(c) if isinstance(c, (float, np.floating)) else str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awsm", default="V2")
    a = ap.parse_args()
    L = ["# RESULTS_SUMMARY (Bölüm 4 veri paketi)\n",
         "*Üretici: `03_Code_Execution/results_summary.py` (kural 6.6: her sayı kayıtlı dosyadan okunur). Kaynaklar: `results/<kampanya>/` koşu kayıtları, "
         f"`results/three_algorithms_{a.awsm}/`, `results/post/`, `04_Results/VALIDATION_REPORT.md`. AWSM = `{a.awsm}` kampanyaları; NSGA-II ve MOEA/D 30 000 FE. "
         "Bu dosya yazarın kullanacağı sayıları verir; yorum ve 'yazılabilir/yazılamaz' sınırları `RESULTS_WRITING_BRIEF.md` içindedir.*\n"]

    # 1. verification
    vr = open(os.path.join(HERE, "..", "04_Results", "VALIDATION_REPORT.md"), encoding="utf-8").read()
    m = re.search(r"## Özet\n\n(.*?)\n\n## ", vr, flags=re.S)
    L.append("## 1. Çözücü doğrulaması (kaynak: VALIDATION_REPORT.md)\n\n" + (m.group(1) if m else "Özet bulunamadı") + "\n")

    # 2. run facts
    L.append("## 2. Koşu bilgileri (kayıtlı meta)\n")
    rows = []
    camps = {"awsm": f"main{a.awsm}", "nsga2": "main", "moead": "mainM"}
    for alg in ALGS:
        fe, wall, sc, rf = [], [], [], []
        n = 0
        for f in glob.glob(os.path.join(RES, camps[alg], "*", "*.npz")):
            z = np.load(f, allow_pickle=False)
            mt = json.loads(str(z["meta"]))
            if mt["alg"] != alg:
                continue
            n += 1
            fe.append(mt["nfe"])
            wall.append(mt["wall_s"])
            if alg == "awsm":
                sc.append(mt["extra"].get("n_scout") or 0)
                rf.append(mt["extra"].get("n_refined") or 0)
        rows.append([NAMES[alg], n, med(fe), med(wall), med(sc) if sc else "—", med(rf) if rf else "—"])
    L.append(table(["Algoritma", "koşu (ana kampanya)", "FE medyan", "süre medyan [s]", "Lévy (öncü) olayı medyan", "inceltme ile eklenen ağırlık medyan"], rows) + "\n")

    # 3. P* per scenario
    P = np.load(os.path.join(POST, "select", "Pstar.npz"))
    rows = []
    for s in P.files:
        p = P[s]
        rows.append([s, len(p), p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()])
    L.append("## 3. Birleşik Pareto cephesi P* (üç algoritma, 30 koşu; yalnız yürütülen aramadan)\n\n" +
             table(["Senaryo", "|P*|", "min f1bar", "maks f1bar", "min λ", "maks λ"], rows) + "\n")

    # 4. algorithm comparison
    tag = f"three_algorithms_{a.awsm}"
    ranks = json.load(open(os.path.join(RES, tag, "ranks.json"), encoding="utf-8"))
    rows = []
    for mname, d in ranks.items():
        mr = d["mean_rank"]
        rows.append([mname, d["N"], d["chi2"], d["p"], d["CD_0.05"]] + [mr[NAMES[x]] for x in ALGS])
    L.append("## 4. Algoritma karşılaştırması (Friedman, blok = 12 ana senaryo; sıra 1 = en iyi)\n\n" +
             table(["Metrik", "N", "χ²", "p", "CD (Nemenyi 0.05)"] + [f"ort. sıra {NAMES[x]}" for x in ALGS], rows) + "\n")
    mt = rcsv(os.path.join(RES, tag, "metrics.csv"))
    for metric in ("HV", "IGD", "SP"):
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in mt:
            if r["group"] == "main":
                g[r["scenario"]][r["alg"]].append(fl(r[metric]))
        rows = [[s] + [med(g[s][x]) for x in ALGS] for s in sorted(g)]
        L.append(f"### 4.{('HV', 'IGD', 'SP').index(metric) + 1} {metric} medyanları (ana senaryolar)\n\n" +
                 table(["Senaryo"] + [NAMES[x] for x in ALGS], rows) + "\n")

    # 5. extremes
    L.append("## 5. Cephe uçları: koşu başına min f1bar ve maks λ'nın medyanı (30 koşu)\n")
    rows = []
    scen = sorted(os.listdir(os.path.join(RES, "main")))
    for s in scen:
        if not os.path.isdir(os.path.join(RES, "main", s)):
            continue
        rec = [s]
        for alg in ALGS:
            mn, mx = [], []
            for f in glob.glob(os.path.join(RES, camps[alg], s, "*.npz")):
                z = np.load(f, allow_pickle=False)
                if json.loads(str(z["meta"]))["alg"] != alg or len(z["front"]) == 0:
                    continue
                mn.append(z["front"][:, 0].min())
                mx.append(z["front"][:, 1].max())
            rec += [med(mn), med(mx)]
        rows.append(rec)
    L.append(table(["Senaryo"] + [f"{NAMES[x]} {q}" for x in ALGS for q in ("min f1bar", "maks λ")], rows) + "\n")

    # 6. convexity
    cv = rcsv(os.path.join(POST, "H_convexity", "front_convexity_measure.csv"))
    rows = []
    for fr in ("P* (union)", "AWSM", "NSGA-II", "MOEA/D"):
        v = [fl(r["fraction_above_hull_0.01"]) for r in cv if r["front"] == fr]
        gp = [fl(r["median_gap"]) for r in cv if r["front"] == fr]
        hv_ = [int(r["hull_vertices"]) for r in cv if r["front"] == fr]
        rows.append([fr, len(v), min(v), max(v), med(gp), min(hv_), max(hv_)])
    L.append("## 6. Ölçülen konvekslik (normalize (f1bar, 1−λ) uzayında alt konveks zarf; kural 2: konvekslik ölçülür, varsayılmaz)\n\n" +
             table(["Cephe", "senaryo", "zarfın >0.01 üstündeki nokta oranı min", "maks", "medyan boşluk (medyan)", "zarf köşe min", "maks"], rows) + "\n")

    # 7. size effect at representative designs
    sw = rcsv(os.path.join(POST, "B_sweep", "mu_sweep.csv"))
    rows = []
    for s in sorted({r["scenario"] for r in sw}):
        for pt in ("q0.00", "q0.25", "q0.50", "q0.75", "q1.00", "knee"):
            rec = [s, pt]
            for mu in ("0.1", "0.2", "0.5", "1.0"):
                v = [fl(r["dP_cr_percent_vs_CT"]) for r in sw if r["scenario"] == s and r["point"] == pt and r["BC"] == "BC1" and r["l_over_h0"] == mu]
                rec.append(v[0] if v else float("nan"))
            rows.append(rec)
    L.append("## 7. Boyut etkisi taraması: BC1 ΔP_cr [%] (CT'ye göre; 0.5 ve 1.0 'geçerlilik sınırı dışı, yalnız eğilim')\n\n" +
             table(["Senaryo (tasarımın optimize edildiği l/h0)", "nokta", "l/h0=0.1", "0.2", "0.5", "1.0"], rows) + "\n")

    # 8. CT <-> MSGT cross evaluation
    cr = rcsv(os.path.join(POST, "C_cross", "ct_msgt_cross.csv"))
    rows = [[r["pair"], r["BC"], r["l_over_h0"], r["quantity"], fl(r["value"])] for r in cr]
    L.append("## 8. CT ↔ MSGT çapraz değerlendirme (aynı HV referansı; λ oranı kütle eşleştirmeli, |Δm|/m ≤ %1)\n\n" +
             table(["Çift", "BC", "l/h0", "Nicelik", "Değer"], rows) + "\n")

    # 9. BC2
    bc = rcsv(os.path.join(POST, "A_bc2", "bc2_reevaluation.csv"))
    rows = []
    for s in sorted({r["scenario"] for r in bc}):
        v = [fl(r["BC2/BC1_N15"]) for r in bc if r["scenario"] == s]
        c = [fl(r["lam_BC2_N18"]) / fl(r["lam_BC2_N15"]) - 1 for r in bc if r["scenario"] == s]
        rows.append([s, min(v), med(v), max(v), max(abs(x) for x in c)])
    L.append("## 9. BC1 ile optimize edilen tasarımların BC2 ile yeniden değerlendirilmesi (N=15; temsilî 6 nokta)\n\n" +
             table(["Senaryo", "λ_BC2/λ_BC1 min", "medyan", "maks", "BC2 N=15→18 göreli değişim (maks mutlak)"], rows) + "\n")

    # 10. ablation
    ab = rcsv(os.path.join(POST, "D_ablation", "l_of_x_ablation.csv"))
    rows = []
    for r_ in ("2.0", "0.5"):
        d1 = [fl(r["dlam_percent_vs_const_l"]) for r in ab if r["r (artificial ratio)"] == r_]
        d2 = [fl(r["dlam_percent_vs_equal_mean_l"]) for r in ab if r["r (artificial ratio)"] == r_]
        rows.append([r_, len(d1), med(d1), min(d1), max(d1), med(d2), min(d2), max(d2)])
    L.append("## 10. l(x) = l[1+(r−1)V_r] ablasyonu ('yapay oran'; Al/SiC C–F; tasarım sabit)\n\n" +
             table(["r", "n", "Δλ% sabit-l'ye göre medyan", "min", "maks", "Δλ% eşit-ortalama-l'ye göre medyan", "min", "maks"], rows) + "\n")

    # 11. benchmarks
    bs = rcsv(os.path.join(POST, "E_bench", "benchmark_summary.csv"))
    bm = rcsv(os.path.join(POST, "E_bench", "benchmark_mass_matched.csv"))
    rows = [[r["scenario"], r["n_nondominated_benchmarks"], fl(r["HV_benchmark"]), fl(r["HV_optimised_Pstar"]), fl(r["HV_ratio"]),
             med([fl(x["lam_opt/lam_bench"]) for x in bm if x["scenario"] == r["scenario"]])] for r in bs]
    L.append("## 11. Referans sütunlar (üniform b=h ölçek ailesi ve doğrusal konik aileler; doğrudan değerlendirme) ve optimize cephe\n\n" +
             table(["Senaryo", "baskın olmayan referans sayısı", "HV referans", "HV P*", "HV oranı", "λ_opt/λ_ref medyan (kütle eşleşmeli, ≤%2)"], rows) + "\n")

    # 12. sections
    sc_ = rcsv(os.path.join(POST, "G_sections", "equal_area_sections.csv"))
    rows = []
    for s in sorted({r["scenario"] for r in sc_}):
        rows.append([s, med([fl(r["circle/rect"]) for r in sc_ if r["scenario"] == s]), med([fl(r["ellipse/rect"]) for r in sc_ if r["scenario"] == s])])
    L.append("## 12. Eşit alanlı dairesel/eliptik kesit dönüşümü (yeniden optimizasyon yok; Critic Round 12 S1)\n\n" +
             table(["Senaryo", "λ_daire/λ_dikdörtgen medyan", "λ_elips/λ_dikdörtgen medyan"], rows) + "\n")

    # 13. profile descriptors and active constraints
    pf = rcsv(os.path.join(POST, "F_profiles", "representative_profiles_summary.csv"))
    rows = []
    for s in sorted({r["scenario"] for r in pf}):
        k = [r for r in pf if r["scenario"] == s and r["point"] == "knee"][0]
        rows.append([s, fl(k["f1bar"]), fl(k["lam"]), fl(k["eps_s_max"]), fl(k["min_h_over_l"]), fl(k["max_D1_over_EI"]),
                     k["n_b_eq_h"], k["n_h_at_min"], k["n_V_at_0"], k["n_V_at_1"], fl(k["Vbar"])])
    L.append("## 13. Diz noktası tasarımlarının betimleyicileri (ε_s ≤ %5 taraması; aktif kısıtlar = kontrol noktası sayıları /6)\n\n" +
             table(["Senaryo", "f1bar", "λ", "ε_s maks", "min h/l", "maks D1/EI", "n(b=h)", "n(h=h_min)", "n(V=0)", "n(V=1)", "V̄"], rows) + "\n")
    allp = [fl(r["eps_s_max"]) for r in pf]
    L.append(f"Tüm temsilî tasarımlarda (n={len(allp)}) ε_s maksimumu: {max(allp):.4g}; %5 eşiğini aşan tasarım sayısı: {sum(1 for x in allp if x > 0.05)}.\n")

    # 14. sensitivity to the number of representative points
    sn = rcsv(os.path.join(POST, "S_sensitivity", "n_points_sensitivity.csv"))
    rows = []
    for s in sorted({r["scenario"] for r in sn}):
        for n in ("5", "9"):
            d = [r for r in sn if r["scenario"] == s and r["n_points"] == n]
            vm = [fl(r["V_mean"]) for r in d]
            rows.append([s, n, vm[0], vm[-1], float(np.corrcoef(np.arange(len(vm)), vm)[0, 1]) if len(vm) > 2 else float("nan")])
    L.append("## 14. Temsilî nokta sayısı duyarlılığı (5 ve 9 nokta): ortalama V_r eğilimi (ilk nokta, son nokta, sıra ile korelasyon)\n\n" +
             table(["Senaryo", "nokta sayısı", "V_mean (kütle ucu)", "V_mean (λ ucu)", "korelasyon"], rows) + "\n")
    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
