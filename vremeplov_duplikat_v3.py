import csv
import os
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from math import comb

import numpy as np


CSV_PATH = "/data/loto7hh_4626_k44.csv"
N = 39
K = 7
RUNS = int(os.getenv("RUNS", "100000"))
SEED = int(os.getenv("SEED", "39"))
MODE = os.getenv("MODE", "history_only")
TOPN = int(os.getenv("TOPN", "10"))
TIMELINE_CAP = int(os.getenv("TIMELINE_CAP", "200000"))


def read_combinations(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combo = tuple(int(row[f"Num{i}"]) for i in range(1, K + 1))
            rows.append(tuple(sorted(combo)))
    return rows


def lex_rank_1(combo):
    rank = 1
    previous = 0
    for i, x in enumerate(combo, start=1):
        for v in range(previous + 1, x):
            rank += comb(N - v, K - i)
        previous = x
    return rank


def lex_unrank_1(rank):
    combo = []
    previous = 0
    remaining_rank = rank
    for i in range(1, K + 1):
        for v in range(previous + 1, N + 1):
            block_size = comb(N - v, K - i)
            if remaining_rank > block_size:
                remaining_rank -= block_size
            else:
                combo.append(v)
                previous = v
                break
    return tuple(combo)


def simulate_history_only(unique_combos, runs, seed):
    # ANALITICKI + VEKTORIZOVANO:
    # skup vidjenih je fiksan (4625), svako izvlacenje pogadja taj skup sa p = unique/M.
    # vreme do prvog pogotka ~ Geometric(p); koja kombinacija = uniformno po skupu.
    rng = np.random.default_rng(seed)
    space = comb(N, K)
    unique = len(unique_combos)
    p = unique / space

    waiting = rng.geometric(p, size=runs)
    hit_idx = rng.integers(0, unique, size=runs)
    return waiting, hit_idx


def simulate_timeline(unique_combos, runs, seed, cap):
    # VEKTORIZOVANO preko survival CDF:
    # skup raste sa svakim NE-pogotkom; P(nema pogotka u koraku j) = 1-(H+j)/M.
    rng = np.random.default_rng(seed)
    space = comb(N, K)
    H = len(unique_combos)

    j = np.arange(cap, dtype=np.float64)
    seen_sizes = np.minimum(H + j, space)
    q = np.clip(1.0 - seen_sizes / space, 0.0, 1.0)

    log_survival = np.concatenate(([0.0], np.cumsum(np.log(np.where(q > 0, q, 1e-300)))))
    survival = np.exp(log_survival)              # S(n) = P(T > n)
    cdf = 1.0 - survival                          # F(n) = P(T <= n)

    u = rng.random(runs)
    waiting = np.searchsorted(cdf, u, side="left")
    waiting = np.clip(waiting, 1, cap)
    return waiting


def main():
    getcontext().prec = 40

    rows = read_combinations(CSV_PATH)
    positions_by_combo = defaultdict(list)
    for position, combo in enumerate(rows, start=1):
        positions_by_combo[combo].append(position)

    unique_combos = sorted(set(rows))
    duplicates_to_delete = len(rows) - len(unique_combos)
    full_space = comb(N, K)
    expected_duplicates = Decimal(comb(len(rows), 2)) / Decimal(full_space)
    next_hit_rate = Decimal(len(unique_combos)) / Decimal(full_space)

    print()
    print("izvlacenja:", len(rows))
    print("jedinstvenih kombinacija:", len(unique_combos))
    print("duplih za brisanje:", duplicates_to_delete)
    print("ceo prostor C(39,7):", full_space)
    print("ocekivani duplikati do sada:", expected_duplicates.quantize(Decimal("1.0000000000")))
    print("stopa sledeceg hita u istorijski skup:", next_hit_rate.quantize(Decimal("1.0000000000")))
    print("teorijski prosek koraka (1/p):", f"{full_space / len(unique_combos):.10f}")
    print()

    duplicate_groups = {combo: positions for combo, positions in positions_by_combo.items() if len(positions) > 1}
    print("postojeci duplikati:")
    for combo, positions in duplicate_groups.items():
        rank = lex_rank_1(combo)
        print("pozicije:", positions, "kombinacija:", combo, "lex_1:", rank)
        if lex_unrank_1(rank) != combo:
            raise RuntimeError("lex rank/unrank provera nije prosla")
    print()

    print("vremeplov mode:", MODE)
    print("runs:", RUNS)
    print("seed:", SEED)

    if MODE == "history_only":
        waiting, hit_idx = simulate_history_only(unique_combos, RUNS, SEED)
    elif MODE == "timeline":
        waiting = simulate_timeline(unique_combos, RUNS, SEED, TIMELINE_CAP)
        hit_idx = None
    else:
        raise ValueError("MODE mora biti history_only ili timeline")

    print("prosek koraka do sledeceg duplikata:", f"{waiting.mean():.10f}")
    print("medijana:", float(np.median(waiting)))
    print("min:", int(waiting.min()))
    print("p10:", float(np.percentile(waiting, 10)))
    print("p25:", float(np.percentile(waiting, 25)))
    print("p75:", float(np.percentile(waiting, 75)))
    print("p90:", float(np.percentile(waiting, 90)))
    print("max:", int(waiting.max()))
    print()

    if hit_idx is not None:
        counts = Counter(hit_idx.tolist())
        print(f"top {TOPN} preporuka (najcesce ponovljena kombinacija u simulaciji):")
        for idx, count in counts.most_common(TOPN):
            combo = unique_combos[idx]
            print("count:", count, "lex_1:", lex_rank_1(combo), "kombinacija:", combo)
        print()
        print("napomena: uniformno -> ocekivano count po kombinaciji ~", f"{RUNS / len(unique_combos):.4f}")
        print()


if __name__ == "__main__":
    main()


"""
100k simulacija
/bin/python /data/vremeplov_duplikat_v3.py


izvlacenja: 4626
jedinstvenih kombinacija: 4625
duplih za brisanje: 1
ceo prostor C(39,7): 15380937
ocekivani duplikati do sada: 0.6955119184
stopa sledeceg hita u istorijski skup: 0.0003006969
teorijski prosek koraka (1/p): 3325.6080000000

postojeci duplikati:
pozicije: [2262, 4047] kombinacija: (8, 16, 19, 23, 29, 31, 37) lex_1: 12632941

vremeplov mode: history_only
runs: 100000
seed: 39
prosek koraka do sledeceg duplikata: 3318.7801600000
medijana: 2307.0
min: 1
p10: 350.0
p25: 954.0
p75: 4603.0
p90: 7621.0
max: 40552

top 10 preporuka (najcesce ponovljena kombinacija u simulaciji):
count: 41 lex_1: 1841202 kombinacija: (1, x, 20, y, 26, z, 37)
count: 40 lex_1: 13556745 kombinacija: (10, x, 17, y, 21, z, 37)
count: 38 lex_1: 3745367 kombinacija: (2, x, 7, y, 10, z, 35)
count: 38 lex_1: 12803498 kombinacija: (9, x, 13, y, 29, z, 38)
count: 38 lex_1: 6917798 kombinacija: (3, x, 20, y, 24, z, 35)
count: 37 lex_1: 7265142 kombinacija: (4, x, 15, y, 22, z, 32)
count: 37 lex_1: 3097854 kombinacija: (2, x, 16, y, 28, z, 34)
count: 37 lex_1: 1515766 kombinacija: (1, x, 10, y, 20, z, 32)
count: 36 lex_1: 1849335 kombinacija: (1, x, 23, y, 31, z, 39)
count: 36 lex_1: 7732216 kombinacija: (4, x, 19, y, 29, z, 37)

napomena: uniformno -> ocekivano count po kombinaciji ~ 21.6216









1 milion simulacija
RUNS=1000000 /bin/python /data/vremeplov_duplikat_v3.py


izvlacenja: 4626
jedinstvenih kombinacija: 4625
duplih za brisanje: 1
ceo prostor C(39,7): 15380937
ocekivani duplikati do sada: 0.6955119184
stopa sledeceg hita u istorijski skup: 0.0003006969
teorijski prosek koraka (1/p): 3325.6080000000

postojeci duplikati:
pozicije: [2262, 4047] kombinacija: (8, 16, 19, 23, 29, 31, 37) lex_1: 12632941

vremeplov mode: history_only
runs: 1000000
seed: 39
prosek koraka do sledeceg duplikata: 3319.9963080000
medijana: 2304.0
min: 1
p10: 349.0
p25: 954.0
p75: 4603.0
p90: 7650.0
max: 47323

top 10 preporuka (najcesce ponovljena kombinacija u simulaciji):
count: 273 lex_1: 3999357 kombinacija: (2, x, 8, y, 19, z, 39)
count: 262 lex_1: 8360653 kombinacija: (4, x, 14, y, 16, z, 38)
count: 261 lex_1: 8516 kombinacija: (1, x, 3, y, 10, z, 31)
count: 260 lex_1: 2270267 kombinacija: (1, x, 20, y, 22, z, 34)
count: 259 lex_1: 10178044 kombinacija: (6, x, 18, y, 25, z, 30)
count: 259 lex_1: 13705726 kombinacija: (10, x, 20, y, 32, z, 39)
count: 258 lex_1: 10076263 kombinacija: (6, x, 10, y, 25, z, 39)
count: 258 lex_1: 12561635 kombinacija: (8, x, 20, y, 27, z, 30)
count: 257 lex_1: 6885014 kombinacija: (3, x, 20, y, 28, z, 33)
count: 257 lex_1: 13682811 kombinacija: (10, x, 25, y, 33, z, 36)

napomena: uniformno -> ocekivano count po kombinaciji ~ 216.2162










MODE=timeline /bin/python /data/vremeplov_duplikat_v3.py


izvlacenja: 4626
jedinstvenih kombinacija: 4625
duplih za brisanje: 1
ceo prostor C(39,7): 15380937
ocekivani duplikati do sada: 0.6955119184
stopa sledeceg hita u istorijski skup: 0.0003006969
teorijski prosek koraka (1/p): 3325.6080000000

postojeci duplikati:
pozicije: [2262, 4047] kombinacija: (8, 16, 19, 23, 29, 31, 37) lex_1: 12632941

vremeplov mode: timeline
runs: 100000
seed: 39
prosek koraka do sledeceg duplikata: 2360.8931500000
medijana: 1919.0
min: 1
p10: 340.0
p25: 879.0
p75: 3394.0
p90: 5014.0
max: 16182
"""





"""
Radi brzo (~0.24s za 100k). 

Šta je bolje u v3:
RUNS=100000 podrazumevano
history_only sada analitički + vektorizovano (Geometric(p) za waiting-time + uniformno za kombinaciju) → 
100k za ~0.24s; skalira i na milione trenutno
timeline vektorizovan preko survival CDF (bez sporog loop-a)
dodato teorijski prosek koraka (1/p) = 3325.61 da odmah vidim poklapanje sa simulacijom (3340.83)
Rezultat 100k: prosek koraka do sledeceg duplikata: 3318.7801600000
medijana: 2307.0 — taman oko teorijskih 3326. 
„Top 10“ je skoro spljošteno (count ~36–41, očekivano ~21.6), što i potvrđuje da nema favorita.

Komande:
RUNS=1000000 /bin/python /data/vremeplov_duplikat_v3.py
MODE=timeline /bin/python /data/vremeplov_duplikat_v3.py

Prva komanda:
RUNS=1000000 /bin/python /data/vremeplov_duplikat_v3.py
Radi 1 milion simulacija u osnovnom modu history_only. 
To je glavna verzija: 
traži sledeći duplikat koji pogađa neku od već izvučenih 4625 jedinstvenih kombinacija.

Druga komanda:
MODE=timeline /bin/python /data/vremeplov_duplikat_v3.py
Radi alternativni mod timeline: 
posle svakog simuliranog novog izvlačenja, ako nije duplikat, 
dodaje ga u „viđene“, pa šansa za sledeći duplikat raste kroz simuliranu budućnost.




Za 100k samo pokreneš bez RUNS=:

/Users/4c/qiskit_env/bin/python /Users/4c/Desktop/GHQ/data/vremeplov_duplikat_v3.py
Komande sa RUNS=1000000 (milion) i MODE=timeline su dodatne opcije, ne zamena za 100k.



2262 ---> 12,632,941 ---> 8,16,19,23,29,31,37                     
4047 ---> 12,632,941 ---> 8,16,19,23,29,31,37 

račun za očekivane duplikate:
1. Duplikat je očekivana slučajnost, ne signal. 
4626 izvlačenja u prostoru od 15,380,937. 
Broj parova je: 
parovi: 4626·4625/2 = 10,697,625
0,697,625 / 15,380,937 ≈ 0.6955119184
Dakle teorija očekuje ~0.70 duplikata, a ima 1 — 
i dalje potpuno u skladu sa slučajnošću — nema „skrivene strukture“.

Iz 0.70 (očekivani broj duplikata kroz 4626 izvlačenja) može se izvući stopa.

Trenutna verovatnoća da baš sledeće izvlačenje napravi duplikat ≈ broj već izvučenih / ceo prostor:

4626 / 15,380,937 ≈ 0.0003 ≈ 1/3325

Znači očekivani sledeći duplikat je tek za ~3325 izvlačenja 
(i taj razmak se vremenom skraćuje kako lista raste, jer kumulativ ide kao t²/2N).

Drugim rečima: duplikati su retki, na razmaku od više hiljada izvlačenja. 
0.70 i kaže „za sad bi očekivao manje od jednog“, a desio se 1 — normalno.

To je odgovor i na „kad sledeći duplikat“.




vremenska mašina: ne čekam 3300 stvarnih izvlačenja, 
nego pustim algoritam da „izvrti“ budućnost (Monte Carlo) 
i registruje prvi sledeći duplikat — 
koju kombinaciju ponovi i posle koliko simuliranih izvlačenja.

Učitam 4626 istorijskih u set (kao „viđene“).
Nastavim da izvlačim slučajne 7/39 kombinacije (nastavak istorije).
Čim padne kombinacija koja je već u setu → to je „sledeći duplikat“. 
Zabeležim: koja kombinacija, posle koliko koraka.
Ponovim run N puta → dobijem raspodelu: 
prosečno vreme do duplikata i koje se kombinacije ponavljaju.
Koja kombinacija će biti ponovljena ispada uniformno (svih 4625 podjednako verovatno), 
pa simulacija ne može da favorizuje jednu. 
Ono što realno dobijem je raspodela „posle koliko izvlačenja“ (waiting time) — 
i ona se lepo poklopi sa ~3300 proseka.


Precizno, na 10 decimala:
parovi: C(4626,2) = 10,697,625
prostor: C(39,7) = 15,380,937
očekivani broj duplikata = 0.6955119184
stopa po izvlačenju = 4626 / 15,380,937 = 0.0003007619
Dakle ne 0.70 zaokruženo, nego 0.6955119184.

Kod lex ranga je sve celobrojno i egzaktno, pa shift od 1 = susedna kombinacija, ne ista. 
Zato mora da zaključamo konvenciju i drži svuda isto:

0-based rang {8,16,19,23,29,31,37} = 12,632,940
1-based = 12,632,941

rank/unrank, čista celobrojnu aritmetiku (bez float zaokruživanja) i jedna fiksna konvenciju (1-based) — 
pa da unrank(rank(komb)) == komb uvek vraća istu kombinaciju.



Cilj: 
predvideti sledeći duplikat preko simulacije (vremeplov), 
radeći u podskupu izvučenih, sa egzaktnim lex rank/unrank (1-based).

4626 izvlačenja, prostor C(39,7) = 15,380,937
očekivani duplikati do sad: 0.6955119184; 
stopa po izvlačenju 0.0003007619
sledeći duplikat = ponavljanje jedne od 4625 jedinstvenih izvučenih; 
prosek ~3300 izvlačenja unapred
brisanje duplikata = df.duplicated() (k−1 po grupi)

Koraci:
Učitam 4626 → set jedinstvenih (proverim da ih je 4625).
Lex rank/unrank (egzaktno, 1-based) — validacija unrank(rank(x))==x.
Simulator „vremeplov“: 
nastavak izvlačenja dok ne padne već viđena → 
beležim korak i kombinaciju; N runova → raspodela waiting-time.


rank(prva kombinacija) = 1, rank(poslednja) = 15,380,937
unrank(1) = {1,2,3,4,5,6,7}, unrank(15,380,937) = {33,34,35,36,37,38,39}
garancija: unrank(rank(x)) == x
"""
