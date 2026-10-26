# 🧪 VALIDATION_SPEC.md (Çözücü Doğrulama Kapısı)

**Kural:** Bu belgedeki ölçütler sağlanmadan optimizasyon kodu çalıştırılamaz (AGENTS.md kural 1). Sonuçlar `03_Code_Execution/test_validation.py` tarafından üretilir ve `04_Results/VALIDATION_REPORT.md`'ye fiilen çalıştırılan değerlerle yazılır.

Kaynak: `00_Reference/mathematics-2021543-revised.md` — Akgöz, B.; Civalek, Ö. *Buckling Analysis of Functionally Graded Tapered Microbeams via Rayleigh–Ritz Method*, Mathematics 2022 (revize taslak). Koniklik (V1, V2, V6): $A(x)=A_L(1-\alpha x/L)$, $I(x)=I_L(1-\alpha x/L)$ (Denk. 35–36; genişlik konikliği, $h$ sabit). V3'te $A\propto(1-\alpha\xi)^2$, $I\propto(1-\alpha\xi)^4$ (aşağıya bakın). Boyutsuz yük: $\lambda = P_{cr}L^2/(EI)$ (homojen).

---

## V1. Tablo 1 — Homojen konik konsol, klasik ($l_0=l_1=l_2=0$), BC1 $\varphi_i = x^{i+1}$
| $\alpha$ | N=2 | N=4 | N=6 | N=8 | N=10 | Kesin [49] |
|---|---|---|---|---|---|---|
| 0 | 2.4860 | 2.4674 | 2.4674 | 2.4674 | 2.4674 | 2.467 |
| 0.1 | 2.4165 | 2.3928 | 2.3928 | 2.3928 | 2.3928 | 2.393 |
| 0.3 | 2.2702 | 2.2351 | 2.2351 | 2.2351 | 2.2351 | 2.235 |
| 0.5 | 2.1078 | 2.0621 | 2.0621 | 2.0621 | 2.0621 | 2.062 |
| 0.7 | 1.9140 | 1.8655 | 1.8653 | 1.8653 | 1.8653 | 1.865 |
| 0.9 | 1.6462 | 1.6229 | 1.6212 | 1.6211 | 1.6211 | 1.621 |
**Ölçüt:** her $(\alpha, N)$ hücresinde 4 ondalık basamakta eşleşme (yuvarlama farkı ±1e-4). Ek kontrol: $\alpha=0$, $N=2$ için analitik $\lambda_1 = 2.4860$, $\lambda_2 = 32.1807$ (Denk. 61).

## V2. Tablo 2 — Homojen konik mesnetli konsol, klasik, BC1 $\varphi_i = x^{i+1}(L-x)$
| $\alpha$ | N=2 | N=4 | N=6 | N=8 | N=10 | Kesin [49] |
|---|---|---|---|---|---|---|
| 0 | 20.9187 | 20.1943 | 20.1907 | 20.1907 | 20.1907 | 20.1907 |
| 0.1 | 19.9908 | 19.1736 | 19.1686 | 19.1686 | 19.1686 | 19.17 |
| 0.3 | 18.0621 | 17.0426 | 17.0354 | 17.0353 | 17.0353 | 17.03 |
| 0.5 | 15.9584 | 14.7452 | 14.7394 | 14.7394 | 14.7394 | 14.74 |
| 0.7 | 13.4459 | 12.1775 | 12.1773 | 12.1772 | 12.1772 | 12.18 |
| 0.9 | 9.7113 | 9.0608 | 9.0307 | 9.0295 | 9.0294 | 9.029 |
**Ölçüt:** 4 ondalık basamakta eşleşme.

## V3. Ref. [40] Tablo 2 (= Mathematics 2022 Tablo 3) — Homojen konik epoksi konsol, MSGT-1/MSGT-2 ($l_0=l_1=l_2=h_1$)
**Geometri ve koniklik (Ref. [40] Denk. 15–16, Tablo 2 başlığı):** dikdörtgen kesit; $h(x)=h_1(1-\alpha x/L)$, $b(x)=b_1(1-\alpha x/L)$ (ikisi birlikte daralır) ⇒ $A=A_1(1-\alpha\xi)^2$, $I=I_1(1-\alpha\xi)^4$. **Bu, V1/V2'deki $I\propto(1-\alpha\xi)$ yasasından farklıdır**; çözücü koniklik yasasını girdi olarak (genel $A(x),I(x)$) almalıdır.
**Girdiler (birebir, Ref. [40] Conclusions girişi, dergi s. 201):** "E=1.44 GPa, v=0.38, L=20h1, b1=2h1, d1=h1, l0=l1=l2=l." (Tabloda $l=h_1$). Boyutsuz yük: $P_{cr}^*=P_{cr}L^2/(EI_1)$. BC1: $x^{i+1}$, BC2: $x^{i+2}$ (C–F), $N=10$.
**Kaynak:** değerler Ref. [40] Tablo 2'dedir (`Papers/sem4802003.pdf`, SHA-256 `3552c4cc34b61c51fee4f31deb9cf64094a55075e3c4a7307c6153704acb0193`, PDF s. 8 / dergi s. 202); Mathematics 2022 Tablo 3 ("Present" sütunu) aynı değerleri 5 ondalıkla verir. [40] Tablo 3 dairesel kesittir (V3b).
| $\alpha$ | CT [40] | MSGT-1=BC1 present | BC1 [40] | MSGT-2=BC2 present | BC2 [40] |
|---|---|---|---|---|---|
| 0 | 2.4674 | 40.38752 | 40.3875 | 41.52736 | 41.5274 |
| 0.2 | 1.8835 | 35.20555 | 35.2055 | 36.04467 | 36.0447 |
| 0.4 | 1.3093 | 29.81004 | 29.8100 | 30.38676 | 30.3868 |
| 0.6 | 0.7566 | 24.04835 | 24.0483 | 24.40082 | 24.4008 |
| 0.8 | 0.2644 | 17.51253 | 17.5125 | 17.67871 | 17.6787 |
**Ölçüt:** BC1/BC2 için 5 anlamlı basamak; CT sütunu 4 ondalık. MCST sütunu kapsam dışı.
**V3b (dairesel kesit, [40] Tablo 3, $d(x)=d_1(1-\alpha\xi)$, $l=d_1$; ek kontrol):** MSGT-1 = 53.0226, 46.3045, 39.2868, 31.7650, 23.1951; MSGT-2 = 54.3880, 47.3112, 39.9799, 32.1895, 23.3959 ($\alpha=0,0.2,\dots,0.8$).
**Durum:** değerlerin çözücüyle yeniden üretilebilirliği çözücü testleriyle belirlenecek. Üretilemezse `DOĞRULANAMADI` ve kullanıcıya rapor; değer elle ayarlanmaz. Poisson: $\nu$ yalnız $G=E/2(1+\nu)$ içinde (PROJECT_STATE Not P1).

## V4. MATLAB eşdeğerliği (`00_Reference/*.m`)
- `cantilevermsgt.m` varsayılan girdileri ($E_L=2000$, $E_R=0.5E_L$, $\nu=0.3$, $h_1=10$, $b_1=h_1$, $L=20h_1$, $l_L=h_1$, $l_R=0.5l_L$, $e=0.3$, $f=0$, $n=1$, $K_w=K_p=0$, $N=15$, BC1) ve `proppedmsgt.m` varsayılan girdileri (homojen $E_R=E_L$, klasik $l_L=0$, $e=0.3$, $N=15$, BC1).
- **Ölçüt:** MATLAB'in verdiği en küçük boyutsuz yükle bağıl fark ≤ 1e-6. MATLAB çıktısı kullanıcıdan alınır veya MATLAB ile çalıştırılır; değer elle girilmez. MATLAB yoksa Python–SymPy ile aynı sembolik akış bağımsız referans olarak koşulur.

## V5. Sayısal sağlamlık
- $N$ = 2…15 yakınsama tablosu (monom tabanı $\xi = x/L$ ile; kötü koşullanma kontrolü: $\mathbf K_G$ koşul sayısı raporlanır).
- Kuadratür nokta sayısına karşı değişim < 1e-10.
- $\mathbf K$, $\mathbf K_G$ simetrik; $\mathbf K_G$ pozitif tanımlı; $P_{cr} > 0$.

## V6. İkincil kontrol (önerilir): Tablo 4–5 (AFG, $E_L=2E_R$, $l_L=2l_R$, $\nu=0.3$, $l_L=h_1$; $\alpha=0.25$ ve $0.75$)
- CT, SGT-BC1, SGT-BC2 sütunları; konsol ve mesnetli konsol; $n = 0, 0.5, 1, 2, 10$. Tablo değerleri `00_Reference/mathematics-2021543-revised.md` içindedir; $L/h_1$ ve boyutsuzlaştırma MATLAB varsayımlarıyla ($L=20h_1$, $\lambda=P L^2/(E_R I_0)$) tutarlı olmalıdır — tutarsızlık çıkarsa kullanıcıya raporlanır.

## V7. Kimlik testleri (Critic önerisi, Round 4)
- Düzgün dikdörtgen kesitte $D_1/EI = 1 + \dfrac{6\,(2+8/15+1)}{1+\nu}\,(l/h)^2$ (≈ $1+21{,}2/(1+\nu)\,(l/h)^2$); çözücünün düzgün kolon $P_{cr}$ oranı bu değerle $D_2$ payı dışında tutarlı olmalı (tolerans: $D_2/(EI L^2)$ terimi kadar; test sınırı Developer'ca belirlenir, değer elle girilmez). Beklenen $
u=0{,}3$: $l/h=0{,}1\to1{,}16$; $0{,}2\to1{,}65$; $0{,}5\to5{,}08$; $1\to17{,}3$.
- $l=0$ ⇒ CT sonucu $\nu$'den bağımsız; $G_i$ faz başına karışım (Denk. 38) birim testi.
- $\varepsilon_s$ ve $\kappa$ kısıt hesabı birim testi (hakem türetmesi; kaynaksız eşik `DOĞRULANAMADI`).
