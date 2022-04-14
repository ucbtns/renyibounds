# 🧠 Bayesian Brains and the Rényi Divergence
We used the Renyi variational inference to evaluate differences in posterior estimates and agent behaviour.  

For pre-requisite, see requirements.txt

## Numerical posterior estimation:

```
python main.py --configs unisim 
```
![mu](https://github.com/ucbtns/renyibounds/blob/main/figures/univ_muq.png)
![sigma](https://github.com/ucbtns/renyibounds/blob/main/figures/univ_sigma_q.png)

## Multiarmed Bandits:

```
python main.py --configs mabsim multisim
```

## Citation:
If you use this code please reference the accompanying [paper](https://watermark.silverchair.com/neco_a_01484.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAr8wggK7BgkqhkiG9w0BBwagggKsMIICqAIBADCCAqEGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM-Jv7DdMeAKozwSVTAgEQgIICcnBBfwDKJI-zgVQeA8TIC-KGUh8n37I5IPDZlsA90DCrzfCPEK-3IMZxH89rUOxU0ujxgNUkormmNMAK-9dvCB2D7JOQkXwOzT-Xb26DNCoyfI1sQZ-BLKr7toeO2C_GG7ufzpPb0XO0UzHyAOH6JAelOZPvFB_QZfw9imoioPmi-Oi0ZXV_gpcXvbzxyoVpuq8ba5ldu4EOzybOlB-MUq_XaNay5wISpkm-K5xKSg46HWLAF8IoqT8jQKLvLwfv_SWmPexE1uCE88mhq2rbi9Q_KPo0e2-lJWSb_ljFwQf5o7T55gJkuuu889lntDbLjrVJrfIk5CTjPlC0mY4ZhvLya-nxciAXjfxO5bHRRbyUVEqwyJjzBGobQ4F8VW64SJHnBmWTLEW1YM7fHoWuErIN6_F2JzY1fhkb3yOG9AQAV44LqFRgP3iIJWhxt6OB5_dRUVANpeM34l7_xu_KX7dAXk1uRj1VxxkIKQiwYBXwwKs7ho2m1EoU6GDzRUIS_WCuy2qt3IfB0qlXyKgeKnQyKE6jPPy0Xz4oTh2QI3RLizQa_eXPgYJaorhFYmeg8-t5LcNRNZCO_KvOe2mcBQwB1iG_UNQ3u_I_uY3BnvkIuhZhNJcBYpLTeXOXQAdNTnY5_gx6b3GqCjMjyC7muIB2AnbrmthZgHJM09liP3U8WVdfAL5kwy0TI5zdKPMV1vWVWr_4mkBlT76xETP1Ze_rn8gwOR2cYR7L-pDk2K7_r7LZREJpj-BUTg61hJaIwCcyeT0J_Hsbr1JRol88xcgsClSCBAlKV01twxce6V-q79qg8ybOmiHjqF59I6YQrUVZ
):
```
@article{10.1162/neco_a_01484,
    author = {Sajid, Noor and Faccio, Francesco and Da Costa, Lancelot and Parr, Thomas and Schmidhuber, Jürgen and Friston, Karl},
    title = "{Bayesian Brains and the Rényi Divergence}",
    journal = {Neural Computation},
    volume = {34},
    number = {4},
    pages = {829-855},
    year = {2022},
    month = {03},
    issn = {0899-7667},
    doi = {10.1162/neco_a_01484},
    url = {https://doi.org/10.1162/neco\_a\_01484},
    eprint = {https://direct.mit.edu/neco/article-pdf/34/4/829/2003077/neco\_a\_01484.pdf},
}
```