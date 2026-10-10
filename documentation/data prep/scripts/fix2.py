p = 'C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s = open(p, encoding='utf-8').read()
R = [
(r"About a fifth of patients with an intracranial aneurysm have more than one \citep{juvela2000}, and the database reflects this:",
 r"A considerable fraction of patients with an intracranial aneurysm have more than one \citep{juvela2000}, and the database reflects this:"),
(r"For a sidewall aneurysm the seeds mark the parent artery upstream and downstream of the sac and the top of the sac. For a bifurcation aneurysm they mark the parent inflow and the two daughter branches. This is the only per-aneurysm manual input, and it takes a few seconds per aneurysm.",
 r"Every aneurysm receives a seed on the parent inflow and one on the top of the sac. A sidewall aneurysm additionally receives one seed on the parent downstream of the sac, and a bifurcation aneurysm one on each of the two daughter branches. This is the only manual input to the removal."),
(r"Altogether 2388 extensions were removed, a median of 8 per vessel and at most 13.",
 r"Altogether 2388 cuts were made, a median of 8 per vessel and at most 13."),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
print('ok')
