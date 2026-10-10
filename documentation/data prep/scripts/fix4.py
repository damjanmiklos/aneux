p = 'C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s = open(p, encoding='utf-8', newline='').read()
a = s.index(r"\includegraphics[width=0.245\textwidth]{figures/ko_p305__in.png}")
b = s.index(r"\label{fig:dp_keepone}")
new = r'''\begin{minipage}{0.49\textwidth}\centering
        \includegraphics[width=\textwidth]{figures/ko_p305__in.png}\\[-1mm]
        {\footnotesize (a) input}
    \end{minipage}\hfill
    \begin{minipage}{0.49\textwidth}\centering
        \includegraphics[width=\textwidth]{figures/ko_p305__1_out.png}\\[-1mm]
        {\footnotesize (b) sample 1}
    \end{minipage}\\[2mm]
    \begin{minipage}{0.49\textwidth}\centering
        \includegraphics[width=\textwidth]{figures/ko_p305__2_out.png}\\[-1mm]
        {\footnotesize (c) sample 2}
    \end{minipage}\hfill
    \begin{minipage}{0.49\textwidth}\centering
        \includegraphics[width=\textwidth]{figures/ko_p305__3_out.png}\\[-1mm]
        {\footnotesize (d) sample 3}
    \end{minipage}
    \caption{Single-aneurysm isolation on a vessel with three aneurysms. (a) The input with all three sacs highlighted in red. (b--d) The three samples generated from it. Each keeps one aneurysm untouched; the wall reconstructed in place of the other two is shown in blue. Everywhere else the sample is identical to the input.}
    '''
s = s[:a] + new + s[b:]
old = r"(c, d) An opening before and after: the oblique, uneven rim is replaced by a pipe section perpendicular to the centreline."
assert old in s
s = s.replace(old, r"(c, d) An opening before and after, seen obliquely from outside the vessel: the uneven rim is replaced by a planar pipe-section cut perpendicular to the centreline.")
open(p, 'w', encoding='utf-8', newline='').write(s)
print('ok')
