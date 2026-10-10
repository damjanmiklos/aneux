p='C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s=open(p,encoding='utf-8',newline='').read()
old=r"The capped vessels were opened by hand in Blender, since a flat cap is easy to recognise visually but awkward to separate reliably from a blunt vessel end by automatic means. The extensions"
new=r"The capped vessels were opened by hand in Blender, since a flat cap is easy to recognise visually but awkward to separate reliably from a blunt vessel end by automatic means. Figure~\ref{fig:dp_uncap} shows a typical difficult case. The cap closes a short, curved side branch that lies against the parent vessel, so the cap is oblique and its rim is irregular. A cut at the plane of the cap would either leave a sliver of cap behind or bite into the neighbouring wall, and the right place to cut had to be judged by eye. The extensions"
assert old in s
s=s.replace(old,new)
anchor=r"\subsection{Removal of flow extensions}"
fig=r'''\begin{figure}[H]
    \centering
    \begin{minipage}{0.49\textwidth}\centering
        \includegraphics[width=\textwidth]{figures/uncap_USFD54_before.png}\[-1mm]
        {\footnotesize (a) as published (capped)}
    \end{minipage}\hfill
    \begin{minipage}{0.49\textwidth}\centering
        \includegraphics[width=\textwidth]{figures/uncap_USFD54_after.png}\[-1mm]
        {\footnotesize (b) opened by hand}
    \end{minipage}
    \caption{One outlet of a capped vessel before and after manual uncapping, seen obliquely from outside the vessel. The flat, tilted cap on a short side branch (a) is removed so that the branch ends in an open rim (b), through whose lumen the dark interior is visible. The rest of the surface is unchanged.}
    \label{fig:dp_uncap}
\end{figure}

'''
assert anchor in s
s=s.replace(anchor,fig+anchor,1)
open(p,'w',encoding='utf-8',newline='').write(s)
