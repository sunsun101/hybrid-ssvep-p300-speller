% ---------- boiler plate stuff ---------- 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aitthesis}[2022/05/03 Asian Institute of Technology Thesis Class]

% ---------- inherit the report package to be used as the base ----------
\LoadClass[12pt]{report}

% ---------- package for setting titleformat and titlespacing ----------
\RequirePackage{titlesec}

% ---------- set word-like font ----------
\RequirePackage[T1]{fontenc}
\RequirePackage{tgheros}%% TeX Gyre Sans Serif
\RequirePackage{tgtermes}%%           TeX Gyre Times

\renewcommand\large{\@setfontsize\large{14pt}{14pt}}
\renewcommand\normalsize{\@setfontsize\normalsize{12pt}{12pt}}
\renewcommand\small{\@setfontsize\small{11pt}{11pt}}

% ---------- set correct margins ----------
\RequirePackage[a4paper, margin=1in, left=1.5in]{geometry}

% ---------- set correct line spacing ----------
\linespread{1.75}

% ---------- set correct chapter style ----------
\renewcommand{\chaptername}{CHAPTER}

\def\@makechapterhead#1{
  {\centering
    \ifnum \c@secnumdepth >\m@ne
        \large\bfseries \@chapapp\space \thechapter
        \\
    \fi
    \interlinepenalty\@M
    \large \bfseries #1
    \\
    \vskip 2em
  }}
\def\@makeschapterhead#1{
  {\centering
    \interlinepenalty\@M
    \large \bfseries  #1\par\nobreak
    \vskip 20pt
  }}

% ---------- set correct section/subsection style ----------

\titleformat{\section}
  {\normalfont\fontsize{12}{1em}\bfseries}{\thesection}{6pt}{}
\titleformat{\subsection}
  {\normalfont\fontsize{12}{1em}\bfseries\itshape}{\thesubsection}{6pt}{}

\titlespacing\section{0pt}{0pt plus 0pt minus 0pt}{-12pt plus 0pt minus 0pt}
\titlespacing\subsection{0pt}{0pt plus 0pt minus 0pt}{-12pt plus 0pt minus 0pt}

% ---------- set correct paragraph style ----------

\setlength\parindent{0pt}
\setlength{\parskip}{12pt}%

% ---------- table of content style ----------
\RequirePackage[titles]{tocloft}

\renewcommand{\cftchapfont}{\bfseries}
\renewcommand{\cftchappagefont}{\bfseries}
\renewcommand{\cftchappresnum}{CHAPTER }
\renewcommand{\cftchapaftersnum}{}
\renewcommand{\cftchapnumwidth}{6em}
\renewcommand{\cftdot}{}

\renewcommand\contentsname{\centering\uppercase{\large{\bf Contents}}}
\renewcommand\listtablename{\centering\uppercase{\large{\bf List of Tables}}}
\renewcommand\listfigurename{\centering\uppercase{\large{\bf List of Figures}}}

\renewcommand{\cftsecindent}{1in}
\renewcommand{\cftsecnumwidth}{2em}
\renewcommand{\cftsubsecindent}{1.5in}
\renewcommand{\cftsubsecnumwidth}{2.5em}
\renewcommand{\cftsubsubsecindent}{2.0in}
\renewcommand{\cftsubsubsecnumwidth}{4.7em}

\renewcommand{\cfttabpresnum}{Table }
\renewcommand{\cfttabaftersnum}{}
\renewcommand{\cfttabnumwidth}{4.5em}
\renewcommand{\cfttabindent}{0em}

\renewcommand{\cftfigpresnum}{Figure }
\renewcommand{\cftfigaftersnum}{}
\renewcommand{\cftfignumwidth}{5em}
\renewcommand{\cftfigindent}{0em}

% ---------- table/figure style ----------

\RequirePackage{caption}
\captionsetup{
  singlelinecheck=false,
  textfont=it
}

\captionsetup[table]{labelfont=bf,name=Table\space,labelsep=newline}
\captionsetup[figure]{labelfont=bf,name=Figure\space,labelsep=newline}

\RequirePackage[bottom]{footmisc}

% ---------- reference style ----------

\RequirePackage{hyperref}
\RequirePackage[natbibapa]{apacite}

\renewcommand\bibsection{\centering{\large{\textbf{REFERENCES}}}}

% ---------- list/enumerate style ----------

\RequirePackage{enumitem}
\setlist[itemize]{noitemsep, topsep=-12pt}
\setlist[enumerate]{noitemsep, topsep=-12pt}