<!--block:B0001-->
# ARS canonical manuscript source snapshot

<!--block:B0002-->
This review artifact contains the manuscript source files and a hash manifest for image assets. It is not the submission PDF.

<!--block:B0003-->
## manuscript/main.tex

<!--block:B0004-->
````latex
% MSc Computational Finance dissertation, September 2026.
% Front matter follows the CS MSc LaTeX template. Body format follows the
% official instructions: 12pt, 1.5 spacing, 2.5 cm margins, A4.
% Compile from the repository root with: make manuscript
\documentclass[12pt,a4paper,oneside]{book}

\usepackage[british]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{array}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{emptypage}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{microtype}
\usepackage{multirow}
\usepackage[authoryear,round]{natbib}
\usepackage{pdflscape}
\usepackage{placeins}
\usepackage{setspace}
\usepackage[margin=2.5cm]{geometry}
\usepackage[titletoc,title]{appendix}
\usepackage[sc]{titlesec}
\usepackage[hidelinks]{hyperref}
\usepackage{doi}
\usepackage[capitalise,noabbrev]{cleveref}

\hypersetup{
  pdftitle={Does News Sentiment Go Beyond the Mean? Development Evidence, Temporal Non-Replication and Economic Limits},
  pdfauthor={Peter Prendergast},
  pdfsubject={MSc Computational Finance dissertation}
}

\graphicspath{{artifacts/}{figures/}{./}}
\onehalfspacing
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\emergencystretch=3em

% Numeric table columns: align on the decimal point and set a real minus sign
% rather than the hyphen a bare text-mode "-" produces.
\sisetup{
  detect-all,
  group-digits = integer,
  group-minimum-digits = 4,
  group-separator = {,},
}
\newcolumntype{d}[1]{S[table-format=#1]}

% Captions: slightly smaller than body text, bold label, hanging indent so a
% wrapped caption lines up under its own first line rather than the label.
\captionsetup{
  font = small,
  labelfont = bf,
  labelsep = period,
  format = hang,
  justification = justified,
  singlelinecheck = false,
  skip = 6pt,
}

% Discourage single lines stranded at the top or foot of a page.
\widowpenalty = 10000
\clubpenalty = 10000
\displaywidowpenalty = 10000

\title{Does News Sentiment Go Beyond the Mean?\\Development Evidence, Temporal Non-Replication and Economic Limits}
\author{Peter Prendergast}
\date{September 2026}

\begin{document}
\pagestyle{plain}
\frontmatter
\hypersetup{pageanchor=false}

\begin{titlepage}
  \newcommand{\HRule}{\rule{\linewidth}{0.5mm}}
  \centering
  \textsc{\LARGE University College London}\\[0.7cm]
  \textsc{\Large Department of Computer Science}\\[0.3cm]
  {\normalsize A thesis submitted in partial fulfilment of the requirements for the degree of Master of Science in Computational Finance, University College London}\\[0.45cm]

  \HRule\\[0.25cm]
  {\LARGE Does News Sentiment Go Beyond the Mean?}\\[0.2cm]
  {\large Development Evidence, Temporal Non-Replication and Economic Limits}\\[0.25cm]
  \HRule\\[0.7cm]

  {\large\textit{Author}}\\
  Peter \textsc{Prendergast}\\[0.8cm]

  \begin{minipage}{0.48\textwidth}
    \begin{flushleft}
      \textit{Academic Supervisor}\\
      Dr Brian \textsc{Healy}\\
      \textsc{Department of Computer Science}\\
      \textsc{University College London}
    \end{flushleft}
  \end{minipage}
  ~
  \begin{minipage}{0.48\textwidth}
    \begin{flushright}
      \textit{Industrial Supervisor}\\
      Dr Peter \textsc{Mitic}\\
      \textsc{Department of Computer Science}\\
      \textsc{University College London}
    \end{flushright}
  \end{minipage}

  \vfill
  {\large\textit{September 2026}}\\[0.45cm]
  \includegraphics[width=0.16\textwidth]{ucl_logo}\\[0.45cm]

  \begin{minipage}{0.92\textwidth}
    \centering
    \small
    This dissertation is submitted as partial requirement for the MSc Computational Finance degree at UCL. It is substantially the result of my own work except where explicitly indicated in the text.

    \medskip
    The dissertation will be distributed to the internal and external examiners, but thereafter may not be copied or distributed except with permission from the author.
  \end{minipage}
\end{titlepage}

\clearpage
\hypersetup{pageanchor=true}
\pagenumbering{arabic}
\setcounter{page}{1}

\section*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}

Financial-news systems usually average story-level sentiment into one
company-day score. Counting the share of stories that cross a negative decision
boundary retains different information. This dissertation asks whether that
negative-story share is conditionally associated with the abnormal return
following a firm's news-assigned session after average sentiment, news volume
and recent prices are controlled, and whether any association is stable through
time. The panel is sparse: 48.3\% of the 512,153 development firm-days carry
one story and the median carries two.

The study uses 715,546 news-bearing firm-days for 570 US equities from FNSPID.
Development ends on 31 December 2019. Nine aggregation rules are compared under
one outcome and inference procedure, followed by daily cross-sectional rank
regressions. The selected estimands are then repeated once on 203,393
firm-days from 2020--2023 under a frozen specification and an outcome-blind
amendment. That later block is chronological but not a pristine holdout because
other project analyses had viewed later outcomes. Reuters/LSEG data for 33 firms
remain a separate portability arm. Economic tests include turnover, costs and
matched-exposure risk comparisons.

Negative-story share alone survives correction in development. With
one-session, five-session and volatility controls, its coefficient is
$-0.00914$, with a 95\% interval of $[-0.01466,-0.00362]$. The frozen
evaluation coefficient is instead $+0.00583$, with interval
$[-0.00426,+0.01593]$ and BH $q=0.515$; the multi-story coefficient and the
nine-rule family also fail. The predeclared evaluation-minus-development
contrast is $+0.01414$, with interval $[+0.00250,+0.02579]$ and $p=0.0173$.
Prospective power to recover the development effect at the first BH-2 hurdle
was only 35.4\%, so the evaluation null is not proof of zero, while the direct
contrast is evidence of temporal instability.

The development return scale is small: a model-free spread is $-0.618$ basis
points per session with an interval crossing zero, and the best break-even cost
is $0.625$ basis points per side against 10 assumed. LSEG estimates are
imprecise, while tuned risk and prompt rules fail their economic gates. The
dissertation's contribution is therefore a preserved temporal
non-replication: a selected historical association survives several
development controls but does not transport to a later FNSPID era or support a
portable, cost-covering rule.

\clearpage
\section*{Acknowledgements}
\addcontentsline{toc}{chapter}{Acknowledgements}

I thank Dr Peter Mitic and Dr Brian Healy for their supervision and for pressing the project to separate an interesting statistical result from a usable investment rule.

\clearpage
\tableofcontents
\clearpage
\listoffigures
\clearpage
\listoftables

\mainmatter
\include{chapters/01_introduction}
\include{chapters/02_literature}
\include{chapters/03_data}
\include{chapters/04_methodology}
\include{chapters/05_results}
\include{chapters/06_conclusion}

\clearpage
\addcontentsline{toc}{chapter}{Bibliography}
\bibliographystyle{ormsv080}
\bibliography{references}

\begin{appendices}
\include{appendices/reproducibility}
\include{appendices/project_summary}
\end{appendices}

\end{document}
````

<!--block:B0005-->
## manuscript/chapters/01_introduction.tex

<!--block:B0006-->
````latex
\chapter{Introduction}
\label{ch:introduction}

\section{The aggregation problem}

Financial news rarely arrives as one clean observation per firm and trading day. A company may be covered in an earnings alert, a longer results story, an analyst reaction and several updates before the next market open. A sentiment pipeline assigns a score to each item, but a portfolio model needs one value for the firm-day. The usual answer is the arithmetic mean. It is easy to calculate and appears to use all available stories. It also removes information.

Two summaries of the same day can disagree even when the day is sparse. A firm-day with one story reduces to whether that story crossed the scorer's negative boundary; the signed mean instead reports how far it fell from neutral, and two stories that are both mildly negative give a mean near zero while both cross the boundary. On a richer day the gap widens: four mildly neutral stories and a day split between two strongly positive and two strongly negative items can share a mean while the negative counts differ. The aggregation rule therefore determines what the return model is allowed to learn.

The distinction matters most on rich days, but this panel is not rich. Of the 512,153 development firm-days, 48.3\% carry a single story and the median carries two, so the comparison is tested mostly where a day's tone has at most two elements. \Cref{ch:results} returns to this directly: the measured association weakens as the day is required to be denser, and the claim is bounded accordingly.

Existing research gives good reasons to examine the negative part of the distribution. Negative language has been linked with return pressure, earnings information and slower price adjustment in several settings \citep{tetlock2007,tetlock2008words,uhl2014,heston2017}. Public bad news also behaves differently from a large price move without identifiable information: the former can be followed by drift, while the latter is more likely to reverse \citep{chan2003news,savor2012information}. These findings do not show that a negative-story share is useful. They do make it a plausible alternative to an average.

The practical standard is higher than statistical detectability. A daily stock signal can have the expected sign and a small standard error while losing money after turnover, or appear to reduce drawdown merely because it holds less market exposure. This dissertation therefore separates incremental information, return-scale size, and value after costs against a fair comparator.

\section{Research question and hypotheses}

The primary research question is:

\begin{quote}
\textbf{Is the share of negative stories in a firm's assigned-session news
flow conditionally associated with the following abnormal-return rank after
controlling for mean sentiment, news volume and recent price dynamics, and is
that association stable from FNSPID development into 2020--2023?}
\end{quote}

For firm $i$ on session $t$, negative-story share is the number of stories given a negative hard label divided by the total number of stories assigned to that firm-session. ``Conditional'' has a specific meaning here. On each session, the abnormal return following the assigned session is regressed on negative-story share while rank-linear terms for mean continuous sentiment and story count are held fixed. The principal robustness specification also holds fixed the firm's lagged one-session return, lagged five-session return and recent volatility. The object of interest is the time-series mean of the daily negative-share coefficient. It is a development-panel association conditional on the selected workflow and observed controls, not a causal effect of publishing negative news or an effect conditional on every possible function of mean sentiment.

Four hypotheses organise the tests, with the fourth stated as two separable claims so that a statistical verdict and an economic one cannot be conflated:

\begin{enumerate}[label=H\arabic*.,leftmargin=1.1cm]
  \item[H1a.] The conditional coefficient on negative-story share is below zero in development after mean sentiment, story count and the recent price path are included.
  \item[H1b.] The unchanged all-firm-day and multi-story coefficients remain negative in the frozen 2020--2023 replication under their declared correction.
  \item[H2.] A model-free high-minus-low negative-share sort, formed within mean-sentiment buckets, has the same negative direction as the development regression estimate.
  \item[H3.] A daily directional implementation of the statistic earns a positive mean return after its own turnover is charged at 10 basis points per side.
  \item[H4a.] The conditional association carries into the Reuters/LSEG blocks, estimated separately by source regime: the transfer coefficients are negative and distinguishable from zero.
  \item[H4b.] A risk rule built from the statistic beats an exposure-matched control in an LSEG block.
\end{enumerate}

H3 and H4 are stated as claims that the evidence can refute. Their cost and matched-control standards are fixed in \cref{ch:methodology} before the corresponding results are reported.

H1a and the downstream hypotheses organise an exploratory aggregation result;
they were not fixed before negative-story share was selected. H1b is narrower:
its estimands, correction and failure rule were frozen before Notebook 75
opened them once. The later period is not a pristine holdout because other
project work had already used its portfolio outcomes, so the result is called a
frozen temporal replication rather than independent confirmation.

\section{Data regimes and scope}

FNSPID is the primary data spine. The cleaned panel contains 715,546
news-bearing firm-days for 570 priced US symbols between 2011 and 2023. Model
and rule development ends on 31 December 2019. The later period supplies the
one-shot replication of the selected statistical estimands and bounded
risk-translation tests already present in the project. The split is
chronological, but it is not presented as an untouched holdout because earlier
project work had observed later outcomes.

Reuters data supplied through LSEG form a separate robustness arm for 33 large US firms. The backward block contains 456 sessions and the recent block contains 167. LSEG has richer and more recent company coverage, but far fewer firms and sessions. Its text is licensed and is neither reproduced in the dissertation nor committed to the accompanying repository. FNSPID and LSEG are never pooled. Combining them would hide changes in source, coverage, scoring and return definition behind a larger sample count.

The study does not claim causality, market capacity or a production trading system, nor that distributional tone is new as a general idea. Prior work already studies within-document tone dispersion, bond-market sentiment dispersion and event-specific disagreement \citep{allee2015,isakin2023,chen2024}.

\section{Contributions}

The project makes three contributions. First, it compares nine firm-day
summaries and examines the selected hard-boundary statistic against rank-linear
mean-sentiment, coverage and recent-price controls. Because single-story days
dominate, this is principally a measurement comparison. Second, it preserves a
frozen temporal failure: the development association changes sign in
2020--2023, and a predeclared contrast tests that change directly. Third, it
translates the development result into return scale, costs and separate-source
tests, retaining the negative economic and portability evidence. The
contribution is the full evidence sequence, not a universal sentiment signal.

\section{Dissertation structure}

\Cref{ch:literature} reviews financial text, negative-news asymmetry, contextual models, aggregation and predictive-finance standards. \Cref{ch:data} compares the two regimes and their limits. \Cref{ch:methodology} defines the aggregators, estimand, inference, accounting and risk tests. \Cref{ch:results} reports statistical, economic and transfer results, including failed rules and post-hoc measurement limits. \Cref{ch:conclusion} answers the question and proposes independent validation centred on story novelty.
````

<!--block:B0007-->
## manuscript/chapters/02_literature.tex

<!--block:B0008-->
````latex
\chapter{Literature Review}
\label{ch:literature}

\section{Why financial text may contain return information}

Prices respond to public information, but a numerical announcement does not exhaust what investors receive. A news story identifies the event, selects facts, gives context and may state whether the outcome beat expectations. Text can therefore convey both fundamentals and the way those fundamentals are framed.

\citet{tetlock2007} provides an early large-sample link between media language and markets. Pessimism in a daily Wall Street Journal column is associated with downward market pressure, unusually high volume and partial reversal. The reversal matters for this dissertation: a text variable can predict the next return without representing a persistent change in value. \citet{tetlock2008words} move from an aggregate column to firm-specific news. The fraction of negative words predicts earnings and returns, with words tied to fundamentals carrying much of the information. That work supplies a direct precedent for measuring a negative proportion, although its proportion is formed from words within documents rather than from stories within a firm-day.

Several later studies establish that the response depends on what produced the price move. \citet{chan2003news} reports drift following identifiable public news, especially bad news, but reversal after large returns without public news. \citet{savor2012information} reaches a related conclusion for major price shocks, using analyst information to distinguish informed from uninformed moves. Together, these findings create an identification problem for a predictive news study. Negative coverage often appears on days when the stock has already fallen. Unless recent returns are controlled, a negative-news statistic may simply label a reversal state.

The effect of media also depends on dissemination and market conditions. \citet{fang2009} relate media coverage breadth to expected returns, while \citet{engelberg2011} use geographic variation in access to local newspapers to show that the distribution channel affects investor trading around earnings. \citet{garcia2013} finds that return predictability from positive and negative language is concentrated in recessions. These studies make universal claims about a news coefficient hard to defend. Coverage, who sees the story and the prevailing state can change its meaning.

Reuters-based evidence is especially relevant to the LSEG arm. Using 3.6 million Reuters articles, \citet{uhl2014} finds that negative sentiment has greater forecasting power than positive sentiment in an index-level time-series model. \citet{heston2017} analyse more than 900,000 Reuters stories and report short daily predictability, longer weekly effects and slower response to negative news. Neither study tests the exact firm-day statistic used here, but both support an expected asymmetry: concentrated negative coverage may retain information that a signed average attenuates.

\section{Measurement: from word lists to contextual models}

Text must be converted into a variable before any return test is possible. A general dictionary counts words that have negative meanings in ordinary English. In finance that can be misleading. \citet{loughran2011liability} show that common business words such as ``liability'' and ``tax'' are often classed as negative by a general dictionary even when their use is routine. A finance-specific word list changes both the labels and the empirical conclusions. Their later survey catalogues further decisions that affect financial text results, including corpus construction, negation, document length and validation against the target task \citep{loughran2016}.

Supervised benchmarks and contextual language models address part of this problem. Financial PhraseBank contains sentences labelled by people with financial knowledge and records disagreement between annotators \citep{malo2014}. FinBERT adapts a pretrained transformer to financial sentiment classification \citep{araci2019}. A separate finance-pretrained FinBERT studied by \citet{huang2023} improves on dictionary and conventional machine-learning methods in labelled-text and market-reaction tasks. The dissertation uses a pinned ProsusAI FinBERT checkpoint to prevent scorer selection from becoming another experiment. The cited models support domain adaptation as a design choice; they do not prove that this checkpoint labels every FNSPID or Reuters headline correctly.

Large language models allow the scoring instruction itself to specify the target. \citet{kirtac2024} compare dictionary, BERT-family and large language models on nearly one million US financial-news articles and report strong results for a large language model. \citet{lopezlira2026}, in an advance-online Journal of Financial Economics article, ask a language model whether a headline is good or bad for the named company and find delayed return response, particularly for negative news and smaller firms, with weaker performance in later periods. These studies motivate the prompt experiments reported later, but they also sharpen the test: a tailored score should beat an existing scorer on the same dates after costs, not merely produce plausible explanations.

Historical scoring by a modern model creates a temporal risk. The model may have absorbed facts that became known after the article date. \citet{glasserman2023} anonymise company identities to separate look-ahead knowledge from distraction caused by background knowledge. Their evidence shows that historical large-language-model sentiment is not automatically leakage safe. The current study therefore treats all prompted results as retrospective development experiments, sends no returns to the scorer and stops when the fixed selection gate fails.

\section{What an average discards}

Most return studies require one signal for an asset and period. The document-level scores are commonly averaged, selected or placed in a time-series model, but the aggregation choice is seldom the object of a controlled comparison. Equal weighting assumes that each additional story contributes one interchangeable observation. It does not distinguish independent news from a rewritten update, and it makes a strong negative item easier to offset with positive or neutral coverage.

The broader literature shows that non-mean text structure can matter, so the contribution here must be stated narrowly. \citet{allee2015} measure how positive and negative words are dispersed through conference-call narratives and find relations with firm performance, reporting choices and user responses. Their unit is word placement within a disclosure, not multiple articles within a day. \citet{isakin2023} relate dispersion in news sentiment to corporate-bond returns and uncertainty. \citet{chen2024} study news-sentiment dispersion around Chinese merger announcements and link it with uncertainty, announcement returns and completion. These papers rule out any claim that sentiment dispersion itself is new.

Other work changes which news is retained. \citet{uhl2021} construct a selected ``top news'' sentiment measure and show that indiscriminately processing a large flow can dilute useful information. Selection is a different answer to the same measurement problem. Rather than choose one story using later return relevance, the present comparison fixes nine return-blind summaries and applies one outcome and inference design to all of them.

Negative-story share has an intuitive but imperfect interpretation. Let four stories receive hard labels $(-1,-1,+1,+1)$. Their mean hard label is zero and their negative share is one half. If the positive stories become neutral, the negative share stays at one half while the mean falls. Conversely, if all stories are mildly negative, the negative share is one although the continuous mean may be close to zero. The two statistics therefore answer different questions. The mean measures the balance and magnitude of signed scores; negative share measures how much of the observed flow crosses the model's negative decision boundary.

This distinction creates the main unresolved confound. \citet{tetlock2011stale} show that stale and reprinted stories receive a different market response and can be followed by reversal. If one adverse event generates several updates, negative-story share can measure editorial repetition rather than several pieces of bad information. Story count alone cannot solve this because it records volume but not family membership. FNSPID's retained checkpoint lacks stable story-family and publisher identifiers. A development regression can estimate information conditional on the mean and count, but it cannot establish information beyond novelty or temporal stability. That limitation governs the interpretation of the selected association and its later non-replication.

\section{From predictability to an investment claim}

An information coefficient is not a portfolio return. A daily cross-sectional association can be too small to cover the trades needed to realise it. This gap is well documented in empirical finance. \citet{korajczyk2004} show that proportional costs and price impact materially change momentum profitability and capacity. \citet{novymarx2016} find that high-turnover anomalies are particularly vulnerable to costs and that more patient implementations can preserve more of the spread. These results support two choices made here: report turnover and net returns beside gross statistics, and calculate the per-side break-even cost rather than treating one assumed cost as exact.

Trying many summaries, thresholds and conditioning variables introduces a second gap between a coefficient and a credible claim. \citet{harvey2016multiple} document how multiple testing weakens isolated return predictors. The Benjamini--Hochberg procedure controls the expected false-discovery proportion within a declared family and is used for the nine-rule comparison and other related tests \citep{benjamini1995}. It is not a cure for an analysis chosen after results are seen. This dissertation therefore reports the selection history and calls the main result exploratory.

Serial dependence also affects uncertainty. Daily coefficient and portfolio series can remain correlated across adjacent sessions. Heteroskedasticity-and-autocorrelation-consistent standard errors follow \citet{newey1987}; portfolio differences use blocks so that resampled observations retain short-run dependence. \citet{politis1994} provide a formal basis for block resampling of weakly dependent time series, although the implementation here uses fixed circular blocks rather than their stationary bootstrap.

A confidence interval that crosses zero can mean either that an effect is small or that the sample is weak. Minimum detectable effects (MDEs) express the effect magnitude a design could detect with chosen size and power \citep{bloom1995}. They are used here for the short LSEG blocks. If a block could detect only a coefficient several times larger than the FNSPID estimate, its null is evidence of low precision, not evidence that the FNSPID-scale association is absent.

\section{Risk management as a separate use case}

A weak directional signal might still identify periods when reducing exposure is useful. That claim requires an independent risk model and a comparator with the same average exposure. Otherwise, any strategy that spends time in cash will usually report lower volatility and smaller losses.

The base used here is a heterogeneous autoregressive (HAR) volatility forecast. \citet{corsi2009har} model realised volatility through daily, weekly and monthly components, capturing persistence with a parsimonious linear structure. Volatility management then scales exposure down when forecast risk is high. \citet{moreira2017} show that inverse-volatility scaling can improve risk-adjusted outcomes across several factors. Neither paper validates sentiment timing. They provide a sentiment-free base against which a news overlay can be judged.

Three comparisons are needed. The first asks whether the overlay improves on the frozen HAR path. The second replaces the time-varying news rule with a constant multiplier chosen from development data to match exposure. The third shifts the observed risk-off schedule through time while preserving how often it is active. Improvement against the first comparison may reflect lower exposure. Improvement against the second and unusually good placement relative to the shifts are stronger evidence of timing.

\section{Research gap}

The located literature establishes that financial language can contain return
information, that negative news often differs from positive news, and that text
structure or selection can matter. It also shows why costs, dependence and
repeated testing must be visible. The contribution here is an evidence
sequence: several return-blind firm-day summaries compared on one development
panel, followed by conditional estimation, a frozen later-era replication,
economic translation and separate-source transfer checks.

The dissertation addresses that bounded gap. It does not introduce a new sentiment model. It asks whether a simple hard-boundary summary has an exploratory association beyond the rank-linear controls used by a mean-based pipeline. The design then tests the useful version of the result through recent-price controls, cost accounting, matched-exposure comparisons, power calculations and LSEG evidence. This ordering makes a null economic answer informative rather than an embarrassment to be tuned away.
````

<!--block:B0009-->
## manuscript/chapters/03_data.tex

<!--block:B0010-->
````latex
\chapter{Data and Panel Construction}
\label{ch:data}

\section{FNSPID primary panel}

FNSPID combines a large public financial-news collection with US equity prices \citep{dong2024}. The published dataset contains about 15.7 million news records and 29.7 million price observations. Those headline counts describe the source archive, not the estimation sample. This dissertation uses a cleaned coherent cohort covering 2011--2023, then requires a valid firm mapping, a tradable adjusted open and a contemporaneous SPY open. The final panel contains 715,546 news-bearing firm-days, 570 priced symbols and 3,262 exchange sessions. Of these firm-days, 512,153 fall in development and 203,393 in the later block. Four development rows lack a complete primary aggregation outcome, leaving 512,149 observations for the nine-rule comparison. Same-day coverage is thin and its distribution matters for how the aggregation comparison should be read: of the 512,153 development firm-days, 247,470 (48.3\%) carry one story, 127,541 (24.9\%) carry two and 137,142 (26.8\%) carry three or more. On the single-story half of the panel a negative-story share is exactly an indicator that the one story crossed the negative boundary, so the aggregation question this dissertation poses is tested mostly where a firm-day has at most two elements to aggregate.

The unit is $(i,t)$: one priced firm and the exchange session to which its news is assigned. Story-level text is never placed in the panel. The scoring checkpoint supplies the symbol, assigned session, recap flag and three FinBERT class probabilities. Headlines remain in a local database. This separation reduces the risk of leaking licensed or copyrighted text into intermediate results.

News timing follows the frozen mapping used to build the scoring checkpoint. When only a publication date is available, the story is assigned to the first New York Stock Exchange session strictly after that date. More precise timestamps retain the checkpoint's earlier mapping. This conservative date-only rule avoids using a story at an open that may have occurred before publication, but it can delay genuinely pre-open news by one session. The retained FNSPID checkpoint contains neither a verifiable availability timestamp nor a precision flag at the row grain, so it cannot recover the affected fraction or repair the mapping after scoring. Throughout the dissertation, ``next-session'' for FNSPID therefore means the abnormal return following the conservatively assigned session, not a guarantee that every interval begins at the first open after a verified publication time.

Daily prices come from the FNSPID archive. A split-adjusted open is calculated as

\begin{equation}
 P^{\mathrm{adj,open}}_{it}=P^{\mathrm{open}}_{it}
 \frac{P^{\mathrm{adj,close}}_{it}}{P^{\mathrm{close}}_{it}}.
 \label{eq:adjusted-open}
\end{equation}

This adjustment handles splits but does not fold dividends into the open-to-open return. The next observation is the immediately following SPY-calendar session, not the firm's next news day. That distinction is material: shifting within a news-bearing panel would create multi-session stock returns for firms with no story on the intervening days. The primary outcome is

\begin{equation}
 r^{A}_{i,t+1}=\left(\frac{P^{\mathrm{adj,open}}_{i,t+1}}
 {P^{\mathrm{adj,open}}_{it}}-1\right)-
 \left(\frac{P^{\mathrm{open}}_{\mathrm{SPY},t+1}}
 {P^{\mathrm{open}}_{\mathrm{SPY},t}}-1\right).
 \label{eq:abnormal-return}
\end{equation}

This stock-minus-SPY return removes the market's open-to-open move but is not a factor-model alpha. It is a short-horizon market-adjusted return of the kind used in event studies, with a simpler benchmark than a fitted factor model \citep{mackinlay1997}. It is chosen because the question concerns the cross-section of firm news over one executable session. A later post-hoc check residualises close-to-close returns against trailing Fama--French betas \citep{famafrench1993}; that construction is a different estimand, not a replacement for \cref{eq:abnormal-return}.

\section{Sentiment observations}

The pinned scorer is the ProsusAI FinBERT financial-sentiment classifier, used without model selection in the final experiment. Both data regimes use model identifier \texttt{ProsusAI/finbert} at revision \texttt{4556d13015211d73\allowbreak dccd3fdd39d39232506f3e43}, loaded from an enforced local snapshot. For story $j$ about firm $i$ on assigned session $t$, the continuous score is

\begin{equation}
 s_{ijt}=p_{ijt}(\mathrm{positive})-p_{ijt}(\mathrm{negative}),
 \qquad -1\leq s_{ijt}\leq 1.
 \label{eq:continuous-score}
\end{equation}

The hard label is the class with the largest of the positive, neutral and negative probabilities. A story is counted as negative only when the negative probability is the largest. This means negative-story share depends on the classifier's decision boundary. A probability of 0.40 can be labelled negative if the other two probabilities are lower, while a similar score may be neutral in another case. No human-labelled audit of the final FNSPID or Reuters samples is available. The study tests the market information in the model's outputs, not the semantic truth of every label.

The development period runs from 3 January 2011 to 31 December 2019 and
spans 2,264 sessions. The evaluation slice contains 203,393 firm-days across
998 sessions from 2 January 2020 to 18 December 2023. The boundary was frozen
before the final aggregation work, but earlier project iterations had viewed
later portfolio outcomes. The development association is therefore
exploratory, and the later slice is not called a pristine holdout. Notebook 75
nevertheless fixed its nine-rule family, two conditional coefficients,
corrections and failure interpretation before opening those particular
evaluation estimands. It was executed once, with no retuning or second run.

\section{Reuters/LSEG robustness arm}

The second regime contains Reuters coverage for 33 large US companies across sectors, supplied through LSEG. It is useful for three reasons. The stories are recent, the same firms can be followed densely through time, and the data carry richer source and availability fields than the retained FNSPID checkpoint. It is a robustness arm, not a replacement for the large cross-section.

Two non-overlapping blocks are retained. The names are relative to one another rather than to FNSPID: both fall after the FNSPID development period, and the ``backward'' block is the earlier of the two. The backward block runs from 2 January 2024 to 24 October 2025 and contains 14,935 firm-day rows across 456 sessions. The recent block runs from 27 October 2025 to 26 June 2026 and contains 5,511 rows across 167 sessions. Both use the same 33 firms. Median story count is 61 in the backward block and 90 in the recent block, compared with 2 in FNSPID development. A one-unit change in negative share therefore has a different empirical support across sources: in FNSPID it often reflects one of two stories changing class, while in LSEG it is a shift across a much denser daily flow.

LSEG event timing uses the first XNYS open at least fifteen minutes after the recorded story availability. Returns are raw open-to-open firm returns in the conditional transfer table, because the retained LSEG aggregate did not provide the exact FNSPID abnormal-return construction. Coefficients are comparable as centred rank associations, but their outcomes are not identical. Figures and tables state this difference. The blocks are not combined with one another or with FNSPID.

The LSEG economic experiments use a 612-session complete 33-firm panel. For the final firm-level rule search, 445 sessions are used for three chronological development folds and 167 are reserved for the forward evaluation block. The selected rule is judged against an equal-weight long portfolio and two development-selected constant-exposure controls. Licensed story text, prompt payloads and model responses stay in the private source archive. Only aggregate counts, coefficients, performance paths and manifests appear in this repository.

\section{Data limitations}

The most important missing variable is story-family identity in FNSPID. The checkpoint has a recap flag, but no stable publisher field, near-duplicate cluster or link from an update to its first release. Several negative rows may therefore be rewrites of one event. Controlling for $\log(1+n_{it})$ separates distribution shape from the amount of coverage, but it cannot separate independent information from repetition. \citet{tetlock2011stale} show that this distinction affects market response.

Other limitations are unequal source density, classifier error and survivorship in the coherent priced cohort. The panel includes only firm-days with news, a valid symbol, a stock open and a SPY open. It does not estimate what happens on no-news days. The 33 LSEG companies are large and heavily covered, so their transfer results say little about smaller firms. Finally, the FNSPID and LSEG periods differ sharply in market conditions. Separate reporting preserves these differences but cannot remove them.
````

<!--block:B0011-->
## manuscript/chapters/04_methodology.tex

<!--block:B0012-->
````latex
\chapter{Methodology}
\label{ch:methodology}

\section{Design order and claim standard}

The analysis proceeds from measurement to development association, frozen
temporal replication, economic scale and transfer. The initial comparison asks
which firm-day summary has the most consistent relation with the next-open
abnormal return. The selected statistic is then placed in a conditional model
and repeated without selection on the later FNSPID era. Only after those tests
are reported is it translated into a portfolio or risk rule.

This order matters because each stage supports a different statement. A corrected information coefficient can support a statistical association. It cannot support a basis-point forecast without a return-scale calculation. A gross portfolio result cannot support economic value without costs. A lower-drawdown path cannot support news timing unless it beats a rule with comparable exposure. LSEG transfer requires a separate estimate and action test; matching signs alone is insufficient.

All main choices are represented by frozen JSON specifications or immutable
aggregate manifests. The aggregation family was repaired and rerun before the
final research question was fixed, so it remains exploratory. Later robustness
tests respond to that opened association. Notebook 75 differs in one bounded
respect: its evaluation estimands, multiplicity families and failure rule were
fixed before those estimands were opened. The evaluation era was not untouched
by the wider project, so this is a frozen temporal replication rather than
independent confirmation. Reported uncertainty remains conditional on the
selected workflow \citep{harvey2016multiple}.

\section{Nine firm-day aggregation rules}

Let $\mathcal{J}_{it}$ be the stories assigned to firm $i$ and session $t$, $n_{it}=|\mathcal{J}_{it}|$, $s_{ijt}$ the continuous score in \cref{eq:continuous-score}, and $h_{ijt}\in\{-1,0,1\}$ the hard negative, neutral or positive label. Eight rules use only the current firm-day; the ninth carries a decayed state through time.

\begin{enumerate}[leftmargin=0.9cm]
  \item \textbf{Mean hard label:}
  $a^{\mathrm{hard}}_{it}=n_{it}^{-1}\sum_j h_{ijt}$.

  \item \textbf{Mean continuous score:}
  $a^{\mathrm{mean}}_{it}=n_{it}^{-1}\sum_j s_{ijt}$.

  \item \textbf{Median continuous score:}
  $a^{\mathrm{median}}_{it}=\operatorname{median}_{j}(s_{ijt})$.

  \item \textbf{Ten-per-cent trimmed mean:} after sorting the scores, $k=\lfloor0.1n_{it}\rfloor$ observations are removed from each tail and the rest are averaged. With small $n_{it}$, $k=0$ and the rule equals the mean.

  \item \textbf{Negative-story share:}
  \begin{equation}
  q^{-}_{it}=\frac{1}{n_{it}}\sum_{j\in\mathcal{J}_{it}}
  \mathbf{1}(h_{ijt}=-1).
  \label{eq:negative-share}
  \end{equation}

  \item \textbf{Dispersion:} the sample standard deviation of $s_{ijt}$, set to zero for a one-story firm-day.

  \item \textbf{Strongest event:} select a non-recap story before a recap, then choose the remaining story with the largest $|s_{ijt}|$. An exact tie with opposite directions is treated as missing rather than broken using the return.

  \item \textbf{Log-count attention:}
  $a^{\mathrm{attn}}_{it}=a^{\mathrm{mean}}_{it}\log(1+n_{it})$. This does not identify an important story; it gives more cross-sectional weight to a mean observed in a larger flow.

  \item \textbf{Decayed state:} first convert the current continuous mean to an impulse $u_{it}=\operatorname{sign}(a^{\mathrm{mean}}_{it})|a^{\mathrm{mean}}_{it}|^{1.5}$. The previous state decays with a three-session half-life. If the new impulse reverses the state's sign, 75\% of the remaining old state is removed before adding the impulse. The rule represents persistent tone while responding quickly to a reversal.
\end{enumerate}

These rules are intentionally simple. A complex learned aggregator would add another training problem and make it harder to determine what information the mean discarded. The strongest-event and attention rules test plausible operational alternatives. Negative share and dispersion are candidate threshold and shape summaries. The median and trimmed mean test robustness to extreme scores, while the decayed state tests persistence.

\section{Initial aggregation comparison}

For each rule $a$, a Spearman rank correlation is calculated across firms on each eligible session:

\begin{equation}
 IC^a_t=\operatorname{corr}_{S}\left(a_{it},r^A_{i,t+1}\right).
 \label{eq:daily-ic}
\end{equation}

The reported information coefficient is $\overline{IC}^a=T^{-1}\sum_t IC^a_t$. Its standard error uses Newey--West HAC with five lags, allowing heteroskedasticity and short serial dependence in the daily series \citep{newey1987}. All nine rules use the same development sample, outcome and horizon. Their two-sided $p$-values form one family. Benjamini--Hochberg correction at $q=0.05$ is applied across that family \citep{benjamini1995}. Story-count strata are descriptive and are not added as further family members.

The sign used for the portfolio translation is fixed by economic meaning: higher positive-score summaries predict higher returns, while higher negative share or dispersion is oriented toward lower returns. It is not chosen from the observed IC sign.

\section{Conditional daily rank regression}

The aggregation comparison cannot show whether negative share is a renamed mean. The direct estimand is constructed through a cross-sectional regression for every session. Let $R_t(x_{it})$ denote an average-tie percentile rank among the available firms on $t$, centred by subtracting one half. The base specification is

\begin{equation}
 \widetilde r^A_{i,t+1}=\alpha_t+\beta_t\widetilde q^{-}_{it}
 +\gamma_t\widetilde{\bar s}_{it}
 +\delta_t\widetilde{\log(1+n_{it})}+\varepsilon_{it},
 \label{eq:conditional-base}
\end{equation}

\noindent where tildes denote centred percentile ranks and $\bar s_{it}$ is mean continuous sentiment. The estimand is

\begin{equation}
 \bar\beta=\frac{1}{T}\sum_{t=1}^{T}\beta_t.
 \label{eq:mean-beta}
\end{equation}

A date must contain at least ten complete firms and a full-rank design. The coefficient series then receives a five-lag HAC standard error. Ranking reduces the influence of return outliers and puts FNSPID and LSEG regressors on a common scale. It also changes the interpretation: $\bar\beta$ is a mean change in next-return rank for a full-rank change in negative share, not a return in percentage points.

The price-path specification adds three centred ranks:

\begin{equation}
 \widetilde r^A_{i,t+1}=\alpha_t+\beta_t\widetilde q^{-}_{it}
 +\gamma_t\widetilde{\bar s}_{it}+\delta_t\widetilde{\log(1+n_{it})}
 +\theta_{1t}\widetilde r^{(1)}_{it}+\theta_{5t}\widetilde r^{(5)}_{it}
 +\phi_t\widetilde\sigma^{(20)}_{it}+\varepsilon_{it}.
 \label{eq:conditional-full}
\end{equation}

\noindent $r^{(1)}$ is the adjusted-open return from $t-1$ to $t$, $r^{(5)}$ is the cumulative adjusted-open return from $t-5$ to $t$, and $\sigma^{(20)}$ is the sample standard deviation of the twenty returns ending at the formation open. All controls end when the outcome begins, so none contains $t$ to $t+1$ information. The full-control row is the single primary reversal-confound test. Intermediate nested specifications are diagnostics.

A later specification, frozen after the headline result, replaces the next-open outcome with a close-to-close market-adjusted return and then with a trailing Fama--French residual of that close-to-close return \citep{famafrench1993}. A third check keeps the next-open outcome but replaces hard share with mean $p(\mathrm{negative})$. The three coefficients form one Benjamini--Hochberg family. They do not join the original nine-rule comparison. Factor betas use the 252 sessions ending at formation, require at least 126 observations, and exclude the outcome session.

The mean-sentiment control enters as one centred rank-linear term. On a single-story firm-day, negative share is a thresholded class indicator derived from the same probability vector, so its coefficient can reflect a nonlinear hard-boundary transformation rather than aggregation across stories. The specification establishes incrementality only relative to its stated rank-linear controls; it does not rule out every nonlinear function of mean sentiment or the class probabilities. A flexible functional-form comparison would be another post-hoc development test and is reserved for an independently specified replication.

For interpretation, a move from the 10th to 90th percentile of negative share changes the regressor rank by 0.8. Multiplying $0.8\bar\beta$ by 100 expresses the fitted change in next-return-rank percentile points. This is a linear translation of the rank model, not a return forecast.

\section{Frozen temporal replication}

Notebook 75 opens the 2020--2023 evaluation slice once for two declared
families. Family A repeats the nine daily information coefficients from
\cref{eq:daily-ic}, with Benjamini--Hochberg correction across nine two-sided
tests. Family B repeats \cref{eq:conditional-base} on all evaluation firm-days
and on the predeclared subset with at least two stories, with BH correction
across those two coefficients. The expected direction is negative. No
evaluation portfolio, alternative outcome, price control, threshold, event
type, scorer or further story-count stratum is computed.

Before the outcome opening, an amendment added power context and a direct
stability contrast. The prospective approximation scales the development HAC
standard error by the square root of the ratio of 2,264 development sessions
to 998 evaluation sessions and uses the first BH-2 hurdle, a two-sided size of
0.025. It is a normal approximation, not a success gate.

For the direct era comparison, the unchanged daily coefficient series are
stacked and the model

\begin{equation}
 \beta_t=\alpha+\Delta D^{\mathrm{eval}}_t+u_t
 \label{eq:temporal-contrast}
\end{equation}

\noindent is estimated with HAC(5) uncertainty, where
$\Delta=\bar\beta_{\mathrm{evaluation}}-
\bar\beta_{\mathrm{development}}$. This separate one-member diagnostic
tests whether the coefficients differ. It avoids treating significance in one
era and non-significance in another as sufficient evidence of a change, and
cannot rescue a failed primary family.

\section{Model-free economic scale}

The double sort avoids a regression. Within each session, firms are assigned to five mean-sentiment groups using average-tie percentile ranks. Within each group, negative share is divided into low, middle and high thirds. Because negative share has many ties, tied values receive fractional allocation so that each third receives one third of the cell weight; no symbol order or return is used to break ties. A cell must contain at least five firms, and a session must have at least three usable mean groups.

The high-minus-low abnormal return is calculated within each mean group. Available group spreads are equally weighted to form one daily spread. The reported mean and interval use HAC(5). The same calculation is repeated for firm-days with at least two stories. The initially declared absolute cut-offs of one third and two thirds produced no sessions with enough overlapping cells; execution stopped before any spread was estimated, and the tie-aware dependent sort was recorded before the replacement result was computed.

\section{Portfolio accounting and costs}

Each aggregation rule is also converted to a daily cross-sectional rank portfolio. Oriented signal ranks are centred and scaled so long and short gross exposure are balanced. Positions formed on $t$ earn the next open-to-open abnormal returns. Let $w_{it}$ be the target weight, $w^{\mathrm{drift}}_{i,t-1}$ the prior weight after the asset return and $c$ the per-side proportional cost. Turnover is reported one-way, as half the traded notional:

\begin{equation}
 \mathrm{TO}_t=\tfrac{1}{2}\sum_i\left|w_{it}-w^{\mathrm{drift}}_{i,t-1}\right|.
 \label{eq:turnover}
\end{equation}

A quoted per-side cost is paid on both sales and purchases, so the charge is $2c\,\mathrm{TO}_t$ and net return is

\begin{equation}
 r^{\mathrm{net}}_{p,t}=\sum_i w_{i,t-1}r^A_{it}-2c\,\mathrm{TO}_t.
 \label{eq:portfolio-net}
\end{equation}

For example, opening a gross-one dollar-neutral book from cash has $\mathrm{TO}_t=0.5$ and costs $c$; liquidating it later adds another $0.5$ and another $c$. The final position is liquidated and charged. The main stock cost is 10 basis points per side. It is a demanding test for a high-turnover daily strategy, not a universal institutional estimate. Break-even cost is the value of $c$ that makes average net return zero, $\bar r^{\mathrm{gross}}_p/(2\overline{\mathrm{TO}})$. Annualised turnover multiplies $\overline{\mathrm{TO}}$ by 252, and both Sharpe ratios use the same annualisation.

Learned no-trade thresholds provide a secondary model-choice check. Logistic regression, gradient boosting and a multilayer perceptron are fitted on development inputs to predict whether an observation should trade. Their evaluation AUC, activity and net returns are compared with a historical fixed band, an activity-matched rule and cash. A label-shuffled neural network is retained as a negative control.

\section{Power and preservation of null results}

For a saved estimate with standard error $SE$, the nominal two-sided 5\% MDE at 80\% power is

\begin{equation}
 MDE_{0.80}=\left(z_{0.975}+z_{0.80}\right)SE=2.8016SE.
 \label{eq:mde}
\end{equation}

For a family of size $m$, a conservative first-BH threshold replaces $z_{0.975}$ with $z_{1-0.05/(2m)}$. This is sufficient for the first ordered test, not exact power for adaptive BH. MDEs are calculated for the LSEG transfer, story-type, earnings-window, publisher and learned-threshold nulls. The frozen audit contains 56 families and 264 tests or decisions closed before the focal analysis, spanning Notebooks 3 to 73. It is retained in the accompanying repository, which also records the focal and later families that the extraction did not capture.

\section{Risk-state translation}

The risk analysis begins with a sentiment-free HAR forecast. Log variance over daily, weekly and monthly windows predicts five-session log variance. The forecast is mapped to a 10\% annual-volatility target, with SPY exposure clipped to $[0.25,1]$. Cash earns zero and trading costs 2 basis points per side. The sentiment overlay cannot raise exposure above the HAR target.

Aggregate pressure on session $t$ is the headline-weighted share of negative FNSPID stories across the market. It is standardised using the preceding 252 sessions, with at least 126 prior observations. The frozen hysteresis rule enters risk-off when the lagged pressure $z$-score exceeds 1.5, remains risk-off until it falls below 0.5, and multiplies HAR exposure by 0.25 while active. Using a lagged state ensures that the exposure decision precedes the return.

Paired circular block bootstraps use twenty-session blocks and 4,999 or 9,999 replications as recorded in each manifest. Return differences and reductions in squared negative returns are reported separately. A constant multiplier selected on development data matches the overlay's mean exposure. Circular-shift placebos move the complete risk-off schedule through the return series, preserving the number and length of episodes. A timing claim requires the observed downside reduction to be unusually large relative to those shifts after correction.

\section{LSEG economic searches and prompt tests}

The exact FNSPID pressure rule is first applied without retuning to the recent and backward LSEG blocks. A later bounded search permits alternative LSEG aggregate rules and a firm-level loss-control rule. Candidate grids, selection scores and fold definitions are fixed in specifications. The aggregate search compares the selected overlay with the frozen HAR path and a development-selected constant multiplier. The firm-level search can reduce exposure to individual companies; it is compared with fully invested equal weight and constant company weights matched to development exposure. Complete gates require tail or utility improvement, an acceptable return lower bound and stability checks. Half-sample and leave-one-company-out results guard against one episode or firm carrying the result.

Prompt experiments test whether the label can be made closer to the economic target. Five Gemma 4 26B prompts ask about direct one-session reaction, delayed reaction, cash-flow revision, surprise catalysts and a red-team consensus view. These variants may use up to five strictly earlier same-company headlines only to judge novelty. A sixth structured prompt asks whether the current headline concerns the target, how it compares with expectations, whether its effect is temporary, and the likely direction of the first tradable price reaction. It deliberately omits novelty and sends no earlier headlines. The output remains a sentiment score on a fixed grid, not a percentage-return forecast. Reuters headlines and company names were sent through OpenRouter to DeepInfra under the recorded no-retention controls; returns were never included.

Only 2024 is used for prompt selection. A prompt must have positive gross and net means, positive net Sharpe, a break-even cost of at least 10 basis points and stable quarterly comparisons with existing Gemma and FinBERT scores. If no prompt passes, the 2025 prompt strategy is not opened. This stopping rule prevents repeated wording changes from migrating into the test period.
````

<!--block:B0013-->
## manuscript/chapters/05_results.tex

<!--block:B0014-->
````latex
\chapter{Results}
\label{ch:results}

\section{Aggregation changes what is detected}

\Cref{tab:aggregation-family} and \cref{fig:aggregation-ic} report the nine-rule development comparison. Negative-story share has a mean daily information coefficient of $-0.00543$, a HAC $t$-statistic of $-3.12$ and a two-sided $p$-value of $0.0018$. It is the only rule to survive Benjamini--Hochberg correction across the nine tests. The sign means that firms with a larger share of negative stories tend to rank lower on next-open abnormal return.

The distinction from a mean is visible in the other rows. Both means sit near $0.003$ with $p$ above $0.09$. The median and decayed state have uncorrected $p$-values below 0.05, but neither survives the family correction, and dispersion is indistinguishable from zero. The result is therefore not that every non-mean statistic works. One particular hard-boundary transformation, the frequency of classifications on the negative side, is more stable than score magnitude in this sample.

\input{artifacts/tab_aggregation_family}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_aggregation_family}
  \caption[Nine-rule information coefficients]{Mean daily cross-sectional rank correlation between each aggregation rule and the next-open abnormal return, FNSPID development. Bars are 95\% Newey--West HAC(5) intervals over 2,264 sessions; muted rules do not survive Benjamini--Hochberg correction across the nine-test family. Source: aggregate Notebook 03 outputs.}
  \label{fig:aggregation-ic}
\end{figure}

The effect is statistically small. An IC of $-0.00543$ says that ranks line up only weakly on an average session. The sample contains more than half a million firm-days, so a weak regularity can be measured precisely. Whether it is large enough to matter is a separate test addressed below.

\section{Association beyond a rank-linear mean}

The daily conditional regression gives the main exploratory estimate for the research question. Estimated over 2,264 development sessions on the 512,153 development panel rows, of which 512,149 have a complete next-open return, the mean coefficient on negative-share rank is $-0.00831$ after mean continuous sentiment and $\log(1+n)$ ranks are included. Its HAC standard error is $0.00295$, giving a 95\% interval of $[-0.01410,-0.00252]$ and $p=0.0049$. The interval and $p$-value are conditional on the selected workflow rather than post-selection-adjusted confirmation.

This coefficient is not a return in basis points. It says that, within an average session, a firm ranked higher on negative share is ranked lower on next-open abnormal return even when its mean score and news volume are held fixed. The mean-sentiment control prevents a day with simply worse average tone from supplying the whole result. The count control prevents a heavily covered company from supplying it merely because more stories create more chances for a negative classification.

The recent-price test makes the association slightly stronger rather than weaker. On the price-complete sample, the base coefficient is $-0.00830$. Adding the lagged one-session return changes it to $-0.00945$; adding the five-session return changes it to $-0.00978$. The full model, which also includes 20-session volatility, gives $-0.00914$ with a standard error of $0.00282$, a 95\% interval of $[-0.01466,-0.00362]$ and $p=0.00118$. \Cref{tab:price-controls} reports the full path.

\begin{table}[htbp]
  \centering
  \caption[Price-path control coefficients]{Negative-story-share coefficient under recent-price controls. All regressors and the outcome are centred within-session percentile ranks.}
  \label{tab:price-controls}
  \small
  \input{tables/tab_reversal_confound}
\end{table}

The full model also estimates a negative one-session return coefficient of $-0.01585$ ($p<0.001$) and a negative volatility coefficient of $-0.01009$ ($p=0.0094$). Mean sentiment is $-0.00031$ ($p=0.880$), while log news count is $0.00315$ ($p=0.0858$). These controls explain their own cross-sectional variation without absorbing the negative-share association. The simplest sceptical account---that negative share is just a label for a firm that recently fell and is about to reverse---is not supported by this specification.

A move from the 10th to the 90th percentile of negative-share rank corresponds to a fitted change of $0.8\times-0.00914=-0.00731$ in next-return rank, or about $-0.731$ percentile points. This is a useful scale check: the estimated ordering effect is less than one percentile point across a wide change in the signal. Precision does not make it large.

\input{artifacts/tab_headline_estimates}

\Cref{tab:headline-estimates} collects the development, temporal-replication
and cross-source estimates.

\section{The frozen temporal replication fails}
\label{sec:temporal-replication}

The selected development association does not persist into 2020--2023. On
203,393 evaluation firm-days across 998 sessions, the unchanged all-firm-day
coefficient is $+0.00583$, with a 95\% HAC interval of
$[-0.00426,+0.01593]$, two-sided $p=0.257$ and two-test BH $q=0.515$. The
predeclared multi-story estimate is also positive, $+0.00295$, with interval
$[-0.00990,+0.01581]$ and $q=0.653$. H1b is rejected under its recorded
direction and correction.

The broader aggregation family fails as well. None of the nine evaluation
information coefficients survives BH correction. Negative-story share has an
IC of $-0.00312$ with $p=0.300$. Mean and median continuous sentiment have
positive uncorrected estimates, but neither clears the first family hurdle.
The outcome therefore licenses no rule reselection or opposite-direction
signal.

A null evaluation coefficient would not by itself show that the eras differ.
The predeclared contrast in \cref{eq:temporal-contrast} addresses that question
directly. Evaluation minus development is $+0.01414$, with HAC standard error
$0.00594$, interval $[+0.00250,+0.02579]$ and $p=0.0173$. The development
mean is $-0.00831$ and the evaluation mean $+0.00583$, so the positive
contrast records both the sign change and its uncertainty. This is a separate
diagnostic, not a way to rescue either failed primary family.

The pre-outcome power calculation also limits the claim. Scaling the
development HAC uncertainty to 998 sessions gave approximately 35.4\% power
to recover the same coefficient at the first BH-2 hurdle; the approximate
80\%-power MDE was $0.01372$, larger than the development estimate. The
evaluation failure is therefore not proof that the later effect is exactly
zero. Taken together with the direct contrast, the defensible reading is
temporal non-replication and instability, not equivalence and not reversal as
a new trading rule.

\section{What the estimate survives, and what it does not}
\label{sec:robustness-diagnostics}

\Cref{tab:robustness-diagnostics} and \cref{fig:estimate-stability} report the diagnostics that bound the estimate. They do not point the same way.

Within development, the inference choice does not matter. Lengthening the
Newey--West lag from the frozen five to ten and then twenty-one leaves the
point estimate at $-0.00831$ and moves $p$ only between $0.0047$ and
$0.0052$. The coefficient is negative in eight of nine development years,
although only 2014 and 2016 are individually distinguishable from zero.
Annual estimates are noisy and show no detectable variation inside
development. That within-period diagnostic does not contradict
\cref{sec:temporal-replication}: the later 998-session block is a different,
predeclared era comparison with materially more information than any one year.

The conditioning restriction matters. Requiring at least two same-day stories reduces the coefficient to $-0.00529$, with a 95\% interval of $[-0.01272,0.00214]$; requiring at least three reduces it to $-0.00183$, with an interval of $[-0.01385,0.01019]$ and $p=0.766$. The $n\geq3$ estimate is about 22\% of the full-sample estimate across roughly 137,000 panel rows. Each restriction removes about half of the remaining rows, so the intervals widen, but the point estimate also moves monotonically towards zero.

This is the most important qualification in the chapter. On a one-story firm-day, negative share is exactly an indicator that the single story was classified negative. Adding it beside one rank-linear mean-sentiment control therefore tests a hard threshold and possible nonlinear score shape as much as an aggregation rule. The full-sample coefficient is substantially a statement about a lone negative label rather than a rich within-day distribution. The current development panel cannot separate those readings without adding another post-hoc specification; the prospectively defined work in \cref{ch:conclusion} is the appropriate test.

\input{artifacts/tab_robustness_diagnostics}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_estimate_stability}
  \caption[Estimate stability under controls and story count]{The same coefficient under two kinds of pressure, drawn on one shared scale. Adding recent-price controls leaves it negative and bounded away from zero; requiring a denser news day moves it towards zero and widens the interval until it spans zero. Emphasised rows are the primary price-path test and the sparsest story-count stratum. Bars are 95\% HAC(5) intervals. Sources: aggregate Notebooks 74 and 79.}
  \label{fig:estimate-stability}
\end{figure}

\section{The association is specific to the next-open hard-share window}
\label{sec:outcome-label-sensitivities}

\Cref{tab:outcome-label-sensitivities} reports three post-hoc measurement tests frozen after the headline result. They form one Benjamini--Hochberg family. They do not join the original nine-rule comparison, and a pass would not have upgraded H1.

Close-to-close hard share is $0.00027$ ($q=0.929$). The trailing Fama--French residual of that close-to-close return \citep{famafrench1993} is $0.00182$ ($q=0.815$). Both intervals cover zero. Close-to-close starts at the assigned session's close, so it misses the open print the primary design uses and includes the next session's daytime.

Mean FinBERT $p(\mathrm{negative})$ on the next-open outcome is $-0.00378$ ($q=0.337$). Mean continuous sentiment is already in the regression, so this is not a renamed mean. None of the three tests survives correction. The exploratory H1 association remains specific to assigned-session, next-open hard share.

\input{artifacts/tab_outcome_label_sensitivities}

\section{The model-free spread is small and noisy}

The double sort provides a return-scale view without regression coefficients. Within each mean-sentiment quintile, firms in the high negative-share third underperform firms in the low third by an average of $0.618$ basis points per session. The HAC standard error is $0.455$ basis points and the 95\% interval is $[-1.510,0.275]$, with $p=0.175$. Four of the five mean-sentiment quintiles have a negative point estimate, but every interval crosses zero, as \cref{fig:double-sort} shows.

When the sample is restricted to firm-days with at least two stories, the pooled spread weakens towards zero, to $-0.253$ basis points with an interval of $[-1.821,1.315]$ and $p=0.752$. The loss of precision is not surprising: the restriction removes almost half of the complete observations, leaving 264,679 firm-days, and the HAC standard error nearly doubles to $0.800$ basis points. The smaller point estimate reproduces the pattern in \cref{tab:robustness-diagnostics} through a second, regression-free route. Both say that singleton negative labels contribute a disproportionate part of the full-sample result.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_double_sort}
  \caption[Double-sort high-minus-low spreads]{High-minus-low negative-share spreads in next-open abnormal return, within mean-sentiment quintiles and pooled. The left panel uses all firm-days and the right requires at least two same-day stories. Points are mean daily spreads and bars are 95\% HAC(5) intervals; negative is the predicted direction. Both pooled estimates have that sign, but every interval crosses zero. Source: aggregate Notebook 76 outputs.}
  \label{fig:double-sort}
\end{figure}

The development regression and double sort do not conflict. The sort coarsens
two continuous signals into buckets, sacrificing statistical efficiency to
make magnitude legible in basis points. Both give a negative development
ordering; the frozen later-era regression does not.

\section{Directional trading fails the cost test}

The gross mean return of the oriented negative-share rank portfolio is $0.890$ basis points per session. Mean one-way drift-aware turnover, defined in \cref{eq:turnover}, is $0.712$ per session, or 179.3 times gross portfolio value per year. Full absolute traded notional is therefore $2\times0.712=1.424$ times gross value per session. At 10 basis points per dollar traded, the mean charge is about 14.23 basis points, reducing mean net return to $-13.34$ basis points and net Sharpe to $-10.41$. The rule's break-even cost is $0.890/(2\times0.712)=0.625$ basis points per side.

No other aggregation rule solves this problem. The nine break-even costs range from $0.116$ to $0.625$ basis points. The charged cost is therefore 16.0 times the best break-even value. \Cref{fig:break-even} shows the complete family on a logarithmic scale. This is stronger evidence than stating that one cost assumption makes returns negative: even a cost one tenth of the assumed value would exceed every observed break-even point.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_break_even_cost_gap}
  \caption[Break-even costs versus the 10 bps charge]{Per-side break-even transaction costs for the nine FNSPID aggregation rules, on a logarithmic scale. The shaded band is the shortfall between the best break-even in the family and the 10-basis-point cost charged to the daily stock portfolios. Source: aggregate Notebook 03 outputs.}
  \label{fig:break-even}
\end{figure}

Learned trade filters do not rescue the directional rule. Evaluation AUC is $0.5005$ for logistic regression, $0.5038$ for gradient boosting and $0.5007$ for the multilayer perceptron. Their net Sharpe ratios are $-2.07$, $-5.47$ and $-1.00$. The historical fixed band is sparse and loses only $0.098$ basis points per session on average, but its net Sharpe is still $-0.108$ and cash is preferred. Against that fixed band, the learned policies lose between $7.71$ and $16.88$ additional basis points per session. This is not a threshold-calibration failure at the margin: the features have almost no evaluation ranking ability.

H3 is rejected for the tested implementation. The negative-share statistic is measurable, but the daily long--short route needed to monetise it trades far too often for its gross spread.

\section{FNSPID risk timing appears only within a narrow boundary}

The frozen HAR target provides a less turnover-intensive use case. Over 998 sessions from 2020 to 2023, the sentiment-free HAR strategy earns 29.85\% net, with a net Sharpe ratio of $0.673$ and maximum drawdown of $-15.73\%$. Adding the negative-pressure hysteresis rule raises total return to 35.20\% and Sharpe to $0.835$, while mean exposure falls from $0.619$ to $0.569$. Maximum drawdown is slightly worse at $-16.36\%$.

The paired mean return difference is $0.366$ basis points per session, with a 95\% block-bootstrap interval of $[-0.996,1.930]$ and $p=0.617$. Squared downside return falls by $5.12\times10^{-6}$ per session, with interval $[2.02\times10^{-6},9.74\times10^{-6}]$ and $p=0.0148$. Against a constant multiplier matched to mean exposure, the return advantage is $0.592$ basis points with an interval of $[-0.635,1.962]$. The data support lower downside in this period, but not a precise return gain.

Temporal stability is weak, as \cref{fig:risk-regime} shows. The overlay beats HAR by $2.731$ basis points per session in 2020 and loses $0.437$ basis points per session over 2021--2023. A walk-forward application to 2013--2019 gives a mean difference of $-0.134$ basis points, interval $[-0.987,0.736]$, although squared downside again improves ($p=0.0052$). Its maximum drawdown falls from $-12.53\%$ to $-8.57\%$.

Circular timing placebos separate the position of the risk-off episodes from their frequency. In 2020--2023, observed squared-downside reduction is 512.5 basis-points squared, or 21.2\% relative to HAR. Only about 1\% of non-zero circular shifts perform as well; after correction across three regimes, $q=0.0301$. \Cref{fig:downside-placebo} places each observed schedule within its own circular-shift null. The 2013--2019 downside timing result has $q=0.0682$ and the LSEG/Gemma result has $q=0.365$. None of the return-timing tests survives correction.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_timing_placebo}
  \caption[Timing placebo versus circular-shift null]{Percentile position of each observed squared-downside reduction within all non-zero circular shifts of the same risk-off schedule. The shaded upper 5\% is a pre-correction reference; $q$ applies Benjamini--Hochberg across the three regimes. Timing evidence survives correction only for FNSPID 2020--2023. Source: aggregate Notebook 45 outputs.}
  \label{fig:downside-placebo}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_risk_regime_stability}
  \caption[Overlay minus base return by period]{Mean return of the sentiment overlay minus its sentiment-free base, by period. The 2013--2019 points compare the walk-forward HAR arms and the 2020--2023 points the frozen HAR target. These are period means rather than tests: only the pooled 2020--2023 downside result survives correction. Sources: aggregate Notebooks 24 and 25.}
  \label{fig:risk-regime}
\end{figure}

The useful-risk gate is not passed: the return interval crosses zero and performance is not stable across subperiods. The risk result is evidence that aggregate pressure happened to mark adverse sessions unusually well in one FNSPID period. It is not a validated deployment rule.

\section{LSEG transfer: development-compatible signs, insufficient precision}

The primary FinBERT conditional coefficients are negative in both LSEG blocks,
matching FNSPID development but not its later era. The backward estimate is
$-0.02861$, with interval $[-0.07283,0.01560]$ and two-test BH $q=0.409$.
The recent estimate is $-0.01573$, with interval $[-0.09453,0.06308]$ and
$q=0.696$. Neither establishes transfer.

An earlier project family bears on the same question and should be read alongside these estimates. A nine-aggregator comparison of the same form as \cref{tab:aggregation-family}, run across two scorers on a 44-firm LSEG Gemma sample, produced 18 tests and no Benjamini--Hochberg survivor; a companion family of two scorers by four metadata filters, applied to the mean continuous aggregator rather than to negative share, produced 8 tests and no survivor. Both are recorded in the project ledger as all-null. That sample is not the 33-firm cohort used here, so it does not settle transfer for this panel, and its blocks and return construction differ again. It does mean that the aggregation comparison has already been executed once on LSEG data without a survivor, and a reader assessing H4 should not treat the imprecision of the two coefficients above as the only LSEG evidence on the question.

The wide intervals reflect limited time-series information. At 5\% size and 80\% power, the backward block's nominal MDE is $0.0632$, 7.6 times the absolute FNSPID baseline coefficient. The recent MDE is $0.1126$, 13.6 times that coefficient. \Cref{fig:cross-source-power} places both estimates beside the FNSPID coefficients and shows the resulting power gap. Failure to detect a FNSPID-scale effect is therefore expected. The estimates do not show equivalence to zero; they show that these blocks can rule out only much larger rank effects.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_cross_source_coefficient_power}
  \caption[Cross-period and cross-source coefficients]{Conditional negative-share estimates with 95\% HAC intervals. The FNSPID evaluation row is the frozen temporal replication; its power entry is the prospective probability of recovering the development effect at the first BH-2 hurdle. LSEG uses next-open raw rather than FNSPID abnormal returns, so the regimes are compared on rank units but never pooled; its entries give 80\%-power MDE multiples. Sources: aggregate Notebooks 71, 75, 77 and 79.}
  \label{fig:cross-source-power}
\end{figure}

Gemma provides a secondary sensitivity. Its backward coefficient is $-0.05443$ with $p=0.0156$, while its recent coefficient is $-0.04663$ with $p=0.263$. The backward value is not promoted to a new primary result: scorer and block comparisons were already specified, the corresponding recent estimate is imprecise, and economic gates below do not pass.

The exact FNSPID risk rule exposes a stronger portability failure. In the recent 167-session LSEG block, the pressure $z$-score never reaches the fixed 1.5 entry threshold. The maximum is 1.349, so there are 0 risk-off sessions and the overlay is identical to HAR. Lowering the threshold after seeing this distribution would answer a different question.

In the backward block the exact rule does activate: 53 of 456 sessions are risk-off, across 18 entries. Net Sharpe rises from $1.291$ for HAR to $1.520$ for the overlay, and maximum drawdown changes from $-11.24\%$ to $-10.83\%$. The paired return difference is only $0.144$ basis points per session, with interval $[-1.295,1.688]$ and $p=0.849$. Downside improvement survives the two-test correction ($q=0.0336$), and the timing placebo does too ($q=0.0482$).

Two diagnostics set the weight this block can carry. The result is concentrated: the two largest of the 18 risk-off episodes, in July--August 2024 and December 2024, supply 65.6\% of the squared-downside reduction between them. Against that, dropping any single episode still leaves a paired interval excluding zero, with $p$ between $0.017$ and $0.049$, and the inference is insensitive to the bootstrap block length, with 10, 20 and 40 sessions giving $p$ between $0.019$ and $0.023$. No single episode carries the result, but a handful do---which is what a genuine crisis-timing mechanism and an accident of placement would both look like in 456 sessions. The backward block is therefore supporting evidence for the risk mechanism; the recent non-activation means the exact state is not an established operating rule.

\section{Tuned LSEG rules do not create incremental economic value}

Allowing the LSEG rule to change does not produce a positive forward result. The aggregate search evaluates 6,672 candidates across three chronological development folds and selects a headline-weighted negative-share hysteresis rule. On the 167-session block, HAR earns $0.926$ basis points per session and 1.241\% in total; the selected overlay earns $0.878$ basis points and 1.168\%. The paired difference is $-0.049$ basis points, with interval $[-0.656,0.633]$. A matched constant has better downside deviation, expected shortfall and drawdown than the news rule. Neither complete comparison gate passes.

The firm-level search is a more direct attempt to save company losses. Its selected rule uses a Gemma attention-weighted negative share, a 63-session firm normaliser and a full cut to cash when the signal exceeds 0.6. In the evaluation block, an always-invested equal-weight portfolio returns 12.40\%, with Sharpe $1.623$, downside deviation $7.54\%$ and drawdown $-6.02\%$. The news rule returns 4.41\%, with Sharpe $0.967$, downside deviation $4.88\%$ and drawdown $-4.34\%$. Read against the fully invested portfolio, this looks like loss protection.

The matched comparator changes that reading. Development-selected constant company weights return 7.49\%, achieve Sharpe $1.619$, downside deviation $4.61\%$ and drawdown $-3.72\%$. Relative to this control, the news rule has 0.63 percentage points more drawdown and 0.27 points more downside deviation. The comparison is if anything generous to the rule: its realised evaluation exposure is $0.576$ against the control's $0.606$, so it carried less risk and still lost on both risk measures. It turns over 99.8 times per year, 25.9 times the matched rule. The paired return difference is $-1.743$ basis points per session, with interval $[-4.727,1.549]$ and $p=0.279$.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_lseg_matched_control_value}
  \caption[LSEG matched-control economic value]{Firm-level LSEG loss control over the 167-session evaluation block. The left panel shows why the always-invested comparison is misleading: the news rule beats the fully invested portfolio on every loss measure but not the exposure-matched constant. The right panel accumulates loss savings, forgone upside, trading cost and the compounding residual into ending wealth against that matched control. Source: aggregate Notebook 82 outputs.}
  \label{fig:lseg-matched-value}
\end{figure}

\Cref{fig:lseg-matched-value} decomposes that comparison. For \$1 million of matched-control capital, the rule avoids \$24,738 on sessions when the control loses. It gives up \$41,126 on sessions when the control gains and incurs \$12,715 of additional cost. After a small reconciliation and compounding residual, ending wealth is \$30,870 lower. The first half loses 3.709 basis points per session against the matched control; the second gains 0.247. Excluding any one of the 33 companies leaves the mean difference negative, ranging from about $-1.26$ to $-2.13$ basis points. All complete economic-value gates fail.

This decomposition answers the loss-saving question directly. The rule does save some losses. Those savings are worth less than the upside forgone and trading cost required to obtain them. Lower raw loss is not incremental economic value.

\section{Prompt engineering does not improve the LSEG result}

All five target-specific prompt variants fail their 2024 selection gate. The best net mean is the delayed five-session reaction prompt at $-4.012$ basis points per session; its gross mean is already $-0.989$ basis points and net Sharpe is $-1.098$. The remaining prompts have net means between $-4.49$ and $-16.46$ basis points. None has positive and stable quarterly net performance.

The structured anticipated-reaction score performs worse. Strict scores are obtained for 3,241 of 3,242 eligible 2024 events across all 33 firms. Gross mean is $-2.590$ basis points per session. After cost it is $-19.666$ basis points, total return is $-40.53\%$ and net Sharpe is $-2.178$. It passes only one of four positive quarters and beats each same-date scorer in only one quarter. Under the stopping rule, no 2025 reaction-score portfolio is computed. Combined recorded model cost for the two prompt experiments is US\$3.16, below the authorised US\$20 ceiling.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_prompt_gate}
  \caption[Prompt selection-gate returns]{Gross and net mean 2024 portfolio returns for the five target-specific prompts and the structured anticipated-reaction score. Every variant is already negative before costs, so the 10-basis-point charge deepens a loss rather than creating one, and none passes the selection gate. Sources: aggregate Notebooks 83 and 84 selection tables.}
  \label{fig:prompt-gate}
\end{figure}

\Cref{fig:prompt-gate} puts the six variants on one scale. These failures reject the tested prompt constructions, not every possible economic instruction. They do show that asking a model directly for the likely stock-price direction does not automatically yield downside protection. A persuasive rationale in structured output is not evidence that the score ranks returns.

\section{Nulls and the final answer}

The frozen audit contains 56 families and 264 tests or decisions spanning Notebooks 3 to 73: 27 within-family BH survivors, and 36 of the 56 families all-null. Two boundaries matter for how it should be read. It excludes the Notebook 71 conditional families that carry the headline coefficient, so no family-level correction is applied to that estimate. It does, however, include the Notebook 45 and 73 families whose corrected $q$-values are reported above as the downside-timing and backward-LSEG results, so those survivors sit inside a documented 264-test search rather than outside it. The extraction predates the focal Notebook 71 conditional families and the final economic and prompt studies, which the repository ledger records separately rather than adding retrospectively to that total. Since the families differ in estimand and date, these totals expose selection pressure but do not estimate a project-wide false-discovery rate.

The principal conditioning nulls are also bounded by power. \Cref{tab:null-mde} reports each family's median MDE and the conservative first-BH threshold from \cref{eq:mde}. The median nominal MDE is 18.8 basis points per unit of negative share for story-type interactions, 15.5 basis points for earnings-window interactions and 0.049 in daily IC units for publisher rules. Some economically relevant effects could therefore remain unresolved, especially in LSEG. By contrast, the learned thresholds are not merely imprecise: their median MDE of 6.9 net basis points per session is smaller than every observed loss against the fixed band, which ran from 7.71 to 16.88 basis points. That null is informative rather than underpowered.

\input{tables/tab_null_mde}

The exploratory development estimate is consistent with H1 under the selected specification: negative-story share is conditionally associated with the abnormal-return rank following the assigned FNSPID session beyond rank-linear mean, count and observed price-path controls. It is not confirmatory evidence because the statistic preceded the final question and the coefficient has no family-level or post-selection adjustment. Close-to-close and FF3 residual ranks do not show the association. H2's directional prediction is met, but its basis-point spread is imprecise. H3 fails because daily implementation cannot cover costs. H4a is unresolved rather than refuted: the transfer coefficients carry the predicted sign but the blocks are too imprecise to distinguish them from zero, and an earlier all-null LSEG aggregation family does not settle the question either. H4b is refuted: every complete economic-value gate fails against an exposure-matched control. The finding motivates prospective measurement work; it is not evidence of a rich-news distribution effect, a tradable rule or validated risk management.
````

<!--block:B0015-->
## manuscript/chapters/06_conclusion.tex

<!--block:B0016-->
````latex
\chapter{Conclusions and Further Work}
\label{ch:conclusion}

\section{Answer to the research question}

The answer is yes in the selected development workflow and no as a stable
cross-era result. Through 2019, negative-story share is conditionally
associated with the following abnormal-return rank beyond rank-linear mean
sentiment, news volume and recent prices. The full-control coefficient is
$-0.00914$, with a 95\% HAC interval of $[-0.01466,-0.00362]$. In the frozen
2020--2023 opening, the unchanged all-firm-day coefficient is instead
$+0.00583$, with interval $[-0.00426,+0.01593]$ and BH $q=0.515$; the
multi-story coefficient and nine-rule family also fail.

The direct era comparison matters more than contrasting two significance
labels. Evaluation minus development is $+0.01414$, with interval
$[+0.00250,+0.02579]$ and $p=0.0173$. Approximate prospective power to recover
the development effect at the first BH-2 hurdle was only 35.4\%, so the
evaluation failure is not proof of an exact zero. Together, the failed
families and direct contrast establish temporal non-replication and
instability under the frozen design.

That conclusion is reinforced by scale and portability. The development
model-free spread is $-0.62$ basis points per session and imprecise; the best
daily rule breaks even at $0.625$ basis points per side against a
10-basis-point charge. LSEG intervals are wide, the recent exact risk rule
never activates, and tuned rules and prompted scores fail their gates. The
development coefficient also weakens as same-day story count rises. A hard
negative boundary remains a useful feature-audit candidate, but this evidence
does not support a stable rich-news distribution effect, live trading rule or
risk limit.

\section{What the project contributes}

Nine aggregation rules are held to one development sample, horizon and
inferential standard. Negative-story share alone survives correction and
remains negative after stated controls. The main contribution is what follows:
the selected families are repeated once on a frozen later era, the adverse
outcome is preserved, and a predeclared HAC contrast tests the era change
directly. Rank translation, costs, LSEG transfer and matched comparators then
show why development detectability is not economic or portable validity.
Against matched LSEG exposure, the firm-level rule ends \$30,870 behind per
\$1 million.

\section{Limits on the result}

The first limit is chronology. The aggregation result was observed before the
final question was settled, and the price-control test was designed after that
result. Notebook 75 froze its own estimands before opening them, but earlier
project work had already used later FNSPID portfolio outcomes. It is therefore
a credible frozen temporal check, not pristine independent confirmation.
Neither its 35.4\% power approximation nor its conventional HAC intervals
corrects for the wider project's full selection history.

The second limit is news structure and functional form. One- and two-story firm-days dominate the panel, while the $n\geq3$ point estimate is about a fifth of the full-sample value and imprecise. On a single-story day, hard share is a thresholded class indicator beside one rank-linear mean control, so the coefficient can capture nonlinear score shape rather than aggregation. The result should not be read as a measured property of dense news flow. FNSPID also lacks stable story-family identifiers: four negative rows may be four disclosures or four versions of one alert. Count controls cannot separate information distribution from repeated attention.

The third limit is measurement. FinBERT supplies model classifications rather than verified labels. Mean $p(\mathrm{negative})$ keeps the next-open sign but is smaller than hard share and does not survive correction, so the hard boundary is doing more work than the probability mass. Close-to-close and FF3 residual ranks are indistinguishable from zero. Date-only FNSPID stories are moved to the next session, and the retained checkpoint cannot report what fraction use that rule. LSEG has precise availability but uses raw rather than FNSPID-style abnormal returns. Cross-source rank coefficients are comparable, not identical estimands.

The economic tests use proportional costs and idealised adjusted-open execution. They omit spread variation, order size, borrow, market impact and cash interest. Most omissions would hurt the daily long--short rule. Ten basis points may be high for liquid LSEG names, but the best break-even is only $0.625$ basis points.

Finally, risk results are path dependent. In backward LSEG, two of eighteen risk-off episodes supply two thirds of the downside reduction. No single episode carries the result, but a handful of adverse periods is a thin base for a mechanism. Timing is weaker in 2013--2019 and absent in recent LSEG. Matched exposure removes one bias, not regime dependence.

\section{Further work}

\subsection{Resolve novelty before changing the return model}

The next experiment should separate independent news from repeated coverage. Reuters/LSEG identifiers, timestamps, update chains and return-blind text similarity can cluster same-company items into first releases, updates and routine stories. Negative share can then be measured across raw stories, unique families and first releases.

Similarity thresholds should be set on hand-audited pairs before returns are joined. A negative family-level coefficient after mean family sentiment and family count are controlled would support the distribution reading. If only raw-story share remains negative, the signal is better described as adverse-news repetition or attention.

FNSPID would require rebuilding the checkpoint, so LSEG is the cheaper start. A 100-company-session audit should measure reliable first-release/update coverage before the full test proceeds.

\subsection{Audit the labels on the target corpora}

A stratified human audit should sample by source, year, confidence, story count and hard class. Reviewers should label target-company direction, expectation-relative direction and mixed effects, reporting agreement rather than forcing consensus. Chapter~5 already compares hard share with mean $p(\mathrm{negative})$ without that audit. The audit is what would interpret the gap. Further prompt work should wait for a specific audited error.

The same prospective design should separate the hard boundary from the mean's functional form. Before returns are opened, it should specify flexible mean-sentiment controls or the full class-probability vector and report singleton and multi-story strata separately. If hard share survives in sufficiently dense firm-days, the aggregation interpretation gains support. If it survives only on singleton days, the result is better described as a classifier-threshold effect. Repeating this choice on the current development outcomes would add another selected test rather than supply confirmation.

\subsection{Collect a test with enough information}

Another short 33-firm block cannot settle transfer. Its MDEs are 7.6 and 13.6 times the FNSPID coefficient; under square-root scaling, matching FNSPID precision would require roughly 58 and 184 times as much time-series information.

The feasible design is a screened universe of several hundred liquid US firms with a later untouched period. Source, mapping, scorer, return and exclusions should be fixed before collection. Source regimes should remain separate, with coefficients compared meta-analytically rather than by row pooling.

The gate should name one coefficient, interval, failure rule and model-free sort. A wide interval remains inconclusive; a precise value near zero rejects portability; a negative coefficient paired with another sub-basis-point spread transfers the measurement result but not the economic claim.

\subsection{Design for the cost budget}

Any new directional test must address the sixteen-fold cost gap. It should trade only return-blind first releases judged material before the open, then hold for a frozen multi-session horizon. Maximum overlap, rebalancing and cost must be fixed. Gross improvement should precede cost optimisation; if event-only gross return is non-positive, portfolio searches should stop.

\subsection{Use news for risk only under a matched-control protocol}

One prospective risk test remains warranted because downside timing survives a circular-shift placebo in FNSPID 2020--2023 and backward LSEG. The $z$-score rule, 2-basis-point cost, sentiment-free volatility base and exposure-matched comparator must be fixed. Success requires corrected downside improvement, a return lower bound and stability across predeclared halves; drawdown alone stays descriptive. Non-activation is a portability result, not permission to lower the threshold.

\section{Closing assessment}

The project began with the hope that a news score might become a useful trading
or risk signal. It ends with a more credible result: a selected development-era
association survives several within-period controls but fails a frozen
2020--2023 replication and changes across eras. That non-replication is not an
embarrassment to be tuned away; it is the central evidence about model risk.

A sentiment feature should be judged first as a measurement, then as a return
association, then for temporal transport, and only then as an economic rule.
This feature clears a selected historical association test, fails temporal
transport, and does not cover its costs. Future work should resolve threshold
shape and story novelty on a genuinely new, better-powered sample before
spending further effort on trading thresholds or prompts.
````

<!--block:B0017-->
## manuscript/appendices/project_summary.tex

<!--block:B0018-->
````latex
\chapter{Project Summary}
\label{app:project-summary}

The administrative project-summary form was not present in the research archive used to assemble this dissertation. This appendix records the final supervised project scope in the same concise form; if the original signed form is recovered, it should replace this page without changing the dissertation results.

\section*{Title}

\textit{Does News Sentiment Go Beyond the Mean? Development Evidence, Temporal Non-Replication and Economic Limits}

\section*{Problem}

Financial-news pipelines commonly average all sentiment scores for a company and day. A hard count of negative classifications preserves different information from that signed average. The project compares those and other firm-day summaries, estimates their exploratory return associations and tests whether any detected relation supports an economically useful trading or risk-management rule.

\section*{Objectives}

\begin{enumerate}
  \item Build a reproducible firm-day panel joining session-assigned financial news, sentiment scores and following-session returns.
  \item Compare several return-blind rules for aggregating multiple same-day stories.
  \item Estimate whether the selected statistic is conditionally associated with returns beyond rank-linear average-sentiment, news-volume and recent-price controls.
  \item Evaluate statistical uncertainty, multiple testing, transaction costs and downside-risk performance.
  \item Test portability on a separate recent Reuters/LSEG company set without pooling source regimes.
\end{enumerate}

\section*{Methods and deliverables}

The project uses FinBERT story scores, daily cross-sectional information coefficients, centred-rank regressions, HAC inference, block bootstraps and Benjamini--Hochberg correction. Economic tests include drift-aware turnover, proportional costs, break-even analysis, HAR volatility targeting, exposure-matched controls and timing placebos. Deliverables are the dissertation, a private clean repository, frozen specifications, selected notebooks, licence-safe aggregate outputs and a provenance ledger.

\section*{Outcome}

The selected negative-story-share statistic has a small exploratory conditional
association with abnormal returns in FNSPID development and remains negative
under recent-price controls. It does not replicate in the frozen 2020--2023
opening; the estimated coefficient changes sign and the direct era contrast
supports temporal instability. The tested daily implementation does not cover
realistic costs. LSEG estimates are imprecise, and tuned LSEG risk and prompt
rules fail their economic gates.
````

<!--block:B0019-->
## manuscript/appendices/reproducibility.tex

<!--block:B0020-->
````latex
\chapter{Reproducibility and Research Record}
\label{app:reproducibility}

\section{Repository boundary}

The accompanying private repository contains the LaTeX source, selected notebooks, reusable analytical helpers, frozen specifications, aggregate result snapshots and licence-safe figures. It excludes raw FNSPID and LSEG text, large row-level panels, API responses, model caches and credentials. The original private research workspace remains the audit archive for authorised full reruns.

Two levels of reproduction are possible. An evidence audit can be performed
from the committed CSV and JSON summaries, specifications, hashes, seeds and
manuscript-generation script. A full computational replay requires the local
FNSPID archive, licensed LSEG inputs and staged parquet panels. Notebook 75 is
not part of that replay: it was opened once, and the exact executed notebook,
source, daily coefficient series and output hashes are preserved in its result
snapshot. Other reruns write to an ignored generated directory and cannot
silently overwrite the aggregate evidence cited here.

The standard checks are:

\begin{verbatim}
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make lint
make validate
make manuscript
\end{verbatim}

Repository validation rejects common row-level data formats, raw-text columns, files above the clean-repository size limit and obvious secrets. Focused tests cover aggregation, daily rank regression, panel timing, turnover, thresholds and risk overlays. A separate prose check rejects a fixed list of stock machine-written phrases; it is a guard, not a substitute for editorial review.

\begin{landscape}
\section{Core provenance}

\Cref{tab:core-result-provenance} ties every number cited in
\cref{ch:results} to the notebook that produced it, the frozen specification
it ran under, the hash of the saved aggregate output and the random seed where
one applies. The pre-Notebook-80 entries were selected from Notebook 78's
extraction. Notebook 75's evaluation families and predeclared stability
contrast, plus the later economic, prompt and measurement-sensitivity rows,
were appended when those studies closed.

\input{tables/tab_core_result_provenance}
\end{landscape}

\section{Reporting units}

The LSEG firm-level economic decomposition is reported in US dollars. The saved
aggregate column is named \texttt{gbp\_per\_million} because the frozen specification
declared a sterling notional, and that column name is preserved rather than rewritten.
No currency conversion was ever applied at any stage: the stored value is a return
difference multiplied by a notional of one million, so the figure is identical under
either label. The underlying cash flows are open-to-open returns on US-listed equities
and are therefore denominated in US dollars, which is the unit used in
\cref{ch:results}. The study models no foreign-exchange exposure, and a sterling label
would have implied one.

\section{Artificial-intelligence assistance}

OpenAI Codex (GPT-5 family, accessed in August 2026) was used as a drafting and analysis assistant. It produced a working manuscript from the author's research plan and aggregate outputs; suggested literature searches and structure; wrote and revised Python and LaTeX; checked bibliographic metadata; generated draft figures; and edited prose. Its output was not treated as empirical evidence. Numerical claims are tied to saved aggregate results, while scholarly claims are tied to the cited papers. The author directed the analysis, authorised data use, set the claim boundaries and retains responsibility for the submitted calculations, citations and wording.

The separate LSEG prompt experiments used Gemma 4 26B FP8 through OpenRouter, routed to DeepInfra. The five-variant experiment sent a selected Reuters headline, the target-company name and up to five strictly earlier same-company headlines for a novelty check. The later structured-reaction experiment sent only the current headline, company name and symbol; it did not assess novelty. No returns were supplied to either scoring prompt. Routing, retention controls and a US\$20 spending cap were recorded before the calls; actual spend for the reported prompt families was approximately US\$3.16. Other manuscript drafting used the clean repository's aggregate evidence rather than licensed story text. This disclosure follows UCL's requirement to acknowledge generative-AI use in assessed work.\footnote{UCL, ``Engaging with Generative AI in your education and assessment'', accessed 13 August 2026: \url{https://www.ucl.ac.uk/study/current-students/exams-and-assessments/assessment-success-guide/engaging-generative-ai-your-education-and-assessment}.}
````

<!--block:B0021-->
## manuscript/tables/tab_core_result_provenance.tex

<!--block:B0022-->
````latex
% Rows through Notebook 79 come from Notebook 78. Rows 80--85 were appended
% from the committed aggregate outputs; their manifests and specs retain full hashes.
\begingroup
\footnotesize
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{4.2cm} r >{\raggedright\arraybackslash}p{5.0cm} >{\raggedright\arraybackslash}p{5.0cm} l >{\raggedright\arraybackslash}p{2.2cm}@{}}
\caption{Provenance of the core reported results}
\label{tab:core-result-provenance} \\
\toprule
Claim & NB & Reported value & Frozen spec & Artifact hash & Seed \\
\midrule
\endfirsthead
\toprule
Claim & NB & Reported value & Frozen spec & Artifact hash & Seed \\
\midrule
\endhead
FNSPID panel dimensions & 01 & 715,546 firm-days; 570 symbols; 3,262 sessions & none & dc5d7433e41c & none / deterministic \\
FNSPID development negative-share IC & 03 & IC -0.005433; HAC t=-3.119 & none & 07151a502bce & 20260731 \\
Direct-signal turnover and break-even & 03 & 179.3x/year one-way; best break-even 0.625 bps/side & none & 07151a502bce & 20260731 \\
HAR benchmark evaluation performance & 21 & Sharpe 0.673; return +29.85\%; drawdown -15.73\% & none & 471b008ef45b & 20260805 \\
HAR x sentiment downside inference & 24 & p=0.0148 & none & cf3a6fb033dd & 20260805 \\
HAR x sentiment evaluation performance & 24 & Sharpe 0.835; return +35.20\%; drawdown -16.36\% & none & e33b8c4e0108 & 20260805 \\
Walk-forward HAR downside result & 25 & drawdown -12.53\% to -8.57\%; p=0.0052 & none & c9bbb3a2a24c & 20260805 \\
FNSPID evaluation downside timing & 45 & q=0.0301 & none & a0defab69499 & 20260805 \\
FNSPID conditional negative-share coefficient & 71 & beta -0.00831 [-0.01410,-0.00252]; p=0.0049; 2,264 sessions & conditional\_\allowbreak{}negative\_\allowbreak{}share\_\allowbreak{}transfer\_\allowbreak{}v1\_\allowbreak{}20260812 & d78b9b1b4f3c & none / deterministic \\
LSEG conditional-transfer coefficients & 71 & backward -0.0286; recent -0.0157; both intervals cross zero & conditional\_\allowbreak{}negative\_\allowbreak{}share\_\allowbreak{}transfer\_\allowbreak{}v1\_\allowbreak{}20260812 & d78b9b1b4f3c & none / deterministic \\
Recent LSEG pressure non-activation & 72 & 0/167 risk-off sessions; maximum z=1.349 vs 1.5 entry & lseg\_\allowbreak{}negative\_\allowbreak{}pressure\_\allowbreak{}har\_\allowbreak{}transfer\_\allowbreak{}v1\_\allowbreak{}20260812 & 69a981ced4f1 & none / deterministic \\
Backward LSEG pressure timing & 73 & BH q=0.0482 & lseg\_\allowbreak{}negative\_\allowbreak{}pressure\_\allowbreak{}har\_\allowbreak{}backward\_\allowbreak{}transfer\_\allowbreak{}v1\_\allowbreak{}20260812 & 524cb0a1f746 & none / deterministic \\
Backward LSEG pressure activation & 73 & 53/456 sessions; 18 entries & lseg\_\allowbreak{}negative\_\allowbreak{}pressure\_\allowbreak{}har\_\allowbreak{}backward\_\allowbreak{}transfer\_\allowbreak{}v1\_\allowbreak{}20260812 & 7faf73cb4846 & none / deterministic \\
Backward LSEG pressure downside inference & 73 & BH q=0.0336 & lseg\_\allowbreak{}negative\_\allowbreak{}pressure\_\allowbreak{}har\_\allowbreak{}backward\_\allowbreak{}transfer\_\allowbreak{}v1\_\allowbreak{}20260812 & 9cd3e8235457 & none / deterministic \\
Conditional-effect yearly direction & 74 & negative in 8/9 development years & none & 8e70cc12d1ed & none / deterministic \\
Conditional-effect multi-story diagnostic & 74 & n>=2 beta -0.00529 [-0.01272,+0.00214] & none & d75411234ca1 & none / deterministic \\
Development double-sort economic magnitude & 76 & all -0.62 [-1.51,+0.27]; n>=2 -0.25 [-1.82,+1.32] bps/session & examiner\_\allowbreak{}facing\_\allowbreak{}analysis\_\allowbreak{}pack\_\allowbreak{}v1\_\allowbreak{}20260812 & 4fac1bedcc4f & 20260812 \\
Learned thresholds versus historical fixed band & 77 & -7.71 to -16.88 net bps/session; MDE 4.59 to 10.04 bps & examiner\_\allowbreak{}facing\_\allowbreak{}analysis\_\allowbreak{}pack\_\allowbreak{}v1\_\allowbreak{}20260812 & b4bf7a431b48 & 20260831 \\
LSEG conditional MDE relative to FNSPID & 77 & 7.6x backward; 13.6x recent; story 18.80 bps; publisher 0.049 IC; earnings 15.52 bps & examiner\_\allowbreak{}facing\_\allowbreak{}analysis\_\allowbreak{}pack\_\allowbreak{}v1\_\allowbreak{}20260812 & b4bf7a431b48 & 20260831 \\
FNSPID reversal-controlled negative-share coefficient & 79 & beta -0.00914 [-0.01466,-0.00362]; p=0.00118; 2,264 sessions & conditional\_\allowbreak{}negative\_\allowbreak{}share\_\allowbreak{}price\_\allowbreak{}path\_\allowbreak{}controls\_\allowbreak{}v1\_\allowbreak{}20260812 & 4b2bfb954830 & none / deterministic \\
FNSPID conditional evaluation replication & 75 & all-days beta +0.00583 [-0.00426,+0.01593], q=0.515; multi-story beta +0.00295, q=0.653 & fnspid\_\allowbreak{}evaluation\_\allowbreak{}beyond\_\allowbreak{}mean\_\allowbreak{}replication\_\allowbreak{}v1 plus pre-execution amendment & 4bdf90ce939e & none / deterministic \\
Evaluation-minus-development stability contrast & 75 & delta +0.01414 [+0.00250,+0.02579]; p=0.0173; HAC(5) & fnspid\_\allowbreak{}evaluation\_\allowbreak{}beyond\_\allowbreak{}mean\_\allowbreak{}replication\_\allowbreak{}v1 plus pre-execution amendment & f7e01d905a32 & none / deterministic \\
LSEG aggregate economic search & 80 & selected 0.878 vs HAR 0.926 net bps/session; both complete gates fail & lseg33\_\allowbreak{}economic\_\allowbreak{}value\_\allowbreak{}rule\_\allowbreak{}search\_\allowbreak{}v1\_\allowbreak{}20260812 & 6d72fec852c7 & 20260912--19 \\
FNSPID economic search & 81 & both complete comparison gates fail & fnspid\_\allowbreak{}economic\_\allowbreak{}value\_\allowbreak{}rule\_\allowbreak{}search\_\allowbreak{}v1\_\allowbreak{}20260812 & b30c628bb69a & 20261012--19 \\
LSEG firm-level loss control & 82 & ending wealth difference -USD 30,869.79 per USD 1m vs symbol-matched constant & lseg33\_\allowbreak{}firm\_\allowbreak{}level\_\allowbreak{}loss\_\allowbreak{}control\_\allowbreak{}search\_\allowbreak{}v1\_\allowbreak{}20260812 & c4f96403b362 & 20260913--24 \\
LSEG five-prompt search & 83 & best net mean -4.012 bps/session; 0/5 candidates eligible & lseg\_\allowbreak{}prompt\_\allowbreak{}economic\_\allowbreak{}value\_\allowbreak{}v1\_\allowbreak{}20260813 & c35c14e61007 & 20260813 \\
Structured reaction prompt & 84 & net mean -19.666 bps/session; selection gate fails & lseg\_\allowbreak{}structured\_\allowbreak{}reaction\_\allowbreak{}score\_\allowbreak{}v1\_\allowbreak{}20260813 & e7414d3b5a2a & 20260813 \\
Post-hoc close-to-close, FF3 and soft-mass family & 85 & 0/3 BH survivors; baseline remains $-0.00831$ & fnspid\_\allowbreak{}development\_\allowbreak{}outcome\_\allowbreak{}and\_\allowbreak{}label\_\allowbreak{}sensitivities\_\allowbreak{}v1\_\allowbreak{}20260813 & 4bdf0f83a0a8 & none / deterministic \\
\bottomrule
\end{longtable}
\endgroup
````

<!--block:B0023-->
## manuscript/tables/tab_null_mde.tex

<!--block:B0024-->
````latex
% Generated by Notebook 77; see its manifest for hashes and seeds.
% Column widths and header line breaks adjusted for the A4 text block; the
% family names, counts and MDE values are reproduced exactly as generated.
\begin{table}[htbp]
\centering
\small
\caption{Minimum detectable effects for the principal null families. The conservative column replaces $z_{0.975}$ in \cref{eq:mde} with the first Benjamini--Hochberg threshold for a family of that size.}
\label{tab:null-mde}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.9cm} d{2.0} d{2.3} d{2.3} >{\raggedright\arraybackslash}p{3.9cm}@{}}
\toprule
Family & {Tests} & {Median} & {Conservative} & Unit \\
 & & {MDE} & {BH MDE} & \\
\midrule
Earnings-window conditioning & 3 & 15.519 & 17.923 & bps interaction vs outside \\
LSEG conditional transfer & 2 & 0.088 & 0.097 & rank coefficient \\
Learned trade/no-trade thresholds & 3 & 6.878 & 7.944 & net bps/session vs historical fixed band \\
Publisher conditioning & 4 & 0.049 & 0.059 & daily cross-sectional IC \\
Story-type conditioning & 12 & 18.802 & 24.878 & bps per unit negative share \\
\bottomrule
\end{tabular}
\end{table}
````

<!--block:B0025-->
## manuscript/tables/tab_reversal_confound.tex

<!--block:B0026-->
````latex
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}l d{-1.5} d{1.5} c d{1.4}@{}}
\toprule
Specification & {$\hat\beta_{neg}$} & {HAC SE} & {95\% CI} & {$p$} \\
\midrule
Full-sample baseline & -0.00831 & 0.00295 & $[-0.01410,\,-0.00252]$ & 0.0049 \\
Price-complete baseline & -0.00830 & 0.00296 & $[-0.01410,\,-0.00251]$ & 0.0050 \\
+ one-session return & -0.00945 & 0.00291 & $[-0.01515,\,-0.00375]$ & 0.0012 \\
+ one- and five-session returns & -0.00978 & 0.00288 & $[-0.01541,\,-0.00414]$ & 0.0007 \\
+ returns and 20-session volatility & -0.00914 & 0.00282 & $[-0.01466,\,-0.00362]$ & 0.0012 \\
\bottomrule
\end{tabular}
% Generated by final_experiments/79_fnspid_development_reversal_confound.ipynb.
% All price controls end at formation open t; ar_open_h1 runs from t to t+1.
````

<!--block:B0027-->
## manuscript/artifacts/tab_aggregation_family.tex

<!--block:B0028-->
````latex
\begin{table}[htbp]
\centering
\caption{The nine firm-day aggregation rules on the FNSPID development panel. $\overline{IC}$ is the mean daily cross-sectional Spearman correlation with the next-open abnormal return; $t$ uses Newey--West HAC(5). BH is Benjamini--Hochberg at $q=0.05$ across the nine rules. Break-even is the per-side cost that sets the mean net return of the corresponding rank portfolio to zero. The rules use 512,145--512,149 firm-days, 2,264 sessions.}
\label{tab:aggregation-family}
\small
\begin{tabular}{@{}l d{-1.5} d{-1.2} d{1.4} c d{1.3}@{}}
\toprule
Aggregation rule & {$\overline{IC}$} & {HAC $t$} & {$p$} & {BH} & {Break-even (bps)} \\
\midrule
Negative-story share & -0.00543 & -3.12 & 0.0018 & \checkmark & 0.625 \\
Decayed sentiment state & 0.00416 & 2.23 & 0.0259 & \textendash & 0.323 \\
Median continuous score & 0.00347 & 2.02 & 0.0431 & \textendash & 0.320 \\
Log-count weighted mean & 0.00328 & 1.81 & 0.0704 & \textendash & 0.245 \\
Trimmed mean & 0.00302 & 1.70 & 0.0889 & \textendash & 0.248 \\
Mean continuous score & 0.00298 & 1.68 & 0.0930 & \textendash & 0.245 \\
Mean hard label & 0.00280 & 1.58 & 0.1149 & \textendash & 0.292 \\
Strongest event & 0.00271 & 1.55 & 0.1205 & \textendash & 0.116 \\
Score dispersion & -0.00149 & -0.92 & 0.3590 & \textendash & 0.279 \\
\bottomrule
\end{tabular}
\end{table}
````

<!--block:B0029-->
## manuscript/artifacts/tab_headline_estimates.tex

<!--block:B0030-->
````latex
\begin{table}[htbp]
\centering
\caption{Headline negative-story-share estimates. FNSPID uses next-open abnormal returns; LSEG uses next-open raw returns. The evaluation rows use BH across the two declared conditional coefficients; the regime contrast is a separate predeclared diagnostic. Intervals are 95\% HAC intervals.}
\label{tab:headline-estimates}
\small
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.9cm} d{-1.5} c l d{4.0}@{}}
\toprule
Test & {Estimate} & {95\% interval} & {$p$ or $q$} & {Sessions} \\
\midrule
Aggregation-family IC & -0.00543 & {\textendash} & $p=0.0018$ & 2264 \\
Conditional baseline & -0.00831 & $[-0.01410,\,-0.00252]$ & $p=0.0049$ & 2264 \\
Full price-path controls & -0.00914 & $[-0.01466,\,-0.00362]$ & $p=0.0012$ & 2264 \\
Evaluation: all firm-days & 0.00583 & $[-0.00426,\,0.01593]$ & $q=0.515$ & 998 \\
Evaluation: at least two stories & 0.00295 & $[-0.00990,\,0.01581]$ & $q=0.653$ & 998 \\
Evaluation minus development & 0.01414 & $[0.00250,\,0.02579]$ & $p=0.0173$ & 3262 \\
Model-free high--low (bps) & -0.618 & $[-1.510,\,0.275]$ & $p=0.175$ & 2264 \\
LSEG backward coefficient & -0.02861 & $[-0.07283,\,0.01560]$ & $q=0.409$ & 456 \\
LSEG recent coefficient & -0.01573 & $[-0.09453,\,0.06308]$ & $q=0.696$ & 167 \\
\bottomrule
\end{tabular}
\end{table}
````

<!--block:B0031-->
## manuscript/artifacts/tab_outcome_label_sensitivities.tex

<!--block:B0032-->
````latex
\begin{table}[htbp]
\centering
\caption{Post-hoc measurement sensitivities on FNSPID development. The baseline row repeats the previously reported next-open hard-share estimate and is not a fourth family member. BH is Benjamini--Hochberg at $q=0.05$ across the three new coefficients. Close-to-close starts at the assigned session close; FF3 residuals use trailing 252-session betas that exclude the outcome session.}
\label{tab:outcome-label-sensitivities}
\small
\begin{tabular}{@{}l d{-1.5} c d{1.4} c c@{}}
\toprule
Test & {Estimate} & {95\% CI} & {$p$} & {$q$} & {BH} \\
\midrule
Next-open hard share (baseline) & -0.00831 & $[-0.01410,\,-0.00252]$ & 0.0049 & -- & -- \\
Close-to-close hard share & 0.00027 & $[-0.00572,\,0.00627]$ & 0.9292 & 0.929 & -- \\
FF3 residual hard share & 0.00182 & $[-0.00404,\,0.00767]$ & 0.5435 & 0.815 & -- \\
Next-open soft negative mass & -0.00378 & $[-0.00844,\,0.00088]$ & 0.1122 & 0.337 & -- \\
\bottomrule
\end{tabular}
\end{table}
````

<!--block:B0033-->
## manuscript/artifacts/tab_robustness_diagnostics.tex

<!--block:B0034-->
````latex
\begin{table}[htbp]
\centering
\caption{Sensitivity of the baseline conditional negative-share coefficient. The upper panel restricts the panel by same-day story count; the lower panel varies the Newey--West lag length on the frozen full sample. All rows use centred within-session percentile ranks and the next-open abnormal return. Panel-row counts precede complete-case filtering; four rows in the full panel lack the return.}
\label{tab:robustness-diagnostics}
\small
\begin{tabular}{@{}l d{-1.5} c d{1.4} d{6.0}@{}}
\toprule
Variation & {$\hat\beta_{neg}$} & {95\% CI} & {$p$} & {Panel rows} \\
\midrule
\multicolumn{5}{@{}l}{\emph{Same-day story count}} \\
\quad All firm-days (frozen) & -0.00831 & $[-0.01410,\,-0.00252]$ & 0.0049 & 512153 \\
\quad At least two stories ($n_{it}\geq2$) & -0.00529 & $[-0.01272,\,0.00214]$ & 0.1629 & 264683 \\
\quad At least three stories ($n_{it}\geq3$) & -0.00183 & $[-0.01385,\,0.01019]$ & 0.7656 & 137142 \\
\addlinespace
\multicolumn{5}{@{}l}{\emph{Newey--West lag length}} \\
\quad 5 lags (frozen) & -0.00831 & $[-0.01410,\,-0.00252]$ & 0.0049 & 512153 \\
\quad 10 lags & -0.00831 & $[-0.01414,\,-0.00248]$ & 0.0052 & 512153 \\
\quad 21 lags & -0.00831 & $[-0.01407,\,-0.00255]$ & 0.0047 & 512153 \\
\bottomrule
\end{tabular}
\end{table}
````

<!--block:B0035-->
## manuscript/references.bib

<!--block:B0036-->
````bibtex
@article{allee2015,
  author  = {Allee, Kristian D. and DeAngelis, Matthew D.},
  title   = {The Structure of Voluntary Disclosure Narratives: Evidence from Tone Dispersion},
  journal = {Journal of Accounting Research},
  year    = {2015},
  volume  = {53},
  number  = {2},
  pages   = {241--274},
  doi     = {10.1111/1475-679X.12072}
}

@article{araci2019,
  author  = {Araci, Dogu},
  title   = {{FinBERT}: Financial Sentiment Analysis with Pre-trained Language Models},
  journal = {arXiv preprint arXiv:1908.10063},
  year    = {2019},
  url     = {https://arxiv.org/abs/1908.10063}
}

@article{benjamini1995,
  author  = {Benjamini, Yoav and Hochberg, Yosef},
  title   = {Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  year    = {1995},
  volume  = {57},
  number  = {1},
  pages   = {289--300},
  doi     = {10.1111/j.2517-6161.1995.tb02031.x}
}

@article{bloom1995,
  author  = {Bloom, Howard S.},
  title   = {Minimum Detectable Effects: A Simple Way to Report the Statistical Power of Experimental Designs},
  journal = {Evaluation Review},
  year    = {1995},
  volume  = {19},
  number  = {5},
  pages   = {547--556},
  doi     = {10.1177/0193841X9501900504}
}

@article{chan2003news,
  author  = {Chan, Wesley S.},
  title   = {Stock Price Reaction to News and No-News: Drift and Reversal after Headlines},
  journal = {Journal of Financial Economics},
  year    = {2003},
  volume  = {70},
  number  = {2},
  pages   = {223--260},
  doi     = {10.1016/S0304-405X(03)00146-6}
}

@article{chen2024,
  author  = {Chen, Yugang and Lu, Jihua and Ma, Weidong and Kumar, Satish and Shahab, Yasir},
  title   = {Dispersion in News Sentiment and {M\&A}s Outcomes},
  journal = {Research in International Business and Finance},
  year    = {2024},
  volume  = {71},
  pages   = {102415},
  doi     = {10.1016/j.ribaf.2024.102415}
}

@article{corsi2009har,
  author  = {Corsi, Fulvio},
  title   = {A Simple Approximate Long-Memory Model of Realized Volatility},
  journal = {Journal of Financial Econometrics},
  year    = {2009},
  volume  = {7},
  number  = {2},
  pages   = {174--196},
  doi     = {10.1093/jjfinec/nbp001}
}

@inproceedings{dong2024,
  author    = {Dong, Zihan and Fan, Xinyu and Peng, Zhiyuan},
  title     = {{FNSPID}: A Comprehensive Financial News Dataset in Time Series},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2024},
  pages     = {4918--4927},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3637528.3671629}
}

@article{engelberg2011,
  author  = {Engelberg, Joseph E. and Parsons, Christopher A.},
  title   = {The Causal Impact of Media in Financial Markets},
  journal = {The Journal of Finance},
  year    = {2011},
  volume  = {66},
  number  = {1},
  pages   = {67--97},
  doi     = {10.1111/j.1540-6261.2010.01626.x}
}

@article{famafrench1993,
  author  = {Fama, Eugene F. and French, Kenneth R.},
  title   = {Common Risk Factors in the Returns on Stocks and Bonds},
  journal = {Journal of Financial Economics},
  year    = {1993},
  volume  = {33},
  number  = {1},
  pages   = {3--56},
  doi     = {10.1016/0304-405X(93)90023-5}
}

@article{fang2009,
  author  = {Fang, Lily and Peress, Joel},
  title   = {Media Coverage and the Cross-section of Stock Returns},
  journal = {The Journal of Finance},
  year    = {2009},
  volume  = {64},
  number  = {5},
  pages   = {2023--2052},
  doi     = {10.1111/j.1540-6261.2009.01493.x}
}

@article{garcia2013,
  author  = {Garc{\'i}a, Diego},
  title   = {Sentiment during Recessions},
  journal = {The Journal of Finance},
  year    = {2013},
  volume  = {68},
  number  = {3},
  pages   = {1267--1300},
  doi     = {10.1111/jofi.12027}
}

@article{glasserman2023,
  author  = {Glasserman, Paul and Lin, Caden},
  title   = {Assessing Look-Ahead Bias in Stock Return Predictions Generated by {GPT} Sentiment Analysis},
  journal = {arXiv preprint arXiv:2309.17322},
  year    = {2023},
  url     = {https://arxiv.org/abs/2309.17322}
}

@article{harvey2016multiple,
  author  = {Harvey, Campbell R. and Liu, Yan and Zhu, Heqing},
  title   = {{\ldots} and the Cross-Section of Expected Returns},
  journal = {The Review of Financial Studies},
  year    = {2016},
  volume  = {29},
  number  = {1},
  pages   = {5--68},
  doi     = {10.1093/rfs/hhv059}
}

@article{heston2017,
  author  = {Heston, Steven L. and Sinha, Nitish Ranjan},
  title   = {News versus Sentiment: Predicting Stock Returns from News Stories},
  journal = {Financial Analysts Journal},
  year    = {2017},
  volume  = {73},
  number  = {3},
  pages   = {67--83},
  doi     = {10.2469/faj.v73.n3.3}
}

@article{huang2023,
  author  = {Huang, Allen H. and Wang, Hui and Yang, Yi},
  title   = {{FinBERT}: A Large Language Model for Extracting Information from Financial Text},
  journal = {Contemporary Accounting Research},
  year    = {2023},
  volume  = {40},
  number  = {2},
  pages   = {806--841},
  doi     = {10.1111/1911-3846.12832}
}

@article{isakin2023,
  author  = {Isakin, Maksim and Pu, Xiaoling},
  title   = {Dispersion in News Sentiment and Corporate Bond Returns},
  journal = {International Review of Financial Analysis},
  year    = {2023},
  volume  = {89},
  pages   = {102761},
  doi     = {10.1016/j.irfa.2023.102761}
}

@article{kirtac2024,
  author  = {Kirtac, Kemal and Germano, Guido},
  title   = {Sentiment Trading with Large Language Models},
  journal = {Finance Research Letters},
  year    = {2024},
  volume  = {62},
  pages   = {105227},
  doi     = {10.1016/j.frl.2024.105227}
}

@article{korajczyk2004,
  author  = {Korajczyk, Robert A. and Sadka, Ronnie},
  title   = {Are Momentum Profits Robust to Trading Costs?},
  journal = {The Journal of Finance},
  year    = {2004},
  volume  = {59},
  number  = {3},
  pages   = {1039--1082},
  doi     = {10.1111/j.1540-6261.2004.00656.x}
}

@article{lopezlira2026,
  author  = {Lopez-Lira, Alejandro and Tang, Yuehua},
  title   = {Can {ChatGPT} Forecast Stock Price Movements? Return Predictability and Large Language Models},
  journal = {Journal of Financial Economics},
  year    = {2026},
  volume  = {184},
  pages   = {104335},
  note    = {Advance online publication; October issue forthcoming at the dissertation search date},
  doi     = {10.1016/j.jfineco.2026.104335}
}

@article{loughran2011liability,
  author  = {Loughran, Tim and McDonald, Bill},
  title   = {When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-{K}s},
  journal = {The Journal of Finance},
  year    = {2011},
  volume  = {66},
  number  = {1},
  pages   = {35--65},
  doi     = {10.1111/j.1540-6261.2010.01625.x}
}

@article{loughran2016,
  author  = {Loughran, Tim and McDonald, Bill},
  title   = {Textual Analysis in Accounting and Finance: A Survey},
  journal = {Journal of Accounting Research},
  year    = {2016},
  volume  = {54},
  number  = {4},
  pages   = {1187--1230},
  doi     = {10.1111/1475-679X.12123}
}

@article{mackinlay1997,
  author  = {MacKinlay, A. Craig},
  title   = {Event Studies in Economics and Finance},
  journal = {Journal of Economic Literature},
  year    = {1997},
  volume  = {35},
  number  = {1},
  pages   = {13--39},
  url     = {https://www.jstor.org/stable/2729691}
}

@article{malo2014,
  author  = {Malo, Pekka and Sinha, Ankur and Korhonen, Pekka and Wallenius, Jyrki and Takala, Pyry},
  title   = {Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts},
  journal = {Journal of the Association for Information Science and Technology},
  year    = {2014},
  volume  = {65},
  number  = {4},
  pages   = {782--796},
  doi     = {10.1002/asi.23062}
}

@article{moreira2017,
  author  = {Moreira, Alan and Muir, Tyler},
  title   = {Volatility-Managed Portfolios},
  journal = {The Journal of Finance},
  year    = {2017},
  volume  = {72},
  number  = {4},
  pages   = {1611--1644},
  doi     = {10.1111/jofi.12513}
}

@article{newey1987,
  author  = {Newey, Whitney K. and West, Kenneth D.},
  title   = {A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix},
  journal = {Econometrica},
  year    = {1987},
  volume  = {55},
  number  = {3},
  pages   = {703--708},
  doi     = {10.2307/1913610}
}

@article{novymarx2016,
  author  = {Novy-Marx, Robert and Velikov, Mihail},
  title   = {A Taxonomy of Anomalies and Their Trading Costs},
  journal = {The Review of Financial Studies},
  year    = {2016},
  volume  = {29},
  number  = {1},
  pages   = {104--147},
  doi     = {10.1093/rfs/hhv063}
}

@article{politis1994,
  author  = {Politis, Dimitris N. and Romano, Joseph P.},
  title   = {The Stationary Bootstrap},
  journal = {Journal of the American Statistical Association},
  year    = {1994},
  volume  = {89},
  number  = {428},
  pages   = {1303--1313},
  doi     = {10.1080/01621459.1994.10476870}
}

@article{savor2012information,
  author  = {Savor, Pavel G.},
  title   = {Stock Returns after Major Price Shocks: The Impact of Information},
  journal = {Journal of Financial Economics},
  year    = {2012},
  volume  = {106},
  number  = {3},
  pages   = {635--659},
  doi     = {10.1016/j.jfineco.2012.06.011}
}

@article{tetlock2007,
  author  = {Tetlock, Paul C.},
  title   = {Giving Content to Investor Sentiment: The Role of Media in the Stock Market},
  journal = {The Journal of Finance},
  year    = {2007},
  volume  = {62},
  number  = {3},
  pages   = {1139--1168},
  doi     = {10.1111/j.1540-6261.2007.01232.x}
}

@article{tetlock2008words,
  author  = {Tetlock, Paul C. and Saar-Tsechansky, Maytal and Macskassy, Sofus},
  title   = {More than Words: Quantifying Language to Measure Firms' Fundamentals},
  journal = {The Journal of Finance},
  year    = {2008},
  volume  = {63},
  number  = {3},
  pages   = {1437--1467},
  doi     = {10.1111/j.1540-6261.2008.01362.x}
}

@article{tetlock2011stale,
  author  = {Tetlock, Paul C.},
  title   = {All the News That's Fit to Reprint: Do Investors React to Stale Information?},
  journal = {The Review of Financial Studies},
  year    = {2011},
  volume  = {24},
  number  = {5},
  pages   = {1481--1512},
  doi     = {10.1093/rfs/hhq141}
}

@article{uhl2014,
  author  = {Uhl, Matthias W.},
  title   = {Reuters Sentiment and Stock Returns},
  journal = {Journal of Behavioral Finance},
  year    = {2014},
  volume  = {15},
  number  = {4},
  pages   = {287--298},
  doi     = {10.1080/15427560.2014.967852}
}

@article{uhl2021,
  author  = {Uhl, Matthias W. and Novacek, Milos},
  title   = {When It Pays to Ignore: Focusing on Top News and Their Sentiment},
  journal = {Journal of Behavioral Finance},
  year    = {2021},
  volume  = {22},
  number  = {4},
  pages   = {461--479},
  doi     = {10.1080/15427560.2020.1821375}
}
````

<!--block:B0037-->
## Binary artifact manifest

<!--block:B0038-->
````text
path	status	sha256	bytes
manuscript/artifacts/fig_aggregation_family.png	present	f341365edf0df4a9b722d6b060982a12a8883c5a16ee892ced7b6a8dee43dcd6	106248
manuscript/artifacts/fig_break_even_cost_gap.png	present	238afad9f96c297d9e5e52ac4a8106b2227dfb98e5358af1d6b4d1ccb75cd470	106882
manuscript/artifacts/fig_cross_source_coefficient_power.png	present	7ce04dd84e2be0049d0f87a60d0150d1744e9c5937fea617da165dc5aab448b5	78953
manuscript/artifacts/fig_double_sort.png	present	6b25cfbd90af03fbb5ba020447af5242e7c6963c489f24597df541da6e27bb39	71175
manuscript/artifacts/fig_estimate_stability.png	present	f53309e59910ed180dcf9bf11cbeace47989bfee280328efb0f6709fd7f8337b	104421
manuscript/artifacts/fig_lseg_matched_control_value.png	present	96f375a2216f63d96868cf815b77c4a44c4e586f3a08d83bdcb99c958a1430e5	123854
manuscript/artifacts/fig_prompt_gate.png	present	3100db153a55694afe629af828d0eee8b13dca7cbf6fe6aee3b8f28a26738dbb	98611
manuscript/artifacts/fig_risk_regime_stability.png	present	8de2683b50263ac559b17889423940990c5b1f39103b67d3f85b6a3908807df8	45022
manuscript/artifacts/fig_timing_placebo.png	present	f60cfb7b2b250bcde9e5f9315f199be3f447a0d2f80739a1c9fb6cb06b0b7817	78340
manuscript/figures/fig_aggregation_ic.png	present	f170a2195749b6db34ec5000a3645e6e444c04fb75fdf916745ec09b1eed1bc9	71209
manuscript/figures/fig_conditional_double_sort.png	present	8bdfcf54f5fbfe7d914b9f69ec3b93acfce185215398e27ea6531388f3a6df4d	91602
manuscript/figures/fig_downside_placebo.png	present	19f4ec7d7c8e1d996f6f88c8f6db00a8d518b46b7127e5a84e03512f307eca86	109562
manuscript/figures/fig_har_sentiment_equity.png	present	e55d9a897f5f6642b4196b602aa08b7d77f90c1501c2ed62d7bd36d016cbca52	346775
manuscript/figures/fig_lseg33_economic_case_and_limit.png	present	f7a5c9a446e3c24f106286d0dcd4332fb4c2c45fe6853784d458679cada41121	299710
manuscript/figures/fig_lseg33_economic_risk_metrics.png	present	fd72bcd79b450abd4990b9f5aee0f455e8e8be30a7ca2adde3e1613b57a49f88	135623
manuscript/figures/fig_lseg33_economic_waterfall.png	present	28a35b1b31dd8a8766f48b86543abc63fc1976d63daade96c5b8610ae27e11f3	135918
manuscript/figures/fig_lseg33_firm_loss_control_evaluation.png	present	6758cad2f91ac110b00da9738af1b04b55141cd406a3c5590be244c7c01d6ebb	305020
manuscript/figures/fig_lseg33_firm_loss_saving.png	present	aeca4a992e5be7c842792baaa07626c19399bada5a0633deb3938ca7d14d9a2d	124884
manuscript/figures/fig_lseg33_prompt_economic_comparison.png	present	0f4edd5d847be4b1fb9cfd7828ffc622fde6e8207597bfc264a9ab4a85cefe4e	96745
manuscript/figures/fig_lseg33_prompt_selection.png	present	78e49677d93364ebdcad75ace69ebe72fce95272b068f4fdfdb5e69b75d5c48a	144928
manuscript/figures/fig_lseg33_risk_activation.png	present	a963ace3fa7b79f6887359ffbdb1dc3164c7838e8b783de69a1e4485fb3a8424	388489
manuscript/figures/fig_lseg33_structured_reaction_score.png	present	2178f682c6d5f0cc11e4890c15636fd45a65074826823c08890ec5eded0fc11e	119768
manuscript/figures/fig_lseg33_timing_placebo.png	present	f5c6e62ff35975f5ef4597a74c4b67cf55c0e618692b2ea8f9adc952ceec5f0f	138151
manuscript/figures/fig_lseg33_tuned_rule_economic_metrics.png	present	5b6fa8518aa5551cb7da59a55907ec4be668cce65e7b9db13f05d41fe49e7af1	106562
manuscript/figures/fig_lseg33_tuned_rule_evaluation.png	present	2815677cf270e779aa180a2e9bd63c35fafaeee1c871f64500d7e53729128a6a	394264
manuscript/figures/fig_reversal_confound.png	present	78b322aa326b93f12db265cb6249c2fa38f3796e39d6dec2dfab2fa36c25758f	149921
manuscript/figures/fig_turnover_breakeven.png	present	7c14adbeb9a3a3fb2fa912698bd5ae797dd68d6c6a0cedcf4303d211cc80de9b	95353
manuscript/ucl_logo.jpg	present	e049aff1be8c83dc28520cf40c70b09904fd5cf3550374c923525dce9f990521	32106
manuscript/ucl_logo.svg	absent	-	0
````
