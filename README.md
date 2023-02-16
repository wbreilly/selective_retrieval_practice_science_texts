## A repository for the analysis and visualization code used in *The limited reach of retrieval practice: Practice of science text material enhances retention of practiced material, but unpracticed material is unaffected.* Reilly W.B., Antony. J.W., & Ranganath, C. (in prep)

### Project Description

Retrieval practice is well known to enhance retention of practiced material, but can enhance or impair retention of unpracticed material. In educational texts, the effects of retrieval practice are not well understood. We explored the effects of retrieval practice in science texts by having participants practice retrieval of "main ideas" and "peripheral ideas" after reading a science text. We predicted that retrieval practice would enhance retention of practiced material in both conditions, relative to a control condition that did not practice any material. Furthermore, we predicted that practice of main ideas would enhance retention of additional information that was not practiced, and that practice of peripheral ideas would imapir retention of additional information. Finally, we collected individual difference variables that we expected to moderate the effects of retrieval practice. 

Our results showed robust increases in retention for practiced material, regardless of whether participants were skilled or less-skilled readers, or had more or less prior knowledge. For unpracticed material, we observed inconsistent evidence that retrieval practice impacted retention. The upside of these results is that educators need not be concerned that retrieval practice (formative assessments) will have unequal benefits for their students, and that practicing material that is peripheral rather than central to a text passage does not have negative effects. The downside is that retrieval practice has rather focal retention benefits.

### Contents

This repository includes the scripts used to munge, analyze, and visualize the data and the resulting figures. It also includes html files that contain the code and output from all statistical analyses.

### Key scripts

`preprocessing.py` Run interactively. Combines the three experimental phases, engineers features, identifies bad data, and outputs clean dataframes.

`publication_ready_plots.py` Produces the manuscript's figures. 

`viruses_stats_imac2.Rmd` Experiment 1 statistical analyses. Control analyses, recall model selection and analyses, multiple choice model selection and analyses. 

`endocrine_stats_imac2.Rmd` Experiment 2 statistical analyses. Control analyses, recall model selection and analyses, multiple choice model selection and analyses. 

`bayes_factor.Rmd` Bayes Factor analyses for Experiment 1 and 2.

### Figure 1. 
<p align="center">
	<img src = "figures/recall_figure.png" width = "70%">
</p>
Recall Performance. Mean idea units recalled on the final free recall test for each type of idea unit for Experiment 1 (viruses text) and Experiment 2 (endocrine text). Main idea units were practiced by the RPm group only. Peripheral idea units were practiced by the RPp group only. The NRP group did not practice any idea units. Panels A and B depict the “testing effect” in that retention of main ideas and peripheral ideas were greatest for the RPm and RPp groups, respectively. Panel C depicts recall of unpracticed idea units (all idea units not included in previous two panels). Error bars indicate standard error of the mean. 


### Figure 2. 
<p align="center">
	<img src = "figures/mc_figure.png" width = "70%">
</p>
Multiple Choice Performance. Mean proportion correct on the multiple-choice final test. Error bars represent standard error of the mean.


### Figure 3. 
<p align="center">
	<img src = "figures/experiment2_3way_figure.png" width = "70%">
</p>
Effects of reading ability and domain knowledge on recall in Experiment 2. Bootstrap analyses of unpracticed idea units in the endocrine text (Experiment 2) revealed a significant three-way interaction between participant group, reading ability, and prior knowledge. For visualization purposes, participants were divided with a median split according to prior knowledge and reading ability. Error bars represent standard error of the mean of each cell.

### Tools

[![](https://img.shields.io/badge/python-3.7.9-blue)](https://www.python.org/)
[![](https://img.shields.io/badge/spyder-4.1.5-blue)](https://www.spyder-ide.org/)
[![](https://img.shields.io/badge/pandas-1.1.3-blue)](https://pandas.pydata.org/)
[![](https://img.shields.io/badge/seaborn-0.11.1-blue)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/matplotlib-3.3.2-blue)](https://matplotlib.org/)

[![](https://img.shields.io/badge/R-4.2.2-blue)](https://www.r-project.org/)
[![](https://img.shields.io/badge/RStudio-2022.07.2-blue)](https://posit.co/products/open-source/rstudio/)
[![](https://img.shields.io/badge/tidyverse-1.3.2-blue)](https://www.tidyverse.org/)
[![](https://img.shields.io/badge/afex-1.2.0-blue)](https://cran.r-project.org/web/packages/afex/index.html)
[![](https://img.shields.io/badge/emmeans-1.8.2-blue)](https://cran.r-project.org/web/packages/emmeans/index.html)
[![](https://img.shields.io/badge/pbkrtest-0.5.1-blue)](https://cran.r-project.org/web/packages/pbkrtest/index.html)
[![](https://img.shields.io/badge/lme4-1.1.31-blue)](https://cran.r-project.org/web/packages/lme4/index.html)
[![](https://img.shields.io/badge/gt-0.8.0-blue)](https://cloud.r-project.org/web/packages/gt/index.html)
[![](https://img.shields.io/badge/rmarkdown-2.18.0-blue)](https://rmarkdown.rstudio.com/)










