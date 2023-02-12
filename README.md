# A repository for the analysis and visualization code used in "The limited reach of retrieval practice: Practice of science text material enhances retention of practiced material, but unpracticed material is unaffected." Reilly W.B., Antony. J.W., & Ranganath, C. (in prep)

## Project Description

Retrieval practice is well known to enhance retention of practiced material, but can enhance or impair retention of unpracticed material. In educational texts, the effects of retrieval practice are not well understood. We explored the effects of retrieval practice in science texts by having participants practice retrieval of "main ideas" and "peripheral ideas" after reading a science text. We predicted that retrieval practice would enhance retention of practiced material in both conditions, relative to a control condition that did not practice any material. Furthermore, we predicted that practice of main ideas would enhance retention of additional information that was not practiced, and that practice of peripheral ideas would imapir retention of additional information. Finally, we collected individual difference variables that we expected to moderate the effects of retrieval practice. 

Our results showed robust increases in retention for practiced material, regardless of whether participants were skilled or less-skilled readers, or had more or less prior knowledge. For unpracticed material, we observed weak and inconsistent evidence that retrieval practice impacted retention. The upside of these results is that educators need not be concerned that retrieval practice (formative assessments) will have unequal benefits for their students, and that practicing material that is peripheral rather than central to a text passage does not have negative effects. The downside is that retrieval practice has rather focal retention benefits.

## Contents

This repository includes the scripts used to munge, analyze, and visualize the data. It also includes html files that contain the code and output from all statistical analyses.  

`preprocessing.py` This script is run interactively. It combines the three experimental phases, engineers features, identifies bad data, and, finally, outputs clean dataframes for plotting in publication_ready_plots.py and computing inferential statistics in R


