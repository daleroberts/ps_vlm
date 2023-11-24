This is a prototype of SAR workflow code called *aludra* that aims to replace the overly-complex [ga_sar_workflow](https://github.com/geoscienceAustralia/ga_sar_workflow) with something much simpler and more flexible. The main GAMMA workflow is contained in the file `run_gamma_workflow` and clocks in at 21 lines of code.

> [!important]  
> Since this is a prototype, this code is not to be shared outside GA's Geodescy group until it has been rewritten to be a more complete solution in the next stage of the project.

## Overview

The projects [ga_sar_workflow](https://github.com/geoscienceAustralia/ga_sar_workflow) + [PyRate](https://github.com/GeoscienceAustralia/PyRate) combine together to perform a *distributed scatterer* approach to InSAR. In this project, we are trying a completely different methodology called the *persistent scatterer* approach.

The aim of the first stage of this project was to: (1) prototype a PS approach using GAMMA and StaMPS to see if this is actually what we want to do (e.g., do the outputs look ok? and can we validate them?), (2) determine if we can write a more flexible and simpler GAMMA workflow that allows us to consider any area of interest on-the-fly, automatically pull-in new observations as they become available, and can scale-up to generate a continental-scale output. We note that ga_sar_workflow is unable to do any of those things.

The second stage of the project is to: (1) Remove the Matlab / StaMPS dependency and implement our own PS approach in Python, (2) Combine the GAMMA driver our PS approach into a single workflow to obtain numerous efficiences.



