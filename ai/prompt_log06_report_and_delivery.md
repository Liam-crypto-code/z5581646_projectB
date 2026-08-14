# Prompt log - Report preparation and delivery checks

## What I wanted

I wanted a Word report grounded in saved output artifacts, with the compulsory
six-section structure, required exhibits and clear limits. I also wanted to prepare
the project for a separate public GitHub repository and Streamlit deployment.

## Prompt(s)

> Please prepare a report evidence pack for Project B without drafting the final
> report prose yet ... Do not invent citations, write final economic interpretation
> in my voice, modify results, or start GitHub actions.

> Please prepare a structured first draft of my FinMosaic Project B report in
> English ... Create report/report.docx as the editable Word source only. Do not
> create or export report/report.pdf yet ... use [STUDENT TO WRITE] placeholders
> for motivation, interpretation, reflection and recommendations.

> Please prepare and deploy my completed FINS5545 Project B ... This folder must
> become its own independent Git repository. It must not push to the parent course
> repository ... Do not push anything until I explicitly provide and approve my new
> GitHub repository URL.

## What the assistant produced

It created an evidence pack and then a report draft with required sections,
figures, tables, captions and traceable results. It also prepared deployment
instructions, checked `.gitignore`, and ran the course hand-in checker. A later
review added verified references for Markowitz, Sharpe, VADER and
Loughran-McDonald.

## What was wrong or risky

An AI-written report cannot be treated as final personal analysis. The report
initially contained student-writing and citation placeholders. The local project
was nested inside the course repository, so pushing to the parent remote would
have been incorrect. The hand-in checker also warned that Python cache files must
be removed before zipping.

## What I changed and why

I kept the report as a Word source until I have personally reviewed its reasoning,
citations, links and wording, and I retained only verifiable references. I created
a separate public repository named `z5581646_projectB` rather than using the course
repository. Before final submission I will export `report.pdf`, add my public
GitHub and Streamlit URLs, remove `__pycache__`/`.pyc` files, run the hand-in
checker again and zip the project root.
