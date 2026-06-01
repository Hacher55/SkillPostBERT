# Raw data

Download the Kaggle datasets and drop the CSVs here. They are gitignored.

- **LinkedIn Job Postings** — https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
- **Indeed Job Postings** — search Kaggle for "Indeed job scrape"

Expected files (rename if needed; `preprocess.py` reads from here):

```
data/raw/linkedin_postings.csv
data/raw/indeed_postings.csv
```

Minimum columns the pipeline uses: a job title, a job description (free text),
and ideally a discipline/category field. The preprocessing step infers
discipline (ME / EE / SE) from title + description keywords when no explicit
field is present.
