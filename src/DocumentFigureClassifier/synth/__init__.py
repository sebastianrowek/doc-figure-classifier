"""
Synthetic training-data generation for the tier-1 figure classifier.

Design notes: docs/synthetic_data_design.md
Labels and rules: docs/labeling_guide_en.md

Synthetic data is training-only. Validation and test sets must be real,
hand-labeled crops -- ``writer.Writer`` enforces this rather than trusting it
to be remembered.
"""
