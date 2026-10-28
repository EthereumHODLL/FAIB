# Retail Credit Policy: Sample Retail Bank (Version 1.0)

This document is the single source of truth for lending rules. The Python engine reads the
parameter block below fresh from disk on every run, and the RAG engine quotes the clauses
verbatim. Editing a value here changes routing and explanations with no code change.

## Policy Parameters

- PARAM min_cibil = 550
- PARAM prime_cibil = 650
- PARAM subprime_max_tenure = 4
- PARAM max_lti = 4.0
- PARAM min_coverage = 1.0
- PARAM prime_shortfall_floor = 0.75
- PARAM min_tenure = 2
- PARAM max_tenure = 20
- PARAM max_loan_amount = 50000000
- PARAM auto_approve_prob = 0.85
- PARAM auto_reject_prob = 0.20
- PARAM min_rationale_chars = 20

## Section 1: Credit Score Requirements

### [CP-1.1] Minimum Credit Score
An applicant must hold a CIBIL score of at least 550. An application with a CIBIL score below 550 does not meet the minimum credit score requirement, unless the Short-Tenure Exception in CP-1.2 applies.

### [CP-1.2] Subprime Short-Tenure Exception
An applicant with a CIBIL score below 550 may be considered only if the requested loan tenure is 4 years or less AND the total asset coverage ratio is at least 1.0x. Such files are never auto-approved and require a mandatory Senior Credit Underwriter decision.

## Section 2: Affordability and Collateral

### [CP-2.1] Loan-to-Income Limit
The requested loan amount must not exceed 4.0 times the applicant's annual income (Loan-to-Income ratio, LTI, of 4.0x or lower).

### [CP-2.2] Minimum Asset Coverage
Total Asset Coverage Ratio is the sum of residential, commercial, luxury and bank liquid asset values divided by the requested loan amount. The ratio must be at least 1.0x.

### [CP-2.3] Prime Collateral Shortfall Review
An applicant with a CIBIL score of 650 or higher whose asset coverage ratio is below 1.0x but at least 0.75x must not be automatically rejected. The file is referred to a Senior Credit Underwriter, who may approve with conditions such as additional collateral or a guarantor, or may reject.

## Section 3: Loan Parameters

### [CP-3.1] Tenure Limits
Loan tenure must be between 2 and 20 years inclusive.

### [CP-3.2] Maximum Loan Amount
A single loan application must not exceed INR 5,00,00,000 (INR 50,000,000).

## Section 4: Data Integrity

### [CP-4.1] Negative or Anomalous Asset Valuations
A negative asset valuation must never be used in a coverage calculation. Such a value is treated as zero for screening, the collateral is flagged as possibly encumbered, and the file is referred to a Senior Credit Underwriter for manual verification before any decision is issued.

## Section 5: Decision Authority

### [CP-5.1] Auto-Approve Authority
An application may be auto-approved only if the model approval probability is above 85% AND no policy breach, exception or data anomaly exists.

### [CP-5.2] Auto-Reject Authority
An application may be auto-rejected only if the model approval probability is below 20% AND at least one clear policy threshold breach exists.

### [CP-5.3] Mandatory Human Review
Every application that is not auto-approved or auto-rejected under CP-5.1 or CP-5.2, including any application with a policy exception, data anomaly, or conflict between the model signal and policy, must be decided by a Senior Credit Underwriter before any financial commitment or adverse action notice is issued.

## Section 6: Communication and Governance

### [CP-6.1] Adverse Action Notice Requirements
A rejection notice must state the principal reasons for the decision and cite the policy clause for each. It must rely only on verified policy breaches. It must not disclose internal model scores, probabilities or algorithm details, and must not speculate about reasons that are not documented.

### [CP-6.2] Standard Conditions of Approval
When a Senior Credit Underwriter approves with conditions, the conditions must be selected from the standard list below.
- CONDITION: Additional collateral to be pledged so that total asset coverage reaches at least 1.0x of the loan amount.
- CONDITION: A creditworthy co-borrower or guarantor to be added to the loan.
- CONDITION: Independent valuation and clear-title verification of all pledged assets before disbursal.
- CONDITION: Loan tenure to be restricted to 4 years or less.
- CONDITION: Reduced sanction amount so that the Loan-to-Income ratio does not exceed 4.0x.

### [CP-6.3] Underwriter Override Documentation
Every human decision must record the underwriter's identifier and a written rationale of at least 20 characters. The decision is binding and is written to the audit log.
