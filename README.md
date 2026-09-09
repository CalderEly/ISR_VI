# ISR_IV
Research Plan Versions: https://docs.google.com/document/d/1cpaLidQlgMXAjLGeAQVMJMn7QAF9FM-7QrrRme_I-cA/edit?usp=sharing 

Institustions for networking:
 1. USC
 2. MSOE

## TODO
### ISSUES

```diff
+ FIXED
- ISSUES PREVENTING RUN
! ISSUES THAT IMPEDE OUTPUT
# OPTIMIZATION TODOS
TODO
```

### Switch to Split-N-MINST
- Keep 10 output layer/ single head
- Full 10 way softmax cross-entropy eval on every task
- Full 10 way argmax at eval
-  #### Metrics
   - 2 Parallel Matricies
   - Full matrix: standard 10 way argmax accuracy. (Recency bias and representational forgetting)
   - Restricted Matrix: argmax restricted to j only. This isolates representational forgetting
   - ACC and BWT compute on both matricies
   - No forward transfer
- 



## DOCS

## Research

Items to research:
 1. Heterogeneuos neurons
 2. Structural plasticity
 3. Methods for measuring success

## PRESENT
