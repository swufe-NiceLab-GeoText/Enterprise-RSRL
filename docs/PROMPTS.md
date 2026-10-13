# LLM Prompt Documentation

This document records the prompt templates used by the current source code for RSRL. The templates below are extracted from active Python assignments, preserving their wording and Python f-string placeholders. They are source templates, not archived requests or model responses from a particular experiment.

The main pipeline has three LLM stages: financial feature selection, enterprise risk scoring, and representative-based semantic edge generation. An additional risk-scoring variant supports the no-industry-rubric ablation.

- [Project overview](../README.md)
- [Paper draft](../DST2027_Enterprise.pdf), Section 4 and Appendix D

## 1. Call configuration

| Stage | Message roles | Source model identifier | Temperature | Maximum output tokens |
| --- | --- | --- | ---: | ---: |
| Batched feature selection | `user` | `deepseek-chat` | 0.3 | 1,000 |
| Final feature selection, if needed | `user` | `deepseek-chat` | 0.3 | 2,000 |
| Industry-rubric risk scoring | `system` + `user` | `deepseek-chat` by default | 0.7 | 500 |
| Semantic edge generation | `user` | `deepseek-chat` | 0.3 | 2,000 |
| Risk scoring without the industry rubric | `system` + `user` | `deepseek-chat` by default | 0.7 | 500 |

The client uses the OpenAI-compatible SDK with `base_url="https://api.deepseek.com"`. The risk-scoring constructors accept a different endpoint and model. These are source defaults, not a guarantee that every cached artifact was generated with the same model revision. The paper refers to DeepSeek-V3; the source does not pin a dated revision. Kimi/Qwen artifact suffixes alone do not establish the complete request configuration that produced them.

The `max_tokens` settings above limit generated responses; they are not limits on input prompt size. No API credentials are included in this document.

### Reading the templates

The fenced blocks contain Python assignments. Values inside `{...}` are filled at runtime; doubled braces `{{` and `}}` become literal JSON braces after f-string evaluation. The surrounding Python variable names and triple quotes are not sent to the model. All prompt templates are presented in English. The semantic edge-generation template has been translated without changing its two edge categories, counts, weight range or JSON fields. `_ENTERPRISE_ID` refers to the original dataset column name, represented with a Unicode escape in the source.

Examples containing `...`, angle-bracket values, or `//` comments describe an output shape rather than valid JSON. The downstream parsers expect the model's actual response to be valid JSON.

## 2. Financial feature selection

**Source:** [`llm_feature_selection`](../src/select_features.py#L7) and [`llm_final_selection`](../src/select_features.py#L127).

The first CSV column is treated as the enterprise identifier and excluded from feature selection. The script computes the following metadata for each remaining column:

| Field | Meaning |
| --- | --- |
| `name` | Input column name |
| `mean`, `std` | Mean and standard deviation for the numeric dtypes recognized by the source; otherwise `null` |
| `missing_rate` | Fraction of missing entries |
| `unique_rate` | Number of distinct non-null values divided by the number of rows |

Features are processed in batches of 20, with a nominal total target of 50. The LLM receives column metadata rather than enterprise rows or outcome labels. All columns after the first are candidates, so the caller is responsible for excluding outcomes or unavailable future information from the input feature table.

### 2.1 Batched selection

Runtime variables:

- `batch_columns`: names in the current batch.
- `batch_stats`: the batch's feature metadata serialized as JSON.
- `all_columns`: all candidate names.
- `target_features`: requested total count, normally 50.
- `batch_size`: 20.

The requested number per batch is computed by the expression in the template; it is not a guarantee that the final feature count is exactly 50.


```python
prompt = f"""
        You are a feature engineering expert. Given the statistical information of the following {len(batch_columns)} features, please select the most valuable features with low similarity.

        Feature Statistics:
        {json.dumps(batch_stats, ensure_ascii=False, indent=2)}
        
        Selection Criteria:
        1. Remove highly similar features (e.g., similar names, similar statistical properties)
        2. Retain features with high information content (high unique_rate, low missing rate)
        3. Consider feature diversity and avoid redundancy
        4. Remove features with highly similar data distributions
        5. Select {min(target_features // (len(all_columns) // batch_size + 1), len(batch_columns))} most important features per batch
        
        Please output strictly in the following JSON format with no other text:
        {{
          "selected_features": ["feature1", "feature2", ...],
          "reason": "brief explanation for selection"
        }}"""
```


The expected response contains `selected_features` and a short `reason`. Markdown fences are stripped before JSON parsing. If an API call or parsing fails, the source retains every feature from that batch.

### 2.2 Final selection

This request is made only if the accumulated selection exceeds the target. `features` contains the first-pass selection, `filtered_stats` contains the matching metadata, and `target_n` is the requested final count.


```python
prompt = f"""
    You are a feature engineering expert. From the following {len(features)} features, select the {target_n} most important features.

    Feature Statistics:
    {json.dumps(filtered_stats, ensure_ascii=False, indent=2)}
    
    Selection Criteria:
    1. Maximum information content (high unique_rate)
    2. Good data quality (low missing rate)
    3. Low similarity between features
    
    Please output strictly in the following JSON format with {target_n} feature names:
    {{
      "selected_features": ["feature1", "feature2", ...]
    }}"""
```


The expected response contains `selected_features`. On failure, the source falls back to the first `target_n` names. Exported CSV columns and the saved feature-name list should be checked separately: the current implementation constructs the CSV column list before removing one item from `selected_features`.

## 3. Risk scoring with an industry rubric

**Source:** [`IndustryStandardLLMEnhancer`](../src/llm_enhancement.py#L28), particularly [`design_industry_standard_prompt`](../src/llm_enhancement.py#L98).

For each enterprise, the script computes a percentile profile over the input dataset and combines a short relative-position description with selected absolute feature values. It then asks for five risk scores and a textual explanation.

### 3.1 Runtime inputs

| Variable | Construction in the source |
| --- | --- |
| `enterprise_id` | Value of the first column of the input feature table; used as a dataset-local identifier in the prompt |
| `feature_values` | Numeric candidate columns, using the dtype checks in `enhance_dataset` |
| `features_display` | First **five** entries of `feature_values`, rounded to four decimal places |
| `context_description` | At most six relative-position descriptions, with selected feature values, percentile positions, and dataset medians |

Percentile ranks are computed with `DataFrame.rank(pct=True)`. Features in the top or bottom 10% are selected first for the context description. If fewer than three extreme features are found, the code supplements them with features farthest from the median percentile. Statistics are computed over the entire supplied input table before downstream train/validation/test splitting.

The rubric below is an embedded experimental scoring rubric. The source does not attach an authoritative external citation or dynamically retrieve standards for these thresholds. The prompt mentions financial ratios such as ROE and current ratio, but the code does not derive those ratios when the supplied columns are accounting amounts; input semantics therefore matter.

### 3.2 System message


```python
system_message = """
You are a Senior Financial Risk Analyst using OBJECTIVE industry standards.

INDUSTRY-STANDARD RISK CRITERIA (use these objective thresholds):

🟢 LOW RISK (Scores 1-3) - Only ~10-15% of companies:
Financial Health:
  - ROE > 15%, ROA > 8%
  - Profit Margin > 10%
  - Revenue Growth > 15% annually
Liquidity:
  - Current Ratio > 2.0
  - Quick Ratio > 1.5
  - Cash/Assets > 0.20
Leverage:
  - Debt-to-Asset < 0.30
  - Debt-to-Equity < 0.50
  - Interest Coverage > 5x
Operational Efficiency:
  - Asset Turnover > 1.0
  - Inventory Turnover > 6x
  - Low operating costs
Market/Systemic:
  - Low beta (< 0.8)
  - Stable earnings
  - Diversified revenue streams

🟡 MEDIUM RISK (Scores 4-7) - ~70-80% of companies (MOST COMMON):
Financial Health:
  - ROE 5-15%, ROA 3-8%
  - Profit Margin 3-10%
  - Modest growth or stable
Liquidity:
  - Current Ratio 1.0-2.0
  - Quick Ratio 0.8-1.5
  - Cash/Assets 0.10-0.20
Leverage:
  - Debt-to-Asset 0.30-0.70
  - Debt-to-Equity 0.50-2.0
  - Interest Coverage 2-5x
Operational Efficiency:
  - Asset Turnover 0.5-1.0
  - Average efficiency
  - Moderate costs
Market/Systemic:
  - Beta 0.8-1.5
  - Some volatility
  - Normal market exposure

🔴 HIGH RISK (Scores 8-10) - Only ~10-15% of companies:
Financial Health:
  - Negative ROE/ROA
  - Operating losses
  - Declining revenue
Liquidity:
  - Current Ratio < 1.0 (insolvency risk!)
  - Quick Ratio < 0.5
  - Cash/Assets < 0.10 (cash crisis!)
Leverage:
  - Debt-to-Asset > 0.70 (overleveraged!)
  - Debt-to-Equity > 2.0
  - Interest Coverage < 2x (default risk!)
Operational Efficiency:
  - Asset Turnover < 0.5
  - Poor management
  - High cost structure
Market/Systemic:
  - Beta > 1.5 (high volatility)
  - Distress signals
  - Industry downturn

SCORING PROCESS:
Step 1: Count how many metrics fall in each category (Low/Medium/High risk zone)
Step 2: Apply decision rules:
  - If 3+ metrics in LOW risk zone → Score 1-3
  - If 3+ metrics in HIGH risk zone → Score 8-10
  - Otherwise → Score 4-7 based on balance
Step 3: Fine-tune within range based on severity

MANDATORY DISTRIBUTION (remember across ALL companies):
- 10-15% receive 1-3 (exceptional performers)
- 70-80% receive 4-7 (normal companies)
- 10-15% receive 8-10 (distressed companies)

CRITICAL RULES:
1. DO NOT give extreme scores unless metrics CLEARLY meet thresholds
2. When in doubt, default to 4-7 range (most companies are average)
3. Use objective thresholds, not subjective judgment
4. Compare metrics to industry standards, not just to dataset percentiles

OUTPUT: JSON only, no explanations.
        """
```


### 3.3 User message


```python
user_message = f"""
Assess Enterprise {enterprise_id}'s financial risk using the objective industry standards provided.

## RELATIVE POSITION IN DATASET
{context_description}

## KEY FINANCIAL METRICS
{json.dumps(features_display, indent=2)}

## ASSESSMENT INSTRUCTIONS

Step 1: For EACH risk dimension, compare the enterprise's metrics to the objective thresholds:

**Financial Health Risk**:
- Check: ROE, ROA, Profit Margin, Revenue Growth
- Apply thresholds: >15% ROE = Low Risk, <0% = High Risk

**Liquidity Risk**:
- Check: Current Ratio, Quick Ratio, Cash/Assets
- Apply thresholds: CR>2.0 = Low Risk, CR<1.0 = High Risk

**Leverage Risk**:
- Check: Debt-to-Asset, Debt-to-Equity, Interest Coverage
- Apply thresholds: D/A<0.3 = Low Risk, D/A>0.7 = High Risk

**Operational Efficiency Risk**:
- Check: Asset Turnover, Inventory Turnover, Cost Ratios
- Apply thresholds: AT>1.0 = Low Risk, AT<0.5 = High Risk

**Market/Systemic Risk**:
- Check: Beta, Volatility, Market Exposure
- Apply thresholds: Beta<0.8 = Low Risk, Beta>1.5 = High Risk

Step 2: Determine risk type based on highest scoring dimension

Step 3: Provide brief reason citing specific metrics vs thresholds

## CRITICAL REMINDERS
- Most companies score 4-7 (this is NORMAL)
- Only give 1-3 if metrics clearly exceed low-risk thresholds
- Only give 8-10 if metrics clearly fall below high-risk thresholds
- Use OBJECTIVE thresholds, not percentile comparisons alone

## OUTPUT FORMAT (JSON only)
{{
  "financial_health_risk": <integer 1-10>,
  "liquidity_risk": <integer 1-10>,
  "leverage_risk": <integer 1-10>,
  "operational_efficiency_risk": <integer 1-10>,
  "market_systemic_risk": <integer 1-10>,
  "risk_type": "<string>",
  "risk_reason": "<brief explanation citing specific metrics vs thresholds>"
}}
        """
```


### 3.4 Expected output and fallback

| Field | Expected value |
| --- | --- |
| `financial_health_risk` | Integer 1-10 |
| `liquidity_risk` | Integer 1-10 |
| `leverage_risk` | Integer 1-10 |
| `operational_efficiency_risk` | Integer 1-10 |
| `market_systemic_risk` | Integer 1-10 |
| `risk_type` | String identifying a risk category |
| `risk_reason` | Short explanation |

Higher scores indicate greater risk. `enterprise_id` is attached locally after parsing. The five numeric scores enter the classifier; `risk_type` and `risk_reason` are excluded from numeric model inputs.

The parser strips Markdown fences, extracts a JSON object, fills missing numeric risk fields with 5, and clips parsed scores to 1-10. Missing `risk_type` becomes `Unknown`. A parsing failure produces scores of 5, `risk_type="Parse Error"`, and a parsing-failure reason. An API failure produces scores of 5, `risk_type="API Error"`, and an API-failure reason. Consequently, downstream score tables can contain fallback values that are not model assessments. `risk_reason` is requested, but is not among the parser's required fields.

## 4. Representative-based semantic edge generation

**Source:** [`clustering_with_llm_edges`](../src/clustering_with_new_edges.py#L32), [`generate_edges_stratified_with_llm`](../src/clustering_with_new_edges.py#L173), and [`design_stratified_prompt`](../src/clustering_with_new_edges.py#L333).

The source standardizes numeric fused features, applies K-means, and chooses the enterprise nearest each cluster centroid. The default entry point requests 100 representatives. For each representative, the LLM is asked to propose five risk-pattern edges and fifteen financial-similarity edges. This is an instruction in the prompt; the response-processing code does not enforce those exact counts.

### 4.1 Runtime inputs

| Variable | Meaning |
| --- | --- |
| `node` | The current representative enterprise |
| `llm_features` | Numeric feature names containing `risk` |
| `base_features` | Remaining numeric features after the source exclusions |
| `llm_desc` | All selected risk features, formatted to two decimal places |
| `base_desc_top10` | First ten base features, formatted to three decimal places |
| `candidates` | Other sampled representatives, excluding the current enterprise |
| `candidate_ids` | IDs of the first twenty candidates |

In this implementation, the candidate set is restricted to representatives. The prompt lists only the first twenty candidate IDs and does **not** include candidate feature vectors, although the Python candidate records contain them. The LLM therefore does not receive full pairwise financial evidence for the requested comparisons. The code also does not consistently exclude the auxiliary numeric `enterprise_id` column from clustering and base-feature discovery; the table above describes the actual selection rules.

### 4.2 User message

This call has a single `user` message. The expert-role sentence is part of that message, rather than a separate system message.


```python
prompt = f"""
        You are an expert in enterprise risk relationship analysis. Analyze enterprises similar to the target enterprise from two perspectives.
        
        ## Target enterprise
        ID: {node[_ENTERPRISE_ID]}
        
        ## Perspective 1: LLM risk assessment (5 dimensions)
        {llm_desc}
        
        ## Perspective 2: Original financial indicators (First 10 shown; total: {len(base_features)} items)
        {base_desc_top10}
        ... (another {len(base_features) - 10} financial indicators)
        
        ## Candidate enterprises
        Total: {len(candidates)} items
        Example ID: {', '.join(candidate_ids[:20])}
        ... (another {len(candidates) - 20} items)
        
        ## Task: Generate 20 edges, split into two categories
        
        **Category A (5 edges): Similar risk patterns**
        - Focus: matching LLM risk scores
        - Example: Both have high leverage_risk + high liquidity_risk
        
        **Category B (15 edges): Similar financial indicators**
        - Focus: Match original financial data
        - Example: Similar debt ratios and cash flows
        
        ## Important rules
        1. Category A must contain exactly 5 edges
        2. Category B must contain exactly 15 edges
        3. Avoid duplicates (If an enterprise matches both categories, place it in category B)
        4. Weight range: 0.6-1.0
        
        ## Output format (strict JSON)
        {{
          "risk_pattern_edges": [
            {{"target_id": "enterprise_id", "weight": 0.85, "reason": "similar high leverage + liquidity"}},
            ... // exactly 5 entries
          ],
          "financial_similarity_edges": [
            {{"target_id": "enterprise_id", "weight": 0.82, "reason": "similar debt structure"}},
            ... // exactly 15 entries
          ]
        }}
    """
```


### 4.3 Expected output and processing

The response is a JSON object with two arrays:

- `risk_pattern_edges`: entries with `target_id`, `weight`, and optional `reason`.
- `financial_similarity_edges`: entries with the same fields.

The source uses an unordered endpoint pair to merge repeated proposals. Given proposal weights `w_1, ..., w_v`, the fused confidence is:

```text
w_fused = 1 - product(1 - w_v)
```

The resulting `sampled_edges.csv` contains `??A`, `??B`, `??`, and `??`. Reasons from repeated proposals are joined with ` | `.

Requests that fail are skipped. Exact edge counts, valid candidate membership, self-loop exclusion, and weight bounds are not enforced by the response parser. The later graph-fusion stage drops endpoints absent from the full node mapping. Current fused graph artifacts retain relation type and endpoints, without forwarding semantic confidence weights into HGT attention.

## 5. Ablation: risk scoring without the industry rubric

**Source:** [`NoStandardLLMEnhancer.design_no_standard_prompt`](../analyze/Label_Test/code/data_process_no_rule/llm_enhancement_no_rule.py#L596).

This variant removes the explicit numerical industry thresholds and score-frequency instructions. It retains the five risk dimensions, dataset-local enterprise ID, first five displayed financial values, relative-position description, and 1-10 score direction. Its API defaults and output parser follow the rubric-based variant.

### 5.1 System message


```python
system_message = """
        You are a financial risk analyst.

        Score enterprise financial risk from 1 to 10 across five dimensions.

        GENERAL GUIDELINES:
        - Lower scores indicate lower risk (better financial health)
        - Higher scores indicate higher risk (financial distress)
        - Consider standard financial analysis principles
        - Evaluate each dimension independently

        OUTPUT: JSON only.
        """
```


### 5.2 User message


```python
user_message = f"""
        Assess Enterprise {enterprise_id}'s financial risk.

        ## FINANCIAL METRICS
        {json.dumps(features_display, indent=2)}

        ## RELATIVE POSITION
        {context_description}

        Evaluate based on:
        - Financial Health: profitability, growth, efficiency
        - Liquidity: short-term solvency, cash management
        - Leverage: debt levels, financial structure
        - Operational Efficiency: asset utilization, cost control
        - Market/Systemic: volatility, market exposure

        Score each dimension (1-10) and provide risk type.
        """
```


Unlike the rubric-based prompt, this request does not spell out JSON field names in its user message. Its parser still expects the same five numeric risk fields and `risk_type`; missing or malformed fields can therefore trigger defaults.

The active semantic-edge prompt in `clustering_no_rule.py` has the same wording as Section 4, differing only in whitespace. That script is a downstream ablation pipeline, not an additional substantive prompt design. Commented-out legacy code in the ablation files is not presented as an active template here.

## 6. Paper-to-source correspondence

The templates above document current behavior. The following distinctions matter when describing the method or reconstructing an experiment:

| Topic | Paper description | Current source / appendix detail |
| --- | --- | --- |
| Risk score scale | Section 4.1 uses `[0, 1]` | Appendix D and the source request integers 1-10; the active main classifier does not first map them to `[0, 1]` |
| Enterprise identity | Main text excludes enterprise identities | The prompt includes an internal dataset ID, not a company name/ticker by design; the actual input table determines the identifier's content |
| Distribution information | Main text excludes class proportions | The system prompt and Appendix D prescribe low/medium/high **risk-score** frequencies; these are not binary outcome label proportions |
| Input to risk scoring | Main text emphasizes percentile profiles | Source sends absolute values, selected percentile descriptions, and medians |
| Financial evidence | Five financial risk dimensions | The source displays only the first five numeric values in the risk-scoring message; it does not guarantee that all rubric ratios are supplied |
| Augmentation candidate scope | Section 4.3 allows neighbors from the node set | Source supplies only representative candidates, with twenty IDs and no candidate feature vectors shown in the prompt |
| Leakage evaluation | Appendix E discusses label permutation | The active label-test script shuffles training and validation labels in its permutation conditions; this diagnostic alone does not establish the absence of all leakage paths |

## 7. Recording a reproducible prompt run

For a future experiment, record the exact model identifier and revision if available, endpoint, message roles, rendered messages, generation parameters, source revision, input feature schema, dataset snapshot, and parser/fallback outcomes. Preserve the representative selection and candidate ordering for edge generation. This repository snapshot stores final CSV artifacts but does not provide a complete request/response archive for every experiment variant.
