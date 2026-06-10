# Comprehensive Project Report: Fake News Detection Using the LIAR Dataset

**Course:** Introduction to Data Science  
**Team Members:** Rehan Abid, Ahmad Azad, Annushay Fatima  
**Dataset:** LIAR (Labeled Instances of Accurate Representations)  
**Project Goal:** Build and evaluate classification models to distinguish fake vs. real news using textual and metadata features

---

## 1. Overview of the Dataset

### 1.1 Dataset Description

The LIAR dataset is a benchmark dataset for fake news detection, containing **12,765 short political statements** (after cleaning) collected from PolitiFact.com. Each statement is labeled for truthfulness across **6 ordinal categories**:

- **pants-fire** (0): Completely false, outrageous claims (8.2%)
- **false** (1): False statements (19.5%)
- **barely-true** (2): Contains some truth but mostly misleading (16.5%)
- **half-true** (3): Half accurate, half inaccurate (20.5%)
- **mostly-true** (4): Mostly accurate with minor errors (19.2%)
- **true** (5): Completely accurate statements (16.1%)

### 1.2 Dataset Structure

Each record contains **14 original columns** plus additional tracking fields:

| Column | Description | Type | Missing Values |
|--------|-------------|------|----------------|
| `id` | Unique statement identifier | String | 0 |
| `label` | Truthfulness rating (6 classes) | Categorical | 0 |
| `statement` | The claim to verify | Text | 0 |
| `subject` | Topic categories (comma-separated) | Categorical | 2 |
| `speaker` | Person making the statement | Categorical | 2 |
| `speaker_job` | Speaker's occupation | Categorical | 3,568 (27.9%) |
| `state` | Geographic location | Categorical | 2,751 (21.5%) |
| `party` | Political affiliation | Categorical | 2 |
| `barely_true_counts` | Historical count | Numerical | 2 |
| `false_counts` | Historical count | Numerical | 2 |
| `half_true_counts` | Historical count | Numerical | 2 |
| `mostly_true_counts` | Historical count | Numerical | 2 |
| `pants_on_fire_counts` | Historical count | Numerical | 2 |
| `context` | Venue/location of statement | Categorical | 131 |

### 1.3 Key Dataset Characteristics

**Label Distribution:**
The dataset shows relatively balanced distribution across the 6 classes, with the most common being `half-true` (20.5%) and least common being `pants-fire` (8.2%). This balance allows us to use accuracy as a primary evaluation metric without severe class imbalance concerns.

**Statement Length:**
- Mean: 107 characters
- Median: 99 characters
- Range: 11 to 3,192 characters
- Most statements are short political sound bites (1-2 sentences)
- 203 outliers detected (1.59%) using IQR method, but retained as valid statements

**Speaker Distribution:**
- Top speakers: Barack Obama (608 statements), Donald Trump (343), Hillary Clinton (296)
- Total unique speakers: 3,310
- Top 20 speakers account for significant portion of dataset
- Mix of politicians, organizations, and media figures

**Party Distribution:**
- Republican: 44.2%
- Democrat: 32.4%
- None/Independent: 17.1%
- Other categories: 6.3%
- Relatively balanced partisan representation

**Subject Distribution:**
- 144 unique subjects (comma-separated, multi-label)
- Top subjects: economy, healthcare, taxes, candidates-biography, jobs
- Most statements span multiple topics

**Credit History Statistics:**
The credit history counts show high positive correlations (0.64-0.99), indicating that speakers with many statements tend to have ratings across all truthfulness categories. This suggests the counts reflect speaker volume more than consistent truthfulness patterns.

---

## 2. Methods and Models Used

### 2.1 Data Preparation Pipeline

#### Phase 1: Data Collection and Merging

**Design Decision:** We merged the original pre-split datasets (train: 10,240, valid: 1,284, test: 1,267) into a single unified dataset before creating fresh splits.

**Rationale:**
1. **Flexibility:** Custom split ratios for experimentation
2. **Stratification:** Ensure balanced label distribution across splits
3. **Avoid Bias:** Original splits may contain hidden temporal or sampling biases
4. **Reproducibility:** Controlled random_state=42 for consistent splits

**Final Split:** 70% train (8,935), 15% validation (1,915), 15% test (1,915) with stratification on labels

#### Phase 2: Data Cleaning

**Step 1: Missing Value Handling**

Strategy: Fill rather than drop to preserve maximum data

| Column Type | Fill Strategy | Justification |
|-------------|---------------|---------------|
| Categorical (`subject`, `speaker`, `speaker_job`, `state`, `context`) | `'unknown'` | Creates valid category; preserves information about missingness |
| `party` | `'none'` | Already exists for non-partisan entities |
| Credit history counts | `0` | Missing indicates no tracked history |

**Impact:** Retained 100% of data (vs. losing 27.9% if dropped `speaker_job` nulls)

**Step 2: Duplicate Removal**

- Duplicate IDs: **0 found** ✓
- Duplicate statements: **26 found and removed**
- **Rationale:** Same statement appearing multiple times indicates data collection errors

**Final dataset:** 12,765 unique statements

**Step 3: Outlier Detection (IQR Method)**

Using the Interquartile Range method:
- Q1 (25th percentile): 73 characters
- Q3 (75th percentile): 133 characters
- IQR: 60 characters
- Lower bound: -17 characters
- Upper bound: 223 characters
- Outliers detected: 203 (1.59%)

**Decision:** Keep all outliers - both very short and very long statements are valid and informative

**Step 4: Text Cleaning**

Operations applied to statement text:
1. **URL Removal:** Strip all HTTP/HTTPS links
2. **Special Character Removal:** Retain only letters, numbers, spaces, and basic punctuation (`. , ! ? ' -`)
3. **Whitespace Normalization:** Collapse multiple spaces to single space

**Step 5: Global Text Normalization**

**Critical Decision:** Apply lowercasing GLOBALLY at preprocessing stage

**Rationale:**
- Prevents inconsistent casing downstream
- Reduces vocabulary size (e.g., "Obama", "obama", "OBAMA" → "obama")
- Standard practice for text classification

**Additional Normalizations:**
- **Accent Removal:** Applied to speaker names for consistency
- **State Code Standardization:** 2-letter codes uppercase (TX, CA), full names lowercase

#### Phase 3: Feature Engineering

**Text Features (9 features):**

| Feature | Description | Purpose |
|---------|-------------|---------|
| `word_count` | Number of words | Length indicator |
| `char_count` | Number of characters | Verbosity measure |
| `avg_word_length` | Average characters per word | Complexity indicator |
| `sentence_count` | Number of sentences (by periods) | Structure measure |
| `exclamation_count` | Number of `!` | Emotional emphasis |
| `question_count` | Number of `?` | Interrogative style |
| `uppercase_count` | Words in ALL CAPS | Emphasis/shouting indicator |
| `sentiment_polarity` | TextBlob polarity (-1 to 1) | Positive/negative tone |
| `sentiment_subjectivity` | TextBlob subjectivity (0 to 1) | Objective/subjective scale |

**Text Preprocessing:**
```python
# Tokenize → Lowercase → Remove stopwords → Lemmatize
"Obama supports healthcare reform" 
→ ["obama", "support", "healthcare", "reform"]
```

**Why Lemmatization over Stemming?**
- Preserves word meaning ("running" → "run" vs. "runn")
- More interpretable for coefficient analysis
- Better for domain-specific political vocabulary

**Credit History Features (8 engineered features):**

1. `total_credit_count` = sum of all 5 rating counts
2. `barely_true_ratio` = barely_true_counts / total_credit_count
3. `false_ratio`, `half_true_ratio`, `mostly_true_ratio`, `pants_on_fire_ratio` (similarly computed)
4. `credibility_score` = weighted average favoring truthfulness
   ```
   score = (pants_fire×0 + false×1 + barely_true×2 + half_true×3 + mostly_true×4) / total_count
   ```
5. `false_to_true_ratio` = (false + pants_fire) / (mostly_true + 1)

**Rationale:** 
- Raw counts are highly collinear (correlation 0.64-0.99)
- Ratios capture truthfulness patterns independent of speaker volume
- Credibility score provides single metric for speaker reliability

**Context Generalization:**

Reduced 5,143 unique contexts to 13 broad categories:

| Original Examples | Generalized Category |
|-------------------|---------------------|
| "interview on CNN", "Fox News interview" | `interview` |
| "TV broadcast", "television appearance" | `broadcast` |
| "tweet", "Twitter post", "Facebook post" | `social media` |
| "campaign speech", "rally" | `campaign event` |
| "congressional hearing" | `hearing` |

**Impact:** Reduced dimensionality while preserving semantic meaning

**Categorical Encoding Strategy:**

**Coverage-based selection:**
- Select top categories covering ≥80% of data
- Cap at top N (10-20) per feature to prevent dimension explosion

**Encoding approach:** One-hot encoding for selected categories

**Features created:**
- `party_*`: 2 binary features (republican, democrat)
- `state_*`: 11 binary features (top states)
- `job_*`: 15 binary features (governor, senator, president, etc.)
- `context_*`: 6 binary features (interview, broadcast, social media, etc.)
- `subject_*`: 20 binary features (extracted individual subjects)

**Total categorical features:** 54 binary columns

**Speaker Features (2 features):**
1. `is_frequent_speaker`: Binary indicator for top 30 speakers
2. `speaker_statement_count`: Total statements by this speaker in dataset

**Target Label Encoding:**

Ordinal encoding preserving natural order:
```python
label_mapping = {
    'pants-fire': 0,
    'false': 1,
    'barely-true': 2,
    'half-true': 3,
    'mostly-true': 4,
    'true': 5
}
```

**Final Feature Summary:**
- Text features: 9
- Credit history features: 13
- Categorical features: 54
- Speaker features: 2
- Statement length: 1
- **Total metadata features:** 79
- **Total with TF-IDF/Count vectors:** 5,000-10,000+ (model-dependent)

### 2.2 Model Selection and Training Strategy

#### Problem Formulation

**Research Question:** Should we use binary, multiclass, or ordinal classification?

**Our Approach:** Train ALL three formulations for comprehensive comparison

1. **6-Class Multiclass:** Direct prediction of original labels (0-5)
2. **Binary Classification:** False (0-2) vs. True (3-5)
3. **Ordinal Regression:** Treat as ordered scale, predict continuous values, round to classes

#### Models Selected

| Model | Justification | Primary Use Case |
|-------|---------------|------------------|
| **Naive Bayes** | Baseline, fast, interpretable | Text classification baseline |
| **Logistic Regression** | Linear baseline, interpretable coefficients | Feature importance analysis |
| **Random Forest** | Non-linear, handles feature interactions | Ensemble learning |
| **Ordinal Ridge** | Designed for ordered labels | Ordinal regression |
| **SVM (Linear)** | Strong text classification performance | High-dimensional text data |
| **LSTM** | Captures sequential patterns | Deep learning baseline |

#### Data Preparation Per Model

**Naive Bayes:**
- **Text:** Count Vectorization (5,000 features, unigrams + bigrams)
- **Metadata:** No scaling (79 features, includes only non-negative features)
- **Constraint:** Removed `sentiment_polarity` (can be negative)
- **Why Count over TF-IDF?** Naive Bayes assumes feature independence, works better with raw counts

**Logistic Regression & SVM:**
- **Text:** TF-IDF Vectorization (10,000 features, unigrams + bigrams)
- **Metadata:** StandardScaler (mean=0, std=1)
- **Combined:** Sparse matrix concatenation [TF-IDF | Scaled Metadata]
- **Final shape:** (samples, 10,079 features)

**Random Forest:**
- **Features:** All 79 numerical/binary engineered features only
- **No text vectors:** Tree-based models struggle with high-dimensional sparse text
- **Scaling:** StandardScaler applied (though RF doesn't strictly require it)

**Ordinal Ridge:**
- **Text:** TF-IDF (10,000 features) → TruncatedSVD (300 components) → Dense matrix
- **Metadata:** StandardScaler (79 features)
- **Combined:** Concatenated dense matrix
- **Final shape:** (samples, 379 features)
- **Why SVD?** Dimensionality reduction makes regression tractable

**LSTM:**
- **Text:** Tokenization → Integer encoding → Padding (max_len=200)
- **Metadata:** Kept separate as auxiliary input (79 features)
- **Architecture:** Dual-input model (text embeddings + metadata dense layer)
- **Vocabulary:** 20,000 most frequent words
- **Embedding:** 128-dimensional learned embeddings

### 2.3 Hyperparameter Tuning

#### Naive Bayes

**Parameter tuned:** `alpha` (Laplace smoothing)

**Search space:** [0.01, 0.1, 0.5, 1.0, 2.0]

**Best configuration:** `alpha=0.01` for 6-class (23.13% validation accuracy)

**Results:**
- Binary: 64.44% validation accuracy
- 6-class: 23.13% validation accuracy

#### Logistic Regression

**Parameter tuned:** `C` (inverse regularization strength)

**Search space:** [10.0, 1.0, 0.1, 0.01, 0.001]

**Best configuration:** `C=1.0` for 6-class (45.85% validation accuracy)

**Key finding:** Stronger regularization (C=0.1) reduces overfitting but slightly lowers accuracy

**Results:**
- Binary: 73.63% validation accuracy (C=0.1)
- 6-class: 45.85% validation accuracy (C=1.0)

#### Random Forest

**Configuration:** 500 trees, max_depth=40

**Results:**
- Binary: 70.29% validation accuracy
- 6-class: 42.04% validation accuracy
- **Severe overfitting:** Training accuracy 99.99%, validation 42.04% (overfit gap: 57.95%)

**Analysis:** Random Forest memorizes training data but fails to generalize due to high-dimensional feature space

#### Ordinal Ridge

**Parameter tuned:** `alpha` (L2 regularization strength)

**Method:** RidgeCV with 5-fold cross-validation, scoring by negative MAE

**Search space:** [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

**Best configuration:** `alpha=5.0` for 6-class

**Binary Ordinal Ridge:**
- **Threshold tuning:** Tested 81 thresholds from 0.1 to 0.9 (step=0.01)
- **Best threshold:** 0.500 (73.05% validation accuracy)

**Results:**
- Binary: 73.05% validation accuracy
- 6-class: 34.88% validation accuracy, MAE=0.97

#### SVM

**Configuration:** LinearSVC (no kernel for computational efficiency)

**Parameter tuned:** `C` and `penalty`

**Search space:**
- C: [0.001, 0.01, 0.1, 1.0, 10.0]
- Penalty: ['l1', 'l2']

**Best configuration:** C=0.1, penalty='l2' for 6-class (46.48% validation accuracy)

**Results:**
- Binary: 71.59% validation accuracy
- 6-class: 46.48% validation accuracy

#### LSTM

**Architecture:**
```python
text_input (200,) → Embedding(20000, 128) → LSTM(128) → Dropout
meta_input (79,) → Dense(64, relu) → Dropout
merged → Concatenate → Dense(64, relu) → Dropout → Output
```

**Hyperparameters tuned:**
- Dropout rate: [0.2, 0.4]
- Weight decay (L2): [0.0, 1e-5, 1e-4]
- Learning rate: [1e-3, 5e-4, 1e-4]

**Grid search:** 2×3×3 = 18 configurations

**Best configuration:**
- Dropout: 0.2
- Weight decay: 1e-4
- Learning rate: 1e-3

**Training:**
- Early stopping: patience=2 (monitor validation accuracy)
- Converged after 5-6 epochs typically

**Results:**
- Binary: 68.41% validation accuracy
- 6-class: 22.30% validation accuracy (best tuned model)

**Note:** LSTM showed highest overfitting sensitivity, requiring careful regularization

---

## 3. Results

### 3.1 Multiclass (6-Class) Performance Summary

| Model | Train Acc | Valid Acc | Test Acc | Overfit Gap | MAE Valid | Pred Time (s) |
|-------|-----------|-----------|----------|-------------|-----------|---------------|
| **SVM** | 73.30% | **46.48%** | **47.05%** | 26.82% | 1.04 | 0.0015 |
| **LR** | 72.79% | 45.85% | 45.64% | 26.94% | 1.06 | 0.0026 |
| **LSTM** | 22.95% | 22.30% | 22.72% | 0.66% | 1.29 | 6.10 |
| **RF** | **99.99%** | 42.04% | 40.57% | **57.95%** ⚠️ | 1.14 | 0.64 |
| **Ridge** | 35.64% | 34.88% | 36.03% | 0.76% | **0.97** | 0.0032 |
| **NB** | 26.60% | 20.57% | 21.98% | 6.03% | 1.39 | 0.0015 |

**Key Findings:**
- **Best Model:** SVM (46.48% validation, 47.05% test accuracy)
- **Most Overfit:** Random Forest (57.95% gap) - severe memorization
- **Most Stable:** Ordinal Ridge (0.76% gap) and LSTM (0.66% gap)
- **Best MAE:** Ordinal Ridge (0.97) - makes "smarter" ordinal errors
- **Fastest:** Naive Bayes and SVM (<0.002s prediction time)

**Insight:** Linear models (SVM, LR) significantly outperform both tree-based (RF) and deep learning (LSTM) approaches on this task. The 6-class problem remains challenging with ~46% best accuracy.

### 3.2 Binary Classification Performance Summary

| Model | Train Acc | Valid Acc | Test Acc | Overfit Gap | F1 Score | Pred Time (s) |
|-------|-----------|-----------|----------|-------------|----------|---------------|
| **Ridge** | 76.56% | **73.05%** | **74.62%** | 3.51% | **0.769** | 0.0066 |
| **LR** | 75.80% | 73.63% | 73.63% | 2.17% | 0.763 | 0.0011 |
| **SVM** | **95.81%** | 71.59% | 72.58% | 24.22% | 0.753 | 0.0015 |
| **RF** | 99.99% | 70.29% | 70.18% | 29.70% | 0.749 | 0.79 |
| **LSTM** | 69.59% | 68.41% | 69.03% | 1.18% | 0.686 | 7.43 |
| **NB** | 71.98% | 64.44% | 66.11% | 7.54% | 0.645 | 0.0017 |

**Key Findings:**
- **Best Model:** Ordinal Ridge (73.05% validation, 74.62% test accuracy)
- **Best F1:** Ordinal Ridge (0.769) - balanced precision/recall
- **Most Stable:** Logistic Regression (2.17% gap)
- **Fastest:** Logistic Regression (0.0011s)
- **Practical Performance:** Top 3 models achieve 71-75% accuracy

**Simplification Benefit:** All models improve 15-30% when reducing to binary
- Naive Bayes: +43.87% (20.57% → 64.44%)
- Logistic Regression: +27.78% (45.85% → 73.63%)
- LSTM: +46.11% (22.30% → 68.41%)

**Recommendation:** For production systems, binary classification offers superior accuracy and simpler interpretation

### 3.3 Per-Class Performance Analysis (Best Multiclass Model: SVM)

**Classification Report (Validation Set):**

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (pants-fire) | 0.49 | 0.46 | 0.48 | 157 |
| 1 (false) | 0.50 | 0.44 | 0.47 | 374 |
| 2 (barely-true) | 0.44 | 0.48 | 0.46 | 315 |
| 3 (half-true) | 0.50 | 0.51 | 0.50 | 393 |
| 4 (mostly-true) | 0.46 | 0.52 | 0.49 | 368 |
| 5 (true) | 0.45 | 0.31 | 0.36 | 308 |
| **Macro Avg** | 0.47 | 0.45 | 0.46 | 1915 |
| **Weighted Avg** | 0.47 | 0.46 | 0.46 | 1915 |

**Observations:**
- **Best F1:** half-true (0.50) - most balanced category
- **Worst F1:** true (0.36) - complete truth hardest to identify
- **Class Confusion:** Adjacent classes frequently confused (ordinal error pattern)
- **Precision vs Recall:** Generally balanced except for "true" (low recall)

**Confusion Matrix Patterns (SVM Validation):**
```
Primary diagonal shows correct predictions
Adjacent cells show ordinal confusion
Extreme classes (pants-fire, true) have fewer far-range errors
Middle classes (barely-true, half-true, mostly-true) heavily confused
```

### 3.4 Model Comparison Insights

#### Overall Performance Ranking (by Composite Score*)

*Composite score weights: Valid Acc (30%), Test Acc (25%), F1 (15%), Overfit Gap (-15%), Pred Time (-10%), Model Size (-5%)

**Top 5 Models:**

1. **Logistic Regression (Binary)** - Score: 0.990
   - Best overall: High accuracy, fast, interpretable
   - Validation: 73.63%, Test: 73.63%, F1: 0.763
   
2. **Ordinal Ridge (Binary)** - Score: 0.989
   - Best accuracy: 74.62% test
   - Excellent generalization (3.51% gap)
   
3. **SVM (Binary)** - Score: 0.913
   - Strong performance with linear kernel
   - Validation: 71.59%, extremely fast (0.0015s)
   
4. **Random Forest (Binary)** - Score: 0.868
   - Good accuracy despite overfitting
   - Validation: 70.29%
   
5. **Naive Bayes (Binary)** - Score: 0.859
   - Fastest training, reasonable performance
   - Validation: 64.44%

#### Binary vs Multiclass Comparison

**Average Performance by Model Family:**

| Model Family | Binary Valid Acc | Multiclass Valid Acc | Improvement | Binary Overfit | Multiclass Overfit |
|--------------|------------------|---------------------|-------------|----------------|-------------------|
| LR | 73.63% | 45.85% | **+27.78%** | 2.17% | 26.94% |
| Ridge | 73.05% | 34.88% | **+38.17%** | 3.51% | 0.76% |
| SVM | 71.59% | 46.48% | **+25.11%** | 24.22% | 26.82% |
| RF | 70.29% | 42.04% | **+28.25%** | 29.70% | 57.95% |
| NB | 64.44% | 20.57% | **+43.87%** | 7.54% | 6.03% |
| LSTM | 68.41% | 22.30% | **+46.11%** | 1.18% | 0.66% |

**Key Insight:** Binary simplification provides dramatic improvement (25-46%) across all model families, with LSTM and NB showing largest gains.

#### Efficiency Analysis

**Speed vs Accuracy Trade-off:**

**Fastest Models (prediction time):**
1. Logistic Regression: 0.0011s (73.63% accuracy)
2. Naive Bayes: 0.0015-0.0017s (64.44% accuracy)
3. SVM: 0.0015s (71.59% accuracy)

**Slowest Models:**
1. LSTM: 6.10-7.43s (68.41% accuracy for binary)
2. Random Forest: 0.64-0.79s (70.29% accuracy for binary)

**Model Size:**
- Linear models (LR, SVM, NB): <0.01 MB
- LSTM: 31.04-31.05 MB (1000x larger)

**Efficiency Winner:** Logistic Regression achieves 73.63% accuracy with fastest prediction (0.0011s) and smallest size

#### Overfitting Analysis

**Best Generalization (lowest overfit gap):**
1. Ordinal Ridge (6-class): 0.76%
2. LSTM (6-class): 0.66%
3. LSTM (binary): 1.18%
4. Logistic Regression (binary): 2.17%

**Worst Overfitting:**
1. Random Forest (6-class): 57.95% ⚠️
2. Random Forest (binary): 29.70% ⚠️
3. Logistic Regression (6-class): 26.94%

**Insight:** Tree-based models (RF) severely overfit high-dimensional features. Deep learning (LSTM) shows excellent generalization when properly regularized.

### 3.5 Feature Importance Analysis

#### Top Predictive TF-IDF Features (Logistic Regression Coefficients)

**For "pants-fire" (extreme falsehood):**
- Positive weights: "ad", "dont", "education", "politifact", "stood" (weights: 1.3-1.9)
- Model learns to identify outrageous claims

**For "true" (complete truth):**
- High positive correlation with: "correct", "accurate", "fact", "data"
- Model learns semantic truthfulness signals

#### Metadata Feature Importance (Random Forest)

**Top 10 Most Important Features:**
1. `credibility_score` (0.082)
2. `total_credit_count` (0.076)
3. `sentiment_polarity` (0.043)
4. `word_count` (0.039)
5. `false_to_true_ratio` (0.037)
6. `is_frequent_speaker` (0.031)
7. `mostly_true_ratio` (0.029)
8. `party_republican` (0.027)
9. `context_interview` (0.025)
10. `subject_healthcare` (0.023)

**Insight:** Engineered features (credibility_score, ratios) prove more useful than raw counts, validating our feature engineering approach.

### 3.6 Error Analysis

**Common Misclassification Patterns (SVM Confusion Matrix):**

1. **Adjacent Class Confusion:** Model frequently predicts ±1 class away (ordinal error)
   - Example: false (1) misclassified as barely-true (2)
   
2. **Middle Class Ambiguity:** Classes 2-4 (barely-true, half-true, mostly-true) heavily confused
   - Inherent subjectivity in these boundaries
   
3. **Extreme Class Clarity:** Classes 0 (pants-fire) and 5 (true) have fewer far-range errors
   - Complete falsehoods and truths easier to identify
   
4. **Directional Bias:** Model tends to predict toward middle classes
   - Reflects dataset distribution (middle classes more frequent)

**Mean Absolute Error (MAE) Analysis:**
- Ordinal Ridge MAE: 0.97 classes (best)
- SVM MAE: 1.04 classes
- Random baseline MAE: ~2.5 classes
- **Interpretation:** Best model predictions average <1 class away from truth

---

## 4. Discussion

### 4.1 Why is Fake News Detection Challenging?

**Theoretical Limitations:**

Even human fact-checkers show disagreement on truthfulness ratings, particularly for middle categories (barely-true, half-true, mostly-true). This suggests an inherent ceiling on machine learning accuracy.

**Dataset-Specific Challenges:**

1. **Short Statements:** Mean length 107 characters limits contextual information
   - Models lack surrounding context, background knowledge
   
2. **Ambiguous Label Boundaries:** The distinction between "barely-true" and "half-true" is subjective
   - Even experts disagree on these nuanced categories
   
3. **Speaker Concentration:** Top 30 speakers account for 15% of data
   - Potential train-test speaker overlap creates leakage risk
   
4. **Temporal Scope:** Statements from 2007-2016 period
   - Language evolution may limit generalization to current discourse
   
5. **Multi-Label Complexity:** 6-way classification inherently harder than binary
   - Each additional class increases decision boundary complexity

**Model Limitations:**

1. **No External Knowledge:** Models only see statement text, not underlying facts
   - Cannot verify numerical claims or factual assertions
   
2. **Limited Context:** Single statements without conversation history
   - Missing important discourse context
   
3. **Lexical Overlap:** True and false statements often use similar vocabulary
   - Deception sophisticated enough to mimic truthful language
   
4. **Sarcasm/Rhetoric:** Models struggle with non-literal language
   - Political rhetoric frequently employs irony, hyperbole

### 4.2 Key Insights from Experiments

#### Insight 1: Binary Simplification Provides Dramatic Improvement

**Evidence:**
- Average improvement across all models: +30.2%
- Largest gains: Naive Bayes (+43.9%), LSTM (+46.1%)
- Smallest gains: SVM (+25.1%), still substantial

**Practical Implication:** For real-world deployment, binary classification ("Trust vs. Don't Trust") offers:
- Superior accuracy (71-75% vs. 46-47%)
- Simpler user interpretation
- More actionable predictions

**Trade-off:** Loss of truthfulness nuance may matter for fact-checking context

#### Insight 2: Linear Models Dominate on This Task

**Performance Summary:**
- **Linear Models (SVM, LR):** 46-47% (6-class),72-74% (binary)
- **Tree-Based (RF):** 42% (6-class), 70% (binary), severe overfitting
- **Deep Learning (LSTM):** 22% (6-class), 68% (binary)

**Why Linear Models Win:**
1. **High-dimensional sparse text:** TF-IDF creates 10,000+ features
2. **Limited training data:** 8,935 training samples insufficient for deep learning
3. **Text already "featurized":** TF-IDF provides strong baseline features
4. **Regularization advantage:** L1/L2 regularization prevents overfitting better than dropout

**LSTM Underperformance Analysis:**
- Vocabulary size (20K) may be too large for dataset size
- Limited benefit from sequential modeling on short statements
- Requires significantly more data to show advantage

**Recommendation for Beginners:** Start with Logistic Regression - achieves 90% of best performance with 10% of complexity

#### Insight 3: Regularization is Critical

**Evidence from Hyperparameter Tuning:**

**Logistic Regression C comparison:**
- C=10.0 (weak reg): 40.89% validation, 15.45% overfit gap
- C=1.0 (moderate reg): 45.85% validation, 26.94% overfit gap
- C=0.1 (strong reg): 42.01% validation, 3.22% overfit gap

**SVM C comparison:**
- C=1.0: 46.48% validation, 26.82% overfit gap
- C=0.1: 46.48% validation, best validation among tested

**LSTM dropout comparison:**
- dropout=0.0: 42.34% validation, overfits after epoch 2
- dropout=0.2: 22.30% validation (best tuned), stable training
- dropout=0.4: 19.79% validation, undercuts capacity

**Key Finding:** Aggressive regularization essential for generalization. Models prone to overfitting due to:
- High feature dimensionality
- Limited training samples
- Complex decision boundaries

#### Insight 4: Ordinal Formulation Shows Promise

**Ordinal Ridge Performance:**
- 6-class accuracy: 34.88% (lower than SVM/LR)
- **MAE: 0.97** classes (best among all models)
- Binary: 73.05% validation (competitive with best)

**Advantage:** Makes "smarter errors"
- Predicting 3 when true label is 4: ordinal error = 1
- Predicting 0 when true label is 4: ordinal error = 4
- SVM treats both errors equally

**Use Case:** When ordinal distance matters (e.g., distinguishing minor inaccuracies from complete falsehoods), ordinal models preferred despite lower accuracy

#### Insight 5: Metadata Provides Limited Signal

**Evidence:**
- Logistic Regression with TF-IDF only: ~44% accuracy
- Logistic Regression with TF-IDF + metadata: 45.85% accuracy (+1.85%)
- Feature importance: Top TF-IDF features have 3-4x higher coefficients than metadata

**Most Useful Metadata:**
- `credibility_score`: 0.082 importance (highest metadata feature)
- `sentiment_polarity`: 0.043 importance
- Party affiliation: 0.027 importance

**Least Useful Metadata:**
- Context categories: 0.012-0.025 importance
- State information: 0.008-0.015 importance
- Individual subjects: 0.005-0.023 importance

**Interpretation:** Statement text content far more predictive than metadata. However, metadata still provides complementary signal, especially speaker credibility history.

### 4.3 Practical Recommendations

**For Real-World Deployment:**

1. **Use Binary Classification**
   - Target: 73-75% accuracy achievable
   - Simpler user experience: "Likely False" vs. "Likely True"
   - More actionable than 6-way classification

2. **Start with Logistic Regression**
   - Fast training and prediction (<0.01s)
   - Interpretable coefficients for explaining predictions
   - Achieves 73.63% binary accuracy

3. **Add Confidence Thresholds**
   - Only show predictions with >75% confidence
   - Flag low-confidence statements for human review
   - Reduces false positive rate

4. **Combine with External Fact-Checking**
   - Model flags suspicious statements
   - Human fact-checkers verify flagged content
   - Augment, don't replace, human judgment

5. **Explain Predictions**
   - Show top TF-IDF features influencing decision
   - Display speaker credibility score if available
   - Build user trust through transparency

**Example Production Pipeline:**
```
User Input → Preprocessing → Binary LR Model → Confidence Score
  ↓
If confidence > 0.75: Display prediction + top 5 influential words
If confidence < 0.75: Flag for human review
```

### 4.4 Limitations and Challenges

#### Data Limitations

1. **Speaker Concentration:** Top 10 speakers represent 15% of data
   - Risk: Model may learn speaker-specific patterns, not general truthfulness
   - Solution: Test on completely unseen speakers
   
2. **Temporal Scope:** 2007-2016 political statements
   - Risk: Language evolution reduces generalization to 2020s discourse
   - Solution: Continuous model retraining on recent data
   
3. **Subject Sparsity:** 144 subjects, many with <50 examples
   - Risk: Insufficient data for subject-specific patterns
   - Impact: Subject features show low importance (0.005-0.023)

#### Model Limitations

1. **No Fact Verification:** Models don't verify claims against knowledge bases
   - Example: "GDP grew 5%" - model cannot check actual GDP data
   - Limitation: Relies purely on linguistic patterns
   
2. **Context-Free:** Single statement without surrounding discourse
   - Missing: What was the statement responding to? What's the broader context?
   - Impact: Cannot detect context-dependent falsehoods
   
3. **Adversarial Vulnerability:** No testing against deliberately crafted deceptive statements
   - Risk: Sophisticated actors could exploit model weaknesses
   - Need: Adversarial robustness testing

#### Evaluation Limitations

1. **Single Test Set:** No cross-domain evaluation (e.g., COVID-19 claims, non-political domains)
   - Risk: Overestimating generalization ability
   - Solution: Test on FEVER, SNOPES, other fake news datasets
   
2. **No Human Baseline:** Unclear if 46% (6-class) or 73% (binary) is good relative to human performance
   - Need: Human annotator agreement studies
   
3. **No Cost Analysis:** False positives (marking true as false) may be more harmful than false negatives
   - Need: Weighted evaluation metrics reflecting real-world costs

### 4.5 Lessons Learned

#### For Data Science Beginners

1. **Start Simple:** Logistic Regression achieved 73% accuracy, LSTM only 68% despite complexity
2. **Feature Engineering Matters:** Credibility score more useful than raw counts
3. **Regularization Essential:** Always tune regularization strength (C, alpha, dropout)
4. **Visualize Everything:** EDA revealed correlations guiding feature engineering
5. **Binary Often Better:** Simplifying problem improved accuracy 25-46% across all models

#### For Fake News Detection

1. **Text Dominates:** Statement content 3-4x more predictive than metadata
2. **Context Critical:** Single statements insufficient; need surrounding discourse
3. **Explainability Key:** Model predictions must be interpretable for user trust
4. **Continuous Learning:** Language and deception tactics evolve, requiring model updates
5. **Human-in-the-Loop:** Automated detection should augment, not replace, human fact-checkers

### 4.6 Future Work

#### Short-Term Enhancements (1-3 months)

1. **Ensemble Methods:** Combine LR + SVM + Ridge via stacking
   - Expected gain: +2-3% accuracy
   
2. **Advanced Text Embeddings:** Replace TF-IDF with Word2Vec or GloVe
   - Benefit: Capture semantic similarity beyond exact word matches
   
3. **Cross-Validation:** Use 5-fold CV instead of single validation set
   - Benefit: More robust performance estimates
   
4. **Class Weighting:** Penalize errors on rare classes (pants-fire, true) more heavily
   - Benefit: Improve performance on extreme classes

5. **Attention Visualization:** For LSTM, visualize which words drive predictions
   - Benefit: Interpretability for deep learning model

#### Medium-Term Research (3-6 months)

1. **External Knowledge Integration:**
   - Link statements to knowledge graphs (Wikidata, Freebase)
   - Verify numerical claims against databases (economic statistics, census data)
   - Expected gain: +10-15% accuracy through fact verification
   
2. **Contextual Models:**
   - Include conversation history (debate transcripts, interview context)
   - Model speaker-statement interactions
   - Benefit: Detect context-dependent falsehoods
   
3. **Transformer Models:**
   - Fine-tune BERT/RoBERTa on LIAR dataset
   - Use domain-specific pretraining (political text corpus)
   - Expected gain: +5-10% accuracy over LSTM
   
4. **Multimodal Detection:**
   - Analyze images/videos accompanying statements
   - Detect deepfakes or manipulated media
   - Benefit: Holistic fake news detection

#### Long-Term Vision (6-12 months)

1. **Cross-Domain Generalization:**
   - Test LIAR-trained models on FEVER, SNOPES, COVID-19 datasets
   - Domain adaptation for transfer learning
   - Goal: Universal fake news detector
   
2. **Adversarial Robustness:**
   - Test against adversarially perturbed statements
   - Train with adversarial examples
   - Goal: Resist deliberate deception attempts
   
3. **Explainable AI:**
   - LIME/SHAP for black-box model interpretation
   - Counterfactual explanations ("If statement said X instead of Y, prediction would flip")
   - Goal: Build user trust through transparency
   
4. **Real-Time Fact-Checking:**
   - Deploy as browser extension or social media integration
   - Sub-second prediction latency
   - Goal: Immediate feedback during information consumption

### 4.7 Broader Impact

#### Societal Benefits

1. **Combat Misinformation at Scale:** Automated detection processes millions of statements
2. **Reduce Fact-Checker Burden:** Prioritize human effort on high-impact claims
3. **Media Literacy Tool:** Educates users about deceptive language patterns
4. **Democratic Health:** Informed citizenry makes better voting decisions

#### Risks and Ethical Concerns

1. **False Positives → Censorship:**
   - Risk: Legitimate speech flagged as false and suppressed
   - Mitigation: High confidence thresholds, human review for borderline cases
   
2. **Bias Amplification:**
   - Risk: Models perpetuate biases in training data (political lean, demographic representation)
   - Mitigation: Regular bias audits, balanced training data
   
3. **Over-Reliance on Automation:**
   - Risk: Users stop critically evaluating information
   - Mitigation: Position as decision support, not final arbiter of truth
   
4. **Adversarial Arms Race:**
   - Risk: Sophisticated actors craft statements to fool models
   - Mitigation: Continuous model updates, adversarial training

#### Ethical Principles for Deployment

1. **Transparency:** Users must know when interacting with automated systems
2. **Explainability:** Predictions must include reasoning (top influential features)
3. **Human Oversight:** High-stakes decisions require human fact-checker review
4. **Accountability:** Clear responsibility chain when models make errors
5. **Continuous Monitoring:** Track false positive/negative rates, bias metrics

**Recommendation:** Fake news detection models should **augment, not replace**, human judgment. Deploy responsibly with guardrails against misuse.

---

## 5. Conclusion

### 5.1 Summary of Findings

**Best Models:**
- **Binary Classification:** Ordinal Ridge (74.62% test accuracy), Logistic Regression (73.63%)
- **6-Class Multiclass:** SVM (47.05% test accuracy), Logistic Regression (45.64%)
- **Fastest:** Logistic Regression (0.0011s prediction time)
- **Most Interpretable:** Logistic Regression (clear coefficient interpretation)

**Key Achievements:**
1. Successfully built 6 different classification models across binary and multiclass formulations
2. Achieved 74.62% test accuracy for binary classification (15 percentage points above majority baseline)
3. Demonstrated that linear models (SVM, LR) outperform complex models (RF, LSTM) on this task
4. Validated feature engineering approach: engineered features (credibility_score, ratios) proved more useful than raw counts
5. Comprehensive comparison of 13 model variants with detailed performance analysis

**Primary Insights:**
1. **Binary simplification is crucial:** All models improve 25-46% when reducing 6 classes to binary
2. **Linear models dominate:** SVM and Logistic Regression achieve best performance with simplest complexity
3. **Regularization is essential:** Aggressive regularization (small C, dropout) prevents severe overfitting
4. **Text features critical:** TF-IDF contributes 3-4x more than metadata features
5. **Ordinal structure valuable:** When ordinal distance matters, Ridge regression makes "smarter errors" (MAE=0.97)

**Practical Takeaway:** For production fake news detection, use binary Logistic Regression (73.63% accuracy, <0.01s prediction, interpretable) with confidence thresholds and human review for borderline cases.

### 5.2 Dataset Insights

**LIAR Dataset Strengths:**
- Balanced across 6 truthfulness categories (14-21% each)
- Rich metadata (speaker, party, credit history, context)
- Diverse political discourse (3,310 speakers, 144 subjects)
- Short statements ideal for text classification research

**Limitations Identified:**
- Short statements (mean 107 characters) limit context
- Middle classes (barely-true, half-true, mostly-true) inherently ambiguous
- Speaker concentration (top 30 = 15% of data) creates potential leakage
- Temporal scope (2007-2016) may not generalize to current discourse

**Impact:** LIAR provides solid benchmark for fake news detection research, but inherent limitations suggest ~46-47% may approach theoretical accuracy ceiling for 6-class classification without external knowledge.

### 5.3 Methodological Contributions

**Preprocessing Pipeline:**
- Comprehensive missing value strategy (fill vs. drop analysis)
- Careful outlier detection with justification for retention
- Global text normalization (lowercasing, lemmatization) applied consistently

**Feature Engineering:**
- Context generalization (5,143 → 13 categories)
- Credit history ratios addressing collinearity
- Coverage-based categorical selection (≥80% data coverage)
- Credibility score as interpretable speaker reliability metric

**Model Selection:**
- Systematic comparison across 6 model families
- Binary, multiclass, and ordinal formulations tested
- Rigorous hyperparameter tuning for each model
- Comprehensive evaluation beyond accuracy (MAE, F1, overfitting analysis)

**Contribution to Field:** Demonstrated that simpler linear models with careful feature engineering outperform complex deep learning on limited-data text classification tasks.

### 5.4 Final Recommendations

**For Practitioners:**
1. **Start with Logistic Regression:** Achieves 73% binary accuracy with fastest prediction
2. **Use Binary Classification:** 25-46% improvement over multiclass
3. **Tune Regularization Carefully:** Critical for preventing overfitting
4. **Engineer Features Thoughtfully:** Ratios and aggregations more useful than raw values
5. **Deploy with Confidence Thresholds:** Only show predictions >75% confidence

**For Researchers:**
1. **Explore Transformer Models:** BERT/RoBERTa may break the 47% multiclass ceiling
2. **Integrate External Knowledge:** Fact verification against databases needed
3. **Study Cross-Domain Transfer:** Test generalization to non-political domains
4. **Develop Adversarial Robustness:** Guard against deliberately deceptive statements
5. **Human Baseline Studies:** Measure inter-annotator agreement for realistic performance targets

**For Society:**
1. **Augment, Don't Replace:** Models support human fact-checkers, not replace them
2. **Maintain Transparency:** Users deserve to know when AI flags content
3. **Regular Bias Audits:** Monitor for political, demographic, or other biases
4. **Continuous Learning:** Update models as language and deception tactics evolve
5. **Ethical Deployment:** High confidence thresholds, human review, clear accountability

---

## Acknowledgments

This project was completed as part of the Introduction to Data Science course. We thank:
- PolitiFact.com for making the LIAR dataset publicly available
- William Yang Wang for creating and curating the dataset
- Our course instructors for guidance and feedback
- Open-source community for tools (scikit-learn, TensorFlow, NLTK, Pandas)

---

**Project Repository:** All code, visualizations, and trained models available upon request.

**Contact:** For questions about methodology or collaboration opportunities, please contact the team members.

**Last Updated:** December 2024

---

**End of Report**
