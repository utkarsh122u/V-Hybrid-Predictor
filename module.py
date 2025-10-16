import pandas as pd

# ---------- Helper: safe string comparison ----------
def _safe_text_equal(a, b):
    """Return 0 if same, 1 if different, None if both missing."""
    a = "" if pd.isna(a) else str(a).strip().lower()
    b = "" if pd.isna(b) else str(b).strip().lower()
    if a == "" and b == "":
        return None
    return 0.0 if a == b else 1.0


# ---------- Helper: Jaccard distance for disease resistance ----------
def _jaccard_distance_from_str(a, b):
    """Compute Jaccard distance between two comma-separated resistance gene lists."""
    if pd.isna(a) and pd.isna(b):
        return 0.0
    if pd.isna(a) or pd.isna(b):
        return 0.5  # one missing → neutral difference

    sa = {x.strip().lower() for x in str(a).split(",") if x.strip()}
    sb = {x.strip().lower() for x in str(b).split(",") if x.strip()}
    if not sa and not sb:
        return 0.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    jaccard_similarity = intersection / union
    return 1 - jaccard_similarity  # distance = 1 - similarity


# ---------- Main function ----------
def add_genetic_distance(df,
                         w_numeric=0.5,
                         w_categorical=0.35,
                         w_disease=0.15,
                         verbose=False):
    """
    Compute a combined genetic distance score for each row in the pairwise DataFrame.
    Each row should have Parent_A_* and Parent_B_* columns.
    Returns the same DataFrame with added columns:
      - genetic_numeric
      - genetic_categorical
      - genetic_disease
      - genetic_distance
    """

    # ---- numeric feature pairs ----
    numeric_pairs = [
        ('Parent_A_Fruit_Weight_g', 'Parent_B_Fruit_Weight_g'),
        ('Parent_A_Brix', 'Parent_B_Brix'),
        ('Parent_A_Yield_kg_per_plant', 'Parent_B_Yield_kg_per_plant'),
        ('Parent_A_Flowering_Time_days', 'Parent_B_Flowering_Time_days'),
        ('Parent_A_Drought_Tolerance_num', 'Parent_B_Drought_Tolerance_num'),
        ('Parent_A_Heat_Tolerance_num', 'Parent_B_Heat_Tolerance_num'),
    ]
    numeric_pairs = [(a, b) for a, b in numeric_pairs if a in df.columns and b in df.columns]

    numeric_dists = []
    for a, b in numeric_pairs:
        vals = pd.concat([df[a].astype(float), df[b].astype(float)], axis=0)
        mn, mx = vals.min(skipna=True), vals.max(skipna=True)
        rng = (mx - mn) if (mx - mn) != 0 else 1.0

        dist = (df[a] - df[b]).abs() / rng
        both_na = df[a].isna() & df[b].isna()
        one_na = df[a].isna() ^ df[b].isna()

        dist.loc[both_na] = 0.0
        dist.loc[one_na] = 0.5
        numeric_dists.append(dist.clip(0, 1))

    genetic_numeric = pd.concat(numeric_dists, axis=1).mean(axis=1) if numeric_dists else pd.Series(0.0, index=df.index)

    # ---- categorical feature pairs ----
    categorical_pairs = [
        ('Parent_A_Species/Type', 'Parent_B_Species/Type'),
        ('Parent_A_Ploidy', 'Parent_B_Ploidy'),
        ('Parent_A_Fruit_Shape', 'Parent_B_Fruit_Shape'),
        ('Parent_A_Fruit_Color', 'Parent_B_Fruit_Color'),
        ('Parent_A_Leaf_Type', 'Parent_B_Leaf_Type'),
        ('Parent_A_Growth_Habit', 'Parent_B_Growth_Habit'),
        ('Parent_A_Pollination_Type', 'Parent_B_Pollination_Type'),
        ('Parent_A_Soil_Pref', 'Parent_B_Soil_Pref'),
        ('Parent_A_Season', 'Parent_B_Season'),
        ('Parent_A_Self_Compat', 'Parent_B_Self_Compat')
    ]
    categorical_pairs = [(a, b) for a, b in categorical_pairs if a in df.columns and b in df.columns]

    cat_dists = []
    for a, b in categorical_pairs:
        sa = df[a].fillna("").astype(str).str.strip().str.lower()
        sb = df[b].fillna("").astype(str).str.strip().str.lower()
        both_empty = (sa == "") & (sb == "")
        one_missing = ((sa == "") ^ (sb == ""))

        dist = (sa != sb).astype(float)
        dist.loc[both_empty] = 0.0
        dist.loc[one_missing] = 0.5
        cat_dists.append(dist)

    genetic_categorical = pd.concat(cat_dists, axis=1).mean(axis=1) if cat_dists else pd.Series(0.0, index=df.index)

    # ---- disease feature ----
    dis_a, dis_b = None, None
    for colpair in [('Parent_A_Disease_Resistance', 'Parent_B_Disease_Resistance'),
                    ('Parent_A_Disease', 'Parent_B_Disease')]:
        if colpair[0] in df.columns and colpair[1] in df.columns:
            dis_a, dis_b = colpair
            break

    if dis_a and dis_b:
        genetic_disease = df.apply(lambda r: _jaccard_distance_from_str(r[dis_a], r[dis_b]), axis=1)
    else:
        genetic_disease = pd.Series(0.0, index=df.index)

    # ---- combine weighted ----
    total_w = w_numeric + w_categorical + w_disease
    w_num, w_cat, w_dis = (w_numeric / total_w, w_categorical / total_w, w_disease / total_w)

    genetic_distance = (w_num * genetic_numeric +
                        w_cat * genetic_categorical +
                        w_dis * genetic_disease).clip(0, 1)

    # ---- attach to DataFrame ----
    df['genetic_numeric'] = genetic_numeric
    df['genetic_categorical'] = genetic_categorical
    df['genetic_disease'] = genetic_disease
    df['genetic_distance'] = genetic_distance

    if verbose:
        print(f"Numeric pairs: {numeric_pairs}")
        print(f"Categorical pairs: {categorical_pairs}")
        print(f"Disease columns: {(dis_a, dis_b)}")
        print(f"Weights (num, cat, dis): {w_num:.2f}, {w_cat:.2f}, {w_dis:.2f}")

    return df

last_msg = """
### 🔬 Overview  
This project is an **AI-driven Hybrid Prediction System** built to revolutionize how plant scientists, breeders, and geneticists identify and evaluate **potential hybrids between two plant species or varieties**.

It uses the combined power of **Machine Learning, Deep Learning, and Bioinformatics** to accurately predict:
- Whether two parent plants can form a **viable hybrid**
- The **expected yield and fruit count per flower**
- Key **phenotypic and genomic outcomes**
- And **scientific feedback** highlighting predicted improvements in hybrid performance

In short, it transforms **traditional guesswork-based hybridization** into a **data-backed predictive science** — saving years of fieldwork and experimentation.

---

### 🌾 How It Works  
The model takes in a set of scientifically measurable parental traits such as:
🌸 Flower size, 🌿 Ovule count, 🌼 Pollen viability, 🌺 Flowering time, and 🧬 Genetic distance between parents.

Using trained **neural network ensembles (ANN + ML hybrid models)**, it learns the underlying relationships between these parameters and real hybrid outcomes.

It then predicts:
- ✅ **Hybrid viability** — whether the cross will succeed
- 🍅 **Expected fruit yield and quality per flower**
- 🌱 **Potential improvements** in plant vigor, fertility, and resilience
- 🧬 **Compatibility insights** between parental genomes and phenotypes

The system captures **complex non-linear relationships** hidden within genetic and morphological data, enabling breeders to visualize biological outcomes with AI precision.

---

- >Traditional: Soil Testing → Field Trial (Years) → Breeding → Validation (~1yr min.)
- >AI Hybrid: Data → Prediction → Validation → Deployment (~2 months)

---
### 🌍 Why It Matters  
Hybrid breeding forms the backbone of **global agriculture and food production** — yet it still heavily depends on **trial-and-error methods** that waste **time, land, and resources**.

Every failed cross means:
- ⛔ **Months of field effort lost**
- 🌾 **Land and resource wastage** on unviable hybrids
- 💸 **Economic losses** due to uncertain outcomes

This model introduces a **data-driven decision layer** that empowers researchers to:
- Predict hybrid success **before physical trials**
- Select **high-potential parent combinations** with confidence
- Understand **compatibility bottlenecks** at both genetic and phenotypic levels
- **Accelerate breeding programs** for crops like tomato, rice, maize, and wheat

Essentially, it helps **save land, time, and cost** — while driving **innovation in crop improvement and sustainability**.

---

### 🚀 Future Scope  
The current version predicts hybrid compatibility and yield outcomes using morphological and genetic inputs.
In upcoming versions, the system aims to evolve into a **complete AI-assisted breeding assistant** by:
- 🔬 Integrating **multi-omics data** (transcriptomics, metabolomics, and epigenetics) for deeper trait prediction.
- 🌿 Expanding to **cross-species hybridization** using phylogenetic and synteny datasets.
- 🖼️ Including **real-time phenotype prediction** from flower or fruit images using CNNs.
- 💻 Offering **interactive simulation tools** where scientists can virtually cross parents and visualize expected hybrid traits.
- 🔗 Connecting to **genomic databases** (e.g., Ensembl Plants, Sol Genomics Network) for automatic feature extraction.

This will make the system a **one-stop AI platform** for hybrid design, analysis, and genetic innovation.

---

### 🌟 Impact  
By combining AI with plant science, this project lays the groundwork for **smart hybrid design** — turning traditional agriculture into a **predictive and sustainable science**.

Its real-world impact includes:
- ⏱️ Reducing breeding cycle time by **over 50%**
- 💧 Minimizing land and resource wastage
- 📈 Improving yield prediction accuracy
- 🧠 Enabling data-informed decisions for crop research
- 🍅 Contributing to **food security and agricultural resilience** worldwide

---

Developed with ❤️ using Streamlit (FRONTEND) & Deep Learning (ANN) in Python

---
"""

def last_msgs():
    return last_msg

def d_info():
    hybrid_disease_resistance = {
        "Low, Moderate": "Indicates variable or inconsistent disease resistance — some tolerance under mild infection, but susceptible under high disease pressure.",

        "TMV, Moderate": "Shows moderate resistance to Tobacco Mosaic Virus; plants may show mild mosaic symptoms but recover without significant yield loss.",

        "Verticillium, Fusarium, Moderate": "Moderate resistance to soil-borne fungal wilts caused by Verticillium and Fusarium; plants may experience partial yellowing or wilt but survive.",

        "Bacterial wilt, Moderate": "Partial resistance to Bacterial wilt (*Ralstonia solanacearum*); plants show delayed symptoms and lower infection rate.",

        "High, Moderate": "Indicates generally strong but not absolute resistance across diseases — performs well under high pathogen pressure.",

        "Moderate": "Shows medium or partial resistance against general disease spectrum; suitable for moderately infected regions.",

        "Nematodes, Moderate": "Moderate resistance to root-knot nematodes; root galling may occur but without severe yield impact.",

        "Broad resistance genes, Moderate": "Carries multiple moderate-effect genes providing wide but partial protection against several diseases.",

        "Low, TMV": "Low resistance to Tobacco Mosaic Virus — high infection risk leading to leaf mottling and reduced vigor.",

        "Verticillium, Low, Fusarium": "Low resistance to Verticillium and Fusarium wilts; plants highly susceptible under infected soil conditions.",

        "Low, Bacterial wilt": "Low resistance to bacterial wilt; plants likely to show sudden wilting and vascular darkening.",

        "Low, High": "Indicates uneven resistance; strong against some pathogens, but weak against others.",

        "Low, Nematodes": "Low resistance to nematodes; heavy root galling expected, leading to reduced growth.",

        "Low": "Poor disease resistance; easily infected by most pathogens.",

        "Low, Broad resistance genes": "Low overall resistance despite possessing some broad-spectrum genes — genes may be weakly expressed or inactive.",

        "Verticillium, TMV, Fusarium": "Resistance against multiple pathogens (Verticillium, TMV, Fusarium); offers multi-disease protection ideal for variable soil conditions.",

        "TMV": "Strong resistance to Tobacco Mosaic Virus; ideal for solanaceous crops like tomato or pepper.",

        "TMV, Bacterial wilt": "Dual resistance to TMV and Bacterial wilt; reduces viral and bacterial losses significantly.",

        "TMV, High": "High resistance to TMV — plants remain symptom-free under most infection conditions.",

        "TMV, Nematodes": "Resistance to both TMV and nematodes; suitable for virus- and pest-prone fields.",

        "TMV, Broad resistance genes": "Carries broad resistance genes including TMV protection; highly durable resistance across multiple pathogens.",

        "Verticillium, Fusarium": "Resistance to Verticillium and Fusarium wilts; helps maintain vascular health and yield stability.",

        "Verticillium, Bacterial wilt, Fusarium": "Triple resistance to fungal and bacterial wilts; highly valuable combination for soil-borne disease management.",

        "Verticillium, High, Fusarium": "High resistance to Verticillium and Fusarium; strong defense against major soil fungi.",

        "Verticillium, Moderate, Fusarium": "Moderate resistance to both Verticillium and Fusarium; reduces disease impact but not complete immunity.",

        "Verticillium, Nematodes, Fusarium": "Multi-resistance to fungal wilts and nematodes; protects root system and vascular function.",

        "Verticillium, Broad resistance genes, Fusarium": "Broad and durable resistance to Verticillium and Fusarium, driven by polygenic defense mechanisms.",

        "Bacterial wilt, High": "Strong resistance to Bacterial wilt; plants remain healthy under high pathogen presence.",

        "Nematodes, Bacterial wilt": "Dual resistance to nematodes and bacterial wilt; ideal for tropical and subtropical cultivation zones.",

        "Bacterial wilt": "Resistance to Bacterial wilt; reduces plant mortality in infected soils.",

        "Bacterial wilt, Broad resistance genes": "Resistance to bacterial wilt combined with broad-spectrum genes; protects against multiple bacterial and fungal pathogens.",

        "High": "Excellent resistance across most common diseases; ensures long-term stability and yield performance.",

        "Nematodes, High": "Strong resistance to nematodes; roots remain healthy with minimal galling.",

        "Broad resistance genes, High": "Carries multiple high-strength resistance genes, offering long-term, multi-pathogen protection.",

        "Nematodes, Broad resistance genes": "Broad and strong resistance to nematodes and other diseases; supports vigorous root growth and resilience.",

        "Nematodes": "Resistance to root-knot nematodes (*Meloidogyne spp.*); protects root system and nutrient uptake.",

        "High, Broad resistance genes": "Top-tier hybrid resistance — combines high-level resistance with broad gene coverage against multiple diseases.",

        "Broad resistance genes": "Possesses polygenic resistance effective against a wide range of pathogens, ensuring durable protection."
    }
    return hybrid_disease_resistance