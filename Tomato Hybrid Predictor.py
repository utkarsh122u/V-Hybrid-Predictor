import streamlit as st
import torch as t
import joblib as jb
import pandas as pd
import module
from module import add_genetic_distance as agd
import torch.nn as nn
import numpy as np

st.set_page_config(page_icon='🍅', page_title='T Hybrid Research')
avg = pd.DataFrame({
    'Hybrid Weight (g)':[55.00],
    'Yield (KG/plant)':[0.95],
    'Brix (°Bx)':[4.20]
}).T
avg.columns = ['Prediction']

season_conditions = {
    "Summer": {"temp": (30, 40), "humidity": (30, 60), "note": "High heat, low humidity"},
    "Spring/Summer": {"temp": (25, 37), "humidity": (40, 70), "note": "Warm to hot, moderate humidity"},
    "Summer/Fall": {"temp": (28, 36), "humidity": (45, 75), "note": "Warm with mild cooling nights"},
    "Year-round (greenhouse)": {"temp": (24, 29), "humidity": (50, 70), "note": "Controlled environment"}
}

def season_overlaps(sss1, sss2):
    s1_min, s1_max = season_conditions[sss1]['temp']
    s2_min, s2_max = season_conditions[sss2]['temp']
    hybrid_min = max(s1_min, s2_min)
    hybrid_max = min(s1_max, s2_max)
    return hybrid_min, hybrid_max

def humidity(sss1, sss2):
    s1_min, s1_max = season_conditions[sss1]['humidity']
    s2_min, s2_max = season_conditions[sss2]['humidity']
    hybrid_min = max(s1_min, s2_min)
    hybrid_max = min(s1_max, s2_max)
    return hybrid_min, hybrid_max

def hybrid_quality_score(yield_value, brix_value, weight_value):
    yield_score = min((yield_value / 6) * 100, 100)
    brix_score = min((brix_value / 9) * 100, 100)
    weight_score = min((weight_value / 120) * 100, 100)

    HQS = 0.4*yield_score + 0.4*brix_score + 0.2*weight_score
    return round(HQS, 2)

columnso = pd.read_csv('sample.csv').columns.tolist()

if "items" not in st.session_state:
    st.session_state["items"] = []

class ANN(nn.Module):
    def __init__(self, num_f, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_f, 526),
            nn.ReLU(),
            nn.BatchNorm1d(526),
            nn.Linear(526, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.network(x)

model_state = t.load('model.pth', map_location='cpu')
x_prep = jb.load('preprocessor.pkl')
y_prep = jb.load('preprocessor0.pkl')

st.image('Tomato Hybrid Predictor.png', width='stretch')
t0, t4, t1, t2, t3 = st.tabs(['📍 About', '🧠 R/W Problem', '🧬 Genomics', '🌱 Phenomics', '🍅 Hybrid'])
with t0:
    st.markdown('###### 🌾 Major Problems Faced in Agriculture[India]')
    with st.expander("🌾 Karnataka — Problems"):
            st.markdown("""
    **1. Repeated droughts and irregular monsoon**  
    **2. Pest and disease outbreaks in tomato, chilli, and ragi**  
    **3. Small and uneven landholdings**  
    **4. Declining soil organic matter**
    """)

    with st.expander("🌿 Bihar — Problems"):
        st.markdown("""
    **1. Flood-prone and waterlogged areas**  
    **2. Low soil fertility in upland regions**  
    **3. Lack of high-yield, resistant hybrids**  
    **4. Fragmented small farms**  
    """)

    with st.expander("🌾 Chhattisgarh — Problems"):
        st.markdown("""
    **1. Poor and uneven soil fertility**  
    **2. High temperature and drought stress**  
    **3. Heavy pest and nematode attacks**  
    **4. Limited irrigation infrastructure**
    """)

    with st.container(border=10):
        st.markdown("""
        **3 Core Problems Every State Faces:**
        - >**Low or Uneven Soil Fertility 🌱**
        - >**Pest and Disease Susceptibility 🦠**
        - >**Small or Fragmented Land Holdings 🏞️**
        """)
    st.divider()
    st.markdown("#### 🔬 Traditional Scientific Solutions to Key Agricultural Problems")
    with st.expander("1. Low or Uneven Soil Fertility 🌱"):
        st.markdown("""
    **Traditional Approach:**  
    - **Soil testing and nutrient-specific fertilization** → Scientists test the soil and apply the exact fertilizers needed to boost crop growth.
    
    **Challenge:**  
    - Requires repeated testing and can be expensive for small farmers.
    """)

    with st.expander("2. Pest and Disease Susceptibility 🦠"):
        st.markdown("""
    **Traditional Approach:**  
    - **Breeding disease-resistant varieties** → Scientists develop and distribute plant varieties that naturally resist common pests and pathogens.
    
    **Challenge:**  
    - Breeding takes years and may not keep up with evolving pathogens.
    """)

    with st.expander("3. Small or Fragmented Land Holdings 🏞️"):
        st.markdown("""
    **Traditional Approach:**  
    - **High-density planting of improved cultivars** → Using compact, high-yielding varieties to maximize production on small plots.
    
    **Challenge:**  
    - Intensive management is needed, and small plots may still face uneven growth.
    """)
    st.divider()
    st.markdown(module.last_msgs())

def event():
    DROUGHT_MAP = {'Very High': 1.0, 'High': 0.75, 'Medium': 0.5, 'Low': 0.25}
    HEAT_MAP = {'Very High': 1.0, 'High': 0.75, 'Medium': 0.5, 'Low': 0.25}

    Parent_A_Drought_Tolerance_num = DROUGHT_MAP.get(dt1, 0.5)
    Parent_B_Drought_Tolerance_num = DROUGHT_MAP.get(dt2, 0.5)

    Parent_A_Heat_Tolerance_num = HEAT_MAP.get(ht1, 0.5)
    Parent_B_Heat_Tolerance_num = HEAT_MAP.get(ht2, 0.5)

    species_match = int(species1 == species2)
    ploidy_match = int(ploidy1 == ploidy2)
    flowering_diff = abs(ftd1 - ftd2)
    flowering_match = int(flowering_diff <= 10)

    color_match = int(color1 == color2)
    shape_match = int(shape1 == shape2)
    leaf_match = int(leaf1 == leaf2)
    growth_match = int(gh1 == gh2)
    soil_match = int(sp1 == sp2)
    drought_match = int(abs(Parent_A_Drought_Tolerance_num - Parent_B_Drought_Tolerance_num) <= 1)
    heat_match = int(abs(Parent_A_Heat_Tolerance_num - Parent_B_Heat_Tolerance_num) <= 1)

    dfffs = pd.DataFrame([{
        'Parent_A_Fruit_Weight_g': w1,
        'Parent_B_Fruit_Weight_g': w2,
        'Parent_A_Brix': brix1,
        'Parent_B_Brix': brix2,
        'Parent_A_Drought_Tolerance_num': Parent_A_Drought_Tolerance_num,
        'Parent_B_Drought_Tolerance_num': Parent_B_Drought_Tolerance_num,
        'Parent_A_Heat_Tolerance_num': Parent_A_Heat_Tolerance_num,
        'Parent_B_Heat_Tolerance_num': Parent_B_Heat_Tolerance_num,
        'Parent_A_Fruit_Color': color1,
        'Parent_B_Fruit_Color': color2,
        'Parent_A_Disease_Resistance': dr1,
        'Parent_B_Disease_Resistance': dr2
    }])

    df = agd(dfffs, verbose=False)
    coo = df[['genetic_numeric', 'genetic_categorical', 'genetic_disease']].iloc[0]  # pick row 0
    gn, gc, gd = coo['genetic_numeric'], coo['genetic_categorical'], coo['genetic_disease']

    lpa = ['', name1, name2, color_match, shape_match, leaf_match, growth_match, soil_match, drought_match, heat_match,
           species1, ploidy1, w1, shape1, color1, leaf1, gh1, ftd1, pt1, sc1, sp1, s1, dt1,
           ht1, y1, brix1, dr1, Parent_A_Drought_Tolerance_num, Parent_A_Heat_Tolerance_num,
           species2, ploidy2, w2, shape2, color2, leaf2, gh2, ftd2, pt1, sc2, sp2, s2, dt2,
           ht2, y2, brix2, dr2,Parent_B_Drought_Tolerance_num, Parent_B_Heat_Tolerance_num,
           gn, gc, gd, pv1, pc1, pm1, cc, ((fd1+fd2)/2), cl2, sl2, oc2, kinship]

    lpa_df = pd.DataFrame([lpa], columns=columnso)
    if 'Unnamed: 0' in lpa_df.columns:
        lpa_df.drop('Unnamed: 0', axis=1, inplace=True)

    def preprocess(data):
        X = x_prep.transform(data)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X

    x_train_tensor = t.tensor(preprocess(lpa_df), dtype=t.float32)

    input_dim = x_train_tensor.shape[1]

    # 🔹 2. Detect OUTPUT dimension
    # =======================================================
    # Access internal transformers from your y_preprocessor
    num_transformer = y_prep.named_transformers_.get('num')
    cat_transformer = y_prep.named_transformers_.get('cat')

    # Count numeric and categorical encoded outputs
    num_features = 0
    if num_transformer is not None and hasattr(num_transformer, 'mean_'):
        num_features = len(num_transformer.mean_)

    cat_features = 0
    if cat_transformer is not None and hasattr(cat_transformer, 'categories_'):
        cat_features = sum(len(cats) for cats in cat_transformer.categories_)

    output_dim = num_features + cat_features

    model = ANN(input_dim, output_dim)
    model.load_state_dict(model_state)
    model.eval()

    device = next(model.parameters()).device
    X_tensor = x_train_tensor.to(device)

    with t.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    def decode_prediction(y_pred, y_prep, original_y_columns):
        n_num = len(num_transformer.mean_) if num_transformer is not None else 0
        n_cat = sum(len(cats) for cats in cat_transformer.categories_) if cat_transformer is not None else 0

        # Split prediction
        num_part = y_pred[:, :n_num] if n_num > 0 else np.empty((y_pred.shape[0],0))
        cat_part = y_pred[:, n_num:] if n_cat > 0 else np.empty((y_pred.shape[0],0))

        # Decode numeric
        if n_num > 0:
            num_decoded = num_transformer.inverse_transform(num_part)
        else:
            num_decoded = np.empty((y_pred.shape[0],0))

        # Decode categorical
        if n_cat > 0:
            cat_part = np.round(cat_part)  # round to nearest integer
            cat_decoded = cat_transformer.inverse_transform(cat_part)
        else:
            cat_decoded = np.empty((y_pred.shape[0],0))

        # Combine
        decoded = np.concatenate([num_decoded, cat_decoded], axis=1)

        # Assign **original column names**
        decoded_df = pd.DataFrame(decoded, columns=original_y_columns)

        return decoded_df

    original_y_columns = [
        'Viability (%)',                   # 89.00175476074219
        'Hybrid Weight (g)',          # 18.382801055908203
        'Yield (KG/plant)',          # 1.3185882568359375
        'Brix (°Bx)',            # 2.404942512512207
        'Hybrid Color',         # Red
        'Hybrid Shape',   # Plum
        'Leaf',         # Regular
        'Growth Habit',   # Determinate
        'Disease Resistance'                  # Moderate
    ]

    decoded_df = decode_prediction(y_pred, y_prep, original_y_columns)
    min, max = season_overlaps(s1, s2)
    hmin, hmax = humidity(s1, s2)
    with t3:
        decoded_df = decoded_df.T
        decoded_df.columns = ["Prediction"]
        decoded_df.iloc[0, 0] = round(decoded_df.iloc[0, 0], 2)
        decoded_df.iloc[1, 0] = round(decoded_df.iloc[1, 0], 2)
        decoded_df.iloc[2, 0] = round(decoded_df.iloc[2, 0], 2)
        decoded_df.iloc[3, 0] = round(decoded_df.iloc[3, 0], 2)

        d_res = decoded_df.loc['Disease Resistance', 'Prediction']
        d_info = module.d_info()

        if d_res is not None:
            decoded_df.loc['Disease Resistance', 'Prediction'] = d_info.get(d_res)

        if min<max and hmin<hmax:
            decoded_df.loc["Temperature", 'Prediction'] = f"{min}°C - {max}°C"
            decoded_df.loc["Humidity", 'Prediction'] = f"{hmin}% - {hmax}%"
        else:
            decoded_df.loc["Temperature", 'Prediction'] = "⚠️ Parents have non-overlapping temperature preferences."

        dfo = st.dataframe(decoded_df, width='stretch')

        predsss = decoded_df.iloc[[1,2,3], :]

        yield_value = decoded_df.loc['Yield (KG/plant)', 'Prediction']
        brix_value = decoded_df.loc['Brix (°Bx)', 'Prediction']
        v_value = decoded_df.loc['Viability (%)', 'Prediction']
        weight_value = decoded_df.loc['Hybrid Weight (g)', 'Prediction']

        HQS = hybrid_quality_score(yield_value, brix_value, weight_value)
        st.metric("🌟 Hybrid Quality Score", f"{HQS}/100", border=5)

        if v_value >= 70 and yield_value >= 4 and brix_value >= 5.5:
            st.success("""
            🍅 **Outstanding Hybrid Alert!**
        
            This hybrid shows **exceptional performance** both in productivity and flavor quality.  
            - 🌿 **Yield:** Extremely high — ideal for commercial-scale and greenhouse farming.  
            - 🍬 **Brix:** Superb sweetness, indicating rich taste and high market appeal.  
            - ⚗️ **Balance:** High yield *without compromising flavor* — a rare and valuable trait.  
            - 💰 **Good to know:** Such hybrids often command **premium market prices** and can significantly enhance **profit per hectare**.  
        
            ✅ *Overall: A top-tier hybrid with excellent genetic potential for large-scale cultivation and consumer satisfaction.*
            """)

        compare = pd.concat([avg, predsss], axis=1, ignore_index=True)
        compare.columns = ['Average', 'Predicted']
        st.subheader('🔬 Comparative Analysis of Predicted Hybrid Traits vs Standard Tomato Cultivar')
        st.bar_chart(compare, x_label='Features', y_label='Quantitative Values')
        st.divider()
        st.area_chart(compare, x_label='Features', y_label='Quantitative Values')

        if dfo:
            st.session_state['items'].append(dfo)
            with t2:
                st.success('✔️ Checkout Hybrid Panel')
        else:
            st.error('❌ Got Some Error !')

with t1:
    with st.container(border=5):
        col1, col2 = st.columns([1, 1], gap='medium')
    with col1:
        st.markdown('<h6><b>Parent A (M)</b></h6>', unsafe_allow_html=True)
        name1 = st.text_input('Species Name', placeholder='Heinz 1706', key='a1')
        species1 = st.selectbox('Species Type', key='34345', options=['S. lycopersicum (Cultivar/Research)',
                                                                      'S. lycopersicum (Heirloom)',
                                                                      'S. lycopersicum (Processing)',
                                                                      'S. peruvianum (Wild)',
                                                                      'S. lycopersicum (Hybrid)',
                                                                      'S. lycopersicum (Cultivar)',
                                                                      'S. lycopersicum (Cherry/Salad)',
                                                                      'S. pimpinellifolium (Wild)',
                                                                      'S. habrochaites (Wild)',
                                                                      'S. cheesmaniae (Wild)'])
        ploidy1 = st.selectbox('Ploidy', options=['Diploid (2n=24)'], key='aa1')
        dt1 = st.selectbox('Drought Tolerance', options=['Medium', 'High', 'Very High', 'Low'], key='aa2')
        ht1 = st.selectbox('Heat Tolerance', options=['Low', 'High', 'Medium'], key='aa3')
        brix1 = st.number_input('Brix (°Bx)', min_value=2.00, step=0.01, max_value=11.00, key='a2')
        dr1 = st.selectbox('Disease Resistance', key='aa4', options=['Moderate','Low','TMV','Fusarium, Verticillium','Bacterial wilt','High','Nematodes','Broad resistance genes'], accept_new_options=True)
        pv1 = st.slider('Pollen viability (%)', min_value=1, max_value=100, key='aa5')
    with col2:
        st.markdown('<h6><b>Parent B (F)</b></h6>', unsafe_allow_html=True)
        name2 = st.text_input('Species Name', placeholder='Brandywine', key='b1')
        species2 = st.selectbox('Species Type', key=34364, options=['S. lycopersicum (Cultivar/Research)',
                                                                    'S. lycopersicum (Heirloom)',
                                                                    'S. lycopersicum (Processing)',
                                                                    'S. peruvianum (Wild)',
                                                                    'S. lycopersicum (Hybrid)',
                                                                    'S. lycopersicum (Cultivar)',
                                                                    'S. lycopersicum (Cherry/Salad)',
                                                                    'S. pimpinellifolium (Wild)',
                                                                    'S. habrochaites (Wild)',
                                                                    'S. cheesmaniae (Wild)'])
        ploidy2 = st.selectbox('Ploidy', options=['Diploid (2n=24)'])
        dt2 = st.selectbox('Drought Tolerance', options=['Medium', 'High', 'Very High', 'Low'])
        ht2 = st.selectbox('Heat Tolerance', options=['Low', 'High', 'Medium'])
        brix2 = st.number_input('Brix (°Bx)', min_value=2.00, step=0.01, max_value=11.00, key='b2')
        dr2 = st.selectbox('Disease Resistance', options=['Moderate','Low','TMV','Fusarium, Verticillium','Bacterial wilt','High','Nematodes','Broad resistance genes'], accept_new_options=True)
    st.divider()
    cc = st.selectbox('Chromosome count (2n)', options=[24])
    kinship = st.number_input('Kinship/Genetic distance (cM)', step=0.01, min_value=0.00, max_value=1.00, key='b3')

with t2:
    with st.container(border=5):
        c1, c2 = st.columns([1, 1], gap='medium')
    with c1:
        st.markdown('<h6><b>Parent A (M)</b></h6>', unsafe_allow_html=True)
        w1 = st.number_input('Weight(g)', key='a11')
        shape1 = st.selectbox('Shape', key='a22', options=['Plum', 'Flattened/Beefsteak', 'Cherry', 'Pear', 'Round',
                                                           'Beefsteak', 'Elongated', 'Oblong'])
        color1 = st.selectbox('Color', key='a33', options=['Red', 'Pink', 'Purple', 'Dark Purple/Red', 'Yellow', 'Orange',
                                                           'Green', 'Striped'])
        leaf1 = st.selectbox('Leaf Type', key='a44', options=['Regular', 'Potato leaf'])
        gh1 = st.selectbox('Growth Habit', key='a55', options=['Determinate', 'Indeterminate', 'Semi-determinate'])
        ftd1 = st.number_input('Flowering Time days', key='s66', min_value=40, max_value=150)
        pt1 = st.selectbox('Pollination Type', key='a77', options=['Self', 'Insect', 'Insect/Self'])
        sc1 = st.segmented_control('Self Compat', key='a88', options=['Yes', 'No'])
        sp1 = st.selectbox('Soil Preference', key='a99', options=['Rich loam', 'Sandy', 'Well-drained', 'Clay loam', 'Loamy'])
        s1 = st.selectbox('Season', key='a10', options=['Summer', 'Spring/Summer', 'Summer/Fall', 'Year-round (greenhouse)'])
        y1 = st.number_input('Yield (KG/plant)', key='a121', min_value=1, max_value=40)
        fd1 = st.number_input('Flower diameter (mm)', key='a122', min_value=6, max_value=22)
        pc1 = st.number_input('Pollen Count', key='a126', min_value=30000, max_value=120000)
        pm1 = st.selectbox('Pollen Morphology', key='a127', options=['Tricolporate, spheroidal'])
        st.button('✜ Prediction', on_click=event, type='secondary')
    with c2:
        st.markdown('<h6><b>Parent B (F)</b></h6>', unsafe_allow_html=True)
        w2 = st.number_input('Weight(g)', key=1)
        shape2 = st.selectbox('Shape', key=2, options=['Plum', 'Flattened/Beefsteak', 'Cherry', 'Pear', 'Round',
                                                       'Beefsteak', 'Elongated', 'Oblong'])
        color2 = st.selectbox('Color', key=3, options=['Red', 'Pink', 'Purple', 'Dark Purple/Red', 'Yellow', 'Orange',
                                                       'Green', 'Striped'])
        leaf2 = st.selectbox('Leaf Type', key=4, options=['Regular', 'Potato leaf'])
        gh2 = st.selectbox('Growth Habit', key=5, options=['Determinate', 'Indeterminate', 'Semi-determinate'])
        ftd2 = st.number_input('Flowering Time days', key=6, min_value=40, max_value=150)
        sc2 = st.segmented_control('Self Compat', key=8, options=['Yes', 'No'])
        sp2 = st.selectbox('Soil Preference', key=9, options=['Rich loam', 'Sandy', 'Well-drained', 'Clay loam', 'Loamy'])
        s2 = st.selectbox('Season', key=10, options=['Summer', 'Spring/Summer', 'Summer/Fall', 'Year-round (greenhouse)'])
        y2 = st.number_input('Yield (KG/plant)', key=11, min_value=1, max_value=40)
        fd2 = st.number_input('Flower diameter (mm)', key=12, min_value=6, max_value=22)
        cl2 = st.number_input('Corolla length (mm)', key=13, min_value=6, max_value=20)
        sl2 = st.number_input('Style length (mm)', key=14, min_value=6, max_value=12)
        oc2 = st.number_input('Ovule count (per flower)', key=16, min_value=18, max_value=50)
with t3:
    if not st.session_state['items']:
        st.info('☝️ No Prediction Yet !')

with t4:
    st.markdown("""
    ##### 🔍 State Overview:
    - **Region:** Western India  
    - **Major Issue:** Soil-borne wilts (Fusarium, Verticillium), TMV spread, fragmented holdings  
    - **Farming Type:** Small-scale open-field tomato cultivation""")
    st.map({'lat':[19],
            'lon':[75]}, zoom=6.5)
    st.markdown("""
    ##### 🌦 Why **Maharashtra**?
    
    - 🦠 **Disease-prone environment:**  
      High humidity and fluctuating temperatures lead to frequent outbreaks of **Fusarium**, **Verticillium**, and **TMV** — reducing tomato yield and quality.  
    
    - 🌾 **Small & fragmented land holdings:**  
      Most tomato farmers cultivate in **0.5–2 acre plots**, limiting scope for large-scale production and mechanization.  
    
    - 🌱 **Fertile but stressed soil:**  
      Continuous fertilizer overuse has caused **nutrient imbalance and soil fatigue**, lowering long-term productivity.  
    
    - 🍅 **Market-oriented cultivation:**  
      Maharashtra supplies tomatoes to major cities like **Mumbai and Pune** — hence **better taste (high Brix)** directly translates to **higher market value and profit**.  
    """)