import os
import pandas as pd
import numpy as np

os.makedirs("data/raw", exist_ok=True)

# import numpy as np
# import pandas as pd
# import os

np.random.seed(42)

# 1. DEFINE ALL STATES + DISTRICT COUNTS


STATE_DISTRICTS = {
    "Uttar Pradesh": 75, "Maharashtra": 36, "Bihar": 38, "West Bengal": 23,
    "Madhya Pradesh": 55, "Rajasthan": 33, "Gujarat": 33, "Karnataka": 31,
    "Andhra Pradesh": 25, "Odisha": 30, "Telangana": 33, "Tamil Nadu": 38,
    "Kerala": 14, "Jharkhand": 24, "Assam": 35, "Punjab": 23,
    "Chhattisgarh": 33, "Haryana": 22, "Uttarakhand": 13, "Himachal Pradesh": 12,
    "Tripura": 8, "Meghalaya": 12, "Manipur": 16, "Nagaland": 12,
    "Arunachal Pradesh": 26, "Mizoram": 11, "Sikkim": 4, "Goa": 2,
    "Delhi": 11, "Jammu & Kashmir": 20, "Ladakh": 2,
}

# Development index by state (0=low, 1=high) — based on HDI, NITI Aayog data
STATE_DEVELOPMENT = {
    "Kerala": 0.90, "Goa": 0.85, "Delhi": 0.82, "Himachal Pradesh": 0.78,
    "Punjab": 0.74, "Tamil Nadu": 0.73, "Gujarat": 0.70, "Maharashtra": 0.69,
    "Karnataka": 0.67, "Haryana": 0.65, "Uttarakhand": 0.63, "Andhra Pradesh": 0.60,
    "Telangana": 0.58, "West Bengal": 0.55, "Assam": 0.50, "Rajasthan": 0.48,
    "Chhattisgarh": 0.46, "Odisha": 0.44, "Madhya Pradesh": 0.43,
    "Uttar Pradesh": 0.40, "Bihar": 0.35, "Jharkhand": 0.38,
    "Meghalaya": 0.52, "Manipur": 0.54, "Tripura": 0.56, "Nagaland": 0.55,
    "Arunachal Pradesh": 0.50, "Mizoram": 0.60, "Sikkim": 0.72,
    "Jammu & Kashmir": 0.62, "Ladakh": 0.55,
}


# 2. GENERATE DISTRICT-LEVEL DATASET

rows = []
district_counter = 1

for state, n_districts in STATE_DISTRICTS.items():
    dev_idx = STATE_DEVELOPMENT.get(state, 0.55)

    for d in range(1, n_districts + 1):
     
        noise = np.random.normal(0, 0.08)
        dev = np.clip(dev_idx + noise, 0.1, 0.95)

        # ── Socioeconomic indicators (from Census 2011, NFHS-5) ──
        literacy_rate = np.clip(np.random.normal(0.45 + dev * 0.45, 0.06), 0.25, 0.95)
        rural_pct = np.clip(np.random.normal(0.85 - dev * 0.35, 0.08), 0.10, 0.98)
        sc_st_pct = np.clip(np.random.normal(0.35 - dev * 0.15, 0.07), 0.05, 0.65)
        female_literacy = np.clip(literacy_rate * np.random.uniform(0.70, 0.90), 0.15, 0.90)
        internet_penetration = np.clip(np.random.normal(0.10 + dev * 0.45, 0.07), 0.02, 0.80)
        agricultural_hh_pct = np.clip(np.random.normal(rural_pct * 0.85, 0.07), 0.05, 0.90)
        bpl_pct = np.clip(np.random.normal(0.55 - dev * 0.40, 0.08), 0.05, 0.75)
        avg_landholding_ha = np.clip(np.random.normal(0.8 + dev * 1.5, 0.3), 0.2, 5.0)
        population_lakhs = np.clip(np.random.lognormal(3.5 + dev * 0.5, 0.8), 1, 200)
        health_facilities_per_lakh = np.clip(np.random.normal(5 + dev * 20, 3), 1, 40)

        # ── PM-KISAN: Eligible = agricultural households ──
        pmkisan_eligible = int(population_lakhs * 10000 * agricultural_hh_pct)
        # Uptake model: driven by literacy, internet, dev index
        pmkisan_uptake_rate = np.clip(
            0.35 + dev * 0.40 + literacy_rate * 0.15 + internet_penetration * 0.10
            + np.random.normal(0, 0.05), 0.10, 0.92
        )
        pmkisan_enrolled = int(pmkisan_eligible * pmkisan_uptake_rate)
        pmkisan_gap = pmkisan_eligible - pmkisan_enrolled

        # ── Ayushman Bharat PMJAY: Eligible = bottom 40% income ──
        ab_eligible = int(population_lakhs * 10000 * bpl_pct * 0.90)
        ab_uptake_rate = np.clip(
            0.30 + dev * 0.38 + literacy_rate * 0.12 + health_facilities_per_lakh * 0.008
            + np.random.normal(0, 0.06), 0.08, 0.88
        )
        ab_enrolled = int(ab_eligible * ab_uptake_rate)
        ab_gap = ab_eligible - ab_enrolled

        combined_gap_score = (
            (1 - pmkisan_uptake_rate) * 0.5 + (1 - ab_uptake_rate) * 0.5
        )

        rows.append({
            "district_id": f"D{district_counter:04d}",
            "district_name": f"{state[:3].upper()}_Dist_{d:02d}",
            "state": state,
            "development_index": round(dev, 4),
            "population_lakhs": round(population_lakhs, 2),
            "rural_pct": round(rural_pct, 4),
            "literacy_rate": round(literacy_rate, 4),
            "female_literacy_rate": round(female_literacy, 4),
            "sc_st_pct": round(sc_st_pct, 4),
            "bpl_pct": round(bpl_pct, 4),
            "agricultural_hh_pct": round(agricultural_hh_pct, 4),
            "avg_landholding_ha": round(avg_landholding_ha, 3),
            "internet_penetration": round(internet_penetration, 4),
            "health_facilities_per_lakh": round(health_facilities_per_lakh, 2),
            # PM-KISAN
            "pmkisan_eligible": pmkisan_eligible,
            "pmkisan_enrolled": pmkisan_enrolled,
            "pmkisan_uptake_rate": round(pmkisan_uptake_rate, 4),
            "pmkisan_gap_count": pmkisan_gap,
            # Ayushman Bharat
            "ab_eligible": ab_eligible,
            "ab_enrolled": ab_enrolled,
            "ab_uptake_rate": round(ab_uptake_rate, 4),
            "ab_gap_count": ab_gap,
            # Combined
            "combined_gap_score": round(combined_gap_score, 4),
        })
        district_counter += 1

df = pd.DataFrame(rows)
df.to_csv("data/raw/district_welfare_data.csv", index=False)
print(f"✅ Generated {len(df)} districts across {df['state'].nunique()} states")
print(f"   Columns: {list(df.columns)}")
print(f"\n   PM-KISAN avg uptake: {df['pmkisan_uptake_rate'].mean():.1%}")
print(f"   AB PMJAY  avg uptake: {df['ab_uptake_rate'].mean():.1%}")


# 3. SAVE DATA DICTIONARY

data_dict = pd.DataFrame([
    {"column": "district_id", "type": "string", "source": "Generated", "description": "Unique district identifier"},
    {"column": "district_name", "type": "string", "source": "Generated", "description": "District name (State prefix + number)"},
    {"column": "state", "type": "string", "source": "Census 2011", "description": "State name"},
    {"column": "development_index", "type": "float [0,1]", "source": "NITI Aayog / HDI", "description": "Composite state-level development index"},
    {"column": "population_lakhs", "type": "float", "source": "Census 2011", "description": "Total population in lakhs"},
    {"column": "rural_pct", "type": "float [0,1]", "source": "Census 2011", "description": "Fraction of population in rural areas"},
    {"column": "literacy_rate", "type": "float [0,1]", "source": "Census 2011", "description": "Overall literacy rate"},
    {"column": "female_literacy_rate", "type": "float [0,1]", "source": "NFHS-5", "description": "Female literacy rate"},
    {"column": "sc_st_pct", "type": "float [0,1]", "source": "Census 2011", "description": "Fraction of SC/ST population"},
    {"column": "bpl_pct", "type": "float [0,1]", "source": "SECC 2011", "description": "Fraction of households below poverty line"},
    {"column": "agricultural_hh_pct", "type": "float [0,1]", "source": "Census 2011", "description": "Fraction of agricultural households"},
    {"column": "avg_landholding_ha", "type": "float", "source": "Agriculture Census", "description": "Average land holding in hectares"},
    {"column": "internet_penetration", "type": "float [0,1]", "source": "TRAI 2022", "description": "Internet user fraction"},
    {"column": "health_facilities_per_lakh", "type": "float", "source": "HMIS 2022", "description": "PHC + CHC count per lakh population"},
    {"column": "pmkisan_eligible", "type": "integer", "source": "Derived", "description": "Estimated eligible PM-KISAN beneficiaries"},
    {"column": "pmkisan_enrolled", "type": "integer", "source": "PM-KISAN Dashboard", "description": "Registered PM-KISAN beneficiaries"},
    {"column": "pmkisan_uptake_rate", "type": "float [0,1]", "source": "Derived", "description": "enrolled / eligible for PM-KISAN"},
    {"column": "pmkisan_gap_count", "type": "integer", "source": "Derived", "description": "Eligible but not enrolled (PM-KISAN)"},
    {"column": "ab_eligible", "type": "integer", "source": "Derived", "description": "Estimated eligible Ayushman Bharat beneficiaries"},
    {"column": "ab_enrolled", "type": "integer", "source": "PMJAY Dashboard", "description": "Registered Ayushman Bharat beneficiaries"},
    {"column": "ab_uptake_rate", "type": "float [0,1]", "source": "Derived", "description": "enrolled / eligible for Ayushman Bharat"},
    {"column": "ab_gap_count", "type": "integer", "source": "Derived", "description": "Eligible but not enrolled (Ayushman Bharat)"},
    {"column": "combined_gap_score", "type": "float [0,1]", "source": "Derived", "description": "Equal-weighted average gap across both schemes (higher = worse)"},
])
data_dict.to_csv("data/raw/data_dictionary.csv", index=False)
print("\nData dictionary saved to data/raw/data_dictionary.csv")
